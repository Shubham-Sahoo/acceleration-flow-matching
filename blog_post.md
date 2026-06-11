# I built latent diffusion from scratch and here is what broke first

*Building DDPM, DDIM, and Flow Matching on a free Colab GPU — the bugs, the numbers, and what the papers skip.*

---

I wanted to understand generative diffusion models at the level where I could actually modify them. Not use a library. Not copy a notebook. Build the pieces from scratch, watch them fail, and figure out why.

This is that story. Both models work now. The path there was not clean.

---

## What I was building

The goal was a latent diffusion setup that could compare three methods on the same architecture:

- **DDPM** — the original 1000-step denoising process
- **DDIM** — same noise schedule, deterministic sampling in 50 steps
- **Flow Matching** — straight-line paths from noise to data, 20-50 steps

All three share a ViT backbone (~5.5M params) and a spatial VAE that compresses images to a 16-channel 8x8 latent space. Datasets: MNIST and CIFAR10, 32x32. Hardware: Google Colab T4, 12GB VRAM.

The architecture is a `LatentViT`: 12 transformer layers, embed_dim=192, 3 heads, class conditioning through an embedding + MLP projection, time conditioning through a small MLP. Both are added to the token sequence before positional embeddings.

---

## The first sign something was wrong

First DDPM run. Loss on epoch 1: 0.282. Loss on epoch 20: 0.272. Total improvement over 20 epochs: 0.010.

That looks like convergence. It is not convergence. It is a model that learned nothing.

For an untrained DDPM predicting noise, the expected loss is around 1.0. The noise target is N(0,1), so if your model predicts zeros, MSE(zeros, N(0,1)) = 1.0 exactly. My loss was 0.282 from the first batch. Something was clamping the outputs before training even started.

The culprit: Post-LayerNorm with 12 transformer layers.

PyTorch's `TransformerEncoderLayer` applies LayerNorm after each sublayer by default. At depth 12 with dropout=0.1, this compounds. Each residual stream rescales and shrinks. After 12 layers, I measured feature std at 0.008. The output projection then maps near-zeros to near-zeros.

```python
x = model.transformer(x)
print("After transformer:", x.std().item())   # 0.008
x = model.output_proj(x)
print("After output_proj:", x.std().item())   # 0.0001
```

The fix was two lines:

```python
encoder_layer = nn.TransformerEncoderLayer(..., norm_first=True)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12,
                                         norm=nn.LayerNorm(embed_dim))
nn.init.normal_(self.output_proj.weight, std=0.02)
nn.init.zeros_(self.output_proj.bias)
```

Pre-LN normalizes before each sublayer. Feature scale stays stable through all 12 layers. DiT uses it. ViT-22B uses it. The rule: deeper than 8-10 layers, use `norm_first=True` by default.

After the fix, the actual DDPM run (109 epochs, LR=1e-4, batch=256):

| Epoch | Loss | Grad Norm |
|-------|------|-----------|
| 1 | 0.384 | 0.268 |
| 5 | 0.282 | 0.119 |
| 10 | 0.277 | 0.105 |
| 50 | 0.272 | 0.065 |
| 75 (best) | **0.269** | 0.054 |
| 100 | 0.275 | 0.051 |
| 109 | 0.273 | 0.048 |

Epoch 1 starts high and drops fast. That is what a real training signal looks like.

---

## Testing the model wrong

After getting real training curves, I encoded a real MNIST image to a latent, passed it through the DDPM sampler, decoded the result. Output was corrupted noise. I thought the model was broken.

It was not. I was testing it wrong.

DDPM is a denoiser trained on noisy inputs `z_t`. A clean latent `z_0` is out-of-distribution. Of course it produces garbage. The correct generation test is:

```python
z_t = torch.randn(10, 16, 8, 8, device=device)  # start from pure noise
labels = torch.arange(0, 10, device=device)
samples = model_ddpm.sample(z_t, labels, t_steps=1000)
```

Obvious in retrospect. Cost several debugging hours.

---

## The class embedding scale problem

After the norm fix and device fix, training worked but the model mostly ignored class labels. Samples for "3" looked roughly the same as "7".

```python
c_emb = model.class_embed(labels)   # std: 0.0196
t_emb = model.time_embed(t)         # std: 0.2457
```

12.5x scale gap. The transformer sees both signals added to each token and learns to rely on the louder one. Root cause: I manually initialized the class embedding with `std=0.02` to match the output projection. Time embedding used default Kaiming init.

Fix: remove the manual init, add an MLP projection matching the time embedding structure:

```python
self.class_embed = nn.Embedding(10, embed_dim)
self.class_proj  = nn.Sequential(
    nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
)
# In forward:
c_emb = self.class_proj(self.class_embed(class_label))
```

The network compensates through LayerNorm over training so this is not fatal, but starting balanced saves epochs.

---

## What the DDPM loss floor actually means

DDPM plateaued at best loss 0.269 (epoch 75). I kept training expecting further drops. The loss barely moved from epoch 20 onward.

This is not a bug. 74% of the 1000 DDPM timesteps have SNR below 1. For those, a 5.5M parameter model on MNIST genuinely cannot predict below ~0.4-0.5. Weighted across all timesteps, the floor is ~0.27.

Here is what the samples look like at different stages:

![DDPM epoch 5 — fragments](assets/ddpm_samples_epoch5.png)
*Epoch 5, loss 0.2817: disconnected fragments with sharp edges. Background correct but no coherent structure.*

![DDPM epoch 50 — strokes forming](assets/ddpm_samples_epoch50.png)
*Epoch 50, loss 0.2719: connected strokes, loosely digit-like but messy. Class conditioning partially working.*

![DDPM epoch 100 — recognizable but inconsistent](assets/ddpm_samples_epoch100.png)
*Epoch 100, loss 0.2748: most digits recognizable. High variance within each class. Some classes (1, 7) look clean, others (0, 2, 6) are still messy.*

The DDPM denoising trajectory runs backward — t=1.0 is pure noise, t=0.0 is the final sample:

![DDPM trajectory epoch 100](assets/ddpm_trajectory_epoch100.png)

Structure emerges in the middle timesteps and the final steps refine. The path is noisier than FM — expected, since DDPM traverses a stochastic SDE rather than a straight ODE.

---

## Flow Matching

Flow Matching trains the same LatentViT backbone with a different objective. Instead of predicting noise, the model predicts velocity: given a point on a straight line between data and noise, which direction is data?

```
z_t = (1-t) * z_0 + t * z_1       (interpolation)
v   = z_1 - z_0                    (constant velocity target)
```

The velocity does not depend on t. That is the key insight. DDPM's score function changes with every timestep. FM's velocity target is the same constant for a given (z_0, z_1) pair. Simpler regression problem.

At inference, you integrate an ODE:

```
z_{t+dt} = z_t + v * dt
```

Because paths are straight, 50 steps gets you what DDPM needs 1000 steps for.

The straight-line insight comes from optimal transport — the Monge problem from 1781 says straight paths minimize transport cost. Three groups published FM independently in 2022 from different angles and all arrived at the same formulation.

### FM training results (100 epochs, LR=1e-4, batch=256)

![FM dashboard](assets/fm_dashboard.png)

| Epoch | Loss | Grad Norm |
|-------|------|-----------|
| 1 | 1.776 | 0.248 |
| 2 | 1.579 | 0.215 |
| 10 | 1.560 | 0.177 |
| 50 | 1.545 | 0.104 |
| 98 (best) | **1.542** | 0.099 |
| 100 | 1.544 | 0.099 |

FM loss is not comparable to DDPM loss — they measure velocity error vs noise prediction error respectively.

Avg epoch time: ~99s on T4. First epoch was 185s (DataLoader warmup). Grad norm decays from 0.248 to a stable 0.099, smooth the whole way.

### What FM samples look like

![FM epoch 5](assets/fm_samples_epoch5.png)
*Epoch 5, loss 1.5625: blurry blobs, correct background, no digit structure.*

![FM epoch 50](assets/fm_samples_epoch50.png)
*Epoch 50, loss 1.5446: class structure clear. 0, 1, 5 already look good. Others forming.*

![FM epoch 100](assets/fm_samples_epoch100.png)
*Epoch 100, loss 1.5435: all 10 digits recognizable, consistent within each class.*

The FM ODE trajectory runs forward — t=0 is noise, t=1 is the sample:

![FM trajectory epoch 100](assets/fm_trajectory_epoch100.png)

Structure solidifies around t=0.5, digit is readable from t=0.6 onward. Noticeably smoother path than DDPM.

---

## The direct comparison

Same architecture, same dataset, same number of epochs:

![FM vs DDPM at epoch 100](assets/comparison_fm_vs_ddpm_epoch100.png)

FM at epoch 100 with 50 steps produces cleaner, more consistent digits than DDPM at epoch 100 with 1000 steps. The digit shapes in FM are sharper and vary less within each class. For classes like 0, 2, 5, 6 the gap is obvious.

This is the practical payoff of straight paths. FM does not just need fewer steps — it also converges to better visual quality at the same training budget on this task.

---

## Euler vs Heun

Euler integration:
```
z_{t+dt} = z_t + v(z_t, t) * dt
```

Heun's method (second-order Runge-Kutta):
```
z_pred    = z_t + v(z_t, t) * dt
z_{t+dt}  = z_t + 0.5 * (v(z_t, t) + v(z_pred, t+dt)) * dt
```

Heun doubles the forward passes per step but significantly improves trajectory quality. FM with Heun at 20 steps beats FM with Euler at 50 steps. If you have the compute budget, Heun is the right default.

---

## An idea that turned out to be circular

I had an idea for adding an "acceleration head" that predicts the error in the velocity prediction and corrects the trajectory at inference. The math:

```
z_next = z_t + v_pred * dt + (v_target - v_pred) * dt
       = z_t + v_target * dt
```

This is just using the ground truth velocity. The acceleration head recovers v_target by computing the residual of v_pred. It works during training because you have v_target. At inference you do not, so the head has nothing to predict.

Any head that "corrects" predictions using the training target is doing this. Worth naming.

Non-degenerate alternatives:
- **Rectified Flow**: train FM, sample (z_0, z_1) pairs from the trained model, retrain. Paths straighten each round. Used in Flux.
- **Heun sampling**: legitimate second-order correction using a second network pass, not the target.
- **Distributional velocity (VRFM)**: model predicts a distribution over velocities, not a point estimate.

---

## Notes on running this on Colab

DDPM: ~109s/epoch after warmup. The 1000-step DDIM evaluation at each epoch makes DDPM slightly slower than FM despite identical training compute.

FM: ~99s/epoch after warmup. First epoch 185s due to DataLoader initialization.

Both: batch size 256 on T4, MNIST 32x32. Comfortable within 12GB VRAM with the SpatialVAE latent compression.

If you use TPU (XLA): you need `ParallelLoader` and `xm.mark_step()` after each batch. Without `mark_step()`, operations queue indefinitely and nothing trains.

The VAE is not the bottleneck. Measured latent statistics: mean=0.003, std=0.998. If your downstream model is struggling, the VAE is almost certainly not why.

---

## What I would do differently

1. Print `output_proj.weight.std()` before starting. If it is near zero, training will flatline.
2. Use `norm_first=True` by default for any transformer deeper than 8 layers.
3. Test generation from pure Gaussian noise at epoch 1, not from encoded images.
4. Match time and class embedding scales at init. LayerNorm compensates over training but starting balanced is free.
5. Watch the first 10 batches of epoch 1. Loss not dropping from a high start means model outputs are collapsed, not that the loss is small.

---

## What comes next

CIFAR10 is next. MNIST is useful for iteration speed but low-resolution grayscale images are too forgiving — many failure modes are not visible. CIFAR10 with RGB and 10 varied classes will stress the class conditioning harder.

After that, video. FM on video is architecturally the same training loop but needs a 3D VAE for temporal compression and 3D positional encodings for the transformer. The production video models (Sora, CogVideoX, Movie Gen, Wan) all use FM. The architecture choices are mostly about token counts at video resolution, not the training objective.

Code: [github.com/shubham-sahoo/acceleration-flow-matching](https://github.com/shubham-sahoo/acceleration-flow-matching)

---

*Shubham Somnath Sahoo — IIT Kharagpur (EE+CS) | ML Research Engineer*

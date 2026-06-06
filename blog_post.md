# I built latent diffusion from scratch and here is what broke first

*A practical account of building DDPM, DDIM, and Flow Matching on a free Colab GPU.*

---

I wanted to understand generative diffusion models at the level where I could actually modify them. Not use a library. Not copy a notebook. Build the pieces from scratch, watch them fail, and figure out why.

This is that story. The model works now. The path there was not clean.

---

## What I was building

The goal was a latent diffusion setup that could compare three methods on the same architecture:

- **DDPM** — the original 1000-step denoising process from Ho et al. 2020
- **DDIM** — same noise schedule, but deterministic sampling in 50 steps
- **Flow Matching** — straight-line paths from noise to data, 20-50 steps

All three share a ViT backbone (~21M params) and a spatial VAE that compresses images to a 16-channel 8x8 latent space. Datasets: MNIST and CIFAR10, 32x32. Hardware: Google Colab T4/V100, 12GB VRAM.

The architecture is a `LatentViT`: 12 transformer layers, embed_dim=384, 6 heads, class conditioning through an embedding, time conditioning through a small MLP.

Here is what I expected: train for 10-20 epochs, see recognizable digits, compare methods.

Here is what happened instead.

---

## The first sign something was wrong

First DDPM run. Loss on epoch 1: 0.282. Loss on epoch 20: 0.272. Total improvement over 20 epochs: 0.010.

That looks like convergence. It is not convergence. It is a model that learned nothing.

For an untrained DDPM predicting noise, the expected loss is around 1.0. The noise target is drawn from N(0,1), so if your model predicts zeros, MSE(zeros, N(0,1)) = 1.0 exactly. That is your baseline.

My loss was 0.282 from the first batch. Something was clamping the outputs before training even started.

The culprit: Post-LayerNorm with 12 transformer layers.

PyTorch's `TransformerEncoderLayer` applies LayerNorm after the attention and feedforward blocks by default (Post-LN). At depth 12 with dropout=0.1, it is not fine. Each LayerNorm rescales and the residual stream slowly collapses. After 12 layers, the feature std was 0.008. The output projection (Linear 384->16) then maps near-zeros to near-zeros.

I measured it directly:

```python
x = model.transformer(x)
print("After transformer:", x.std().item())   # 0.008
x = model.output_proj(x)
print("After output_proj:", x.std().item())   # 0.0001
```

MSE between near-zero outputs and N(0,1) noise should be ~1.0 but with the specific initialization, it started at 0.282 and barely moved.

The fix was two lines:

```python
encoder_layer = nn.TransformerEncoderLayer(
    ...
    norm_first=True   # Pre-LN instead of Post-LN
)
self.transformer = nn.TransformerEncoder(
    encoder_layer,
    num_layers=depth,
    norm=nn.LayerNorm(embed_dim)  # final norm
)

# Reset output projection
nn.init.normal_(self.output_proj.weight, std=0.02)
nn.init.zeros_(self.output_proj.bias)
```

Pre-LN normalizes before each sublayer instead of after. Feature scale stays stable throughout the depth. This is why DiT uses it. This is why ViT-22B uses it. The rule: if your transformer is deeper than 8-10 layers, default to Pre-LN.

After the fix:
```
Epoch 1: 0.339  (real loss now — model is actually far from solution)
Epoch 2: 0.281  (dropped 0.057, actually learning)
Epoch 7: 0.275
```

The 0.339 start was a relief. That is what an untrained model looks like.

---

## The device mismatch

While debugging the loss, I also hit this:

```
RuntimeError: Input type (torch.FloatTensor) and weight type
(torch.cuda.FloatTensor) should be the same
```

The `DDPMSpatialLatent` sampler has `device='cpu'` as a default argument. I instantiated it without passing `device='cuda'`. The noise schedule tensors (betas, alphas) live on CPU. The model lives on GPU. First batch fails.

```python
# Wrong
model_ddpm = DDPMSpatialLatent(model, vae, num_steps=1000)

# Right
model_ddpm = DDPMSpatialLatent(model, vae, num_steps=1000, device='cuda')
```

After this, I added an explicit check at the start of every training run:

```python
assert next(model.parameters()).device.type == 'cuda'
assert model_ddpm.betas.device.type == 'cuda'
```

Saves ten minutes of confusion later.

---

## Testing the model wrong

After getting real training curves, I ran a visual test. I encoded a real MNIST image to a latent, passed it through the DDPM sampler, decoded the result. The output was corrupted noise.

I spent time thinking the model was broken.

It was not. I was testing it wrong.

DDPM is a denoiser. It is trained on noisy inputs `z_t` and learns to predict the noise. Passing a clean encoded latent `z_0` is out-of-distribution. The model has never seen clean latents during training. Of course it produces garbage.

The correct generation test:

```python
z_t = torch.randn(10, 16, 8, 8, device=device)  # start from pure noise
labels = torch.arange(0, 10, device=device)
samples = model_ddpm.sample(z_t, labels, t_steps=1000)
```

This is obvious once you think about what DDPM actually does. I did not think about it carefully the first time.

---

## The class embedding problem

After the norm fix and the device fix, training worked but the model was mostly ignoring class labels. Samples for different digits looked nearly identical.

I measured the embedding scales:

```python
c_emb = model.class_embed(labels)       # std: 0.0196
t_emb = model.time_embed(t)             # std: 0.2457
```

The class embedding was 12.5x weaker than the time embedding. The transformer sees both signals added to each token. When one is 12x louder, the network leans on the louder one.

Root cause: I had manually initialized the class embedding with `std=0.02` to match the output projection. The time embedding used default Kaiming init.

The fix was to remove the manual init and add a small MLP projection to match the time embedding structure:

```python
self.class_embed = nn.Embedding(10, embed_dim)
# removed: nn.init.normal_(self.class_embed.weight, std=0.02)

self.class_proj = nn.Sequential(
    nn.Linear(embed_dim, embed_dim),
    nn.GELU(),
    nn.Linear(embed_dim, embed_dim)
)

# In forward:
c_emb = self.class_proj(self.class_embed(class_label))
```

The network does self-calibrate through LayerNorm over training, so this is not fatal. But starting with matched scales means the model picks up class signal faster.

---

## What the loss floor actually means

DDPM on MNIST, after all fixes, plateaued at about 0.271 around epoch 13. I kept training expecting it to drop further. It did not.

This is not a bug. It is the architecture ceiling.

74% of the 1000 DDPM timesteps have SNR below 1. For those timesteps, a 21M parameter model on MNIST genuinely cannot predict much better than ~0.4-0.5. Averaged across all timesteps, the weighted floor is around 0.25-0.30.

Visual quality at loss 0.271: disconnected stroke shapes, class conditioning partially working but digits not yet legible. You need roughly loss 0.15 to see connected strokes and around 0.10 for clean digits. Getting there from 0.271 requires more capacity or longer training with LR decay.

---

## Switching to Flow Matching

Flow Matching uses the same LatentViT backbone with a different training objective.

DDPM learns to predict the noise that was added at each timestep. The path from noise to data is curved and stochastic. It takes 1000 steps at inference.

Flow Matching learns to predict velocity. Given a point on a straight line between data and noise, which direction is data? The line is:

```
z_t = (1 - t) * z_0 + t * z_1
```

where z_0 is data and z_1 is noise. The velocity is constant: `v = z_1 - z_0`. The regression target does not depend on t. You just need to predict which direction to move.

At inference, you run an ODE:

```
z_{t+dt} = z_t + v * dt
```

Because the paths are straight, 20-50 steps gets you where 1000 DDPM steps get you.

The straight-line idea comes from optimal transport. The Monge problem (1781) asks: how do you move mass between two distributions while minimizing total work? The answer is straight lines. Three groups published FM simultaneously in 2022, all arriving at the same conclusion from different directions.

### FM training results (MNIST, 100 epochs)

LR=1e-4, batch size 256, same LatentViT backbone.

| Epoch | Loss | Grad Norm |
|-------|------|-----------|
| 1 | 1.776 | 0.248 |
| 2 | 1.579 | 0.215 |
| 10 | 1.560 | 0.177 |
| 50 | 1.545 | 0.104 |
| 98 (best) | 1.542 | 0.099 |
| 100 | 1.544 | 0.099 |

Note: FM loss is not comparable to DDPM loss — they measure different things (velocity error vs noise prediction error).

Average epoch time: 99 seconds on T4. The first epoch took 185 seconds (dataset loading + JIT warmup). From epoch 2 onward it was stable at 95-105 seconds.

Grad norm decays from 0.248 to a stable 0.099-0.101 by epoch 100. Smooth decay, no spikes. That is a healthy training signal.

### What the outputs actually look like

Epoch 5 (loss 1.5625): blurry blobs with no recognizable digit shape. Correct background color (black) but no structure.

![Epoch 5 samples](assets/fm_samples_epoch5.png)

Epoch 50 (loss 1.5446): connected strokes, class structure visible. You can tell what most digits are trying to be. Some classes like 0, 1, 5 are already quite clear.

![Epoch 50 samples](assets/fm_samples_epoch50.png)

Epoch 100 (loss 1.5435): all 10 digits are recognizable. The model generates consistent shapes within each class. 0, 1, 2, 5, 7 look clean. 3, 4, 6, 8, 9 have some variation but are readable.

![Epoch 100 samples](assets/fm_samples_epoch100.png)

And here is the training progress as a GIF:

![FM training progress](assets/fm_training_progress.gif)

### What the ODE trajectory looks like

At epoch 100, the ODE trajectory from t=0 (noise) to t=1 (sample) looks like this:

![FM ODE trajectory at epoch 100](assets/fm_trajectory_epoch100.png)

t=0.0 is pure noise. By t=0.3 you can see rough shape. By t=0.5 a stroke structure is visible. From t=0.6 onward the digit is readable and the remaining steps are refinements. The path is smooth, which is what you want — a sudden jump at any t would indicate the model learned a shortcut rather than a consistent velocity.

---

## Euler vs Heun sampling

Euler integration takes one step per dt:

```
z_{t+dt} = z_t + v(z_t, t) * dt
```

Heun's method does a half-step, evaluates velocity there, then corrects:

```
z_pred = z_t + v(z_t, t) * dt
z_{t+dt} = z_t + 0.5 * (v(z_t, t) + v(z_pred, t+dt)) * dt
```

This doubles the forward passes per step but significantly improves trajectory quality. FM with Heun at 20 steps is better than FM with Euler at 50 steps. If you have the compute budget, Heun is the right default.

---

## An idea that turned out to be circular

At one point I had an idea: add a second head that predicts "acceleration" (the error in the velocity prediction) and add it at inference to correct the trajectory. The math:

```
correction = v_target - v_pred
z_next = z_t + v_pred * dt + correction * dt
       = z_t + v_target * dt
```

This is just using the ground truth velocity. There is no information added. The acceleration head is computing the residual of the velocity head to recover the true answer. During training you have v_target, so it looks like it works. At test time you do not, so the head has nothing to predict.

Anything that "corrects" predictions using the training target is doing this. Worth naming because it is easy to miss.

Non-degenerate alternatives:

- **Rectified Flow**: train FM, generate (z_0, z_1) pairs with the trained model, retrain on those pairs. Each round straightens paths further. Used in Flux.
- **Heun sampling**: second-order correction that uses a second network pass, not the ground truth target.
- **Distributional velocity (VRFM, arXiv:2502.09616)**: model predicts a distribution over velocities, not a point estimate. Non-degenerate because the velocity head models a conditional distribution, not just its mean.

---

## Notes on running this on Colab

Batch size 256 on a T4 with MNIST worked fine. Each epoch took about 99 seconds after warmup. The first epoch took 185 seconds due to DataLoader initialization.

If you use TPU (XLA) instead of GPU on Colab, you need `ParallelLoader` and explicit `xm.mark_step()` calls after each batch. Without `mark_step()`, operations queue indefinitely. With correct XLA setup the speedup is 5-10x over T4.

The VAE is not the bottleneck. Measured latent statistics on MNIST: mean = 0.003, std = 0.998. Near-perfect N(0,1). If your downstream model is struggling, the VAE is almost certainly not the reason.

---

## What I would do differently

1. Check `output_proj.weight.std()` before the first training run. If it is near zero, nothing will train.

2. Use Pre-LN by default for any transformer deeper than 8 layers. It is the right default for modern architectures.

3. Test generation from pure Gaussian noise in epoch 1, not from encoded images. Catch the "testing wrong" bug before it costs debugging time.

4. Match embedding scales at initialization. Time and class embeddings should have similar std before training starts.

5. Look at the loss for the first 10 batches of epoch 1. If it is not dropping from a high starting value, something is wrong with the model outputs.

---

## Where this goes next

FM on CIFAR10 is next. MNIST is good for fast iteration but the images are too simple — bad models can partially hide. CIFAR10 with 3 channels and 10 varied classes is a harder test.

After that, video. FM on video requires a 3D VAE (temporal + spatial compression) and 3D positional encodings, but the training loop is identical. The production video models (Sora, CogVideoX, Movie Gen, Wan) all use FM. The architecture choices are mostly about handling token counts at video length and resolution.

The code is at: [github.com/shubham-sahoo/acceleration-flow-matching](https://github.com/shubham-sahoo/acceleration-flow-matching)

---

*Shubham Somnath Sahoo — IIT Kharagpur (EE+CS) | ML Research Engineer*

"""
training_callbacks.py
─────────────────────
Plug-and-play callback system for DDPM / Flow Matching training.
Handles directory structure, logging, sample saving, and GIF generation.

USAGE
─────
from training_callbacks import TrainingCallback

cb = TrainingCallback(
    experiment_name="ddpm_mnist_v1",
    model_type="ddpm",          # "ddpm" | "fm" | "vae"
    dataset="mnist",
    config={"lr": 1e-3, "batch_size": 256, "embed_dim": 384}
)

# In your training loop:
for epoch in range(epochs):
    for x, y in loader:
        loss, _ = model_sampler.forward_loss(x, y, criterion)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        cb.on_batch_end(loss.item(), grad_norm.item())

    cb.on_epoch_end(
        epoch=epoch,
        model_sampler=model_sampler,
        optimizer=optimizer,
        device=device,
        sample_labels=torch.arange(10, device=device),  # optional
    )

cb.on_train_end(model_sampler)
"""

import os
import time
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from google.colab import drive as _colab_drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# ─────────────────────────────────────────────────────────────────────────────
# Directory layout
# ─────────────────────────────────────────────────────────────────────────────
#
#  experiments/
#  └── {experiment_name}_{timestamp}/
#      ├── config.json                  ← hyperparameters + metadata
#      ├── log.jsonl                    ← one JSON line per epoch
#      ├── checkpoints/
#      │   ├── epoch_010.pth
#      │   └── best.pth
#      ├── samples/
#      │   ├── epoch_005.png
#      │   └── epoch_010.png
#      ├── plots/
#      │   ├── loss_curve.png
#      │   ├── grad_norm.png
#      │   └── lr_schedule.png
#      └── gifs/
#          └── training_progress.gif
#


class TrainingCallback:
    """
    Attaches to any training loop to capture:
      • per-epoch loss, grad norm, LR
      • sample grids saved as PNG
      • FM trajectory snapshots
      • loss/grad-norm plots
      • training-progress GIF
      • full checkpoint with optimizer state
      • JSON log for easy post-processing
    """

    def __init__(
        self,
        experiment_name: str,
        model_type: str = "ddpm",       # "ddpm" | "fm" | "vae"
        dataset: str = "mnist",
        config: dict = None,
        base_dir: str = None,
        save_every_n_epochs: int = 5,
        checkpoint_every_n_epochs: int = 10,
        sample_every_n_epochs: int = 5,
        n_sample_rows: int = 2,         # how many rows of samples per class
        n_sample_steps_ddpm: int = 1000,
        n_sample_steps_fm: int = 50,
        drive_sync: bool = False,       # copy to Google Drive if in Colab
        drive_path: str = "/content/drive/MyDrive/experiments",
        latent_channels: int = 16,
        latent_h: int = 8,
        latent_w: int = 8,
    ):
        self.experiment_name = experiment_name
        self.model_type = model_type.lower()
        self.dataset = dataset
        self.config = config or {}
        self.save_every_n_epochs = save_every_n_epochs
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.sample_every_n_epochs = sample_every_n_epochs
        self.n_sample_rows = n_sample_rows
        self.n_sample_steps_ddpm = n_sample_steps_ddpm
        self.n_sample_steps_fm = n_sample_steps_fm
        self.drive_sync = drive_sync and IN_COLAB
        self.drive_path = drive_path
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w

        # ── timestamp ──
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{experiment_name}_{self.timestamp}"

        # ── base directory ──
        if base_dir is None:
            base_dir = "/content/experiments" if IN_COLAB else "./experiments"
        self.run_dir = Path(base_dir) / run_name

        # ── sub-directories ──
        self.dirs = {
            "checkpoints": self.run_dir / "checkpoints",
            "samples":     self.run_dir / "samples",
            "plots":       self.run_dir / "plots",
            "gifs":        self.run_dir / "gifs",
            "trajectories": self.run_dir / "trajectories",
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # ── log file ──
        self.log_path = self.run_dir / "log.jsonl"
        self.config_path = self.run_dir / "config.json"

        # ── in-memory state ──
        self.epoch_losses: list = []
        self.batch_losses: list = []
        self.grad_norms_epoch: list = []    # per-batch inside epoch
        self.grad_norms: list = []          # per-epoch mean
        self.lrs: list = []
        self.epoch_times: list = []
        self.best_loss = float("inf")
        self.current_epoch = 0
        self._epoch_start = time.time()
        self._batch_count = 0

        # ── save config ──
        meta = {
            "experiment_name": experiment_name,
            "run_name": run_name,
            "model_type": model_type,
            "dataset": dataset,
            "timestamp": self.timestamp,
            "run_dir": str(self.run_dir),
            **self.config,
        }
        with open(self.config_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n{'─'*60}")
        print(f"  TrainingCallback initialised")
        print(f"  Run dir : {self.run_dir}")
        print(f"  Model   : {model_type}  |  Dataset: {dataset}")
        print(f"{'─'*60}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Per-batch hook
    # ─────────────────────────────────────────────────────────────────────────

    def on_batch_end(self, loss: float, grad_norm: float = None):
        """Call at the end of every training batch."""
        self.batch_losses.append(loss)
        if grad_norm is not None:
            self.grad_norms_epoch.append(grad_norm)
        self._batch_count += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Per-epoch hook
    # ─────────────────────────────────────────────────────────────────────────

    def on_epoch_end(
        self,
        epoch: int,
        model_sampler=None,
        optimizer=None,
        device: str = "cuda",
        sample_labels=None,         # torch.Tensor of class labels for sampling
        vae=None,                   # override if sampler doesn't hold VAE
    ):
        """Call at the end of every training epoch."""
        self.current_epoch = epoch + 1   # 1-indexed for display
        epoch_time = time.time() - self._epoch_start
        self._epoch_start = time.time()

        # ── aggregate ──
        avg_loss = float(np.mean(self.batch_losses)) if self.batch_losses else float("nan")
        avg_grad = float(np.mean(self.grad_norms_epoch)) if self.grad_norms_epoch else float("nan")
        lr = optimizer.param_groups[0]["lr"] if optimizer else float("nan")

        self.epoch_losses.append(avg_loss)
        self.grad_norms.append(avg_grad)
        self.lrs.append(lr)
        self.epoch_times.append(epoch_time)

        # ── reset batch accumulators ──
        self.batch_losses.clear()
        self.grad_norms_epoch.clear()

        # ── console ──
        print(
            f"  Epoch {self.current_epoch:3d} │ "
            f"loss {avg_loss:.6f} │ "
            f"grad {avg_grad:.4f} │ "
            f"lr {lr:.2e} │ "
            f"{epoch_time:.1f}s"
        )

        # ── JSON log ──
        entry = {
            "epoch": self.current_epoch,
            "loss": avg_loss,
            "grad_norm": avg_grad,
            "lr": lr,
            "epoch_time_s": epoch_time,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # ── best checkpoint ──
        if avg_loss < self.best_loss and model_sampler is not None:
            self.best_loss = avg_loss
            self._save_checkpoint(model_sampler, optimizer, epoch, tag="best")

        # ── periodic checkpoint ──
        if self.current_epoch % self.checkpoint_every_n_epochs == 0:
            self._save_checkpoint(
                model_sampler, optimizer, epoch,
                tag=f"epoch_{self.current_epoch:04d}"
            )

        # ── samples ──
        if self.current_epoch % self.sample_every_n_epochs == 0 and model_sampler is not None:
            self._save_samples(model_sampler, device, sample_labels, avg_loss)

            if self.model_type == "fm":
                self._save_trajectory(model_sampler, device, sample_labels)

        # ── plots (save every epoch, cheap) ──
        self._save_plots()

        # ── drive sync ──
        if self.drive_sync:
            self._sync_to_drive()

    # ─────────────────────────────────────────────────────────────────────────
    # End-of-training hook
    # ─────────────────────────────────────────────────────────────────────────

    def on_train_end(self, model_sampler=None):
        """Call once training is complete."""
        print(f"\n{'─'*60}")
        print(f"  Training complete  │  Best loss: {self.best_loss:.6f}")

        # ── final plots ──
        self._save_plots()

        # ── GIF ──
        if PIL_AVAILABLE:
            self._make_gif()
        else:
            print("  PIL not available — skipping GIF (pip install Pillow)")

        # ── summary JSON ──
        summary = {
            "best_loss": self.best_loss,
            "final_loss": self.epoch_losses[-1] if self.epoch_losses else None,
            "total_epochs": self.current_epoch,
            "total_time_s": sum(self.epoch_times),
            "run_dir": str(self.run_dir),
        }
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  Results saved to : {self.run_dir}")
        if self.drive_sync:
            self._sync_to_drive()
        print(f"{'─'*60}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # VAE-specific hook
    # ─────────────────────────────────────────────────────────────────────────

    def on_vae_epoch_end(
        self,
        epoch: int,
        vae,
        test_loader,
        optimizer=None,
        device: str = "cuda",
        recon_loss: float = None,
        kl_loss: float = None,
    ):
        """Specialised epoch hook for VAE training."""
        self.current_epoch = epoch + 1
        epoch_time = time.time() - self._epoch_start
        self._epoch_start = time.time()

        avg_loss = float(np.mean(self.batch_losses)) if self.batch_losses else float("nan")
        lr = optimizer.param_groups[0]["lr"] if optimizer else float("nan")

        self.epoch_losses.append(avg_loss)
        self.lrs.append(lr)
        self.epoch_times.append(epoch_time)
        self.batch_losses.clear()
        self.grad_norms_epoch.clear()

        print(
            f"  Epoch {self.current_epoch:3d} │ "
            f"total {avg_loss:.6f} │ "
            f"recon {recon_loss:.6f} │ "
            f"kl {kl_loss:.6f} │ "
            f"lr {lr:.2e}"
        )

        # ── save VAE reconstructions ──
        if self.current_epoch % self.sample_every_n_epochs == 0:
            self._save_vae_reconstructions(vae, test_loader, device)

        self._save_plots()

        if self.drive_sync:
            self._sync_to_drive()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, model_sampler, optimizer, epoch, tag: str):
        path = self.dirs["checkpoints"] / f"{tag}.pth"
        model = model_sampler.model if hasattr(model_sampler, "model") else model_sampler
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": self.epoch_losses[-1] if self.epoch_losses else None,
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, path)
        print(f"    ✓ Checkpoint saved: {path.name}")

    def _save_samples(self, model_sampler, device, sample_labels, current_loss):
        """Generate and save a sample grid."""
        try:
            model_sampler.model.eval()
            n_classes = 10

            with torch.no_grad():
                rows = []
                for _ in range(self.n_sample_rows):
                    z = torch.randn(
                        n_classes,
                        self.latent_channels,
                        self.latent_h,
                        self.latent_w,
                        device=device
                    )
                    if sample_labels is None:
                        labels = torch.arange(n_classes, device=device)
                    else:
                        labels = sample_labels[:n_classes].to(device)

                    if self.model_type == "fm":
                        samples = model_sampler.sample(
                            z, labels,
                            t_steps=self.n_sample_steps_fm
                        )
                    else:
                        samples = model_sampler.sample(
                            z, labels,
                            t_steps=self.n_sample_steps_ddpm
                        )
                    rows.append(samples.cpu())

            # ── plot ──
            fig, axes = plt.subplots(
                self.n_sample_rows, n_classes,
                figsize=(n_classes * 1.5, self.n_sample_rows * 1.5)
            )
            if self.n_sample_rows == 1:
                axes = axes[np.newaxis, :]

            for r, row_samples in enumerate(rows):
                for c in range(n_classes):
                    img = row_samples[c]
                    if img.shape[0] == 1:
                        axes[r, c].imshow(img[0], cmap="gray", vmin=-1, vmax=1)
                    else:
                        axes[r, c].imshow(
                            (img.permute(1, 2, 0).numpy() + 1) / 2
                        )
                    if r == 0:
                        axes[r, c].set_title(str(c), fontsize=8)
                    axes[r, c].axis("off")

            method = self.model_type.upper()
            steps = (
                self.n_sample_steps_fm if self.model_type == "fm"
                else self.n_sample_steps_ddpm
            )
            fig.suptitle(
                f"{method} | Epoch {self.current_epoch} | "
                f"Loss {current_loss:.4f} | {steps} steps",
                fontsize=10
            )
            plt.tight_layout()

            out = self.dirs["samples"] / f"epoch_{self.current_epoch:04d}.png"
            plt.savefig(out, bbox_inches="tight", dpi=120)
            plt.close(fig)
            print(f"    ✓ Samples saved:    {out.name}")

        except Exception as e:
            print(f"    ✗ Sample generation failed: {e}")
        finally:
            model_sampler.model.train()

    def _save_trajectory(self, model_sampler, device, sample_labels):
        """Save FM denoising trajectory (t=0 → t=1) for one sample."""
        try:
            model_sampler.model.eval()
            label = (
                torch.tensor([1], device=device)
            )

            z_t = torch.randn(
                1, self.latent_channels,
                self.latent_h, self.latent_w,
                device=device
            )

            n_steps = self.n_sample_steps_fm
            dt = 1.0 / n_steps
            n_snapshots = 10
            snapshot_every = max(1, n_steps // n_snapshots)
            snapshots = []
            timesteps_saved = []

            with torch.no_grad():
                for i, t_val in enumerate(
                    np.linspace(0, 1 - dt, n_steps)
                ):
                    if i % snapshot_every == 0:
                        img = model_sampler.vae.decode_only(z_t)
                        snapshots.append(img[0].cpu())
                        timesteps_saved.append(t_val)

                    t_tensor = torch.full((1,), t_val, device=device)
                    v = model_sampler.model(z_t, label, t_tensor)
                    z_t = z_t + v * dt

                # final
                img = model_sampler.vae.decode_only(z_t)
                snapshots.append(img[0].cpu())
                timesteps_saved.append(1.0)

            n = len(snapshots)
            fig, axes = plt.subplots(1, n, figsize=(n * 2, 2.2))
            for i, (snap, t) in enumerate(zip(snapshots, timesteps_saved)):
                if snap.shape[0] == 1:
                    axes[i].imshow(snap[0], cmap="gray", vmin=-1, vmax=1)
                else:
                    axes[i].imshow((snap.permute(1, 2, 0).numpy() + 1) / 2)
                axes[i].set_title(f"t={t:.2f}", fontsize=7)
                axes[i].axis("off")

            fig.suptitle(
                f"FM Trajectory — Epoch {self.current_epoch}",
                fontsize=9
            )
            plt.tight_layout()

            out = self.dirs["trajectories"] / f"epoch_{self.current_epoch:04d}.png"
            plt.savefig(out, bbox_inches="tight", dpi=120)
            plt.close(fig)
            print(f"    ✓ Trajectory saved: {out.name}")

        except Exception as e:
            print(f"    ✗ Trajectory save failed: {e}")
        finally:
            model_sampler.model.train()

    def _save_vae_reconstructions(self, vae, test_loader, device):
        """Save VAE original vs reconstruction grid."""
        try:
            vae.eval()
            x, _ = next(iter(test_loader))
            x = x[:8].to(device)

            with torch.no_grad():
                x_recon, _, _, _ = vae(x)

            fig, axes = plt.subplots(2, 8, figsize=(16, 4))
            for i in range(8):
                for row, img_t in enumerate([x, x_recon]):
                    img = img_t[i].cpu()
                    if img.shape[0] == 1:
                        axes[row, i].imshow(
                            img[0], cmap="gray", vmin=-1, vmax=1
                        )
                    else:
                        axes[row, i].imshow(
                            (img.permute(1, 2, 0).numpy() + 1) / 2
                        )
                    axes[row, i].axis("off")
                    if i == 0:
                        axes[row, i].set_ylabel(
                            "Original" if row == 0 else "Recon",
                            fontsize=8
                        )

            fig.suptitle(f"VAE Reconstruction — Epoch {self.current_epoch}", fontsize=10)
            plt.tight_layout()

            out = self.dirs["samples"] / f"vae_epoch_{self.current_epoch:04d}.png"
            plt.savefig(out, bbox_inches="tight", dpi=120)
            plt.close(fig)
            print(f"    ✓ VAE recon saved:  {out.name}")

        except Exception as e:
            print(f"    ✗ VAE recon save failed: {e}")
        finally:
            vae.train()

    def _save_plots(self):
        """Save loss, grad-norm, and LR plots."""
        epochs_x = list(range(1, len(self.epoch_losses) + 1))
        if not epochs_x:
            return

        # ── loss curve ──
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs_x, self.epoch_losses, color="#2196F3", linewidth=1.5,
                label="train loss")
        if self.best_loss < float("inf"):
            ax.axhline(self.best_loss, color="#F44336", linestyle="--",
                       linewidth=0.8, label=f"best {self.best_loss:.4f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title(
            f"{self.experiment_name} — Loss Curve\n"
            f"{self.model_type.upper()} | {self.dataset}"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.dirs["plots"] / "loss_curve.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

        # ── grad norm ──
        if self.grad_norms:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(epochs_x, self.grad_norms, color="#4CAF50", linewidth=1.2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean Grad Norm")
            ax.set_title("Gradient Norm")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.dirs["plots"] / "grad_norm.png",
                        dpi=120, bbox_inches="tight")
            plt.close(fig)

        # ── LR schedule ──
        if any(not math.isnan(lr) for lr in self.lrs):
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(epochs_x, self.lrs, color="#FF9800", linewidth=1.2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning Rate")
            ax.set_title("LR Schedule")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.dirs["plots"] / "lr_schedule.png",
                        dpi=120, bbox_inches="tight")
            plt.close(fig)

        # ── combined dashboard ──
        self._save_dashboard()

    def _save_dashboard(self):
        """Single-image dashboard: loss + grad + LR + latest samples."""
        sample_files = sorted(self.dirs["samples"].glob("epoch_*.png"))
        has_samples = bool(sample_files)

        rows = 3 if has_samples else 2
        fig = plt.figure(figsize=(14, rows * 3))
        gs = gridspec.GridSpec(rows, 2, figure=fig, hspace=0.4, wspace=0.3)

        epochs_x = list(range(1, len(self.epoch_losses) + 1))

        # loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs_x, self.epoch_losses, color="#2196F3", linewidth=1.5)
        if self.best_loss < float("inf"):
            ax1.axhline(self.best_loss, color="#F44336", linestyle="--",
                        linewidth=0.8, label=f"best {self.best_loss:.4f}")
            ax1.legend(fontsize=7)
        ax1.set_title("Loss", fontsize=9)
        ax1.set_xlabel("Epoch", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # grad norm
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs_x, self.grad_norms, color="#4CAF50", linewidth=1.2)
        ax2.set_title("Grad Norm", fontsize=9)
        ax2.set_xlabel("Epoch", fontsize=8)
        ax2.grid(True, alpha=0.3)

        # LR
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs_x, self.lrs, color="#FF9800", linewidth=1.2)
        ax3.set_title("Learning Rate", fontsize=9)
        ax3.set_xlabel("Epoch", fontsize=8)
        ax3.set_yscale("log")
        ax3.grid(True, alpha=0.3)

        # epoch time
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.bar(epochs_x, self.epoch_times, color="#9C27B0", alpha=0.7)
        ax4.set_title("Epoch Time (s)", fontsize=9)
        ax4.set_xlabel("Epoch", fontsize=8)
        ax4.grid(True, alpha=0.3, axis="y")

        # latest sample image
        if has_samples:
            latest = sample_files[-1]
            img = plt.imread(str(latest))
            ax5 = fig.add_subplot(gs[2, :])
            ax5.imshow(img)
            ax5.set_title(
                f"Latest Samples — {latest.stem}", fontsize=9
            )
            ax5.axis("off")

        fig.suptitle(
            f"{self.experiment_name}  |  "
            f"{self.model_type.upper()}  |  {self.dataset}  |  "
            f"Epoch {self.current_epoch}",
            fontsize=11
        )
        plt.savefig(self.run_dir / "dashboard.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    def _make_gif(self):
        """Combine sample PNGs into a training-progress GIF."""
        if not PIL_AVAILABLE:
            return
        frames_paths = sorted(self.dirs["samples"].glob("epoch_*.png"))
        if len(frames_paths) < 2:
            return
        try:
            frames = [Image.open(str(p)).convert("RGBA") for p in frames_paths]
            out = self.dirs["gifs"] / "training_progress.gif"
            frames[0].save(
                str(out),
                save_all=True,
                append_images=frames[1:],
                duration=400,
                loop=0,
                optimize=True,
            )
            print(f"    ✓ GIF saved:        {out.name}  ({len(frames)} frames)")
        except Exception as e:
            print(f"    ✗ GIF creation failed: {e}")

    def _sync_to_drive(self):
        """Copy run directory to Google Drive."""
        try:
            import shutil
            dest = Path(self.drive_path) / self.run_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in self.run_dir.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(self.run_dir)
                    target = dest / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(f), str(target))
        except Exception as e:
            print(f"    ✗ Drive sync failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    def load_log(self) -> list:
        """Return all epoch log entries as a list of dicts."""
        entries = []
        if self.log_path.exists():
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        return entries

    def print_summary(self):
        """Print a compact summary table to console."""
        entries = self.load_log()
        if not entries:
            print("No log entries yet.")
            return
        print(f"\n{'Epoch':>6} {'Loss':>10} {'GradNorm':>10} {'LR':>10} {'Time(s)':>8}")
        print("─" * 50)
        for e in entries:
            print(
                f"{e['epoch']:>6} "
                f"{e['loss']:>10.6f} "
                f"{e.get('grad_norm', float('nan')):>10.4f} "
                f"{e.get('lr', float('nan')):>10.2e} "
                f"{e.get('epoch_time_s', 0):>8.1f}"
            )
        print(f"\nBest loss: {self.best_loss:.6f}  |  Run dir: {self.run_dir}\n")

    @property
    def run_path(self) -> str:
        return str(self.run_dir)
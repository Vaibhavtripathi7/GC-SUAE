"""
Training engine for GC-SUAE and the ablation baselines.

Supports AMP, gradient accumulation, a momentum encoder with TAGCL lambda
annealing, cosine LR with warmup, checkpointing, and CSV logging.
"""

import copy
import csv
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Unified trainer for GC-SUAE and all ablation baselines.

    For ablation baselines (which don't use TAGCL), pass use_tagcl=False.
    For GC-SUAE, pass use_tagcl=True to enable momentum encoder + TAGCL loss.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        output_dir: str,
        use_tagcl: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.cfg         = cfg
        self.output_dir  = output_dir
        self.use_tagcl   = use_tagcl
        self.device      = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(output_dir, exist_ok=True)

        self.model    = model.to(self.device)
        self.loss_fn  = loss_fn
        self.train_loader = train_loader
        self.val_loader   = val_loader

        # Momentum encoder (EMA copy of main model for TAGCL)
        if use_tagcl:
            self.momentum_model = copy.deepcopy(model).to(self.device)
            for p in self.momentum_model.parameters():
                p.requires_grad_(False)
        else:
            self.momentum_model = None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.get("learning_rate", 1e-4),
            weight_decay=cfg.get("weight_decay", 1e-5),
        )

        # LR scheduler with linear warmup + cosine decay
        total_steps = cfg.get("epochs", 100) * len(train_loader)
        warmup_steps = cfg.get("warmup_epochs", 5) * len(train_loader)
        self.scheduler = self._build_scheduler(total_steps, warmup_steps)

        # AMP
        self.scaler = GradScaler() if cfg.get("amp", True) else None

        # Training state
        self.epoch          = 0
        self.global_step    = 0
        self.best_metric    = -float("inf") if cfg.get("best_mode", "max") == "max" else float("inf")
        self.best_epoch     = 0
        self.accumulate_steps = cfg.get("accumulate_steps", 4)

        # TAGCL annealing
        self.lambda_tagcl_max    = cfg.get("lambda_tagcl_max", 0.5)
        self.lambda_tagcl_anneal = cfg.get("lambda_tagcl_anneal_epochs", 30)

        # Logging
        self.log_path = os.path.join(output_dir, "training_log.csv")
        self._init_log()

        print(f"[Trainer] Model: {type(model).__name__} | "
              f"Device: {self.device} | "
              f"TAGCL: {use_tagcl} | "
              f"Epochs: {cfg.get('epochs', 100)}")

    def _build_scheduler(self, total_steps: int, warmup_steps: int):
        """Linear warmup followed by cosine decay."""
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _init_log(self):
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "val_loss", "val_mse", "val_sam",
                "lambda_tagcl", "lr", "epoch_time_s"
            ])

    def _log_epoch(self, row: dict):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([row.get(k, "") for k in [
                "epoch", "train_loss", "val_loss", "val_mse", "val_sam",
                "lambda_tagcl", "lr", "epoch_time_s"
            ]])

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """EMA update: θ_mom ← m·θ_mom + (1-m)·θ_main"""
        m = self.cfg.get("tagcl_momentum", 0.999)
        for p_main, p_mom in zip(
            self.model.parameters(), self.momentum_model.parameters()
        ):
            p_mom.data.mul_(m).add_(p_main.data, alpha=1.0 - m)

    def _get_tagcl_lambda(self, epoch: int) -> float:
        """Linear anneal from 0 to lambda_tagcl_max over first N epochs."""
        if epoch >= self.lambda_tagcl_anneal:
            return self.lambda_tagcl_max
        return self.lambda_tagcl_max * (epoch / self.lambda_tagcl_anneal)

    def _forward_momentum(self, batch: dict) -> torch.Tensor:
        """Run momentum encoder on batch; returns latent z_key."""
        if self.momentum_model is None:
            return None
        out = self.momentum_model(batch)
        return out["latent"] if isinstance(out, dict) else out[1]

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)
        self.optimizer.zero_grad()

        for i, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward
            if self.scaler:
                with autocast():
                    if hasattr(self.model, "encode"):
                        output = self.model(batch)
                    else:
                        recon, z = self.model(batch)
                        output = {"recon_iirs": recon, "latent": z,
                                  "abundances": torch.zeros(1), "recon_feo": batch["feo"],
                                  "recon_dem": batch["dem"]}

                    z_momentum = None
                    if self.use_tagcl and self.momentum_model is not None:
                        z_momentum = self._forward_momentum(batch)

                    loss, components = self.loss_fn(output, batch, z_momentum)
                    loss = loss / self.accumulate_steps

                self.scaler.scale(loss).backward()
            else:
                if hasattr(self.model, "encode"):
                    output = self.model(batch)
                else:
                    recon, z = self.model(batch)
                    output = {"recon_iirs": recon, "latent": z,
                              "abundances": torch.zeros(1), "recon_feo": batch["feo"],
                              "recon_dem": batch["dem"]}

                z_momentum = None
                if self.use_tagcl and self.momentum_model is not None:
                    z_momentum = self._forward_momentum(batch)

                loss, components = self.loss_fn(output, batch, z_momentum)
                (loss / self.accumulate_steps).backward()

            if (i + 1) % self.accumulate_steps == 0 or (i + 1) == n_batches:
                if self.scaler:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.get("grad_clip_norm", 1.0)
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.get("grad_clip_norm", 1.0)
                    )
                    self.optimizer.step()
                self.optimizer.zero_grad()

            if self.use_tagcl:
                self._update_momentum_encoder()

            self.scheduler.step()
            self.global_step += 1
            total_loss += components["loss_total"]

            if self.global_step % self.cfg.get("log_every_n_steps", 10) == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  Step {self.global_step:5d} | "
                    f"loss={components['loss_total']:.5f} | "
                    f"mse={components['loss_mse']:.5f} | "
                    f"sam={components['loss_sam']:.4f} | "
                    f"tagcl={components['loss_tagcl']:.4f} | "
                    f"lr={lr:.2e}"
                )

        return {"train_loss": total_loss / n_batches}

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        total_mse, total_sam, total_loss = 0.0, 0.0, 0.0

        from models.architectures import SpectralAngleMapperLoss
        sam_fn = SpectralAngleMapperLoss().to(self.device)

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if hasattr(self.model, "encode"):
                output = self.model(batch)
                recon = output["recon_iirs"]
            else:
                recon, _ = self.model(batch)
                output = {"recon_iirs": recon, "latent": torch.zeros(1),
                          "abundances": torch.zeros(1),
                          "recon_feo": batch["feo"], "recon_dem": batch["dem"]}

            loss, components = self.loss_fn(output, batch, z_momentum=None)
            total_loss += components["loss_total"]
            total_mse  += torch.nn.functional.mse_loss(recon, batch["iirs"]).item()
            total_sam  += sam_fn(recon, batch["iirs"]).item()

        n = len(self.val_loader)
        return {
            "val_loss": total_loss / n,
            "val_mse":  total_mse / n,
            "val_sam":  total_sam / n,
        }

    def fit(self):
        """Full training loop."""
        epochs = self.cfg.get("epochs", 100)
        best_mode = self.cfg.get("best_mode", "max")
        best_metric_name = self.cfg.get("best_metric", "val_loss")

        # For val_loss we want minimum
        if best_metric_name == "val_loss":
            best_mode = "min"
            self.best_metric = float("inf")

        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            t0 = time.time()

            # Anneal TAGCL weight
            tagcl_lambda = self._get_tagcl_lambda(epoch)
            self.loss_fn.set_lambda_tagcl(tagcl_lambda)

            print(f"\nEpoch [{epoch:03d}/{epochs}] | λ_tagcl={tagcl_lambda:.4f}")

            train_metrics = self.train_epoch()
            val_metrics   = self.validate()
            epoch_time    = time.time() - t0

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"  ↳ train_loss={train_metrics['train_loss']:.5f} | "
                f"val_mse={val_metrics['val_mse']:.5f} | "
                f"val_sam={val_metrics['val_sam']:.4f} | "
                f"time={epoch_time:.1f}s"
            )

            # Logging
            self._log_epoch({
                "epoch":        epoch,
                "train_loss":   train_metrics["train_loss"],
                "val_loss":     val_metrics["val_loss"],
                "val_mse":      val_metrics["val_mse"],
                "val_sam":      val_metrics["val_sam"],
                "lambda_tagcl": tagcl_lambda,
                "lr":           lr,
                "epoch_time_s": epoch_time,
            })

            # Checkpointing
            if epoch % self.cfg.get("save_every_n_epochs", 10) == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch:03d}.pth")

            # Best model tracking
            current = val_metrics.get(best_metric_name, val_metrics["val_loss"])
            is_best = (
                (best_mode == "max" and current > self.best_metric) or
                (best_mode == "min" and current < self.best_metric)
            )
            if is_best:
                self.best_metric = current
                self.best_epoch  = epoch
                self.save_checkpoint("best_model.pth")
                print(f"  ✓ New best {best_metric_name}={current:.5f} at epoch {epoch}")

        print(f"\n[Trainer] Training complete. Best epoch: {self.best_epoch} "
              f"({best_metric_name}={self.best_metric:.5f})")

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.output_dir, filename)
        torch.save({
            "epoch":       self.epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "cfg":         self.cfg,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.epoch       = ckpt.get("epoch", 0)
        self.best_metric = ckpt.get("best_metric", self.best_metric)
        print(f"[Trainer] Loaded checkpoint from epoch {self.epoch}")

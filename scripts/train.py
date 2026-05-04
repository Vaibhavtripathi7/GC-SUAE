"""
Main training entry point for a single model.

Usage:
    python scripts/train.py --config configs/gcsuae_default.yaml
    python scripts/train.py --config configs/gcsuae_default.yaml --resume <checkpoint.pth>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader, random_split

from data.dataset import LunarMultimodalDataset
from losses.tagcl import CombinedLoss
from models.architectures import build_model, GC_SUAE
from utils.trainer import Trainer, set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict):
    data_cfg = cfg["data"]

    dataset = LunarMultimodalDataset(
        iirs_hdr    = data_cfg["iirs_hdr"],
        iirs_qub    = data_cfg["iirs_qub"],
        dem_path    = data_cfg["dem_path"],
        feo_path    = data_cfg["feo_path"],
        band_start  = data_cfg.get("band_start", 0),
        band_end    = data_cfg.get("band_end", 86),
        patch_size  = data_cfg.get("patch_size", 64),
        stride      = data_cfg.get("stride", 32),
    )

    n = len(dataset)
    n_train = int(data_cfg.get("train_split", 0.8) * n)
    n_val   = int(data_cfg.get("val_split",   0.1) * n)
    n_test  = n - n_train - n_val

    g = torch.Generator().manual_seed(cfg.get("experiment", {}).get("seed", 42))
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=g)

    nw = data_cfg.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=nw, pin_memory=True)

    return train_loader, val_loader, test_loader, dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_cfg = cfg.get("experiment", {})
    seed = exp_cfg.get("seed", 42)
    set_seed(seed)

    output_dir = os.path.join(exp_cfg.get("output_dir", "outputs/"), exp_cfg.get("name", "run"))
    os.makedirs(output_dir, exist_ok=True)

    # Save config copy
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    print(f"[Main] Experiment: {exp_cfg.get('name', 'run')}")
    print(f"[Main] Output dir: {output_dir}")

    # Data
    train_loader, val_loader, test_loader, dataset = build_dataloaders(cfg)
    print(f"[Main] Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # Model
    model_cfg = cfg["model"]
    model_cfg["n_bands"]  = cfg["data"].get("n_bands", 86)
    model = build_model(model_cfg["name"], model_cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Main] Model: {model_cfg['name']} | Params: {n_params:,}")

    # Loss
    train_cfg = cfg["training"]
    tagcl_cfg = {
        "feo_threshold":   cfg["data"].get("feo_positive_threshold", 0.05),
        "slope_threshold": cfg["data"].get("slope_positive_threshold", 0.1),
        "temperature":     train_cfg.get("tagcl_temperature", 0.07),
        "queue_size":      train_cfg.get("tagcl_queue_size", 4096),
        "d_model":         model_cfg.get("latent_dim", 128),
        "momentum":        train_cfg.get("tagcl_momentum", 0.999),
    }
    use_tagcl = model_cfg["name"] == "GC_SUAE"
    loss_fn = CombinedLoss(
        lambda_sam   = train_cfg.get("lambda_sam", 0.1),
        lambda_tagcl = 0.0,   # annealed by trainer
        lambda_dem   = train_cfg.get("lambda_dem", 0.05),
        lambda_feo   = train_cfg.get("lambda_feo", 0.1),
        lambda_lmm   = train_cfg.get("lambda_lmm", 0.2),
        tagcl_cfg    = tagcl_cfg if use_tagcl else {},
    )

    # Trainer
    trainer = Trainer(
        model        = model,
        loss_fn      = loss_fn,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = {**train_cfg, **exp_cfg},
        output_dir   = output_dir,
        use_tagcl    = use_tagcl,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit()

    print(f"\n[Main] Training complete. Best model saved to {output_dir}/best_model.pth")


if __name__ == "__main__":
    main()

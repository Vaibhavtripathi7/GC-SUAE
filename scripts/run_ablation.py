"""
Ablation runner: trains each model variant sequentially, evaluates it, and
writes the comparison table and figures.

Usage:
    python scripts/run_ablation.py --config configs/ablation.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from data.dataset import EndmemberLibrary, LunarMultimodalDataset
from losses.tagcl import CombinedLoss
from models.architectures import build_model
from utils.evaluation import (
    build_mineral_map,
    cluster_latents,
    compute_cluster_spectra,
    compute_sam_reconstruction,
    extract_latents,
    figure1_main_results,
    figure2_mineral_map,
    figure3_ablation_table_figure,
    nearest_endmember_sam_deg,
    print_results_table,
    spatial_coherence,
)
from utils.trainer import Trainer, set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_one_model(model_name: str, model_cfg_override: dict, shared_cfg: dict,
                   train_loader, val_loader, full_loader, dataset, endmember_lib,
                   output_dir: str, device: torch.device, seed: int = 42) -> dict:
    """Train one model variant and return its evaluation metrics."""

    print(f"\n{'='*60}")
    print(f"  TRAINING: {model_name}")
    print(f"  Description: {model_cfg_override.get('description', '')}")
    print(f"{'='*60}")

    model_out_dir = os.path.join(output_dir, model_name.lower())
    os.makedirs(model_out_dir, exist_ok=True)

    # Build model
    n_bands    = shared_cfg["data"].get("n_bands", 86)
    latent_dim = model_cfg_override.get("latent_dim", 64)
    patch_size = shared_cfg["data"].get("patch_size", 64)

    # The NoTAGCL ablation reuses the GC_SUAE architecture; only the loss differs.
    # build_model only knows the base name, so remap the variant name here.
    build_name = "GC_SUAE" if model_name == "GC_SUAE_NoTAGCL" else model_name

    model = build_model(build_name, {
        "n_bands":    n_bands,
        "latent_dim": latent_dim,
        "patch_size": patch_size,
        "n_minerals": 6,
        "d_model":    256,
        "n_heads":    8,
    })

    # Loss
    is_gcsuae = model_name in ("GC_SUAE", "GC_SUAE_NoTAGCL")
    use_tagcl = (model_name == "GC_SUAE")
    tagcl_cfg = {
        "feo_threshold":   shared_cfg["data"].get("feo_positive_threshold", 0.05),
        "slope_threshold": shared_cfg["data"].get("slope_positive_threshold", 0.1),
        "temperature":     0.07,
        "queue_size":      4096,
        "d_model":         latent_dim,
        "momentum":        0.999,
    } if use_tagcl else {}

    loss_fn = CombinedLoss(
        lambda_sam   = 0.1,
        lambda_tagcl = 0.0,
        lambda_dem   = model_cfg_override.get("lambda_dem", 0.05),
        lambda_feo   = model_cfg_override.get("lambda_feo", 0.1),
        lambda_lmm   = model_cfg_override.get("lambda_lmm", 0.2) if is_gcsuae else 0.0,
        tagcl_cfg    = tagcl_cfg,
    )

    train_cfg = {
        "epochs":                   model_cfg_override.get("epochs", 65),
        "batch_size":               shared_cfg["data"].get("patch_size", 8),
        "accumulate_steps":         4,
        "learning_rate":            1e-4,
        "weight_decay":             1e-5,
        "amp":                      True,
        "grad_clip_norm":           1.0,
        "warmup_epochs":            5,
        "lambda_tagcl_max":         model_cfg_override.get("lambda_tagcl_max", 0.5),
        "lambda_tagcl_anneal_epochs": 30,
        "tagcl_momentum":           0.999,
        "log_every_n_steps":        20,
        "save_every_n_epochs":      model_cfg_override.get("epochs", 65),  # only save at end
        "best_metric":              "val_loss",
        "best_mode":                "min",
    }

    trainer = Trainer(
        model        = model,
        loss_fn      = loss_fn,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = train_cfg,
        output_dir   = model_out_dir,
        use_tagcl    = use_tagcl,
        device       = device,
    )

    # Skip retraining if this variant already finished: a metrics.json plus a
    # saved checkpoint means we can just reload the weights and re-evaluate.
    metrics_path = os.path.join(model_out_dir, "metrics.json")
    ckpt_path    = os.path.join(model_out_dir, "best_model.pth")
    if os.path.exists(metrics_path) and os.path.exists(ckpt_path):
        print(f"[Ablation] Found existing metrics + checkpoint for {model_name}; "
              f"skipping training and reloading weights.")
        trainer.load_checkpoint(ckpt_path)
    else:
        trainer.fit()
        # Evaluate the best-val-loss weights, not whatever the last epoch left,
        # so this path and the reload path above measure the same model.
        trainer.load_checkpoint(ckpt_path)

    # Evaluation
    print(f"\n[Ablation] Evaluating {model_name}...")

    latents, feo_means, slope_means = extract_latents(model, full_loader, device)
    labels, centers, cluster_metrics = cluster_latents(latents, n_clusters=10, seed=seed)
    sam_metrics = compute_sam_reconstruction(model, val_loader, device)

    # Mineral identification
    cluster_spectra = compute_cluster_spectra(model, full_loader, labels, 10, device)
    mineral_assignments = endmember_lib.identify(cluster_spectra, sam_threshold=0.15)

    n_identified = sum(1 for x in mineral_assignments if x is not None)
    mineral_id_acc = n_identified / len(mineral_assignments)
    nearest_sam = nearest_endmember_sam_deg(cluster_spectra, endmember_lib.spectra)
    coherence   = spatial_coherence(labels, dataset.indices, dataset.stride)

    metrics = {
        **cluster_metrics,
        **sam_metrics,
        "mineral_id_accuracy":   round(mineral_id_acc, 4),
        "mineral_assignments":   mineral_assignments,
        "n_identified_clusters": n_identified,
        "nearest_endmember_sam_deg_mean": round(float(nearest_sam.mean()), 3),
        "nearest_endmember_sam_deg_min":  round(float(nearest_sam.min()), 3),
        "spatial_coherence":     round(float(coherence), 4),
        "seed":                  seed,
    }

    # Save metrics. numpy/torch scalars aren't JSON-serializable, so unwrap any
    # value exposing .item() to a native Python number first.
    with open(os.path.join(model_out_dir, "metrics.json"), "w") as f:
        json.dump(
            {k: (v.item() if hasattr(v, "item") else v)
             for k, v in metrics.items() if not isinstance(v, list)},
            f, indent=2,
        )

    print(f"  Silhouette:  {metrics['silhouette_score']:.4f}")
    print(f"  DB Index:    {metrics['davies_bouldin_index']:.4f}")
    print(f"  Mean SAM:    {metrics['mean_sam_deg']:.3f}°")
    print(f"  Mineral ID:  {n_identified}/10 clusters identified "
          f"(nearest-endmember SAM mean {nearest_sam.mean():.2f}°, min {nearest_sam.min():.2f}°)")
    print(f"  Coherence:   {coherence:.4f}")

    return metrics, labels, cluster_spectra, feo_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override experiment.seed (used for multi-seed runs).")
    parser.add_argument("--models", default=None,
                        help="Comma-separated subset of model keys from the config, e.g. "
                             "'pooled_attn_fusion,gcsuae_no_tagcl'. Default: all.")
    parser.add_argument("--output-dir", default=None,
                        help="Override experiment.output_dir.")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers. Use 0 where the start method is not fork.")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed   = args.seed if args.seed is not None else cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    output_dir = args.output_dir or cfg.get("experiment", {}).get("output_dir", "outputs/ablation")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Ablation] seed={seed}  output_dir={output_dir}")

    if args.models:
        wanted = [k.strip() for k in args.models.split(",")]
        missing = [k for k in wanted if k not in cfg["models"]]
        if missing:
            sys.exit(f"Unknown model keys {missing}; available: {list(cfg['models'])}")
        cfg["models"] = {k: cfg["models"][k] for k in wanted}

    # Build shared dataset
    data_cfg = cfg["data"]
    dataset = LunarMultimodalDataset(
        iirs_hdr   = data_cfg["iirs_hdr"],
        iirs_qub   = data_cfg["iirs_qub"],
        dem_path   = data_cfg["dem_path"],
        feo_path   = data_cfg["feo_path"],
        band_start = 0,
        band_end   = data_cfg.get("n_bands", 86),
        patch_size = data_cfg.get("patch_size", 64),
        stride     = data_cfg.get("stride", 32),
        min_valid_fraction = data_cfg.get("min_valid_fraction", 1.0),
        pixel_size_m = tuple(data_cfg.get("pixel_size_m", (79.8, 94.0))),
        feo_max_valid = data_cfg.get("feo_max_valid", None),
    )

    n = len(dataset)
    n_train = int(data_cfg.get("train_split", 0.8) * n)
    n_val   = int(data_cfg.get("val_split",   0.1) * n)
    n_test  = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, _ = random_split(dataset, [n_train, n_val, n_test], generator=g)

    bs, nw = 4, args.num_workers
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw)
    full_loader  = DataLoader(dataset,  batch_size=bs, shuffle=False, num_workers=nw)

    # Load endmember library
    endmember_lib = EndmemberLibrary(
        data_cfg["endmember_path"],
        n_bands=data_cfg.get("n_bands", 86),
    )

    # Run all models
    all_results     = {}
    gcsuae_data     = {}  # save GC-SUAE specific data for figure 2

    for model_key, model_cfg_override in cfg["models"].items():
        model_name = model_cfg_override["name"]
        metrics, labels, cluster_spectra, feo_means = run_one_model(
            model_name, model_cfg_override, cfg,
            train_loader, val_loader, full_loader,
            dataset, endmember_lib, output_dir, device, seed=seed,
        )
        all_results[model_name] = metrics

        if model_name == "GC_SUAE" and labels is not None:
            gcsuae_data = {
                "labels":           labels,
                "cluster_spectra":  cluster_spectra,
                "feo_means":        feo_means,
                "mineral_assign":   metrics["mineral_assignments"],
            }

    # Print results table
    print_results_table(all_results)

    # Save full results JSON
    results_path = os.path.join(output_dir, "ablation_results.json")
    saveable = {
        m: {k: (v.item() if hasattr(v, "item") else v)
            for k, v in metrics.items() if not isinstance(v, list)}
        for m, metrics in all_results.items()
    }
    with open(results_path, "w") as f:
        json.dump(saveable, f, indent=2)
    print(f"\n[Ablation] Full results saved → {results_path}")

    # Generate figures
    if gcsuae_data:
        figure1_main_results(
            results         = all_results,
            cluster_spectra = gcsuae_data["cluster_spectra"],
            feo_array       = gcsuae_data["feo_means"],
            labels          = gcsuae_data["labels"],
            n_clusters      = 10,
            output_path     = os.path.join(output_dir, "Figure_1_Main_Results.png"),
        )

        mineral_map = build_mineral_map(
            labels              = gcsuae_data["labels"],
            mineral_assignments = gcsuae_data["mineral_assign"],
            spatial_grid        = dataset.spatial_grid,
            mineral_names       = endmember_lib.MINERAL_NAMES,
        )
        figure2_mineral_map(
            mineral_map         = mineral_map,
            dem_data            = dataset.dem_data,
            feo_data            = dataset.feo_data,
            mineral_assignments = gcsuae_data["mineral_assign"],
            mineral_names       = endmember_lib.MINERAL_NAMES,
            output_path         = os.path.join(output_dir, "Figure_2_Mineral_Map.png"),
        )

    figure3_ablation_table_figure(
        results     = all_results,
        output_path = os.path.join(output_dir, "Figure_3_Ablation_Table.png"),
    )

    print(f"\n[Ablation] All done. Figures and results in: {output_dir}/")


if __name__ == "__main__":
    main()

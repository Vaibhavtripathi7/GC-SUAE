"""
Aggregate multi-seed ablation results into a mean ± std table.

Each seed run writes <root>/seed<N>/<model>/metrics.json (see run_ablation.py
--seed / --output-dir). This script collects them, reports mean ± std per model
and metric, and runs a paired test between two named models across seeds.

Usage:
    python scripts/aggregate_seeds.py --root outputs/ablation_seeds \
        --compare PooledAttnFusion GC_SUAE_NoTAGCL
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

METRICS = [
    ("silhouette_score",               "Silhouette ↑",   "{:.3f}"),
    ("davies_bouldin_index",           "DB ↓",           "{:.3f}"),
    ("spatial_coherence",              "Coherence ↑",    "{:.3f}"),
    ("mean_sam_deg",                   "Recon SAM (°) ↓","{:.2f}"),
    ("nearest_endmember_sam_deg_mean", "EM SAM (°) ↓",   "{:.2f}"),
]


MODEL_ORDER = ["Unimodal2DCNN", "Unimodal3DCNN", "EarlyFusion", "LateFusion",
               "PooledAttnFusion", "GC_SUAE_NoTAGCL", "GC_SUAE"]


def load_runs(root: str) -> dict:
    """
    Returns {model: {metric: [values across seeds]}} plus the seed list.

    Reads each model's own metrics.json (<root>/seed<N>/<model>/metrics.json)
    rather than the per-seed ablation_results.json, which only lists the
    models of the most recent (possibly partial) invocation.
    """
    runs = defaultdict(lambda: defaultdict(list))
    seeds = sorted(os.path.basename(d) for d in glob.glob(os.path.join(root, "seed*")) if os.path.isdir(d))
    found = defaultdict(list)
    for seed in seeds:
        for path in sorted(glob.glob(os.path.join(root, seed, "*", "metrics.json"))):
            with open(path) as f:
                metrics = json.load(f)
            model = _model_name(os.path.basename(os.path.dirname(path)))
            found[model].append(seed)
            for key, _, _ in METRICS:
                if key in metrics:
                    runs[model][key].append(metrics[key])
    ordered = {m: runs[m] for m in MODEL_ORDER if m in runs}
    ordered.update({m: v for m, v in runs.items() if m not in ordered})
    for m, s in found.items():
        if len(s) != len(seeds):
            print(f"  note: {m} present in {len(s)}/{len(seeds)} seeds ({', '.join(s)})")
    return ordered, seeds


def _model_name(dirname: str) -> str:
    """Output dirs are lower-cased model names; map back to the canonical name."""
    for m in MODEL_ORDER:
        if m.lower() == dirname:
            return m
    return dirname


def fmt(values, spec):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return spec.format(values[0])
    return f"{spec.format(values.mean())} ± {spec.format(values.std(ddof=1))}"


def markdown_table(runs):
    header = "| Model | n | " + " | ".join(label for _, label, _ in METRICS) + " |"
    sep    = "|:---|:---:|" + "|".join(":---:" for _ in METRICS) + "|"
    rows = [header, sep]
    for model, metrics in runs.items():
        n = max((len(v) for v in metrics.values()), default=0)
        cells = [fmt(metrics[k], spec) if k in metrics else "–" for k, _, spec in METRICS]
        rows.append(f"| {model} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def latex_table(runs):
    cols = "l" + "c" * len(METRICS)
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
             "Model & " + " & ".join(label for _, label, _ in METRICS) + " \\\\", "\\midrule"]
    for model, metrics in runs.items():
        cells = [fmt(metrics[k], spec).replace("±", "$\\pm$") if k in metrics else "--"
                 for k, _, spec in METRICS]
        lines.append(model.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def paired_test(runs, a: str, b: str):
    from scipy import stats
    print(f"\nPaired comparison across seeds: {a} vs {b}")
    for key, label, spec in METRICS:
        va, vb = runs[a].get(key, []), runs[b].get(key, [])
        n = min(len(va), len(vb))
        if n < 2:
            print(f"  {label:18s} n={n}: too few seeds for a test")
            continue
        va, vb = np.asarray(va[:n]), np.asarray(vb[:n])
        t, p = stats.ttest_rel(va, vb)
        d = (va - vb).mean() / ((va - vb).std(ddof=1) + 1e-12)
        print(f"  {label:18s} n={n}  Δ={spec.format((va - vb).mean())}  "
              f"paired t={t:.2f}  p={p:.4f}  Cohen d={d:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Directory containing seed*/ subfolders")
    ap.add_argument("--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"), default=None)
    ap.add_argument("--latex", action="store_true", help="Also print a LaTeX tabular")
    args = ap.parse_args()

    runs, seeds = load_runs(args.root)
    if not runs:
        raise SystemExit(f"No seed*/<model>/metrics.json found under {args.root}")
    print(f"Seeds found: {seeds}\n")
    print(markdown_table(runs))
    if args.latex:
        print("\n" + latex_table(runs))
    if args.compare:
        paired_test(runs, *args.compare)

    out = os.path.join(args.root, "aggregate.json")
    with open(out, "w") as f:
        json.dump({m: {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                           "values": v} for k, v in met.items()} for m, met in runs.items()},
                  f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os, pandas as pd, matplotlib.pyplot as plt, numpy as np

ROOT = "/home/deeps/deepmvi-seminar"
datasets = ["electricity", "airq"]

for ds in datasets:
    csv_path = os.path.join(ROOT, "baseline_results", ds, "metrics.csv")
    out_dir = os.path.join(ROOT, "baseline_results", ds)
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "mae_bar_log.png")
    if not os.path.exists(csv_path):
        print(f"Skipped {ds}: metrics.csv not found at {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    if "MAE" not in df.columns:
        print(f"Skipped {ds}: MAE column missing in {csv_path}")
        continue
    methods = df["method"].astype(str).tolist()
    maes = df["MAE"].astype(float).tolist()
    # clamp very small/zero values so they appear on log scale plot
    eps = 1e-8
    maes_clamped = [max(x, eps) for x in maes]
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.bar(methods, maes_clamped)
    ax.set_yscale("log")
    ax.set_ylabel("MAE (log scale)")
    ax.set_xlabel("Method")
    ax.set_title(f"Baseline MAE comparison ({ds}) — log scale")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Saved:", out_png)

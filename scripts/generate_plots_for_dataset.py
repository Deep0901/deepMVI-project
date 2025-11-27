#!/usr/bin/env python3
"""
generate_plots_for_dataset.py

Usage:
  python3 generate_plots_for_dataset.py --dataset electricity
Produces plots in: plots/<dataset>/
"""
import os, sys, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150

ROOT = os.path.expanduser("/home/deeps/deepmvi-seminar")
p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
args = p.parse_args()

DATA_DIR = os.path.join(ROOT, "data", args.dataset)
OUT_DIR = os.path.join(ROOT, "output_deepmvi", args.dataset)
PLOTS_DIR = os.path.join(ROOT, "plots", args.dataset)
os.makedirs(PLOTS_DIR, exist_ok=True)

X_path = os.path.join(DATA_DIR, "X.npy")
IMP_path = os.path.join(OUT_DIR, "imputed.npy")

if not os.path.exists(X_path):
    print("ERROR: cannot find", X_path); sys.exit(1)
if not os.path.exists(IMP_path):
    print("ERROR: cannot find", IMP_path); sys.exit(1)

X = np.load(X_path)
imputed = np.load(IMP_path)

nan_mask = np.isnan(X)
diff = imputed - X
abs_diff = np.abs(diff)
overall_mae = float(np.nanmean(abs_diff))

# summary CSV
metrics = {"rows":[X.shape[0]], "cols":[X.shape[1]], "overall_MAE":[overall_mae], "missing_frac":[float(nan_mask.mean())]}
pd.DataFrame(metrics).to_csv(os.path.join(PLOTS_DIR, "imputation_summary.csv"), index=False)

# 1 missingness mask
plt.figure(figsize=(6,4))
plt.title("Original missingness mask (1 = observed, 0 = missing)")
plt.imshow((~nan_mask).astype(int).T, aspect='auto', origin='lower')
plt.xlabel("Time index"); plt.ylabel("Series index"); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "missingness_mask.png")); plt.close()

# 2 diff heatmap
plt.figure(figsize=(6,4))
vmin = np.nanpercentile(diff, 1)
vmax = np.nanpercentile(diff, 99)
plt.imshow(diff.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
plt.colorbar(label='imputed - original'); plt.xlabel("Time"); plt.ylabel("Series"); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "diff_heatmap.png")); plt.close()

# 3 example series (first, mid, last)
n_series = X.shape[1]
examples = [0, max(0,n_series//2), n_series-1]
for idx in examples:
    plt.figure(figsize=(9,3))
    t = np.arange(min(300, X.shape[0]))
    plt.plot(t, X[:len(t), idx], label="Original")
    plt.plot(t, imputed[:len(t), idx], label="Imputed", linestyle='--')
    plt.xlabel("Time"); plt.ylabel("Value"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"series_{idx}_comparison.png")); plt.close()

# 4 histogram absolute errors
errs = abs_diff.flatten(); errs = errs[~np.isnan(errs)]
plt.figure(figsize=(6,4)); plt.hist(errs, bins=60); plt.xlabel("Absolute error"); plt.ylabel("Count"); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "abs_error_histogram.png")); plt.close()

# 5 observed vs imputed sample
obs_mask = ~np.isnan(X)
idxs = np.column_stack(np.nonzero(obs_mask))
if idxs.shape[0] > 0:
    sample_n = min(3000, idxs.shape[0])
    sel = np.random.choice(np.arange(idxs.shape[0]), size=sample_n, replace=False)
    sel_idx = idxs[sel]
    obs_vals = X[sel_idx[:,0], sel_idx[:,1]]
    imp_vals = imputed[sel_idx[:,0], sel_idx[:,1]]
    plt.figure(figsize=(5,5)); plt.scatter(obs_vals, imp_vals, s=6)
    mn = min(obs_vals.min(), imp_vals.min()); mx = max(obs_vals.max(), imp_vals.max())
    plt.plot([mn,mx],[mn,mx], linestyle='--')
    plt.xlabel("Original"); plt.ylabel("Imputed"); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "observed_vs_imputed_scatter.png")); plt.close()

# 6 top10 series by MAE
series_mae = np.nanmean(abs_diff, axis=0)
order = np.argsort(series_mae)[::-1][:10]
plt.figure(figsize=(8,3)); plt.bar(range(len(order)), series_mae[order])
plt.xticks(range(len(order)), [str(int(o)) for o in order], rotation=45); plt.ylabel("MAE"); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "top10_series_mae.png")); plt.close()

print("Saved plots into", PLOTS_DIR)

#!/usr/bin/env python3
"""
generate_presentation_plots.py

Generates plots to explain and demonstrate DeepMVI imputation results.
Saves images to /home/deeps/deepmvi-seminar/plots/

Plots produced:
 - missingness_mask.png           : heatmap of original observed/missing pattern
 - diff_heatmap.png               : heatmap of (imputed - original)
 - series_{i}_comparison.png      : before/after time-series plots for 3 example series
 - abs_error_histogram.png        : histogram of absolute errors
 - observed_vs_imputed_scatter.png: scatter of observed vs imputed (sampled)
 - top10_series_mae.png           : bar chart of top 10 series by MAE
 - imputation_summary.csv         : CSV of summary metrics
"""
import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150

ROOT = "/home/deeps/deepmvi-seminar"
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output_deepmvi")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Paths
X_path = os.path.join(DATA_DIR, "X.npy")
IMP_path = os.path.join(OUT_DIR, "imputed.npy")

if not os.path.exists(X_path):
    print("ERROR: cannot find X.npy at", X_path); sys.exit(1)
if not os.path.exists(IMP_path):
    print("ERROR: cannot find imputed.npy at", IMP_path); sys.exit(1)

# Load
X = np.load(X_path)
imputed = np.load(IMP_path)

# Compute masks and diffs
nan_mask = np.isnan(X)
diff = imputed - X
abs_diff = np.abs(diff)

overall_mae = float(np.nanmean(abs_diff))
changed_mask = abs_diff > 1e-6
changed_frac = float(changed_mask.mean())

# Save metrics summary
metrics = {
    "rows": [X.shape[0]],
    "cols": [X.shape[1]],
    "overall_MAE": [overall_mae],
    "changed_fraction": [changed_frac],
    "missing_fraction": [float(nan_mask.mean())]
}
summary_df = pd.DataFrame(metrics)
summary_csv = os.path.join(PLOTS_DIR, "imputation_summary.csv")
summary_df.to_csv(summary_csv, index=False)

# 1) Missingness mask
plt.figure(figsize=(6,4))
plt.title("Original missingness mask (1 = observed, 0 = missing)")
plt.imshow((~nan_mask).astype(int).T, aspect='auto', origin='lower')
plt.xlabel("Time index"); plt.ylabel("Series index"); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "missingness_mask.png"))
plt.close()

# 2) Diff heatmap (imputed - original)
plt.figure(figsize=(6,4))
plt.title("Imputed - Original (heatmap) - clipped for visualization")
vmin = np.nanpercentile(diff, 1)
vmax = np.nanpercentile(diff, 99)
plt.imshow(diff.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
plt.colorbar(label='imputed - original'); plt.xlabel("Time"); plt.ylabel("Series"); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "diff_heatmap.png"))
plt.close()

# 3) Example series plots (choose first, middle, last)
n_series = X.shape[1]
examples = [0, max(0, n_series//2), n_series-1]
for idx in examples:
    plt.figure(figsize=(9,3))
    plt.title(f"Series {idx}: Original vs Imputed (first 300 timesteps)")
    t = np.arange(min(300, X.shape[0]))
    plt.plot(t, X[:len(t), idx], label="Original")
    plt.plot(t, imputed[:len(t), idx], label="Imputed", linestyle='--')
    plt.xlabel("Time"); plt.ylabel("Value"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"series_{idx}_comparison.png"))
    plt.close()

# 4) Histogram of absolute errors
plt.figure(figsize=(6,4))
plt.title("Histogram of absolute errors (imputed - original)")
errs = abs_diff.flatten(); errs = errs[~np.isnan(errs)]
plt.hist(errs, bins=60)
plt.xlabel("Absolute error"); plt.ylabel("Count"); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "abs_error_histogram.png"))
plt.close()

# 5) Observed vs Imputed scatter (sampled)
plt.figure(figsize=(5,5))
plt.title("Observed vs Imputed (sampled)")
obs_mask = ~np.isnan(X)
idxs = np.column_stack(np.nonzero(obs_mask))
if idxs.shape[0] > 0:
    sample_n = min(3000, idxs.shape[0])
    rng = np.random.default_rng(0)
    sel = rng.choice(np.arange(idxs.shape[0]), size=sample_n, replace=False)
    sel_idx = idxs[sel]
    obs_vals = X[sel_idx[:,0], sel_idx[:,1]]
    imp_vals = imputed[sel_idx[:,0], sel_idx[:,1]]
    plt.scatter(obs_vals, imp_vals, s=6)
    mn = min(obs_vals.min(), imp_vals.min()); mx = max(obs_vals.max(), imp_vals.max())
    plt.plot([mn,mx], [mn,mx], linestyle='--')
plt.xlabel("Original"); plt.ylabel("Imputed"); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "observed_vs_imputed_scatter.png"))
plt.close()

# 6) Top 10 series by MAE
series_mae = np.nanmean(abs_diff, axis=0)
order = np.argsort(series_mae)[::-1][:10]
plt.figure(figsize=(8,3))
plt.title("Top 10 series by MAE")
plt.bar(range(len(order)), series_mae[order])
plt.xticks(range(len(order)), [str(int(o)) for o in order], rotation=45)
plt.ylabel("MAE"); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "top10_series_mae.png"))
plt.close()

print("Plots saved in:", PLOTS_DIR)
print("Files:")
for f in sorted(os.listdir(PLOTS_DIR)):
    print(" -", f)


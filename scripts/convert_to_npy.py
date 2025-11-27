#!/usr/bin/env python3
import sys, os, numpy as np, pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: convert_to_npy.py <input-file> [--time-index colname_or_idx]")
    sys.exit(1)

inp = Path(sys.argv[1])
time_idx = None
if len(sys.argv) >= 3 and sys.argv[2].startswith("--time-index"):
    parts = sys.argv[2].split("=")
    time_idx = parts[1] if len(parts) > 1 else None

print("Reading", inp)
if inp.suffix.lower() in ['.csv', '.txt']:
    df = pd.read_csv(inp, index_col=0 if time_idx is None else None)
elif inp.suffix.lower() in ['.parquet']:
    df = pd.read_parquet(inp)
elif inp.suffix.lower() == '.npy':
    arr = np.load(inp)
    # assume .npy is already X; create A from non-NaN
    X = np.nan_to_num(arr, 0.0)
    A = (~np.isnan(arr)).astype(int)
    np.save('data/X.npy', X)
    np.save('data/A.npy', A)
    print("Saved data/X.npy and data/A.npy (from .npy input)")
    sys.exit(0)
else:
    raise SystemExit("Unsupported file type: " + inp.suffix)

# If df has a time column index as first column, try to set it:
if time_idx:
    if time_idx.isdigit():
        idx = int(time_idx)
        df = pd.read_csv(inp, index_col=idx)
    else:
        if time_idx in df.columns:
            df = df.set_index(time_idx)

# convert to matrix: rows=time, cols=series
X = df.values.astype(float)
A = (~np.isnan(X)).astype(int)
X_filled = np.nan_to_num(X, 0.0)

os.makedirs('data', exist_ok=True)
np.save('data/X.npy', X_filled)
np.save('data/A.npy', A)
print("Saved data/X.npy and data/A.npy")

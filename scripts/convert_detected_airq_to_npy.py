#!/usr/bin/env python3
import os, sys, numpy as np, pandas as pd
SRC = os.environ.get("AIRQ_SRC")
DST_DIR = "/home/deeps/deepmvi-seminar/data/airq"
os.makedirs(DST_DIR, exist_ok=True)

print("Converting source:", SRC)
if not os.path.exists(SRC):
    raise SystemExit("Source file not found: " + SRC)

ext = os.path.splitext(SRC)[1].lower()
# load based on extension, with fallbacks
if ext in (".csv", ".txt"):
    try:
        df = pd.read_csv(SRC, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(SRC, sep='\s+', engine="python")
elif ext in (".parquet",):
    df = pd.read_parquet(SRC)
elif ext == ".npy":
    arr = np.load(SRC, allow_pickle=True)
    if isinstance(arr, np.ndarray):
        X = arr.astype(float)
        A = (~np.isnan(X)).astype(int)
        np.save(os.path.join(DST_DIR, "X.npy"), np.nan_to_num(X, 0.0))
        np.save(os.path.join(DST_DIR, "A.npy"), A)
        print("Saved X.npy and A.npy from .npy file")
        raise SystemExit(0)
else:
    raise SystemExit("Unsupported file type: " + ext)

# drop non-numeric columns (if any)
for c in list(df.columns):
    if not np.issubdtype(df[c].dtype, np.number):
        print("Dropping non-numeric column:", c)
        df = df.drop(columns=[c])

X = df.values.astype(float)
A = (~np.isnan(X)).astype(int)
np.save(os.path.join(DST_DIR, "X.npy"), np.nan_to_num(X, 0.0))
np.save(os.path.join(DST_DIR, "A.npy"), A)
print("Saved:", os.path.join(DST_DIR, "X.npy"), os.path.join(DST_DIR, "A.npy"))
print("Shape:", X.shape, "missing_frac:", float(np.isnan(X).mean()))

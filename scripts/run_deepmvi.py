#!/usr/bin/env python3
"""
Fixed run_deepmvi.py — non-recursive CPU monkeypatch.

Loads /home/deeps/deepmvi-seminar/data/X.npy, calls transformer.transformer_recovery(X),
and saves result to /home/deeps/deepmvi-seminar/output_deepmvi/imputed.npy.

This version uses the real torch module inside the wrapper to avoid recursion.
"""
import os, sys, numpy as np, importlib

DATA_X = os.path.expanduser("/home/deeps/deepmvi-seminar/data/X.npy")
OUT_DIR = os.path.expanduser("/home/deeps/deepmvi-seminar/output_deepmvi")
OUT_PATH = os.path.join(OUT_DIR, "imputed.npy")

if not os.path.exists(DATA_X):
    print("ERROR: data file not found at", DATA_X)
    sys.exit(2)

repo_path = os.path.expanduser("/home/deeps/deepmvi-seminar/DeepMVI")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

try:
    transformer = importlib.import_module("transformer")
except Exception as e:
    print("ERROR importing transformer module from", repo_path)
    print(e)
    sys.exit(3)

# Import the real torch module (separate reference) and call its device constructor inside lambda.
# This avoids calling transformer.torch.device (which we may be overriding) — preventing recursion.
try:
    real_torch = __import__("torch")
    transformer.torch.device = lambda *args, **kwargs: real_torch.device("cpu")
    print("Patched transformer.torch.device -> CPU (using real torch.device)")
except Exception as e:
    print("Warning: failed to patch transformer.torch.device; proceeding anyway. Error:", e)

# Load data and run
X = np.load(DATA_X)
print("Loaded X.npy shape:", X.shape)

try:
    imputed = transformer.transformer_recovery(X)
except Exception as e:
    print("ERROR while running transformer.transformer_recovery():")
    raise

os.makedirs(OUT_DIR, exist_ok=True)
np.save(OUT_PATH, imputed)
print("Saved imputed output to", OUT_PATH)
print("Imputed shape:", imputed.shape)

#!/usr/bin/env python3
"""
Safe patched runner for DeepMVI transformer.py.

- Makes a temporary patched copy of DeepMVI/transformer.py replacing
  "torch.device('cuda:%d'%0)" with "torch.device('cpu')"
- Imports the patched file directly (file-based import) to avoid modifying installed modules
- Calls transformer_recovery on /home/deeps/deepmvi-seminar/data/X.npy
- Saves output to /home/deeps/deepmvi-seminar/output_deepmvi/imputed.npy
"""
import os, sys, io, importlib.util, numpy as np, tempfile, shutil

ROOT = "/home/deeps/deepmvi-seminar"
REPO = os.path.join(ROOT, "DeepMVI")
SRC = os.path.join(REPO, "transformer.py")
DATA_X = os.path.join(ROOT, "data", "X.npy")
OUT_DIR = os.path.join(ROOT, "output_deepmvi")
OUT_PATH = os.path.join(OUT_DIR, "imputed.npy")

if not os.path.exists(SRC):
    print("ERROR: transformer.py not found at", SRC); sys.exit(2)
if not os.path.exists(DATA_X):
    print("ERROR: data file not found at", DATA_X); sys.exit(3)

# read source
with open(SRC, "r", encoding="utf-8") as f:
    src_text = f.read()

# replace the explicit cuda device constructor with cpu
patched_text = src_text.replace("device = torch.device('cuda:%d'%0)", "device = torch.device('cpu')")

# Fallback: also replace common variants (defensive)
patched_text = patched_text.replace("torch.device('cuda:%d'%0)", "torch.device('cpu')")
patched_text = patched_text.replace("torch.device('cuda')", "torch.device('cpu')")

# write patched file to a temp path inside repo (so relative imports still work)
patched_path = os.path.join(REPO, "_patched_transformer.py")
with open(patched_path, "w", encoding="utf-8") as f:
    f.write(patched_text)

# import the patched module by file path to avoid altering sys.modules for "transformer"
spec = importlib.util.spec_from_file_location("_patched_transformer", patched_path)
patched_mod = importlib.util.module_from_spec(spec)
sys.modules["_patched_transformer"] = patched_mod
spec.loader.exec_module(patched_mod)

print("Imported patched transformer from", patched_path)

# load data
X = np.load(DATA_X)
print("Loaded X.npy shape:", X.shape)

# call transformer_recovery
try:
    imputed = patched_mod.transformer_recovery(X)
except Exception as e:
    print("ERROR while running patched transformer_recovery():")
    raise

# save output
os.makedirs(OUT_DIR, exist_ok=True)
np.save(OUT_PATH, imputed)
print("Saved imputed output to", OUT_PATH)
print("Imputed shape:", imputed.shape)

# clean up patched file (optional) — keep it for debugging, comment out if you want to remove
# os.remove(patched_path)

#!/usr/bin/env python3
"""
Patched runner v2 for DeepMVI transformer.py.

- Ensures DeepMVI repo dir is on sys.path so top-level imports (utils, loader, model) succeed.
- Replaces cuda device lines with CPU.
- Imports the patched module by file path and runs transformer_recovery.
- Saves result to /home/deeps/deepmvi-seminar/output_deepmvi/imputed.npy
"""
import os, sys, importlib.util, numpy as np

ROOT = "/home/deeps/deepmvi-seminar"
REPO = os.path.join(ROOT, "DeepMVI")
SRC = os.path.join(REPO, "transformer.py")
DATA_X = os.path.join(ROOT, "data", "X.npy")
OUT_DIR = os.path.join(ROOT, "output_deepmvi")
OUT_PATH = os.path.join(OUT_DIR, "imputed.npy")

# sanity checks
if not os.path.exists(SRC):
    print("ERROR: transformer.py not found at", SRC); sys.exit(2)
if not os.path.exists(DATA_X):
    print("ERROR: data file not found at", DATA_X); sys.exit(3)

# read and patch source
with open(SRC, "r", encoding="utf-8") as f:
    src_text = f.read()

patched_text = src_text.replace("device = torch.device('cuda:%d'%0)", "device = torch.device('cpu')")
patched_text = patched_text.replace("torch.device('cuda:%d'%0)", "torch.device('cpu')")
patched_text = patched_text.replace("torch.device('cuda')", "torch.device('cpu')")

# write patched file into repo so relative/top-level imports work
patched_path = os.path.join(REPO, "_patched_transformer.py")
with open(patched_path, "w", encoding="utf-8") as f:
    f.write(patched_text)

# ensure REPO is on sys.path *before* importing the patched module so "import utils" resolves
if REPO not in sys.path:
    sys.path.insert(0, REPO)
    # also insert parent of repo just in case
    parent = os.path.dirname(REPO)
    if parent not in sys.path:
        sys.path.insert(0, parent)

# import patched module by file location
spec = importlib.util.spec_from_file_location("_patched_transformer", patched_path)
patched_mod = importlib.util.module_from_spec(spec)
# ensure module-level imports use our sys.path additions
sys.modules["_patched_transformer"] = patched_mod
try:
    spec.loader.exec_module(patched_mod)
except Exception as e:
    print("ERROR while importing patched transformer. Traceback follows:")
    raise

print("Imported patched transformer from", patched_path)

# load data and call recovery
X = np.load(DATA_X)
print("Loaded X.npy shape:", X.shape)

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

# Keep the patched file for debugging; remove if you want:
# os.remove(patched_path)

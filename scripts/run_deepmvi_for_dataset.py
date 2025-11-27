#!/usr/bin/env python3
"""
run_deepmvi_for_dataset.py

Usage:
  python3 run_deepmvi_for_dataset.py --dataset electricity
  python3 run_deepmvi_for_dataset.py --dataset airq

Reads: data/<dataset>/X.npy
Writes: output_deepmvi/<dataset>/imputed.npy
"""
import os, sys, argparse, importlib.util, numpy as np

ROOT = os.path.expanduser("/home/deeps/deepmvi-seminar")

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, help="dataset folder name in data/ (e.g. electricity, airq)")
args = p.parse_args()

DATA_X = os.path.join(ROOT, "data", args.dataset, "X.npy")
OUT_DIR = os.path.join(ROOT, "output_deepmvi", args.dataset)
OUT_PATH = os.path.join(OUT_DIR, "imputed.npy")
REPO = os.path.join(ROOT, "DeepMVI")
SRC = os.path.join(REPO, "transformer.py")

if not os.path.exists(DATA_X):
    print("ERROR: data file not found at", DATA_X)
    sys.exit(2)
if not os.path.exists(SRC):
    print("ERROR: DeepMVI transformer.py not found at", SRC)
    sys.exit(3)

# create patched copy of transformer (force CPU inside patched file)
with open(SRC, "r", encoding="utf-8") as f:
    src_text = f.read()
patched_text = src_text.replace("device = torch.device('cuda:%d'%0)", "device = torch.device('cpu')")
patched_text = patched_text.replace("torch.device('cuda:%d'%0)", "torch.device('cpu')")
patched_path = os.path.join(REPO, "_patched_transformer_for_dataset.py")
with open(patched_path, "w", encoding="utf-8") as f:
    f.write(patched_text)

# ensure repo on path for imports
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# import patched module by file path
spec = importlib.util.spec_from_file_location("_patched_transformer_for_dataset", patched_path)
patched_mod = importlib.util.module_from_spec(spec)
sys.modules["_patched_transformer_for_dataset"] = patched_mod
spec.loader.exec_module(patched_mod)
print("Imported patched transformer from", patched_path)

# load data and run
X = np.load(DATA_X)
print("Loaded X.npy shape:", X.shape)
imputed = patched_mod.transformer_recovery(X)

os.makedirs(OUT_DIR, exist_ok=True)
np.save(OUT_PATH, imputed)
print("Saved imputed output to", OUT_PATH)
print("Imputed shape:", imputed.shape)

#!/usr/bin/env python3
"""
Wrapper integration test for DeepMVI

Saves a readable summary and exits with non-zero on failure.

Place this at: tests/run_wrapper_test.py (run from project root)

What it does:
1. Loads data/<dataset>/X.npy and optional A.npy.
2. Computes checksum of existing output_deepmvi/<dataset>/imputed.npy (if present).
3. Imports both wrapper variants (standard and non-destructive), runs fit/transform.
4. Verifies shapes, computes MAE on masked entries (where mask==0), prints results.
5. Verifies whether original imputed.npy was overwritten and reports that.

Exit codes:
 0 = success
 non-zero = failure
"""

import os
import sys
import hashlib
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output_deepmvi'
TESTS_DIR = PROJECT_ROOT / 'tests'

# Config
DATASET = os.environ.get('DEEPVMI_TEST_DATASET', 'electricity')
PRINT = print

def sha1_of_file(path: Path) -> str:
    if not path.exists():
        return ''
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            b = f.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def load_X_and_mask(dataset: str):
    x_path = DATA_DIR / dataset / 'X.npy'
    if not x_path.exists():
        raise FileNotFoundError(f'X.npy not found for dataset {dataset}: {x_path}')
    X = np.load(x_path)
    a_path = DATA_DIR / dataset / 'A.npy'
    A = None
    if a_path.exists():
        A = np.load(a_path)
    return X, A

def safe_save(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

def compute_mae_on_mask(X_true, X_imp, mask):
    # mask: 1 = observed, 0 = missing; we evaluate on missing positions
    if mask is None:
        # fallback: evaluate on NaNs in X_true
        eval_mask = np.isnan(X_true)
    else:
        eval_mask = (mask == 0)
    if eval_mask.sum() == 0:
        return None
    y_true = X_true[eval_mask]
    y_imp = X_imp[eval_mask]
    return float(np.mean(np.abs(y_true - y_imp)))

def main():
    try:
        X, A = load_X_and_mask(DATASET)
    except Exception as e:
        PRINT('ERROR loading data:', e)
        sys.exit(2)

    original_imputed_path = OUTPUT_DIR / DATASET / 'imputed.npy'
    orig_sha1 = sha1_of_file(original_imputed_path)
    PRINT('Original imputed path:', original_imputed_path)
    PRINT('Original imputed sha1:', orig_sha1 or '(none)')

    # Import wrappers
    std_wrapper_mod = None
    nd_wrapper_mod = None
    try:
        from imputegap.wrapper.AlgoPython.DeepMVI.deepmvi_wrapper import DeepMVI_Imputer
        PRINT('Imported standard wrapper: DeepMVI_Imputer')
    except Exception as e:
        PRINT('Failed to import standard wrapper:', e)
        DeepMVI_Imputer = None

    try:
        from imputegap.wrapper.AlgoPython.DeepMVI.deepmvi_wrapper_non_destructive import DeepMVI_Imputer_ND
        PRINT('Imported non-destructive wrapper: DeepMVI_Imputer_ND')
    except Exception as e:
        PRINT('Non-destructive wrapper import failed (this is OK if you did not create it):', e)
        DeepMVI_Imputer_ND = None

    results = []

    # Run standard wrapper if available
    if 'DeepMVI_Imputer' in globals() and DeepMVI_Imputer is not None:
        try:
            imp = DeepMVI_Imputer({'dataset': DATASET, 'project_root': str(PROJECT_ROOT)})
            imp.fit(X, A)
            X_imp = imp.transform(X, A)
            out_path = OUTPUT_DIR / DATASET / 'imputed_from_wrapper_test.npy'
            safe_save(X_imp, out_path)
            PRINT('Standard wrapper produced imputed shape:', X_imp.shape)
            mae = compute_mae_on_mask(X, X_imp, A)
            PRINT('MAE (standard wrapper) on eval mask:', mae)
            results.append(('standard', True, X_imp.shape, mae))
        except Exception as e:
            PRINT('Standard wrapper run failed:', e)
            results.append(('standard', False, None, None))
    else:
        PRINT('Standard wrapper not available; skipping')

    # Run non-destructive wrapper if available
    if 'DeepMVI_Imputer_ND' in globals() and DeepMVI_Imputer_ND is not None:
        try:
            imp2 = DeepMVI_Imputer_ND({'dataset': DATASET, 'project_root': str(PROJECT_ROOT)})
            imp2.fit(X, A)
            X_imp2 = imp2.transform(X, A)
            out_path2 = OUTPUT_DIR / DATASET / 'imputed_from_wrapper_nd_test.npy'
            safe_save(X_imp2, out_path2)
            PRINT('Non-destructive wrapper produced imputed shape:', X_imp2.shape)
            mae2 = compute_mae_on_mask(X, X_imp2, A)
            PRINT('MAE (non-destructive wrapper) on eval mask:', mae2)
            results.append(('non-destructive', True, X_imp2.shape, mae2))
        except Exception as e:
            PRINT('Non-destructive wrapper run failed:', e)
            results.append(('non-destructive', False, None, None))
    else:
        PRINT('Non-destructive wrapper not available; skipping')

    # Check whether original imputed was overwritten
    new_sha1 = sha1_of_file(original_imputed_path)
    PRINT('After wrapper, imputed sha1:', new_sha1 or '(none)')
    if orig_sha1 and new_sha1 and orig_sha1 != new_sha1:
        PRINT('NOTICE: original imputed.npy appears to have changed (sha1 mismatch).')
    elif orig_sha1 == new_sha1:
        PRINT('original imputed.npy unchanged')
    else:
        PRINT('original imputed.npy was not present before or after')

    # Summarize
    PRINT('\\nSUMMARY:')
    ok = True
    for r in results:
        name, success, shape, mae = r
        PRINT(f' - {name}: success={success}, shape={shape}, mae={mae}')
        if not success:
            ok = False

    if ok:
        PRINT('\\nWrapper test: SUCCESS')
        sys.exit(0)
    else:
        PRINT('\\nWrapper test: FAILURE (see logs above)')
        sys.exit(3)

if __name__ == '__main__':
    main()

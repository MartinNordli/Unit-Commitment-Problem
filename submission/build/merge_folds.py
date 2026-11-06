"""Merge per-fold out-of-fold files (name_y<fold>.npy) into one run and log it. Usage: merge_folds.py <name> <oof|fwd> [extra json]"""
import sys; sys.path.insert(0, "src"); sys.dont_write_bytecode = True
import glob, json, re, numpy as np
from common import load_cases, ROOT
from harness import report
name, view = sys.argv[1], sys.argv[2]; extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
files = sorted(glob.glob(f"{ROOT}/artifacts/{view}/{name}_y*.npy")); assert files, name
oof = np.full((2922, 14, 168), np.nan, np.float32)
for f in files:
    a = np.load(f); ok = np.isfinite(a).all((1, 2)); assert not (np.isfinite(oof[ok]).all((1, 2))).any(), f"overlap in {f}"; oof[ok] = a[ok]
c = load_cases(); np.save(f"{ROOT}/artifacts/{view}/{name}.npy", oof)
report(name, oof, c["Y"].astype(np.float32), c, extra={**extra, "folds_merged": [re.search(r"_y(\d+)", f).group(1) for f in files], "forward": view == "fwd"})

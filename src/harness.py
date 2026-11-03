"""Locked evaluation harness: folds, metric, experiment log. Listed in harness.lock; changes need the human's approval.

Validation views
  loyo     leave one year out, 2015-2022, 7-day purge on both sides. Same folds as the earlier project.
  forward  train strictly before the validation year (2019-2022). The closest analogue to the real test split.
"""
import hashlib, json, os, subprocess
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(ROOT, "harness.lock"); FOLDS = os.path.join(ROOT, "harness", "folds.json")

def _sha(p): return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()

def verify_lock():
    want = json.load(open(LOCK)); bad = [p for p, h in want.items() if _sha(p) != h]
    if bad: raise RuntimeError(f"Locked files changed: {bad}. These need the human's approval.")

def _load(view):
    verify_lock(); d = json.load(open(FOLDS))[view]
    return {int(k): (np.array(v["train"]) - 1, np.array(v["val"]) - 1) for k, v in d.items()}  # Run No is 1-based, arrays 0-based

def folds(): return _load("loyo")
def forward_folds(): return _load("forward")
def auc(y, p): return float(roc_auc_score(np.asarray(y).ravel(), np.asarray(p).ravel()))

def _git():
    try:
        h = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        d = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain", "--", "src", "harness", "harness.lock"], text=True, stderr=subprocess.DEVNULL).strip()
        return h, bool(d)
    except Exception:
        return None, True

def report(name, oof, Y, c, log=True, extra=None):
    """Micro AUC overall and per year, appended to experiments.jsonl with the commit it came from."""
    verify_lock(); y = np.array([int(s[:4]) for s in c["start"]])[:2922]
    idx = np.where(np.isfinite(np.asarray(oof)).all(axis=(1, 2)))[0]
    r = {"name": name, "oof_auc": auc(Y[idx], oof[idx]), "per_year": {int(k): round(auc(Y[idx][y[idx] == k], oof[idx][y[idx] == k]), 5) for k in sorted(set(y[idx]))}, "n_cases": int(len(idx))}
    r["worst_year"] = min(r["per_year"].values()); r["commit"], r["dirty"] = _git()
    if extra: r.update(extra)
    if log:
        with open(os.path.join(ROOT, "experiments.jsonl"), "a") as f: f.write(json.dumps(r) + "\n")
    print(json.dumps(r)); return r

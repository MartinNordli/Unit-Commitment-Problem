"""The acceptance gate. A candidate passes only if it beats the reference by more than measured seed noise, improves at
least 6 of 8 years, and does not lose more than the noise when trained strictly on the past."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import ROOT
a = argparse.ArgumentParser(); a.add_argument("cand"); a.add_argument("ref"); a.add_argument("--noise", type=float, required=True); a.add_argument("--cand-fwd"); a.add_argument("--ref-fwd"); a = a.parse_args()
rows = {}
for l in open(os.path.join(ROOT, "experiments.jsonl")):
    r = json.loads(l); rows[r["name"]] = r
c, f = rows[a.cand], rows[a.ref]; d = c["oof_auc"] - f["oof_auc"]; ys = [y for y in c["per_year"] if y in f["per_year"]]; up = sum(c["per_year"][y] > f["per_year"][y] for y in ys)
msg = [f"OOF {c['oof_auc']:.5f} vs {f['oof_auc']:.5f}, delta {d:+.5f} (noise {a.noise:.5f})", f"years up {up}/{len(ys)}", f"worst year {c['worst_year']} vs {f['worst_year']}"]
ok = d > a.noise and up >= 6 and not c.get("dirty", True)
if c.get("dirty", True): msg.append("REFUSED: result came from uncommitted code")
if a.cand_fwd:
    fd = rows[a.cand_fwd]["oof_auc"] - rows[a.ref_fwd]["oof_auc"]; msg.append(f"forward delta {fd:+.5f}"); ok = ok and fd > -a.noise
print(" | ".join(msg)); print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

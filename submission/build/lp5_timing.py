"""R25: solve times of LP v5 against v4 and the projected LP stage of a notebook run on 4 cores."""
import sys; sys.path.insert(0, "src"); sys.dont_write_bytecode = True
import numpy as np
names = sys.argv[1:] or ["lp5", "lp5_w90", "lp5_w110"]
t4 = np.load("artifacts/lp/lp4_b10.npz")["time"]
print(f"lp4_b10 mean {t4.mean():.2f} s, median {np.median(t4):.2f}, max {t4.max():.1f}")
for n in names:
    z = np.load(f"artifacts/lp/{n}.npz"); t = z["time"]; st = z["int_status"] if "int_status" in z.files else None
    print(f"{n:12s} mean {t.mean():.2f} s, median {np.median(t):.2f}, p99 {np.percentile(t, 99):.1f}, max {t.max():.1f} (case {int(z['idx'][t.argmax()])}), "
          f"cases > 60 s: {(t > 60).sum()}, > 300 s: {(t > 300).sum()}, not proven optimal: {int((st > 0).sum()) if st is not None else '?'}, ratio to v4 mean {t.mean() / t4.mean():.2f}")

"""Check the notebook's data loading, LP and features against the research code and artifacts. Reads only."""
import os, re, sys, time
S, R = sys.argv[1], "/cluster/home/martnal/sintef"
os.chdir(R); os.environ["UC_DATA_DIR"] = R + "/data"; os.environ["UC_WORK_DIR"] = S + "/w0"
src = open(S + "/nb1.py").read()
for a, b in dict(TREE_GROUPS='("base","lp","ls")', NET_GROUPS='("base","lp","lps","mf")', NET_EPOCHS="24", TREE_WEIGHT=".7", STUDENT_ID="x", TEAM="x", RUNTIME="x").items(): src = src.replace(f"@@{a}@@", b)
cells = [re.sub(r"^.*\n", "", c, count=1) for c in re.split(r"^# %%", src, flags=re.M)[1:] if "[markdown]" not in c.split("\n")[0]]
env = {}
# find also has to see the two csv files in the repo root
code = "\n".join(cells[:3]).replace('DATA_DIR = Path(os.environ.get("UC_DATA_DIR", "."))', f'DATA_DIR = Path("{R}/data")')
code = code.replace('find("prediction_mapping.csv")', f'"{R}/prediction_mapping.csv"')
exec(code, env)
import numpy as np
c, ref = env["CASES"], np.load("artifacts/cases.npz")
for k in c:
    same = np.array_equal(c[k], ref[k]); print("cases", k, "identical" if same else "DIFFERENT", flush=True)
exec(cells[3].split("t0 = time.time()\nZ = solve_all")[0], env)
rng = np.random.default_rng(0); idx = np.sort(rng.choice(3647, 40, replace=False))
for name, ws in (("lp4_b10", 1.0), ("lp4_w90", .9), ("lp4_w110", 1.1)):
    z = np.load(f"artifacts/lp/{name}.npz"); env["_LP_ARGS"].update(ws=ws, keep=("u", "pfrac", "V", "dual", "conv", "flows")); bad = {k: 0 for k in env["_LP_ARGS"]["keep"]}
    for i in idx:
        r = env["solve_case"](int(i))
        for k in bad: bad[k] += not np.array_equal(r[k], z[k][i])
    print(name, "cases with any difference out of", len(idx), bad, flush=True)
exec(cells[4].split("t0 = time.time()\nX_ALL")[0], env)
sys.path.insert(0, "src"); sys.dont_write_bytecode = True
from features import build as rb
from common import load_cases as rl
z = np.load("artifacts/lp/lp4_b10.npz"); sens = [np.load(f"artifacts/lp/lp4_w{w}.npz") for w in (90, 110)]
X1, n1 = rb(rl(), z, None, sens)
X2, n2 = env["build"](c, {k: z[k] for k in ("u", "pfrac", "V", "dual", "conv", "flows")}, [{"u": s["u"]} for s in sens])
print("feature names identical", n1 == n2, len(n1), "| values identical", np.array_equal(X1, X2))

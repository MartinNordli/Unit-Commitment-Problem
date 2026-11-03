"""Does the notebook's ST-Net run with float16 autocast (GPUs without bfloat16) and on CPU, and how slow would CPU be? Random data, real shapes."""
import os, re, sys, time
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np, torch, xgboost as xgb, math
src = open("submission/build/nb1.py").read()
cell = next(c for c in re.split(r"^# %%", src, flags=re.M)[1:] if "class Net(" in c and "[markdown]" not in c.split("\n")[0])
code = cell.split("t0 = time.time()\nX = X_ALL")[0].split("\n", 1)[1]
env = dict(UNITS=["Hogga_G1","Tokke_G1","Tokke_G2","Tokke_G3","Tokke_G4","Lio_G1","Byrte_G1","Vinje_G1","Vinje_G2","Vinje_G3","Haukeli_G1","Songa_G1","Kjela_G1","Vesle Kjela_G1"],
           PRIORITY=[("Vinje_G1","Vinje_G2"),("Vinje_G2","Vinje_G3"),("Tokke_G1","Tokke_G3"),("Tokke_G3","Tokke_G2"),("Tokke_G3","Tokke_G4")], DEVICE="cuda", torch=torch, nn=torch.nn, F=torch.nn.functional, math=math, np=np)
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn; env.update(AveragedModel=AveragedModel, get_ema_multi_avg_fn=get_ema_multi_avg_fn)
exec(code, env)
def run(dev, dtype, n=64):
    env["DEVICE"] = dev; env["AMP_DTYPE"] = dtype
    X = torch.randn(n, 14, 168, 70).to(dev); Y = (torch.rand(n, 14, 168) > .5).float().to(dev); t = time.time()
    m = env["fit_net"](X, Y, 0, 1); p = env["predict_net"](m, X.cpu()); return round(time.time() - t, 1), bool(np.isfinite(p).all())
print("gpu float16", run("cuda", torch.float16), flush=True)
torch.set_num_threads(4); s, ok = run("cpu", torch.float16); print("cpu", s, "s for 2 train steps + predict on 64, finite", ok, flush=True)
t = time.time(); env["DEVICE"] = "cpu"; X = torch.randn(32, 14, 168, 70); net = env["Net"](70); opt = torch.optim.AdamW(net.parameters()); 
for _ in range(3): opt.zero_grad(); l, _v = net(X); l.mean().backward(); opt.step()
step = (time.time() - t) / 3; print("cpu s per train step", round(step, 2), "=> 3 seeds x 24 epochs x 92 steps, hours", round(step * 3 * 24 * 92 / 3600, 1), flush=True)
A = np.random.rand(6868224, 66).astype(np.float32); y = (np.random.rand(len(A)) > .5).astype(np.float32); d = xgb.DMatrix(A, label=y); t = time.time()
xgb.train(dict(objective="binary:logistic", device="cpu", tree_method="hist", eta=.08, grow_policy="lossguide", max_leaves=63, max_depth=0, min_child_weight=50., subsample=.5, colsample_bytree=.8, max_bin=255, verbosity=0, nthread=4), d, num_boost_round=5); r = (time.time() - t) / 5
print("xgb cpu s per round", round(r, 1), "=> 3 seeds x 300 rounds, hours", round(r * 900 / 3600, 1), flush=True)

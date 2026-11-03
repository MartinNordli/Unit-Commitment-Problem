# %% [markdown]
# # Predicting hydropower unit commitment in Tokke-Vinje: project report
#
# **Name:** Martin Nordli Almenningen  
# **Student id:** 562274  
# **Kaggle team name:** [197] sur salamander  
# **Group:** 197  
# **Course:** TDT4173 Modern Machine Learning in Practice
#
# This notebook is the report. It was run once, and every figure and most tables below are computed in the cells from the raw data and the saved results of my experiments. The hand-written tables quote the experiment log.
#
# ## Contents
# 1. Summary
# 2. The task
# 3. Exploratory data analysis
#    - 3.1 Search domain knowledge
#    - 3.2 Understand how the data was generated
#    - 3.3 Check if the data is intuitive
#    - 3.4 Explore individual features
#    - 3.5 Explore pairs and groups of features
#    - 3.6 Clean up features
# 4. Validation
# 5. Feature engineering (the LP surrogate, extraction and selection)
# 6. All models and algorithms tried
# 7. The final models and how the hyperparameters were chosen
# 8. Model interpretation (SHAP, permutation importance, partial dependence)
# 9. Error analysis
# 10. Robustness
# 11. Reproducibility of the short notebooks
# 12. Limitations

# %%
import sys, json, re, warnings
sys.path.insert(0, "src"); sys.dont_write_bytecode = True; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
from common import load_cases, years, auc, RES, INFLOW, MINFLOW, MINVOL
from lp_surrogate import UNITS
from features import TOPO, UP, DOWN, PMAX, PLANTS, group

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.dpi": 110, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2})
def heat(ax, M, xl, yl, title, vmin=None, vmax=None, fmt="{:.3f}", cmap="Blues", size=7):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax); ax.grid(False); ax.set_xticks(range(len(xl)), xl); ax.set_yticks(range(len(yl)), yl); ax.set_title(title)
    lo, hi = np.nanmin(M) if vmin is None else vmin, np.nanmax(M) if vmax is None else vmax
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=size, color="white" if (M[i, j] - lo) / (hi - lo + 1e-12) > .6 else INK)
    return im

pd.set_option("display.max_colwidth", 110); pd.set_option("display.width", 220)
c = load_cases(); Y = c["Y"]; P = c["price"]; yr = years(c); N_TR = 2922; ytr = yr[:N_TR]; YRS = list(range(2015, 2023)); starts = pd.to_datetime(c["start"])
rows = {}
for l in open("experiments.jsonl"):
    d = json.loads(l); rows[d["name"]] = d
XT, NT = np.load("artifacts/oof/XR_ls.npy").astype(np.float64), np.load("artifacts/oof/N6_min.npy").astype(np.float64); B = .7 * XT + .3 * NT
Z = np.load("artifacts/lp/lp4_b10.npz"); LPF = Z["pfrac"].astype(np.float64); LPU = Z["u"].astype(np.float64)
XF, NF = np.load("artifacts/fwd/XR_ls_f.npy").astype(np.float64), np.load("artifacts/fwd/N6f_min.npy").astype(np.float64)
print("LOYO blend", round(auc(Y, B), 5), "| forward blend", round(auc(Y[np.isfinite(XF).all((1, 2))], (.7 * XF + .3 * NF)[np.isfinite(XF).all((1, 2))]), 5))

# %% [markdown]
# ## 1. Summary
#
# **The main idea.** Solving the true problem (a mixed-integer program in SHOP) for every case is not possible without SHOP, but its linear relaxation is. I wrote a linear program (LP) of the whole watercourse, solved it three times for every case with HiGHS, and gave the solution to the models as features. The LP has no trained parameters: it reads only the case inputs and the topology file. The machine learning models then learn what separates SHOP from the LP.
#
# **The result.** Features built by hand, without an optimiser, reached a best blended cross-validation AUC of 0.986 and public leaderboard 0.98906. Adding the LP took the blend to 0.994 and 0.99638. The LP solution was the largest single step in the project.
#
# | Submission | Model | LOYO AUC (2015-22) | Forward AUC (2019-22) | Public LB |
# |---|---|---|---|---|
# | **Short notebook 1** | XGBoost + ST-Net, logit blend 0.7/0.3 | 0.99400 | 0.99379 | **0.99638** |
# | **Short notebook 2** | Same, ST-Net without absolute price columns | 0.99399 | 0.99379 | 0.99635 |
# | B2, earlier blend | 3 GBDT + 2 ST-Net (nested weights) | 0.99377 | - | 0.99626 |
# | E021a, best without LP | LightGBM + ST-Net (nested weights) | 0.98584 | - | 0.98906 |
#
# Models were selected on LOYO and forward validation. The public leaderboard was used only as a check. The two submissions are close to identical (Spearman 0.99889 between them), so notebook 2 is a small hedge against one specific risk (Section 10), not a second independent bet.
#
# Limitations are stated in Section 12. The most important ones are that the design of the LP was chosen while looking at labels from all training years, so no part of 2015-2022 is truly untouched, and that SHOP's behaviour at negative prices cannot be checked with these labels.

# %% [markdown]
# ## 2. The task
#
# SHOP is SINTEF's short-term hydropower scheduling software. For Tokke-Vinje (17 reservoirs, 9 plants, 14 units, 1000 MW) it solves a one-week schedule for each day from 2015-01-01 to 2024-12-25. The task is to predict the on/off decision for each of the 14 units in each of the 168 hours of a case. Scoring is micro-averaged ROC AUC over all (case, unit, hour) values.
#
# - **Train:** 2922 cases from 2015-2022 with SHOP's decisions as labels.
# - **Test:** 725 cases from 2023-01-01 to 2024-12-25. The split is purely in time, so this is extrapolation to new years.
# - **Inputs:** hourly prices and inflows, daily reservoir volumes and water values, minimum flow and volume constraints, and the watercourse topology.
#
# Since grading uses the private leaderboard and the two parts of the test set are not known, I optimised for generalisation to new years rather than for the public score.

# %% [markdown]
# ## 3. Exploratory data analysis
# ### 3.1 Search domain knowledge
#
# Sources: the task description, the dataset documentation, the SHOP documentation linked from it, and the topology files (`Tokke_topology.pdf`, `Tokke_Vinje_topology.yaml`). What I took from them:
#
# - **Water value.** Water used now cannot be sold later. The water value is the estimated future worth of a reservoir's marginal water, in EUR/MWh. Roughly, a unit should run when the price exceeds the value of the water it would use. This makes price against water value the core comparison.
# - **Cascades.** Water released from a plant reaches a lower reservoir, where it is still worth something. The true cost of running therefore depends on the watercourse, not just the reservoir feeding the plant.
# - **Efficiency and head.** A unit converts flow to power along a concave curve that depends on head (the height of the water) and on losses in tunnels and penstocks. Best efficiency is below maximum power, and units in a plant share tunnel losses.
# - **Non-convex parts.** Start costs, minimum power and the on/off status itself are what make the true problem a mixed-integer program. Identical units are a source of symmetry, which solvers usually break with a fixed order.
# - **Environmental constraints.** Minimum flows and minimum volumes can force production or bypass whatever the price is.
# - **Flood weeks.** If a reservoir is about to spill, water is close to free, so units can run at very low prices.

# %% [markdown]
# ### 3.2 Understand how the data was generated
#
# The labels are the output of an optimiser, SHOP v17.17.0, on one fixed model of the watercourse, so they are a deterministic function of the inputs and not a noisy measurement of the real world. This has two consequences that shaped the whole project:
#
# 1. In principle nothing is random in the label, so any remaining error is something I have not modelled. This motivates rebuilding the optimisation problem instead of only learning from examples.
# 2. Only what SHOP received can be a legitimate input. For a case starting on day $d$, this is the 168 hours of price, inflow and constraints, the start volumes at $d$, and the water values at the end of the week, $d+7$ (as the dataset documentation states). I read later volumes and intermediate water values as outside the input, and the feature code cannot reach them.
#
# Cases roll daily, so neighbouring cases share 6 of 7 days. The cell below shows that this matters for validation, and that SHOP often changes its mind about the same hour when it is re-solved one day later, since the horizon and start state have moved.

# %%
agree = {f"{l} day(s) later": float((Y[l:, :, :168 - 24 * l] == Y[:-l, :, 24 * l:]).mean()) for l in (1, 2, 3, 6)}
uc = pd.read_csv("data/kernel/Unit_commitment_decisions.csv", usecols=["starttime", "total_calculation_time"], parse_dates=["starttime"])
print("Share of identical decisions for the same clock hour when solved again:", {k: round(v, 3) for k, v in agree.items()})
print("SHOP solve time per case, seconds:", uc.total_calculation_time.describe()[["min", "50%", "max"]].round(1).to_dict())
print("Median per year:", uc.groupby(uc.starttime.dt.year).total_calculation_time.median().round(1).to_dict())
print("Label shape", Y.shape, "| share on", round(Y.mean(), 4), "| case on-share constant through week:", round(((Y.sum(2) == 0) | (Y.sum(2) == 168)).mean(), 3))
i = UNITS.index
print("\nFixed order among identical units, share of hours where the first is on whenever the second is:")
for a, b in [("Vinje_G1", "Vinje_G2"), ("Vinje_G2", "Vinje_G3"), ("Tokke_G1", "Tokke_G3"), ("Tokke_G3", "Tokke_G2"), ("Tokke_G3", "Tokke_G4"), ("Tokke_G4", "Tokke_G2")]:
    print(f"  {a} >= {b}: {(Y[:, i(a)] >= Y[:, i(b)]).mean():.4f}")

# %% [markdown]
# Reading these outputs:
#
# - Only 88 % of decisions for the same clock hour survive a re-solve one day later, and 74 % six days later. Target hours near the horizon end depend heavily on the case. So predictions for the same clock hour should not be expected to agree across cases, and a later experiment that smoothed them across neighbouring cases (Section 6) made every year worse.
# - A unit-week is constant (all on or all off) in 59 % of cases, so getting the weekly level right matters a lot.
# - Identical units follow a nearly deterministic start order. Without this, the model has no way of telling Tokke G1 from G2. I used it directly in the LP and as a structural constraint in ST-Net's output. The last pair holds only 98.1 % of hours, so I imposed it in the LP but left it out of the net.

# %% [markdown]
# ### 3.3 Check if the data is intuitive
#
# The two figures below test the domain story from 3.1. Do units run when the price exceeds the water value? Does on-share follow the market?

# %%
ri = {r: k for k, r in enumerate(RES)}; wk = np.arange(0, len(P), 7); dates = starts[wk]; wv = c["wv_end"]
fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw=dict(hspace=.25))
ax[0].plot(dates, P.mean(1)[wk], color=BLUE, label="Weekly mean price"); ax[0].plot(dates, wv.mean(1)[wk], color=ORANGE, label="Mean water value at end of week")
ax[0].axvspan(starts[N_TR], starts[-1], color=GRID, alpha=.6, lw=0); ax[0].text(starts[N_TR + 15], ax[0].get_ylim()[1] * .88, "Test years", fontsize=8, color=MUTED); ax[0].set_ylabel("EUR/MWh"); ax[0].set_title("Water values follow the market, and the years differ sharply"); ax[0].legend(loc="upper left")
ax[1].plot(starts[:N_TR][wk[wk < N_TR]], Y[wk[wk < N_TR]].mean((1, 2)), color=AQUA); ax[1].axvspan(starts[N_TR], starts[-1], color=GRID, alpha=.6, lw=0); ax[1].set_ylabel("Share of unit-hours on"); ax[1].set_title("SHOP's on-share by week (labels hidden in test years)"); ax[1].set_ylim(0, 1); plt.show()

# %%
fig, axs = plt.subplots(3, 3, figsize=(10, 6.2), sharex=True, sharey=True, gridspec_kw=dict(hspace=.3, wspace=.1)); bins = np.arange(-40, 41, 4); rowsT = []
for ax, pl in zip(axs.ravel(), PLANTS):
    u0 = [k for k, u in enumerate(UNITS) if TOPO.gen_plant[u] == pl][0]; w = wv[:N_TR, ri[UP[u0]]]; d = (P[:N_TR] - w[:, None]).ravel(); y = Y[:, u0].ravel(); b = np.digitize(d, bins); xs, ys = [], []
    for k in range(1, len(bins)):
        if (b == k).sum() > 200: xs.append((bins[k - 1] + bins[k]) / 2); ys.append(y[b == k].mean())
    ax.plot(xs, ys, color=BLUE); ax.axvline(0, color=MUTED, lw=.8); ax.set_title(pl); ax.set_ylim(-.03, 1.03)
    on = Y[:, u0].astype(bool); mix = (on.sum(1) >= 12) & (on.sum(1) <= 156); th = np.array([(np.median(P[k][on[k]]) + np.median(P[k][~on[k]])) / 2 for k in np.where(mix)[0]])
    wu = wv[:N_TR, ri[UP[u0]]][mix]; wd = wv[:N_TR, ri[DOWN[u0]]][mix] if DOWN[u0] else np.zeros(mix.sum()); A = np.c_[wu, wd, np.ones(len(wu))]; co = np.linalg.lstsq(A, th, rcond=None)[0]
    rowsT.append(dict(plant=pl, weeks=int(mix.sum()), coef_wv_up=co[0], coef_wv_down=co[1], intercept=co[2], R2=1 - ((th - A @ co) ** 2).sum() / ((th - th.mean()) ** 2).sum()))
fig.suptitle("Share of hours on against price minus upstream water value (EUR/MWh), first unit of each plant, 2015-2022", fontsize=11, fontweight="bold", x=.09, ha="left", y=.97); plt.show()
display(pd.DataFrame(rowsT).round(2))

# %% [markdown]
# Mostly yes, with informative exceptions:
#
# - Water values track the market (correlation 0.93 between weekly price and mean water value), so a high price alone does not mean a unit should run. The relevant quantity is relative.
# - On-share rises across zero margin, as expected. Byrte, Songa, Kjela and Vesle Kjela are close to clean steps. Hogga, Tokke, Lio, Vinje and Haukeli still run 15-50 % of the time at a strongly negative margin. This could be forced production from minimum flow and spill, which a simple margin cannot see, and is what a model of the whole watercourse is for. Vinje's first unit also stays below 0.85 at positive margins.
# - The table is a rough check. For each plant-week with mixed on/off hours I take a threshold midway between the median price of on hours and off hours, then regress it on upstream and downstream water value. The upstream coefficient is the larger one for every plant, but Vinje (0.53 and 0.50) and Tokke (0.58 and 0.26) also give the downstream value clear weight. This estimator is crude, so it is only a hint.
#
# My first reading was that the value of water in a reservoir should be converted to money with that reservoir's own energy factor, locally or summed down the cascade. An LP built that way scored far worse than one with telescoping values (LP alone: AUC 0.606 and 0.743 for the two first conventions, against 0.932 for telescoping, Section 5). So the intuitive reading of the water values was wrong, and only testing against the labels showed it.

# %% [markdown]
# ### 3.4 Explore individual features

# %%
h = pd.read_csv("data/kernel/Historical_day_ahead_price_2015_2025.csv", parse_dates=["date"]).set_index("date").iloc[:, 0]; h.index = pd.to_datetime(h.index, utc=True); h = h["2015":"2024-12-31"]; g = h.groupby(h.index.year)
display(pd.DataFrame({"min": g.min(), "median": g.median(), "mean": g.mean(), "max": g.max(), "% negative hours": g.apply(lambda s: (s < 0).mean() * 100)}).round(2).rename_axis("Price, EUR/MWh"))
months = starts[:N_TR].month.values; ml = "J F M A M J J A S O N D".split()
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(wspace=.25, width_ratios=[1, 1.15]))
heat(ax[0], np.array([[Y[ytr == y, g_].mean() for y in YRS] for g_ in range(14)]), [str(y)[2:] for y in YRS], UNITS, "Share of hours on, by year", 0, 1, "{:.2f}")
heat(ax[1], np.array([[Y[months == m, g_].mean() for m in range(1, 13)] for g_ in range(14)]), ml, UNITS, "Share of hours on, by month", 0, 1, "{:.2f}", size=6.5); plt.show()
hh = (P[:N_TR] - np.median(P[:N_TR], 1, keepdims=True)); q = pd.read_csv("data/kernel/Historical_inflow_1958_2025.csv", parse_dates=["date"]).set_index("date"); q.index = pd.to_datetime(q.index, utc=True); q15 = q["2015":"2022"]
fig, ax = plt.subplots(1, 3, figsize=(11, 3), gridspec_kw=dict(wspace=.28))
ax[0].plot(range(1, 13), q15.sum(axis=1).groupby(q15.index.month).mean(), color=BLUE); ax[0].set_xticks(range(1, 13), ml); ax[0].set_title("Total inflow by month"); ax[0].set_ylabel("m3/s")
ax[1].hist(np.arcsinh(h.loc["2015":"2022"] / 10), bins=60, color=BLUE, density=True, label="2015-22", alpha=.75); ax[1].hist(np.arcsinh(h.loc["2023":] / 10), bins=60, color=ORANGE, density=True, label="2023-24", alpha=.6); ax[1].set_title("Price, asinh(price/10)"); ax[1].legend()
ll = Y.mean(2).ravel(); ax[2].hist(ll, bins=40, color=BLUE); ax[2].set_title("Unit on-share within a week"); ax[2].set_yscale("log"); ax[2].set_ylabel("unit-weeks (log)"); plt.show()
print("Hours with negative prices inside cases, train:", round(float((P[:N_TR] < 0).mean()) * 100, 3), "% | test:", round(float((P[N_TR:] < 0).mean()) * 100, 2), "%")
print("Switches per unit-week:", {u: round(float(v), 1) for u, v in zip(UNITS, np.abs(np.diff(Y, axis=2)).sum(2).mean(0))})

# %% [markdown]
# What this shows:
#
# - **Strong regimes.** Median price ranged from 7 EUR/MWh (2020, flooded and nearly free) to 181 EUR/MWh (2022, energy crisis), and on-share dropped from 0.60 to 0.34, in the opposite direction of what price alone would suggest. Price level alone cannot explain the labels.
# - **Heavy tails, both ways.** Prices reach 898 EUR/MWh in 2024 and -62 in 2023. An `asinh` transform handles both.
# - **The test set is different in one way that matters:** negative prices are 2.2 % of test hours and 0.014 % of train hours. Otherwise, test price levels stay close to the range training has seen (maximum 898 against 844 EUR/MWh). Section 10 returns to this.
# - **Inflow is seasonal,** with a large peak in May and June and a smaller one in autumn. The effect on on-share depends on the plant: Tokke G1 peaks in May (0.83), while Lio and Vinje G3 dip then. So season alone does not set the level. Base rates also differ widely between units (Vinje G3 26 %, Lio 75 %) and between years for one unit (Kjela 14 % in 2021, 73 % in 2018).
# - **The on-share is bimodal** with spikes at 0 and 1, which agrees with 3.2.

# %% [markdown]
# ### 3.5 Explore pairs and groups of features

# %%
wvc = np.corrcoef(wv.T); labc = np.corrcoef(Y.transpose(1, 0, 2).reshape(14, -1))
vmat = np.array(pd.read_csv("data/kernel/Historical_volume_2015_2024.csv", index_col=0)[RES]); mx = np.array([float(TOPO.res[r]["max_vol"]) for r in RES])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(wspace=.28))
heat(ax[0], labc, [u.replace("_G", "") .replace("Vesle Kjela", "VK")[:6] for u in UNITS], UNITS, "Correlation of on/off labels between units", -.1, 1, "{:.1f}", size=6.5); ax[0].tick_params(axis="x", rotation=90)
ax[1].scatter(P.mean(1), wv.mean(1), s=5, c=np.where(np.arange(len(P)) < N_TR, BLUE, ORANGE), alpha=.5, linewidths=0); ax[1].set_xscale("symlog", linthresh=30); ax[1].set_yscale("symlog", linthresh=30); ax[1].set_xlabel("Weekly mean price, EUR/MWh (symlog)"); ax[1].set_ylabel("Mean water value, EUR/MWh (symlog)"); ax[1].set_title("Price and water value by case")
for lab, col in (("Train", BLUE), ("Test", ORANGE)): ax[1].scatter([], [], s=20, c=col, label=lab)
ax[1].legend(); plt.show()
print("Correlation between weekly mean price and mean water value:", round(float(np.corrcoef(P.mean(1), wv.mean(1))[0, 1]), 2)); print("Correlation between reservoirs' water values, min and median:", round(float(wvc.min()), 2), round(float(np.median(wvc)), 2))
print("Share of hours with a non-zero minimum flow:", {k: round(float(v), 2) for k, v in zip(MINFLOW, (c["minflow"] > 0).mean((0, 1)))}); print("Share of hours with a non-zero minimum volume:", {k: round(float(v), 2) for k, v in zip(MINVOL, (c["minvol"] > 0).mean((0, 1)))})
print("Reservoir fill, mean over days:", {k: round(float(v), 2) for k, v in zip(RES, (vmat / mx).mean(0))})

# %% [markdown]
# - Units in the same plant are strongly correlated, and so are Tokke and Vinje, which are hydraulically linked (0.4-0.8). Correlations elsewhere are low (0.1-0.4). Units are decided together, so sharing information across them (ST-Net's attention, the system features in the trees) should help, and did.
# - Water values across reservoirs correlate at a median of 0.88 and at least 0.76, which is why ratios and differences carry the signal rather than the individual values.
# - Constraints are calendar-driven and mostly a fixed function of date. As separate features, they add very little (E004 and R3 in Section 6). Their real effect is on the water balance, which the LP handles.
# - In the scatter of price against water value, test cases mostly sit in the same band as train cases, with a few at unusually low water value for their price.
#
# **Drift check (adversarial validation).** A classifier tries to tell test hours from train hours, on alternating 60-day blocks so overlapping cases are kept together. High AUC means the feature set is easy to tell apart, which is a warning, though not proof of harm. Calendar and unit columns are left out because alternating blocks let the classifier memorise which dates are test.

# %%
dr = json.load(open("artifacts/drift.json")); display(pd.DataFrame(dr).T.assign(auc=lambda d: d.auc.astype(float).round(3)).rename_axis("Feature set").rename(columns={"n": "columns", "auc": "adversarial AUC", "top": "most used columns"}))

# %% [markdown]
# @@DRIFT_TEXT@@

# %% [markdown]
# ### 3.6 Clean up features
#
# - **Missing values and duplicates.** No gaps or duplicate timestamps in any file in the case windows. `load_cases` asserts this for all 3647 cases, and all target columns are strictly 0 or 1. No imputation was needed.
# - **Constant columns.** Volumes for `Hyljelihyl` and `Vatjern` are constant, since the documentation says they have no records and are assumed 50 % full. This is misleading for Haukeli, whose upstream reservoir is Vatjern, so `fill_up` carries no information for that unit (and `fill_down` none for Kjela, through Hyljelihyl). I kept the columns, since a tree can ignore them. The LP covers the water situation through the balance of the whole watercourse.
# - **Time alignment.** All files are in UTC. Local Oslo time (with daylight saving) is derived for `hour_local` and `weekday_local`, since the market follows local time.
# - **Scaling.** Prices and water values are heavy-tailed and go negative, so absolute levels use `asinh(x/10)`. Case-relative price columns divide by the case's interquartile range. For the net, every column is standardised on the training fold and clipped to ±6.
# - **Feature removal.** See Section 5.3. Groups were kept or removed through the acceptance gate, and absolute price columns were dropped from the trees because they add nothing measurable there but are what separates train from test most sharply.

# %% [markdown]
# ## 4. Validation
#
# All models use the same locked folds, and the fold file, the metric and the lock are protected by SHA256 hashes checked at run time.
#
# - **Leave-one-year-out (LOYO):** 8 folds, 2015-2022, each validating on a full year. A 7-day purge on both sides removes every training case that shares an hour with validation.
# - **Forward:** train strictly before the validation year, for 2019, 2020, 2021 and 2022. This is the closest analogue to the actual test split. It is reported separately because LOYO lets a model learn from later years, and forward validation sometimes disagrees.
# - **Acceptance gate:** a change is accepted only if LOYO improves by more than measured seed noise (0.0001 for XGBoost, measured with 5 seeds), at least 6 of 8 years improve, forward does not lose more than the noise, and the run came from committed code. Hypotheses and expected effects were written in the experiment log before each run.
# - **Label control:** with shuffled whole-case labels, AUC falls to 0.589, which is what unit and hour base rates alone can give. So no leakage is visible.
#
# I did not tune on the public leaderboard, and it never decided between models. Blending weights in earlier work were nested (chosen on 7 years and tested on the 8th). For the final model I fixed 0.7 after seeing a flat range of weights (Section 7), so that number is not nested.

# %% [markdown]
# ## 5. Feature engineering
# ### 5.1 Feature extraction: the LP surrogate
#
# For each case I solve a linear program of the full watercourse for 168 hours:
#
# $$\max \sum_t \lambda_t \sum_{g} P_{g,t} \; + \; \sum_r v_r V_{r,168} \; - \; \text{flow costs} \; - \; \text{switching costs}$$
#
# subject to water balance for all 17 reservoirs, volume bounds, minimum flow, minimum volume, and a relaxed commitment $u_{g,t}\in[0,1]$. Power is a concave piecewise-linear function of flow, built per unit from the turbine and generator curves in the YAML file at the head implied by the start volume, including penstock and shared-tunnel losses. Head stays fixed within a week. Everything comes from the inputs SHOP gets and the YAML file. No labels and no fitted parameters are used.
#
# **Telescoping water values.** The terminal value is the key modelling decision. Water that ends the week in reservoir $r$ is worth its water value at the first plant below it, plus the value of the reservoir below that plant:
#
# $$\text{value}(r) = w_r \cdot \text{conv}(\text{first plant below } r) + \text{value}(\text{reservoir below that plant})$$
#
# so the marginal value of running a plant nets out to about price minus its upstream water value. This matched the labels far better than the two conventions I tried first:
#
# | LP version | What it has | LP alone, AUC of $u$ | AUC of power fraction |
# |---|---|---|---|
# | `lp_local` | Local value at each reservoir | 0.6062 | 0.6495 |
# | `lp_global` | Global value down the cascade | 0.7431 | 0.7581 |
# | `lp_tele` | Telescoping values | 0.9323 | 0.9470 |
# | `lp_v2`, `lp_v3` | Added per-unit efficiency curves, head loss and priority order | 0.9442, 0.9558 | 0.9561, 0.9688 |
# | **`lp4_b10`, final** | Added corrected Tokke order and river costs, bypass cost 10 | **0.9576** | **0.9705** |
# | `lp4_b100`, `lp4_b1000` | Bypass cost sweep | 0.9527, 0.9517 | 0.9656, 0.9646 |
# | `milp_v2`, `mp4_b10` | Integer schedule (MILP) on v2 and v4 | 0.9296, 0.9441 | 0.9445, 0.9606 |
#
# Scores are on all 2922 training cases and need no fitting. They are all from this one set of labels, which I return to in Section 12. The relaxed LP scored higher than the integer MILP, since a fractional $u$ acts like a soft score, and adding MILP features to the trees did not help (L3 below).
#
# **Solves per case:** the final features use three: water values at x1.0, x0.9 and x1.1.

# %%
interp = json.load(open("artifacts/interp_final.json")); G = interp["groups"]
display(pd.DataFrame([dict(group=k, columns=len(v), examples=", ".join(v[:7]) + (" ..." if len(v) > 7 else "")) for k, v in G.items()]).set_index("group"))
print("Dropped from trees (ST-Net in notebook 1 keeps them):", open("artifacts/scale_cols.txt").read().strip())

# %% [markdown]
# ### 5.2 Features built on top of the LP
#
# - **Schedule (`lp_*`, 36 columns):** LP commitment and power, its neighbours in time and rolling means, the marginal cost of running from reservoir shadow prices (`lp_margin`, `lp_margin_ratio`), hours to the nearest LP switch, the LP's output for the plant, for neighbouring plants and for the whole system, the fill trajectory of connected reservoirs, and bypass flow.
# - **Sensitivity (`ls_*`, 7 columns):** the same LP with water values scaled by 0.9 and 1.1. A decision that stays the same under both is firm. One that flips is a near-tie, which is where the LP is most likely to disagree with SHOP. `ls_mean` is the mean of the three schedules.
# - **Market and calendar (15):** ranks within week and day, deviations from rolling means, price relative to the week, and local time.
# - **Unit and water state (14):** unit identity, capacity, start cost per MW, price against water value, fill and inflow for the connected reservoirs.
#
# Earlier, without the LP, I built features by hand for break-even prices, hydraulic state in hours of full-capacity production, neighbouring plants and whole-system state (Section 6.1). The LP replaced all of them with one mechanism.

# %% [markdown]
# ### 5.3 Feature selection
#
# Groups were removed or added by the gate, one hypothesis at a time. All with XGBoost or ST-Net at 3 seeds, unless noted.

# %%
def row(n, ref=None):
    r = rows[n]; return r["oof_auc"], (r["oof_auc"] - rows[ref]["oof_auc"]) if ref else np.nan
T = [("Add `ls_*` (trees)", "XR_ls", "XR_min", "XR_ls_f", "XR_min_f", "Accepted. 66 features, 7/8 years"),
     ("Add `ls_*`, keep `mf_*`, `lps_*` (trees)", "XR_sens2", "XR_ref", "XR_sens_f2", "XR_ref_f", "Passes, but no better than the row above with 13 fewer features"),
     ("Drop `mf_*` constraint columns (trees)", "XR_nomf", "XR_ref", "XR_nomf_f", "XR_ref_f", "Inside noise, dropped for simplicity"),
     ("Drop `lps_*` plant shares (trees)", "XR_nolps", "XR_ref", "XR_nolps_f", "XR_ref_f", "Inside noise, dropped for simplicity"),
     ("Drop both (trees)", "XR_min", "XR_ref", "XR_min_f", "XR_ref_f", "No change, 59 vs 72 features"),
     ("Keep absolute price columns (trees)", "XR_scale", "XR_ref", "XR_scale_f", "XR_ref_f", "Inside noise, so left out of the trees"),
     ("Keep absolute price columns (net)", "N4_v4_scale", "N3_v4_mf", "N4f_e24_scale", "N3f3_e24", "Clear gain, kept in the net in notebook 1"),
     ("`base,lp,ls` only (net)", "N6_min", "N4_v4_scale", "N6f_min", "N4f_e24_scale", "Accepted. 7/8 years, 4/4 forward"),
     ("Add `ls_*`, keep `mf_*`, `lps_*` (net)", "N5_ls", "N4_v4_scale", "N5f_ls", "N4f_e24_scale", "Fails, 4/8 years"),
     ("Drop absolute price columns again (net, hedge)", "N7_noabs", "N6_min", "N7f_noabs", "N6f_min", "Fails the gate by design. Used as the hedge in notebook 2")]
display(pd.DataFrame([{"Change": a, "LOYO": rows[n]["oof_auc"], "d LOYO": rows[n]["oof_auc"] - rows[rf]["oof_auc"], "d forward": rows[nf]["oof_auc"] - rows[rff]["oof_auc"], "columns": rows[n].get("n_features"), "Decision": note} for a, n, rf, nf, rff, note in T]).round(5).set_index("Change"))

# %% [markdown]
# One finding I had wrong to begin with. I expected removing absolute price levels to help everywhere, since they are the biggest drift signal. For the trees, they indeed add nothing. For the net, however, forward AUC rose by 0.0007 with them (mostly from 2020), so they stayed. The hedge in notebook 2 is what I would use if that decision turned out wrong in the test years.

# %% [markdown]
# ## 6. All models and algorithms tried
#
# I worked in two phases. In the first phase (physics features and no optimiser), the models competed on hand-built features. In the second phase, every model got the LP as input.
#
# ### 6.1 Phase 1: without the LP
#
# All results are LOYO over 8 years unless marked *tuning* (2018, 2020 and 2022 only, used for early selection). The gate noise here was 0.00084 since LightGBM was more unstable on those features.
#
# | Id | Model and idea | AUC | Outcome and what I learned |
# |---|---|---|---|
# | E000 | LightGBM on base features | 0.95415 | Reference. 2020 is the weakest year (0.92178) |
# | E006 | Logistic regression (25 % of rows) | 0.89708 *tuning* | Baseline. Non-linear interactions matter |
# | E006 | Random forest (10 % of rows) | 0.93927 *tuning* | Baseline, below LightGBM's 0.94791 on the same folds |
# | E007 | XGBoost, same style of settings | 0.94888 *tuning* | On par with LightGBM, correlated 0.98, so no use in an ensemble then |
# | E001 | Break-even price per plant | 0.95701 | Accepted, 8/8 years. Biggest help for Vinje |
# | E002 | Separate stage-1 model for weekly level | -0.00163 *tuning* | Rejected, all folds worse. The hourly model learns level better itself |
# | E003 | Hydraulic state in hours of full capacity | 0.96009 | Accepted, 8/8 years. 2020 +0.016, spill pressure explains running at low prices |
# | E004 | Inflow anomaly and constraint columns | -0.00029 *tuning* | Rejected, inside noise |
# | E005 | Monotone constraints on price | +0.00068 *tuning* | Rejected, below noise and very slow to train |
# | E008 | UC-Net: dilated convolutions, Transformer, FiLM, priority head, aux head | 0.95893 | Different errors from LightGBM. Public LB not submitted alone |
# | E009 | E001 + E003 together | 0.96295 | Accepted. Public LB 0.97252 |
# | E010 | Scale-free features, no absolute price levels | 0.96409 | Gate fail, 4/8 years. Better in the extreme years only |
# | E011 | Optuna tuning of LightGBM | 0.96389 | Rejected. All the gain sat in the folds that chose the parameters, the other 5 averaged -0.00014 |
# | E012 | LightGBM + UC-Net logit blend, nested | 0.97847 | Accepted. Public LB 0.98506 |
# | E013 | Neighbouring plant economics and spill pressure | 0.97919 | Accepted, +0.016 and 8/8 years. Largest feature gain in this phase |
# | E014 | E013 + UC-Net, nested | 0.98254 | Accepted. Public LB 0.98845 |
# | E015 | UC-Net with 3 nets per seed | 0.96461 | Passes. Reduced variance |
# | E018 | UC-Net on E013 features | 0.96019 | Passes formally. Replaced by ST-Net |
# | E019 | ST-Net: one token per (unit, hour), attention across units | 0.98199 (3-seed ensemble 0.98425) | Accepted. Alone beats E014 |
# | E020 | Per-plant blend weights | 0.98345 | Passes. Barely above noise |
# | E021a | E013 + ST-Net, nested | **0.98584** | Best without LP. Public LB **0.98906** |
# | - | Overlap smoothing across neighbouring cases | 0.98151 | Rejected. Worse in 8/8 years |
# | E022 | ST-Net with 3 other seeds | 0.98187 (3-seed ensemble 0.98407) | Six seeds pooled add nothing measurable, so 3 are enough |
# | E023 | ST-Net on E013 features | -0.00042 *tuning* | Rejected |
# | E024 | ST-Net with pairwise ranking loss, direct for AUC | -0.00034 *tuning* | Rejected |
# | E025 | Per-plant blend weights on E021a | 0.98604 | Rejected, below noise |
# | E026 | LightGBM with whole-watercourse state | 0.98118 | Accepted as single model. Considered for notebook 2 and rejected: about 0.012 below the final blend |
# | E027 | E026 + ST-Net | 0.98608 | Below noise, not used |
#
# ### 6.2 Phase 2: with the LP
#
# All numbers below are read from the experiment log. "Forward" is train-before only. Rows are in the order they ran.

# %%
DESC = [("L0_base", "L0_base_fwd", "LightGBM", "No LP, base features", "Reference for what the LP adds"),
    ("L1_base_lp", None, "LightGBM", "Base + LP telescoping", "+0.037. By far the largest step in the project"),
    ("L2_v2", "L2_v2_fwd", "LightGBM", "LP v2", "Adopted"),
    ("L2_v2_shuffle", None, "LightGBM", "LP v2, labels shuffled", "Control, about 0.59 as expected"),
    ("L3_v2_milp", None, "LightGBM", "LP v2 + integer MILP schedule", "Rejected, no gain"),
    ("L4_noscale", None, "LightGBM", "LP v2, no absolute prices", "-0.00017, small loss"),
    ("L5_v3", None, "LightGBM", "LP v3", "Adopted"),
    ("L6_v3_noscale", None, "LightGBM", "LP v3, no absolute prices", "Inside noise"),
    ("L7_v4_mf", None, "LightGBM", "LP v4 + required flow columns", "Adopted LP v4"),
    ("L8_v4", None, "LightGBM", "LP v4", "Reference"),
    ("L9_v4_mf_s3", "L10f", "LightGBM", "LP v4 + `mf_*`, 3 seeds", "Blend member in B2"),
    ("X1_v4_mf", "X1f_fwd", "XGBoost", "Same features", "Blend member in B2. Fast on GPU, so became the main tree"),
    ("C1_v4_mf", "C2f", "CatBoost", "Same features", "Blend member in B2. Weak forward, dropped later"),
    ("N1_stnet_lp", None, "ST-Net", "LP v2, 1 seed", "First LP net"),
    ("N2_stnet_v3", None, "ST-Net", "LP v3, 2 seeds", "Improves with the LP"),
    ("N3_v4_mf", "N3f3_e24", "ST-Net", "LP v4, no absolute prices, e24", "Blend member in B2"),
    ("N4_v4_scale", "N4f_e24_scale", "ST-Net", "With absolute prices, e24", "Blend member in B2. Forward +0.0007"),
    ("N8_e12", "N8f_e12", "ST-Net", "No absolute prices, e12", "Rejected. 12 epochs worse in LOYO"),
    ("N9_e12_scale", "N9f_e12_scale", "ST-Net", "With absolute prices, e12", "Rejected, R20: e24 wins on same seeds"),
    (None, "N10f_e8", "ST-Net", "e8", "Rejected, too few epochs"),
    (None, "N11f_e16", "ST-Net", "e16", "Equal to e24 without prices. Not pursued"),
    (None, "N12f_d96", "ST-Net", "d=96 instead of 64", "Rejected, R15, 1/4 years"),
    ("N5_ls", "N5f_ls", "ST-Net", "Keep all + `ls_*`", "Gate fail, R19"),
    ("N6_min", "N6f_min", "ST-Net", "`base,lp,ls`", "**Accepted. Final net, notebook 1**"),
    ("N7_noabs", "N7f_noabs", "ST-Net", "`base,lp,ls`, no absolute prices", "**Hedge, notebook 2**"),
    (None, "P1f_pseudo", "XGBoost", "Pseudo-labels from validation year", "Rejected, no gain over X3f 0.99297"),
    ("XR_ref", "XR_ref_f", "XGBoost", "Clean reference, 72 columns, 5 seeds", "Gate reference, seed noise measured here"),
    ("XR_r800", "XR_r800_f", "XGBoost", "eta 0.04, 800 rounds", "Rejected, R13, inside noise"),
    ("XR_ls", "XR_ls_f", "XGBoost", "`base,lp,ls`, no absolute prices", "**Accepted. Final trees**")]
tab = pd.DataFrame([dict(Model=m, Features=f, LOYO=rows[a]["oof_auc"] if a else np.nan, Forward=rows[b]["oof_auc"] if b else np.nan, Note=n) for a, b, m, f, n in DESC]).round(5); display(tab)

# %%
pts = [("Logistic regression*", .89708, 0), ("Random forest*", .93927, 0), ("LightGBM, base (E000)", .95415, 0), ("LP alone, power fraction (no training)", auc(Y, LPF[:N_TR]), 2), ("UC-Net (E008)", .95893, 0), ("LightGBM, physics features (E009)", .96295, 0), ("LightGBM + neighbours (E013)", .97919, 0), ("LightGBM, whole system (E026)", .98118, 0), ("ST-Net (E019)", .98199, 0), ("LightGBM + UC-Net (E012)", .97847, 0), ("LightGBM + ST-Net (E021a)", .98584, 0),
    ("LightGBM + LP v4", rows["L8_v4"]["oof_auc"], 1), ("CatBoost + LP v4", rows["C1_v4_mf"]["oof_auc"], 1), ("XGBoost + LP v4", rows["XR_ref"]["oof_auc"], 1), ("ST-Net + LP v4", rows["N4_v4_scale"]["oof_auc"], 1), ("XGBoost final trees", rows["XR_ls"]["oof_auc"], 1), ("ST-Net final", rows["N6_min"]["oof_auc"], 1), ("Blend B2", rows["B2_gbdt3_nn2_nested"]["oof_auc"], 1), ("Final blend, notebook 1", auc(Y, B), 1)]
pts = sorted(pts, key=lambda p: p[1]); cols = [BLUE, ORANGE, YELLOW]; fig, ax = plt.subplots(figsize=(9, 5.4))
ax.hlines(range(len(pts)), .88, [p[1] for p in pts], color=GRID, lw=1); ax.scatter([p[1] for p in pts], range(len(pts)), s=42, c=[cols[p[2]] for p in pts], zorder=3, edgecolors="#fcfcfb", linewidths=1.5); ax.set_yticks(range(len(pts)), [p[0] for p in pts]); ax.set_xlim(.885, 1.0); ax.set_xlabel("Cross-validated AUC, LOYO"); ax.grid(axis="y", visible=False); ax.set_title("With the LP, every model scores above the best earlier blend")
for k, p in enumerate(pts): ax.text(p[1] + .0015, k, f"{p[1]:.4f}", va="center", fontsize=7.5, color=MUTED)
for lab, col in (("Without LP", BLUE), ("With LP", ORANGE), ("LP alone", YELLOW)): ax.scatter([], [], s=30, c=col, label=lab)
ax.legend(loc="lower right"); plt.show(); print("* Three tuning folds only, not all 8 years, so not directly comparable with the others.")

# %% [markdown]
# Two different predictor families are in the final blend (gradient-boosted trees and a neural network). Logistic regression, random forest, LightGBM, XGBoost, CatBoost, UC-Net, ST-Net and the LP itself were all tested.
#
# Things considered but not run. Synthetic data was discussed and written down as optional: (P1) pushing the cheapest hours of training weeks below zero and labelling with the LP, and (P2) scaling price and water value together as augmentation. Noise would change the correct label, rolling cases already act as time shifts, and cases labelled by the LP hold none of the SHOP-minus-LP gap that the models learn. P1 could not be validated on labels at all, since validation years barely contain negative hours.

# %% [markdown]
# ## 7. The final models and how the hyperparameters were chosen
#
# **XGBoost** (66 features: `base,lp,ls` without the 6 absolute price columns). Trained on long-format (case, unit, hour) rows, 3 seeds averaged in logit space.
#
# | Parameter | Value | How it was obtained |
# |---|---|---|
# | `learning_rate`, rounds | 0.08, 300 | Set in E000 and kept. R13 (0.04, 800) was inside noise. Learning curves (Section 10) show validation still rises very slowly to about 1000 rounds and never falls |
# | `max_leaves`, `grow_policy` | 63, lossguide | Set for LightGBM in E000 and copied to XGBoost. E011 (Optuna) found no gain outside the folds that chose it, so I stopped tuning trees |
# | `min_child_weight` | 50 | Set when moving from LightGBM to XGBoost, as a rough counterpart to LightGBM's `min_data_in_leaf` of 500. Not tuned |
# | `subsample`, `colsample_bytree`, `reg_lambda` | 0.5, 0.8, 1.0 | Same as E000, not tuned |
# | Seeds | 3 | Seed std of 0.00003 LOYO and 0.00006-0.00009 forward (R1), so more seeds add nothing visible |
#
# **ST-Net** (70 features: `base,lp,ls`). A token per (unit, hour). Four blocks, each with a dilated convolution over the week (dilation 1, 2, 4, 8) conditioned on a unit summary, and attention across the 14 units in the same hour with a learned unit-to-unit bias. One Transformer layer over the week follows. The output enforces the priority among identical units from 3.2 by subtracting a positive amount from the parent unit's logit. An auxiliary head predicts each unit's weekly on-share.
#
# | Parameter | Value | How it was obtained |
# |---|---|---|
# | Epochs | 24 | Compared on the same seeds (R20): e24 beats e12 on two seed sets, LOYO and forward. I claimed earlier that e24 overfits. That came from comparing 1 seed with 3, and the claim does not hold |
# | `d`, blocks, heads | 64, 4, 4 | Chosen to be fast enough for repeated runs. d=96 was worse (R15, 1 of 4 years) |
# | Dropout, weight decay | 0.1, 0.01 | Carried over from E019. Single-seed forward checks with dropout 0.25 (+0.00016) and with weight decay 0.1, dropout 0.2 and 16 epochs (+0.0006) looked better, but their e24 reference turned out to be an unlucky seed: the same setup with 3 seeds is +0.0012 above it (R10), above both checks. They were not repeated with 3 seeds, so these values were kept, not shown to be best |
# | Learning rate, batch, EMA | 1e-3 with warmup and cosine decay, 32, 0.995 | Carried over from E008/E019 |
# | Aux weight | 0.3 | Carried over from E008 |
# | Seeds | 3 | Seed sets differ up to 0.00035 alone. In the blend, this nearly vanishes, and six seeds add nothing (R14) |
#
# **LP.** Values from the YAML file wherever it specifies them. The bypass cost was swept over 10, 100 and 1000 with the LP scored alone, and the documented value of 10 was the best. The ±10 % scale for the sensitivity solves was set before running, and not tuned.
#
# **Blend weight, 0.7 trees and 0.3 net in logit space.** Validation is flat over a wide range, as the table below shows. Nested weights on an earlier model set came out at 0.65-0.70 in all 8 folds. I fixed 0.7 before the last round and kept it. It isn't nested for the final pair, but the range is too flat for this to matter much.

# %%
f = lambda a, b, w: w * a + (1 - w) * b; okf = np.isfinite(XF).all((1, 2)); W = [.5, .6, .7, .8, .9, 1.]
display(pd.DataFrame({"Tree weight": W, "LOYO": [auc(Y, f(XT, NT, w)) for w in W], "Forward": [auc(Y[okf], f(XF, NF, w)[okf]) for w in W]}).round(5).set_index("Tree weight").T)
print("Correlation between tree and net logits:", round(np.corrcoef(XT.ravel(), NT.ravel())[0, 1], 3))

# %% [markdown]
# ## 8. Model interpretation
#
# Three methods on the final models. All were trained before 2022 and explained on held-out 2022, seed 0.
#
# ### 8.1 SHAP on the trees

# %%
S = np.load("artifacts/shap_final.npz", allow_pickle=True); sj = json.load(open("artifacts/shap_final.json")); nm = S["names"]; imp = S["imp"]; o = np.argsort(-imp)[:20][::-1]; gc = {"base": BLUE, "lp": ORANGE, "ls": AQUA}
fig, ax = plt.subplots(figsize=(8.5, 4.8)); ax.barh(range(20), imp[o], color=[gc[group(nm[k])] for k in o], height=.72); ax.set_yticks(range(20), nm[o]); ax.set_xlabel("Mean |SHAP|, log-odds"); ax.grid(axis="y", visible=False); ax.set_title("Top 20 features in the final XGBoost")
for lab, gg in (("Base", "base"), ("LP schedule", "lp"), ("LP sensitivity", "ls")): ax.barh([0], [0], color=gc[gg], label=lab)
ax.legend(loc="lower right"); plt.show(); print("Share of total mean |SHAP| by group:", {k: round(v, 3) for k, v in sj["groups"].items()})
g_ = sj["gain"]; print("Top 8 by gain:", sorted(g_, key=g_.get, reverse=True)[:8])

# %% [markdown]
# Most of the model's attribution falls on features the LP supplies: 77 % for LP schedule and LP sensitivity together, with all market, calendar and unit columns sharing the rest.
#
# - `ls_mean` is first: the mean of the LP schedule at water values x0.9, x1.0 and x1.1. It acts as a soft version of the LP decision that knows how firm the decision is.
# - `lp_margin_ratio` and `lp_margin_roll5` compare the price with the LP's marginal cost from shadow prices. So the model refines the LP's economics near a tie, instead of only copying the schedule.
# - `lp_sys_t` (system production in that hour) shows the model relies on system state.
# - `unit_index` and the unit columns adjust for the base rate of each unit, and for the start order from 3.2.
# - `start_hours_wv` is start cost in hours of production at the water value. I expected a high value to discourage starting. The next figure shows the opposite: a high value goes with more running. High values occur when the water value is near zero, so this column seems to work mostly as a marker of cheap water, not as a start cost effect.
#
# Gain ranks `lp_u` and neighbouring `lp_u_*` high, since a few early splits with large gain sit on them, while `ls_mean` is used more consistently.

# %%
top = [str(x) for x in nm[S["top"]]]; show = ["ls_mean", "lp_margin_ratio", "lp_margin_roll5", "lp_sys_t", "lp_u", "start_hours_wv"]; fig, axs = plt.subplots(2, 3, figsize=(11, 5.4), gridspec_kw=dict(hspace=.35, wspace=.22))
for ax, n in zip(axs.ravel(), show):
    j = top.index(n); x, s = S["x"][:, j], S["sv"][:, j]; hi = np.quantile(x, .995); lo = np.quantile(x, .005); ax.scatter(x, s, s=3, alpha=.18, c=BLUE, linewidths=0); ax.set_xlim(*{"lp_margin_ratio": (0, 4), "start_hours_wv": (-.05, 2)}.get(n, (lo - (hi - lo) * .03, hi + (hi - lo) * .03))); ax.axhline(0, color=MUTED, lw=.8); ax.set_title(n)
    if n in interp["pdp"]:
        q = interp["pdp"][n]; base_ = np.mean(q["y"]); ax.plot(q["x"], np.array(q["y"]) - base_, color=ORANGE, label="Partial dependence, centred")
        ax.legend(loc="best")
axs[0, 0].set_ylabel("SHAP, log-odds"); axs[1, 0].set_ylabel("SHAP, log-odds"); fig.suptitle("SHAP dependence and partial dependence", fontsize=11, fontweight="bold", x=.09, ha="left", y=.96); plt.show()

# %% [markdown]
# Most curves have the shape the physics predicts:
# 
# - `ls_mean` clusters near 0, 1/3, 2/3 and 1, since it is the mean of three mostly binary schedules, and each step up raises the odds of running. The largest step is from 1/3 to 2/3.
# - `lp_margin_roll5` and `lp_margin_ratio` rise sharply where the price passes the LP's marginal cost (margin 0, ratio 1), then flatten. Far from that point, more margin tells the model little.
# - `lp_sys_t` rises with total system output, and `lp_u` matters mostly in the range from off to partly on.
# - The orange partial dependence is computed from 20 000 rows and agrees in direction with SHAP. They differ in size since partial dependence sweeps a feature alone, through combinations that do not occur in the data.
#
# ### 8.2 Per-unit attribution

# %%
un = S["unit"]; ks = [top.index(n) for n in top[:12]]; M = np.array([[np.abs(S["sv"][un == g_, k]).mean() for k in ks] for g_ in range(14)]); fig, ax = plt.subplots(figsize=(11, 4.6)); heat(ax, M, [top[k] for k in ks], UNITS, "Mean |SHAP| by unit and feature, top 12 features", fmt="{:.2f}"); ax.tick_params(axis="x", rotation=40); plt.setp(ax.get_xticklabels(), ha="right"); plt.show()

# %% [markdown]
# `ls_mean` leads for every unit with a similar size. The differences that stand out make sense: `unit_index` matters most for the last units in the start order (Tokke G4 0.78, Vinje G3 0.70), and `start_hours_wv` barely matters for Haukeli (0.11), which rarely switches.
#
# ### 8.3 Permutation importance: trees, net and blend
#
# A fresh interpretation run (R24). Each (case, unit) gets its week-long feature track from another random (case, unit). The drop in AUC on 2022 is a sensitivity. Shuffled inputs are not physically consistent, so this is a stress as much as an explanation. Group rows are the mean of 3 shuffles.
#
# My hypothesis, written before running: both models lean mainly on the LP schedule, and the net loses more than the trees when market price columns are shuffled, because it has the absolute columns.

# %%
gi = pd.DataFrame(interp["group"]).T[["xgb", "net", "blend"]]; gi["columns"] = [len(G[k]) for k in gi.index]; print("Unshuffled AUC on 2022, seed 0:", {k: round(v, 5) for k, v in interp["base"].items()}); display(gi.sort_values("blend").round(5).rename_axis("Shuffled group"))
si = pd.DataFrame(interp["single"]).T; si = si.sort_values("blend").head(15)[::-1]; fig, ax = plt.subplots(figsize=(9, 4.8)); yy = np.arange(len(si)); w_ = .26
for k, (m_, col) in enumerate((("xgb", BLUE), ("net", ORANGE), ("blend", AQUA))): ax.barh(yy + (k - 1) * w_, -si[m_], height=w_ * .9, color=col, label={"xgb": "XGBoost", "net": "ST-Net", "blend": "Blend"}[m_])
ax.set_yticks(yy, si.index); ax.grid(axis="y", visible=False); ax.set_xlabel("Loss in AUC when this one feature is shuffled"); ax.set_title("Top 15 single features by loss in the blend, on 2022"); ax.legend(loc="lower right"); plt.show()

# %% [markdown]
# @@PERM_TEXT@@

# %% [markdown]
# ### 8.4 One week, side by side
#
# A case with a median blend AUC. The figure compares SHOP's decision with the LP's relaxed schedule and with the out-of-fold blend, in a validation fold where the case was unseen.

# %%
ca = np.array([auc(Y[k].ravel(), B[k].ravel()) if 0 < Y[k].mean() < 1 else np.nan for k in range(N_TR)]); k = int(np.nanargmin(np.abs(ca - np.nanmedian(ca)))); fig, axs = plt.subplots(4, 1, figsize=(11, 8), sharex=True, gridspec_kw=dict(hspace=.28, height_ratios=[1, 2, 2, 2]))
axs[0].plot(P[k], color=BLUE, label="Price"); wvu = wv[k, ri[UP[UNITS.index("Tokke_G1")]]]; axs[0].axhline(wvu, color=ORANGE, label="Water value above Tokke"); axs[0].set_ylabel("EUR/MWh"); axs[0].set_ylim(0, P[k].max() * 1.45); axs[0].legend(loc="upper right", ncol=2); axs[0].set_title(f"Case starting {c['start'][k]}, blend AUC for this case {ca[k]:.4f}")
for ax, M, tt in ((axs[1], Y[k], "SHOP's decision (label)"), (axs[2], LPU[k], "LP relaxed commitment, no training"), (axs[3], 1 / (1 + np.exp(-B[k])), "Final blend P(on)")):
    ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest"); ax.grid(False); ax.set_yticks(range(14), UNITS, fontsize=6.5); ax.set_title(tt)
axs[3].set_xticks(range(0, 168, 24), [f"day {d + 1}" for d in range(7)]); plt.show()

# %% [markdown]
# This is the mechanism in one case. The LP has most of the week's plan without training: Tokke, Lio and Byrte are almost exactly right. It is wrong in places. It holds Hogga and the Vinje units off for the first day or more while SHOP runs them, and it switches Haukeli off late in day 7. The blend keeps the plan, moves toward SHOP in those places (Hogga and Vinje become uncertain instead of off, and Haukeli stays on), and is partly right about the short Kjela runs the LP mostly misses. It still misses the short off periods for Vinje G3 and Hogga in days 5-7.

# %% [markdown]
# ## 9. Error analysis

# %%
fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.6), gridspec_kw=dict(wspace=.3, width_ratios=[1.35, 1])); M = np.array([[auc(Y[ytr == y, g_], B[ytr == y, g_]) for y in YRS] + [auc(Y[:, g_], B[:, g_])] for g_ in range(14)]); heat(ax[0], M, [str(y)[2:] for y in YRS] + ["all"], UNITS, "Blend AUC by unit and year, out of fold", .95, 1., "{:.3f}", cmap="Blues", size=6.5)
L = [auc(Y[:, g_], LPF[:N_TR, g_]) for g_ in range(14)]; Bl = [auc(Y[:, g_], B[:, g_]) for g_ in range(14)]; X_ = [auc(Y[:, g_], XT[:, g_]) for g_ in range(14)]; N_ = [auc(Y[:, g_], NT[:, g_]) for g_ in range(14)]; yy = np.arange(14)
ax[1].hlines(yy, L, Bl, color=GRID, lw=1.4); ax[1].scatter(L, yy, s=38, c=YELLOW, label="LP alone", zorder=3); ax[1].scatter(Bl, yy, s=38, c=BLUE, label="Blend", zorder=3); ax[1].set_yticks(yy, UNITS); ax[1].invert_yaxis(); ax[1].grid(axis="y", visible=False); ax[1].legend(loc="lower left"); ax[1].set_title("What learning adds to the LP"); ax[1].set_xlabel("AUC, 2015-22"); plt.show()
display(pd.DataFrame({"LP alone": L, "XGBoost": X_, "ST-Net": N_, "Blend": Bl}, index=UNITS).round(4).sort_values("Blend").T)
wm = P[:N_TR].mean(1); bins = np.quantile(wm, np.linspace(0, 1, 9)); pb = np.digitize(wm, bins[1:-1]); mo = starts[:N_TR].month.values; ex = lambda a, m: auc(Y[m], a[m])
fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.2), gridspec_kw=dict(wspace=.3))
ax[0].plot(range(8), [ex(B, pb == k) for k in range(8)], color=BLUE, label="Blend"); ax[0].plot(range(8), [ex(LPF[:N_TR], pb == k) for k in range(8)], color=YELLOW, label="LP alone"); ax[0].set_xticks(range(8), [f"{bins[k + 1]:.0f}" for k in range(8)], rotation=45); ax[0].set_xlabel("Weekly mean price up to, EUR/MWh"); ax[0].set_ylabel("AUC"); ax[0].set_title("By price octile"); ax[0].legend()
ax[1].plot(range(1, 13), [ex(B, mo == m) for m in range(1, 13)], color=BLUE, label="Blend"); ax[1].plot(range(1, 13), [ex(LPF[:N_TR], mo == m) for m in range(1, 13)], color=YELLOW, label="LP alone"); ax[1].set_xticks(range(1, 13), ml); ax[1].set_title("By month"); ax[1].legend()
pb_ = 1 / (1 + np.exp(-B)); bn = np.linspace(0, 1, 21); ix = np.digitize(pb_.ravel(), bn) - 1; mp = [pb_.ravel()[ix == k].mean() for k in range(20)]; my = [Y.ravel()[ix == k].mean() for k in range(20)]; ax[2].plot([0, 1], [0, 1], color=GRID, lw=1); ax[2].plot(mp, my, color=BLUE, marker="o", ms=4); ax[2].set_xlabel("Predicted P(on)"); ax[2].set_ylabel("Observed share on"); ax[2].set_title("Calibration, out of fold"); plt.show()
print("Blend AUC per year:", {y: round(auc(Y[ytr == y], B[ytr == y]), 5) for y in YRS}); lv = lambda A: np.broadcast_to(A.mean(2, keepdims=True), A.shape); print("AUC if only the weekly level were known, true level:", round(auc(Y, lv(Y)), 4), "| blend's level:", round(auc(Y, lv(B)), 4))

# %% [markdown]
# @@ERROR_TEXT@@

# %% [markdown]
# ## 10. Robustness
# ### 10.1 Regime holdouts
#
# Train on one regime and test on another, on XGBoost. The LP's advantage should hold up when it has to extrapolate, since it has no trained parameters. Note that the test sets here overlap with years that were available when the LP was designed.

# %%
rr = pd.read_csv("artifacts/robust_regime.csv").set_index("holdout"); rr.columns = ["n train", "n val", "No LP", "LP with prices", "LP, no prices", "LP alone (no training)"]; display(rr)
rc = json.load(open("artifacts/robust_curves.json")); fig, ax = plt.subplots(1, 2, figsize=(11, 3.3), gridspec_kw=dict(wspace=.28)); rs = ["50", "100", "200", "300", "500", "800", "1200"]
for (y, d), col in zip(rc["rounds"].items(), (BLUE, ORANGE, AQUA, YELLOW)): ax[0].plot([int(r) for r in rs], [d[r][1] for r in rs], color=col, label=f"{y}")
ax[0].axvline(300, color=MUTED, lw=.8); ax[0].text(320, ax[0].get_ylim()[0] + .0004, "Used: 300", fontsize=8, color=MUTED); ax[0].set_xscale("log"); ax[0].set_xlabel("Rounds"); ax[0].set_ylabel("Validation AUC"); ax[0].set_title("Forward validation by rounds"); ax[0].legend(ncol=2)
sz = rc["size"]; ax[1].plot([int(k) for k in sz], list(sz.values()), color=BLUE, marker="o", ms=5); ax[1].set_xlabel("Training years"); ax[1].set_ylabel("AUC on 2022"); ax[1].set_title("Five or seven years beat one to three"); plt.show()

# %% [markdown]
# - Without LP features, regime holdouts score 0.91-0.93. With them, they score 0.99. The gap is much larger than any model choice in Section 6.
# - The rounds curve never falls: from 300 to 1200 it rises by at most 0.0002 to about 1000 rounds. There is no overfitting with rounds, and 300 is if anything slightly underfit, but R13 showed more rounds is inside noise.
# - Five or seven training years beat one to three (0.9917 and 0.9923 against about 0.990), so the final models train on all 8.
#
# ### 10.2 Stress tests
#
# Trained before 2022 and evaluated on half of 2022's cases. Inputs are perturbed, and **the LP is re-solved on the perturbed inputs**, so these reflect what the pipeline would actually do. Labels are unchanged, so a change here is a sensitivity and not a true score: at scaled inputs, SHOP's true decision may well differ.

# %%
sf, sn = json.load(open("artifacts/stress_final.json")), json.load(open("artifacts/stress_noabs.json")); kk = ["scale_0.5", "scale_2", "scale_4", "price_noise", "negative_cheapest"]
display(pd.DataFrame({"XGBoost": [sf[k]["xgb"] for k in kk], "ST-Net, notebook 1": [sf[k]["net"] for k in kk], "ST-Net, notebook 2": [sn[k]["net"] for k in kk], "Blend, notebook 1": [sf[k]["blend"] for k in kk], "Blend, notebook 2": [sn[k]["blend"] for k in kk], "LP schedule changed": [sf[k]["lp_u_change"] for k in kk]}, index=["Price & water value x0.5", "Price & water value x2", "Price & water value x4", "Noise on price", "Cheapest 10 % of hours pushed below zero"]).round(5))
print("Unperturbed blend on this half of 2022: notebook 1", round(sf["base"]["blend"], 5), "| notebook 2", round(sn["base"]["blend"], 5))
print("Mean P(on) in the hours pushed negative: before", round(sf["negative_cheapest"]["p_on_before"], 3), "after", round(sf["negative_cheapest"]["p_on_after"], 3), "| LP:", round(sf["negative_cheapest"]["lp_on_before"], 3), "to", round(sf["negative_cheapest"]["lp_on_after"], 3))
s1 = pd.read_csv("artifacts/submission_1.csv"); T1 = s1.iloc[:, 1:].values.reshape(725, 14, 168); P2 = P[N_TR:]; neg = np.broadcast_to(P2[:, None] < 0, T1.shape); print("\nOn the actual test set (no labels). Mean P(on) in negative-price hours:", round(T1[neg].mean(), 3), "| all other hours:", round(T1[~neg].mean(), 3), "| LP power fraction in negative hours:", round(LPF[N_TR:][neg].mean(), 3))

# %% [markdown]
# - **Scaling by 0.5, 2 or 4** barely moves the result, which is expected: scaled price and water value together leave the LP schedule almost unchanged, and it is the main input. The trees are close to scale-free, since most of their columns are ratios, ranks or LP output.
# - **The earlier model E021a fell 0.0229 on negative prices** under the same perturbation, measured on all validation years. With the LP, the fall is 0.0072 on half of 2022. The test sets differ, so this is only indicative.
# - **Negative prices stay the largest unknown.** Producing at a negative price costs money, and both the LP and the models react by switching off, with P(on) falling from 0.14 to 0.01. On the true test set, predicted P(on) in negative hours is 0.14 against 0.48 elsewhere. The direction is reasonable, but how strongly SHOP reacts cannot be verified. It may keep units on for flood, minimum flow or start cost reasons.
# - **Price noise** costs 0.0075 because it really changes the economics. It is the least relevant of these, since SHOP and the features see exactly the same price in the real task.
#
# **The hedge in notebook 2.** The hypothesis for notebook 1 is that the absolute price level helps the net (R6, R21). If the level misleads it in the test years, notebook 2 removes that risk: the blend takes less damage at x4 scaling (-0.00177 against -0.00231) and on negative prices (-0.00668 against -0.00722), for almost no cost in validation (LOYO -0.00001, forward 0.00000). Both are the same in the LP, so this hedge doesn't protect against the LP failing. I considered the older LP-free LightGBM (E026) as a hedge against that and rejected it: its LOYO is about 0.012 lower, and its predecessor E013 fell 0.029 on negative prices (indicative, as the test set differs).
#
# Public leaderboard: 0.99638 and 0.99635, in line with validation saying they are equal.

# %% [markdown]
# ## 11. Reproducibility of the short notebooks
#
# Both notebooks are self-contained: all the code is inside, they start from the raw data, need no external data and run offline. They solve the LP in parallel on up to 4 cores with HiGHS and store the solutions in `work/`, so an interrupted run can resume.
#
# | Check | Notebook 1 | Notebook 2 |
# |---|---|---|
# | Empty directory, 4 cores, NVIDIA H100 | 50.6 min in total, LP 47.6 | 42.3 min in total, LP 39.2 |
# | Peak memory (sampled every 15 s, excludes LP workers) | About 11 GB RAM, about 4 GB GPU | 10.6 GB RAM, about 4 GB GPU |
# | XGBoost against research run | Identical logits | Identical logits |
# | ST-Net against research run | Spearman 0.99999, largest logit difference 0.125 | Spearman 0.99999, largest logit difference 0.145 |
# | LP schedules against research run | Identical | Not compared separately |
#
# GPU training is not bit-exact, which explains the small net differences. They do not move the blend measurably.
#
# **Requirement of 12 hours, 4 CPU cores, 32 GB.** @@CPU_TEXT@@
#
# The GPU is optional: the code uses bf16 autocast where available and falls back to float16 on an older GPU, or to CPU. The float16 and CPU paths were tested on random data at real shapes. A run on a real smaller GPU was not tested. Software versions are in `requirements.txt` (Python 3.12, PyTorch 2.6.0 with CUDA 12.4, XGBoost 3.4.1, SciPy 1.18.1 with HiGHS).

# %% [markdown]
# ## 12. Limitations
#
# - **No untouched holdout.** The structure of the LP (telescoping, head losses, priority order) was chosen by scoring it against labels from all training years. LOYO and forward scores are therefore somewhat optimistic for the LP-based models. The public leaderboard is the only independent check, and it agrees with the gain, but it was also consulted a few times.
# - **Negative prices** are 2.2 % of test hours and nearly absent from training. They can't be validated with labels (Section 10.2). Also, negative hours by test year are 1.99 % for 2023 and 2.35 % for 2024, and I do not know the public/private split, so the private score may react differently.
# - **Weakest units.** Vinje G3, Hogga and Kjela (Section 9). A likely reason is what the LP cannot represent: start costs and integer decisions at small margins. I did not establish this.
# - **The LP's own simplifications.** Head is fixed within a week, efficiency is a concave approximation, and the commitment is relaxed. The ML models correct some of this, but only what the training regimes have shown.
# - **Blend weight** is not nested for the final pair, though the validation is flat around it.
# - **Interpretation.** SHAP explains the trees only. Permutation importance covers both, but a shuffle breaks physical consistency. Correlated features share credit, so individual rankings should be read loosely.
# - **Runtime on a normal PC.** @@CPU_LIMIT@@
#
# **Tools.** Claude Code (an AI coding assistant) was used for writing and running code, analyses and drafting this report. All numbers come from logged runs, and hypotheses were written down before they were run.

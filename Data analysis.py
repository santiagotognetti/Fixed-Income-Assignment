
"""QMiFI hand-in: Switzerland and France, Nelson-Siegel analysis."""
import glob, os, re, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

MATS = [1, 2, 3, 5, 7, 10, 20]      # maturities in years
FREQ = "ME"                           # month-end sampling (lags below are in months)
TAU_STAR = 2.5                       # curvature loading peaks here (Diebold-Li style, 2-3y)
LAMBDA = 1.7933 / TAU_STAR           # per year; peak of ((1-e^-x)/x - e^-x) is at x=1.7933
LAGS = [1, 6, 12]
START, END = None, None              # e.g. "2005-01-01"; None = full common sample


def load_swiss(path="Swiss Yield Data.xlsx"):
    raw = pd.read_excel(path, header=None)
    hdr = raw.index[raw[0].astype(str).str.strip().eq("Maturity")][0]
    cols = raw.iloc[hdr, 1:].astype(str).str.extract(r"(\d+)")[0].astype(int).tolist()  # "1 year" -> 1
    df = raw.iloc[hdr + 2:].copy()
    df.columns = ["date"] + cols
    df.index = pd.to_datetime(df.pop("date"), errors="coerce")
    df = df[df.index.notna()].apply(pd.to_numeric, errors="coerce")
    return df.loc[:, ~df.columns.duplicated()]

def load_france(pattern="Webstat_Export_en_FM.D.FR.EUR.FR2.BB.FRMOYTEC*.HSTA.xlsx"):
    out = {}
    for f in glob.glob(pattern):
        m = int(re.search(r"TEC(\d+)\.", f).group(1))
        d = pd.read_excel(f)
        s = pd.Series(pd.to_numeric(d["obs_value"], errors="coerce").values,
                      index=pd.to_datetime(d["time_period"]), name=m)
        out[m] = s.dropna()
    return pd.DataFrame(out).sort_index()

def to_monthly(df):
    return df.dropna(how="all").resample(FREQ).last()

ch = to_monthly(load_swiss())
fr = to_monthly(load_france())
ch, fr = ch[MATS], fr[MATS]
idx = ch.dropna().index.intersection(fr.dropna().index)
if START: idx = idx[idx >= START]
if END:   idx = idx[idx <= END]
data = {"Switzerland": ch.loc[idx], "France": fr.loc[idx]}
print("Common sample:", idx.min().date(), "to", idx.max().date(), "| obs:", len(idx))

fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for a, (c, d) in zip(ax, data.items()):
    for m in [1, 5, 10, 20]: a.plot(d.index, d[m], label=f"{m}y")
    a.set_title(c); a.legend(); a.axhline(0, c="k", lw=.5)
ax[0].set_ylabel("Yield (%)"); plt.tight_layout(); plt.savefig("results/fig1_yields.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
for c, d in data.items(): ax.plot(MATS, d.mean(), marker="o", label=c)
ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Mean yield (%)"); ax.legend()
plt.tight_layout(); plt.savefig("results/fig2_avg_curve.png", dpi=200); plt.close()

def acf(x, k): return x.autocorr(k)
rows = []
for c, d in data.items():
    for m in MATS:
        r = {"country": c, "maturity": m, "mean": d[m].mean(), "std": d[m].std()}
        for k in LAGS: r[f"rho{k}"] = acf(d[m], k)
        rows.append(r)
sf = pd.DataFrame(rows)
sf.to_csv("results/stylized_facts_table.csv", index=False)
print("\nSF (i),(iii),(iv),(v): mean / std / autocorrelations\n", sf.round(2).to_string(index=False))

for c, d in data.items():
    corr = d.corr(); corr.to_csv(f"results/crosscorr_{c}.csv")
    print(f"\nSF (vi) cross-correlations {c}\n", corr.round(2))
    mean = d.mean()
    print(f"SF (i) {c}: concave & increasing average curve? increasing =",
          bool(np.all(np.diff(mean.values) > 0)),
          "| concave =", bool(np.all(np.diff(mean.values) / np.diff(MATS) == np.sort(np.diff(mean.values) / np.diff(MATS))[::-1])))

def shape(y):
    dy = np.diff(y.values)
    if np.all(dy > 0): return "upward"
    if np.all(dy < 0): return "inverted"
    s = np.sign(dy); ch_ = (np.diff(s) != 0).sum()
    return "humped" if ch_ == 1 and s[0] > 0 else "inv-humped" if ch_ == 1 else "S/other"
for c, d in data.items():
    print(f"\nSF (ii) shapes {c}:\n", d.apply(shape, axis=1).value_counts())

def loadings(tau, lam=LAMBDA):
    tau = np.asarray(tau, float); x = lam * tau
    l2 = (1 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(tau), l2, l2 - np.exp(-x)])
B = loadings(MATS)

def fit_ns(d):
    beta = np.linalg.lstsq(B, d.values.T, rcond=None)[0].T
    fac = pd.DataFrame(beta, index=d.index, columns=["level", "slope", "curv"])
    fit = pd.DataFrame(beta @ B.T, index=d.index, columns=d.columns)
    return fac, fit
factors, fitted = {}, {}
for c, d in data.items(): factors[c], fitted[c] = fit_ns(d)

for c, d in data.items():
    res = d - fitted[c]
    print(f"\nNS fit {c}: RMSE per maturity (pp):\n", np.sqrt((res**2).mean()).round(3).to_dict())
    emp = pd.DataFrame({"level_vs_10y": [factors[c].level.corr(d[10])],
                        "slope_vs_(1y-10y)": [factors[c].slope.corr(d[1] - d[10])],
                        "curv_vs_(2*2y-1y-10y)": [factors[c].curv.corr(2*d[2] - d[1] - d[10])]})
    print(f"Factor vs empirical proxies {c}:\n", emp.round(3).to_string(index=False))
    # SF re-check on fitted yields
    print(f"Fitted-yield std/rho1/rho12 {c}:\n",
          pd.DataFrame({"std": fitted[c].std(), "rho1": fitted[c].apply(lambda s: s.autocorr(1)),
                        "rho12": fitted[c].apply(lambda s: s.autocorr(12))}).round(2))

fig, ax = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
for j, c in enumerate(data):
    for i, f in enumerate(["level", "slope", "curv"]):
        ax[i, j].plot(factors[c].index, factors[c][f]); ax[i, j].set_title(f"{c}: {f}")
plt.tight_layout(); plt.savefig("results/fig3_factors.png", dpi=200); plt.close()

summ = []
for c, f in factors.items():
    for k in f.columns:
        summ.append({"country": c, "factor": k, "mean": f[k].mean(), "std": f[k].std(),
                     "std_of_changes": f[k].diff().std(), "rho1": f[k].autocorr(1),
                     "rho12": f[k].autocorr(12), "ADF_p": adfuller(f[k].dropna())[1]})
fs = pd.DataFrame(summ); fs.to_csv("results/factor_summary.csv", index=False)
print("\nFactor summary (vol, persistence, unit-root test)\n", fs.round(3).to_string(index=False))

var_out, fc_rows = [], []
for c, f in factors.items():
    m = VAR(f).fit(1)
    A = pd.DataFrame(m.coefs[0], index=f.columns, columns=[f"{x}(t-1)" for x in f.columns])
    A.to_csv(f"results/VAR1_A_{c}.csv")
    eig = np.abs(np.linalg.eigvals(m.coefs[0])).max()
    print(f"\nVAR(1) {c}: A matrix\n{A.round(3)}\nmax |eigenvalue| = {eig:.3f}")
    print("t-values:\n", m.tvalues.round(2).iloc[1:].T.rename(columns=lambda x: x))
    print("Residual correlation:\n", pd.DataFrame(m.resid).corr().round(2))

def oos(c, h=1, init=0.5):
    f, d = factors[c], data[c]; n = len(f); t0 = int(n * init); err = {"VAR": [], "AR": [], "RW": []}
    for t in range(t0, n - h):
        tr = f.iloc[:t + 1]
        va = VAR(tr).fit(1); pv = va.forecast(tr.values[-1:], h)[-1]
        pa = []
        for k in tr.columns:
            y = tr[k]; X = np.column_stack([np.ones(len(y) - 1), y.values[:-1]])
            b = np.linalg.lstsq(X, y.values[1:], rcond=None)[0]; v = y.values[-1]
            for _ in range(h): v = b[0] + b[1] * v
            pa.append(v)
        actual = d.iloc[t + h].values
        for name, p in [("VAR", pv), ("AR", np.array(pa)), ("RW", tr.values[-1])]:
            err[name].append(actual - B @ p)
    return {k: np.sqrt(np.mean(np.square(v))) for k, v in err.items()}
fc = []
for c in data:
    for h in [1, 6, 12]:
        r = oos(c, h); fc.append({"country": c, "h": h, **r})
fc = pd.DataFrame(fc); fc.to_csv("results/forecast_rmse.csv", index=False)
print("\nOut-of-sample yield RMSE (pp), expanding window (second half of sample)\n", fc.round(3).to_string(index=False))

# Simple risk/return view for the investment advice (holding-period excess return proxy, carry-based)
print("\nLatest curve snapshot:")
print(pd.DataFrame({c: data[c].iloc[-1] for c in data}).round(3))
print("\nFigures and tables saved in ./results/")

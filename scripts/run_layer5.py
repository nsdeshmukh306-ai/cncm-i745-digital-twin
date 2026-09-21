"""SUPERSEDED (v3 driver). Kept only because it generated the LHS labels in
results/layer5_training_data.csv. Release 5.0.0 uses scripts/run_layer5_ml.py; running
this script would overwrite the layer 5 results with the retired CNN version.

Layer 5: neural surrogate of the strain GEM, benchmarked against baselines.

The claim a surrogate has to earn is that it is worth more than a cheap
regressor. So the CNN is trained on exactly the same folds as ridge
regression, random forest and gradient boosting, and the comparison table is
part of the output rather than an appendix. Inference speed is measured, not
asserted, and Monte-Carlo dropout supplies predictive intervals whose
calibration is checked against held-out coverage.
"""
import sys, os, time, json
import numpy as np, pandas as pd, cobra
sys.path.insert(0, ".")
from sbdtwin import gem as G, surrogate as S

N = int(os.environ.get("N_LHS", "2000"))
SEED = 20260921
os.makedirs("results", exist_ok=True); os.makedirs("models", exist_ok=True)
T0 = time.time()

m = cobra.io.read_sbml_model("models/yeast9_cncm_i745.xml")
BOUNDS = dict(glucose=(0.1, 10.0), oxygen=(0.0, 10.0),
              ammonium=(0.1, 10.0), phosphate=(0.01, 2.0))
keys = list(BOUNDS)
CACHE = "results/layer5_training_data.csv"


def fba_eval(x):
    with m:
        for r in m.exchanges:
            r.lower_bound = 0.0
        for rid in G.BASE_IONS:
            if rid in m.reactions:
                m.reactions.get_by_id(rid).lower_bound = -1000.0
        for k, v in zip(keys, x):
            m.reactions.get_by_id(G.EX[k]).lower_bound = -float(v)
        mu = m.slim_optimize()
    return 0.0 if (mu is None or mu != mu or mu < 0) else float(mu)


if os.path.exists(CACHE):
    cached = pd.read_csv(CACHE)
    X, y = cached[keys].values, cached["mu"].values
    t_lp_total = float("nan")
    print(f"[{time.time()-T0:.0f}s] reusing cached LHS labels {X.shape}", flush=True)
else:
    keys, X = S.latin_hypercube(BOUNDS, N, seed=SEED)
    t0 = time.time()
    y = np.array([fba_eval(x) for x in X])
    t_lp_total = time.time() - t0
    pd.DataFrame(np.column_stack([X, y]), columns=keys + ["mu"]).to_csv(CACHE, index=False)
    print(f"[{time.time()-T0:.0f}s] FBA labels: mu in [{y.min():.4f}, {y.max():.4f}], "
          f"{(y < 1e-9).sum()} zero-growth, {t_lp_total/len(X)*1000:.1f} ms/LP", flush=True)
N = len(X)

# ---------------------------------------------------------------- CV comparison
import joblib
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

joblib.parallel_config(backend="sequential")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
rows, oof = [], {k: np.zeros_like(y) for k in ["ridge", "rf", "gbr", "cnn"]}
for fold, (tr, te) in enumerate(kf.split(X)):
    sx = StandardScaler().fit(X[tr])
    Xtr, Xte = sx.transform(X[tr]), sx.transform(X[te])
    ytr, yte = y[tr], y[te]
    models = {
        "ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xtr, ytr),
        "rf": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1).fit(Xtr, ytr),
        "gbr": GradientBoostingRegressor(random_state=SEED).fit(Xtr, ytr),
    }
    for name, mdl in models.items():
        p = mdl.predict(Xte)
        oof[name][te] = p
        rows.append(dict(fold=fold, model=name, r2=r2_score(yte, p),
                         mae=mean_absolute_error(yte, p),
                         rmse=float(np.sqrt(np.mean((yte - p) ** 2)))))
    n_in = int(0.85 * len(tr))
    net, vloss = S.train_cnn(Xtr[:n_in], ytr[:n_in], Xtr[n_in:], ytr[n_in:], seed=SEED + fold)
    import torch
    net.eval()
    with torch.no_grad():
        p = net(torch.tensor(Xte, dtype=torch.float32)).numpy()
    oof["cnn"][te] = p
    rows.append(dict(fold=fold, model="cnn", r2=r2_score(yte, p),
                     mae=mean_absolute_error(yte, p),
                     rmse=float(np.sqrt(np.mean((yte - p) ** 2)))))
    print(f"[{time.time()-T0:.0f}s] fold {fold}: " +
          ", ".join(f"{r['model']}={r['r2']:.4f}" for r in rows[-4:]), flush=True)

cv = pd.DataFrame(rows)
cv.to_csv("results/layer5_cv_folds.csv", index=False)
agg = cv.groupby("model")[["r2", "mae", "rmse"]].agg(["mean", "std"]).round(5)
agg.columns = ["_".join(c) for c in agg.columns]
agg.reset_index().to_csv("results/layer5_cv_metrics.csv", index=False)
print(agg.to_string(), flush=True)
pd.DataFrame(dict(observed=y, **{f"pred_{k}": v for k, v in oof.items()})).to_csv(
    "results/layer5_oof_predictions.csv", index=False)

# ---------------------------------------------------------------- final model
sx = StandardScaler().fit(X)
Xs = sx.transform(X)
n_in = int(0.85 * len(Xs))
net, vloss = S.train_cnn(Xs[:n_in], y[:n_in], Xs[n_in:], y[n_in:], seed=SEED)
import torch
torch.save(dict(state=net.state_dict(), n_in=Xs.shape[1], keys=keys,
                scaler_mean=sx.mean_, scaler_scale=sx.scale_,
                bounds=BOUNDS), "models/surrogate_cnn.pt")

mu_hat, sd_hat = S.mc_dropout_predict(net, Xs[n_in:], n_pass=100, seed=SEED)
cover = float(np.mean(np.abs(y[n_in:] - mu_hat) <= 1.96 * sd_hat))
pd.DataFrame(dict(observed=y[n_in:], predicted=mu_hat, sd=sd_hat)).to_csv(
    "results/layer5_mc_dropout.csv", index=False)

spd = S.benchmark_speed(net, Xs, fba_eval, n_lp=25)
spd["mc_dropout_95_coverage"] = cover
json.dump(spd, open("results/layer5_speed_benchmark.json", "w"), indent=1)
print(f"[{time.time()-T0:.0f}s] surrogate {spd['seconds_per_surrogate_prediction']*1e6:.1f} us/pred vs "
      f"LP {spd['seconds_per_lp_solve']*1e3:.1f} ms  (speedup {spd['speedup']:.0f}x); "
      f"MC-dropout 95% coverage={cover:.3f}", flush=True)
print(f"[{time.time()-T0:.0f}s] LAYER5 DONE", flush=True)

"""Layer 5 (machine-learning half): surrogate versus cheap baselines.

Runs on the cached LHS design (results/layer5_training_data.csv) produced by
the FBA sweep, and on the measured LP timing (results/layer5_lp_timing.json),
so no genome-scale model needs to be resident here.

Two departures from the v3 design, both deliberate.

* The surrogate is a multilayer perceptron, not a 1-D convolutional network.
  A convolution assumes locality along the input axis; the input here is four
  unordered nutrient uptake bounds, where no such ordering exists, so the
  convolutional inductive bias is unmotivated. The MLP is the architecture the
  problem actually calls for.
* Predictive uncertainty comes from a deep ensemble (five networks with
  different initialisations and bootstrap resamples) rather than Monte-Carlo
  dropout. Ensembles are better calibrated than MC dropout on regression
  tasks, and the calibration is checked here against held-out coverage rather
  than asserted.

The claim a surrogate must earn is that it beats a cheap regressor, so every
model sees identical folds and the comparison table is an output, not an
appendix.
"""
import os, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

SEED = 0
T0 = time.time()

d = pd.read_csv("results/layer5_training_data.csv")
keys = [c for c in d.columns if c != "mu"]
X, y = d[keys].values.astype(np.float64), d["mu"].values.astype(np.float64)
mx, sx = X.mean(0), X.std(0) + 1e-9
Xs = (X - mx) / sx
print(f"[{time.time()-T0:.0f}s] {len(X)} samples, inputs {keys}, "
      f"mu {y.min():.4f}-{y.max():.4f}, {(y < 1e-9).sum()} zero-growth", flush=True)


def mlp(seed):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                     solver="adam", learning_rate_init=2e-3, batch_size=64,
                     max_iter=800, early_stopping=True, n_iter_no_change=40,
                     validation_fraction=0.15, random_state=seed))


def fit_ensemble(Xtr, ytr, n=5, seed=SEED):
    rng = np.random.default_rng(seed)
    nets = []
    for i in range(n):
        idx = rng.choice(len(Xtr), len(Xtr), replace=True)
        nets.append(mlp(seed + i).fit(Xtr[idx], ytr[idx]))
    return nets


def ensemble_predict(nets, X):
    P = np.stack([m.predict(X) for m in nets])
    return P.mean(0), P.std(0)


MODELS = {
    "ridge": lambda: RidgeCV(alphas=np.logspace(-4, 3, 40)),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
    "gradient_boosting": lambda: GradientBoostingRegressor(random_state=SEED),
}

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
rows, oof = [], np.full((len(y), len(MODELS) + 1), np.nan)
for fold, (tr, te) in enumerate(kf.split(Xs)):
    for j, (name, mk) in enumerate(MODELS.items()):
        p = mk().fit(Xs[tr], y[tr]).predict(Xs[te])
        oof[te, j] = p
        rows.append(dict(fold=fold, model=name, r2=r2_score(y[te], p),
                         mae=mean_absolute_error(y[te], p)))
    nets = fit_ensemble(Xs[tr], y[tr], n=5, seed=SEED + 10 * fold)
    p, _ = ensemble_predict(nets, Xs[te])
    oof[te, -1] = p
    rows.append(dict(fold=fold, model="mlp_ensemble", r2=r2_score(y[te], p),
                     mae=mean_absolute_error(y[te], p)))
    print(f"[{time.time()-T0:.0f}s] fold {fold}: " +
          ", ".join(f"{r['model']} R2={r['r2']:.4f}" for r in rows[-4:]), flush=True)

folds = pd.DataFrame(rows)
folds.to_csv("results/layer5_cv_folds.csv", index=False)
cv = folds.groupby("model").agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
                                mae_mean=("mae", "mean"), mae_std=("mae", "std")).reset_index()
cv = cv.sort_values("r2_mean", ascending=False)
cv.to_csv("results/layer5_cv_metrics.csv", index=False)
print(cv.round(5).to_string(index=False), flush=True)

pd.DataFrame(np.column_stack([y, oof]),
             columns=["observed"] + list(MODELS) + ["mlp_ensemble"]).to_csv(
    "results/layer5_oof_predictions.csv", index=False)

# ---- final ensemble on 80% of the data, intervals checked on the held-out 20%
tr = np.arange(len(Xs)) % 5 != 0
nets = fit_ensemble(Xs[tr], y[tr], n=5, seed=SEED)
mu_hat, sd_hat = ensemble_predict(nets, Xs[~tr])
cover = float(np.mean(np.abs(y[~tr] - mu_hat) <= 1.96 * sd_hat))
pd.DataFrame(dict(observed=y[~tr], predicted=mu_hat, sd=sd_hat)).to_csv(
    "results/layer5_ensemble_uncertainty.csv", index=False)

import pickle
with open("models/surrogate_mlp_ensemble.pkl", "wb") as fh:
    pickle.dump(dict(nets=nets, keys=keys), fh)

# ---- measured inference speed against the measured LP time
Xq = Xs[:1024]
nets[0].predict(Xq[:8])
t0 = time.perf_counter()
ensemble_predict(nets, Xq)
t_nn = (time.perf_counter() - t0) / len(Xq)
lp = json.load(open("results/layer5_lp_timing.json"))
spd = dict(seconds_per_surrogate_prediction=t_nn,
           seconds_per_lp_solve=lp["seconds_per_lp_solve"],
           speedup=lp["seconds_per_lp_solve"] / t_nn,
           n_surrogate=int(len(Xq)), n_lp=int(lp["n_lp"]),
           ensemble_95_coverage=cover, n_ensemble=len(nets))
json.dump(spd, open("results/layer5_speed_benchmark.json", "w"), indent=1)
print(f"[{time.time()-T0:.0f}s] surrogate {t_nn*1e6:.1f} us/pred vs LP "
      f"{lp['seconds_per_lp_solve']*1e3:.0f} ms ({spd['speedup']:.0f}x); "
      f"ensemble 95% coverage={cover:.3f}", flush=True)
print(f"[{time.time()-T0:.0f}s] LAYER5 DONE", flush=True)

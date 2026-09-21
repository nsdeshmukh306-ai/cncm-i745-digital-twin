"""Layer 4: host-interaction kinetics - Bayesian calibration and sensitivity.

(a) Toxin A proteolysis by the 54 kDa serine protease is calibrated with emcee
    against two citable quantities from Castagliuolo et al. (1999): 60%
    reversal of the toxin-induced resistance drop after a 1 h pre-exposure,
    and 73% of the proteolytic activity being protease-dependent. Reported
    quantities are posterior medians with 95% credible intervals.
(b) The NF-kB / cytokine module is NOT fitted - there is no citable
    quantitative time course to fit it to. It is analysed instead with Sobol
    variance decomposition, which says which parameters would have to be
    measured to make its predictions trustworthy.
(c) The barrier-function link to Layer 2 is evaluated over the polyamine flux
    the genome-scale model actually predicts.
"""
import sys, os, time, json
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sbdtwin import kinetics as K

os.makedirs("results", exist_ok=True)
T0 = time.time()
rng = np.random.default_rng(7)

# ------------------------------------------------------------ (a) calibration
import emcee
ndim, nwalk, nstep, burn = 3, 24, 3000, 800
p0 = np.array([np.log(50.0), np.log(50.0), np.log(0.15)]) + 0.25 * rng.standard_normal((nwalk, ndim))
sampler = emcee.EnsembleSampler(nwalk, ndim, K.log_post_tcda_constraints)
sampler.run_mcmc(p0, nstep, progress=False)
chain = sampler.get_chain(discard=burn, flat=True)
acc = float(np.mean(sampler.acceptance_fraction))
try:
    tau = sampler.get_autocorr_time(quiet=True)
except Exception:
    tau = np.array([np.nan] * ndim)
print(f"[{time.time()-T0:.0f}s] MCMC: {chain.shape[0]} samples, acceptance={acc:.2f}, "
      f"tau={np.round(tau, 1)}", flush=True)

vmax, km, ks = np.exp(chain).T
post = pd.DataFrame(dict(vmax=vmax, km=km, k_spont=ks))
sub = post.sample(min(2000, len(post)), random_state=1)
post_pred = pd.DataFrame([dict(
    half_life_h=K.half_life(r.vmax, r.km, r.k_spont),
    frac_1h=K.fraction_degraded(r.vmax, r.km, r.k_spont, 1.0),
    frac_4h=K.fraction_degraded(r.vmax, r.km, r.k_spont, 4.0),
    protease_share=K.protease_share(r.vmax, r.km, r.k_spont))
    for r in sub.itertuples()])
post.to_csv("results/layer4_posterior_samples.csv", index=False)

rows = []
for name, v in list(post.items()) + list(post_pred.items()):
    q = np.nanpercentile(v, [2.5, 50, 97.5])
    rows.append(dict(quantity=name, median=round(float(q[1]), 4),
                     lo95=round(float(q[0]), 4), hi95=round(float(q[2]), 4)))
summ = pd.DataFrame(rows)
summ.to_csv("results/layer4_posterior_summary.csv", index=False)
print(summ.to_string(index=False), flush=True)

# posterior-predictive toxin trajectory
t_grid = np.linspace(0, 8, 81)
traj = np.array([K.simulate_tcda(r.vmax, r.km, r.k_spont, t_end=8.0, n=81)[1]
                 for r in sub.head(400).itertuples()])
pd.DataFrame(dict(t_h=t_grid,
                  median=np.median(traj, axis=0),
                  lo95=np.percentile(traj, 2.5, axis=0),
                  hi95=np.percentile(traj, 97.5, axis=0))
             ).to_csv("results/layer4_toxin_trajectory.csv", index=False)

# ------------------------------------------------------------ (b) Sobol on NF-kB
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

NAMES = ["k_act", "k_deg", "k_sup", "k_il1", "k_il1d", "k_tnf", "k_tnfd",
         "k_il10", "k_il10d", "k_fb"]
CENTRE = np.array([1.5, 1.0, 2.33, 1.33, 1.0, 1.4, 1.0, 1.33, 1.0, 0.1])
problem = dict(num_vars=len(NAMES), names=NAMES,
               bounds=[[c * 0.4, c * 2.5] for c in CENTRE])
X = sobol_sample.sample(problem, 256, calc_second_order=False)
Y = np.array([K.suppression(x) for x in X])
ok = np.isfinite(Y)
Si = sobol_analyze.analyze(problem, np.where(ok, Y, np.nanmean(Y[ok])),
                           calc_second_order=False, print_to_console=False)
pd.DataFrame(dict(parameter=NAMES, S1=Si["S1"], S1_conf=Si["S1_conf"],
                  ST=Si["ST"], ST_conf=Si["ST_conf"])
             ).sort_values("ST", ascending=False).to_csv(
    "results/layer4_sobol.csv", index=False)
print(f"[{time.time()-T0:.0f}s] Sobol: n={len(X)} evaluations, "
      f"suppression range {np.nanmin(Y):.3f}-{np.nanmax(Y):.3f}", flush=True)

# prior-predictive interval on suppression (explicitly NOT a fit)
qs = np.nanpercentile(Y, [2.5, 50, 97.5])
json.dump(dict(suppression_prior_predictive=dict(
    median=float(qs[1]), lo95=float(qs[0]), hi95=float(qs[2]), n=int(ok.sum()),
    note="prior-predictive under uniform parameter ranges; not calibrated to data")),
    open("results/layer4_nfkb_prior_predictive.json", "w"), indent=1)

# ------------------------------------------------------------ (c) barrier index
pol = pd.DataFrame([dict(polyamine_flux=f, barrier_index=K.barrier_index(f))
                    for f in np.round(np.linspace(0, 3, 61), 3)])
pol.to_csv("results/layer4_barrier_curve.csv", index=False)
print(f"[{time.time()-T0:.0f}s] LAYER4 DONE", flush=True)

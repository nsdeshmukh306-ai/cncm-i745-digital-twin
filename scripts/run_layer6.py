"""Layer 6: S. boulardii vs C. difficile competition and colonisation resistance."""
import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sbdtwin import competition as C

os.makedirs("results", exist_ok=True)
T0 = time.time()

# reference time courses: with and without the yeast
rows = []
for label, y0, pp in [("dosed_S_boulardii", (25.0, 0.05, 0.05, 0.0), None),
                      ("undosed_S_boulardii", (25.0, 0.05, 0.05, 0.0), {"Y_in": 0.0}),
                      ("C_difficile_alone", (25.0, 0.0, 0.05, 0.0), {"Y_in": 0.0})]:
    t, y, p = C.simulate(pp, t_end=200.0, y0=y0)
    for i in range(0, len(t), 4):
        rows.append(dict(scenario=label, t_h=float(t[i]), substrate=float(y[0][i]),
                         boulardii=float(y[1][i]), c_difficile=float(y[2][i]),
                         toxinA=float(y[3][i])))
pd.DataFrame(rows).to_csv("results/layer6_timecourses.csv", index=False)

base_out = C.outcome()
alone = C.simulate({"Y_in": 0.0}, t_end=250.0, y0=(25.0, 0.0, 0.05, 0.0))[1]
tox_red = 1.0 - base_out["A_final"] / max(float(alone[3][-1]), 1e-12)
print(f"[{time.time()-T0:.0f}s] baseline: {base_out['classification']}, "
      f"C. difficile {base_out['C_final']:.4g} gDW/L, toxin A reduced {100*tox_red:.1f}%", flush=True)

# outcome map over inhibition strength x transit rate
grid = C.exclusion_boundary(np.round(np.linspace(0.0, 3.0, 16), 3),
                            np.round(np.linspace(0.02, 0.5, 16), 4))
g = pd.DataFrame(grid)
g.to_csv("results/layer6_outcome_map.csv", index=False)
print(g.classification.value_counts().to_string(), flush=True)

# critical dilution rate at which C. difficile is excluded, per k_ox
crit = []
for k, sub in g.groupby("k_ox"):
    exc = sub[sub.classification == "C_difficile_excluded"]
    crit.append(dict(k_ox=float(k),
                     D_min_exclusion=float(exc.D.min()) if len(exc) else np.nan,
                     D_max_exclusion=float(exc.D.max()) if len(exc) else np.nan,
                     n_excluded=int(len(exc))))
pd.DataFrame(crit).to_csv("results/layer6_exclusion_boundary.csv", index=False)

# minimum daily dose required to exclude C. difficile
dose = []
for D in [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    for g in [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        o = C.outcome({"D": D, "Y_in": C.dose_to_Y_in(g, D)})
        dose.append(dict(D=D, g_per_day=g, Y_in=round(C.dose_to_Y_in(g, D), 4), **o))
dd = pd.DataFrame(dose)
dd.to_csv("results/layer6_dose_response.csv", index=False)
mind = []
for D, sub in dd.groupby("D"):
    exc = sub[sub.classification == "C_difficile_excluded"]
    mind.append(dict(D=float(D),
                     min_dose_g_per_day=float(exc.g_per_day.min()) if len(exc) else np.nan))
pd.DataFrame(mind).to_csv("results/layer6_min_dose.csv", index=False)
print("minimum dose for exclusion (g dry biomass/day):",
      {r["D"]: r["min_dose_g_per_day"] for r in mind}, flush=True)

# univariate sensitivity of toxin burden to each parameter
sens = []
for par, vals in [("k_ox", [0.2, 0.5, 0.8, 1.2, 2.0]),
                  ("k_prot", [0.3, 0.6, 1.2, 2.4, 4.8]),
                  ("D", [0.05, 0.08, 0.10, 0.15, 0.25]),
                  ("S_in", [10.0, 18.0, 25.0, 35.0, 50.0]),
                  ("mu_y", [0.20, 0.28, 0.35, 0.45, 0.55])]:
    for v in vals:
        o = C.outcome({par: v})
        sens.append(dict(parameter=par, value=v, **o))
pd.DataFrame(sens).to_csv("results/layer6_sensitivity.csv", index=False)
print(f"[{time.time()-T0:.0f}s] LAYER6 DONE", flush=True)

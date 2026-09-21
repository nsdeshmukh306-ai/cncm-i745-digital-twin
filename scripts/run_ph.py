"""pH-dependent bioenergetics of gastric transit (Layer 3, stomach zone)."""
import sys, time
import numpy as np, pandas as pd, cobra
sys.path.insert(0, ".")
from sbdtwin import gem as G, bioenergetics as B

T0 = time.time()
m = cobra.io.read_sbml_model("models/yeast9_cncm_i745.xml")
ng = B.ngam_reaction(m)
print(f"[{time.time()-T0:.0f}s] NGAM reaction: {ng.id} {ng.name} bounds={ng.bounds}", flush=True)
rows = []
for k_leak in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
    rows += B.growth_vs_pH(m, G.set_medium, k_leak,
                           pH_grid=np.round(np.arange(1.5, 7.6, 0.25), 2),
                           carbon=("glucose", G.GUT_GLUCOSE), oxygen=0.1)
d = pd.DataFrame(rows)
d.to_csv("results/layer3_ph_bioenergetics.csv", index=False)
surv = {float(k): B.survival_pH(g.to_dict("records")) for k, g in d.groupby("k_leak")}
print("survival pH by leak coefficient:", {k: round(v, 2) for k, v in surv.items()}, flush=True)
print(d[np.isclose(d.pH, 2.0)][["k_leak", "ngam_extra", "pump_cost_kJ_per_mol", "mu"]].round(4).to_string(index=False), flush=True)
pd.DataFrame([dict(k_leak=k, survival_pH=v) for k, v in surv.items()]).to_csv(
    "results/layer3_survival_pH.csv", index=False)
print(f"[{time.time()-T0:.0f}s] PH DONE", flush=True)

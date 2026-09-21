"""Single-gene deletion screen: per-gene context rollback (fast path)."""
import sys, os, time
import numpy as np, pandas as pd, cobra
sys.path.insert(0, ".")
from sbdtwin import gem as G

PHASE = sys.argv[1]
LABEL = "Yeast9_base" if PHASE == "base" else "CNCM_I745"
T0 = time.time()
m = cobra.io.read_sbml_model(G.YEAST9_SBML if PHASE == "base"
                             else "models/yeast9_cncm_i745.xml")
G.set_medium(m, carbon=("glucose", G.GUT_GLUCOSE), oxygen=G.GUT_OXYGEN)
wt = float(m.slim_optimize())
print(f"[{time.time()-T0:.0f}s] {LABEL} wt mu={wt:.6f}", flush=True)

rows = []
gids = [g.id for g in m.genes if len(g.reactions) > 0]
for i, gid in enumerate(gids):
    with m:
        m.genes.get_by_id(gid).knock_out()
        v = m.slim_optimize()
    v = 0.0 if (v is None or v != v or v < 0) else float(v)
    rows.append((gid, v, v / wt))
    if (i + 1) % 250 == 0:
        print(f"    {i+1}/{len(gids)} ({time.time()-T0:.0f}s)", flush=True)
d = pd.DataFrame(rows, columns=["gene", "growth", "ratio"])
d["classification"] = np.where(d.growth < 1e-6, "essential",
                      np.where(d.ratio < 0.5, "growth_reduced", "neutral"))
d["model"], d["wt_growth"] = LABEL, round(wt, 6)
d.to_csv(f"results/layer2_essentiality_{PHASE}.csv", index=False)
print(f"[{time.time()-T0:.0f}s] {LABEL}: n={len(d)} "
      f"{d.classification.value_counts().to_dict()}", flush=True)

"""Layer 2 driver: strain model, phenotype validation, capacity, essentiality.

Run as: python scripts/run_layer2.py <phase>   with phase in {base, strain}

Only ONE genome-scale model is ever resident: the two phases run as separate
processes so the operating system reclaims the memory in between. The strain
phase reloads the SBML and applies the correction in place rather than copying.
"""
import sys, os, time
import numpy as np, pandas as pd, cobra
sys.path.insert(0, ".")
from sbdtwin import gem as G

PHASE = sys.argv[1] if len(sys.argv) > 1 else "base"
os.makedirs("results", exist_ok=True); os.makedirs("models", exist_ok=True)
T0 = time.time()
LABEL = "Yeast9_base" if PHASE == "base" else "CNCM_I745"

m = cobra.io.read_sbml_model(G.YEAST9_SBML)
snap = G.snapshot_gpr(m)
print(f"[{time.time()-T0:.0f}s] loaded Yeast9: {len(m.reactions)} rxns, "
      f"{len(m.metabolites)} mets, {len(m.genes)} genes", flush=True)

if PHASE == "strain":
    edits = G.correct_in_place(m)
    pd.DataFrame(edits).to_csv("results/layer2_gpr_edits.csv", index=False)
    cobra.io.write_sbml_model(m, "models/yeast9_cncm_i745.xml")

active = sum(1 for r in m.reactions if r.bounds != (0.0, 0.0))
pd.DataFrame([dict(model=LABEL, reactions=len(m.reactions), active_reactions=active,
                   metabolites=len(m.metabolites), genes=len(m.genes),
                   genes_with_reactions=sum(1 for g in m.genes if len(g.reactions) > 0))]
             ).to_csv(f"results/layer2_model_summary_{PHASE}.csv", index=False)

# ---------------------------------------------------- phenotype sweep
rows = []
for src in ["glucose", "galactose", "maltose", "sucrose", "trehalose", "glycerol",
            "ethanol", "acetate", "raffinose", "melibiose"]:
    rows.append(dict(substrate=src, role="carbon", model=LABEL,
                     mu=round(G.growth(m, carbon=(src, 10.0), oxygen=1000.0), 6)))
for nsrc in ["ammonium", "asparagine"]:
    rows.append(dict(substrate=nsrc, role="nitrogen", model=LABEL,
                     mu=round(G.growth(m, carbon=("glucose", 10.0),
                                       nitrogen=(nsrc, 10.0), oxygen=1000.0), 6)))
pd.DataFrame(rows).to_csv(f"results/layer2_phenotype_{PHASE}.csv", index=False)
print(f"[{time.time()-T0:.0f}s] phenotype sweep done", flush=True)

# ---------------------------------------------------- capacity sweep
cap = []
for v in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
    for src in ["glucose", "galactose", "maltose", "sucrose"]:
        with m:
            G.capacity_from_snapshot(m, snap, v, apply_loss=(PHASE == "strain"))
            G.set_medium(m, carbon=(src, 10.0), oxygen=1000.0)
            mu = m.slim_optimize()
        cap.append(dict(v_per_paralogue=v, model=LABEL, substrate=src,
                        mu=round(0.0 if (mu is None or mu != mu) else float(mu), 6)))
pd.DataFrame(cap).to_csv(f"results/layer2_capacity_{PHASE}.csv", index=False)
if PHASE == "strain":
    with m:
        tc = G.capacity_from_snapshot(m, snap, 1.0)
    pd.DataFrame(tc).to_csv("results/layer2_transporter_capacity.csv", index=False)
print(f"[{time.time()-T0:.0f}s] capacity sweep done", flush=True)

# ---------------------------------------------------- pFBA at gut conditions
from cobra.flux_analysis import pfba
with m:
    G.set_medium(m, carbon=("glucose", G.GUT_GLUCOSE), oxygen=G.GUT_OXYGEN)
    s = pfba(m)
pd.DataFrame([dict(model=LABEL, reaction=rid, name=m.reactions.get_by_id(rid).name,
                   abs_flux=round(float(v), 5))
              for rid, v in s.fluxes.abs().sort_values(ascending=False).head(40).items()]
             ).to_csv(f"results/layer2_pfba_{PHASE}.csv", index=False)
print(f"[{time.time()-T0:.0f}s] pFBA done", flush=True)

print(f"[{time.time()-T0:.0f}s] LAYER2 {PHASE.upper()} DONE", flush=True)

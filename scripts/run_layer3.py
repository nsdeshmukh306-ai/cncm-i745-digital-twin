"""Layer 3: expression-constrained zone metabolism with uncertainty propagation.

Design
------
The three intestinal zones are constrained by genome-wide expression from GEO
GSE18 (Gasch et al. 2000), mapped onto Yeast9 GPRs by E-Flux. Instead of one
point estimate per zone, each zone is evaluated over an ensemble of expression
draws sampled within the between-sample SD of the matched GEO arrays, so every
reported flux carries an interval that reflects real measurement uncertainty.

Zone contrasts are then tested per reaction with Welch's t-test and
Benjamini-Hochberg FDR control, which is what makes "this pathway differs
between ileum and colon" a statistical claim rather than a visual one.

The stomach is deliberately NOT modelled by expression: no acid-shock condition
exists in GSE18, and the dominant effect of pH 2 is bioenergetic, not
transcriptional. It is handled by sbdtwin.bioenergetics instead.
"""
import sys, os, time
import numpy as np, pandas as pd, cobra
sys.path.insert(0, ".")
from sbdtwin import gem as G, eflux as E, bioenergetics as B

N_DRAWS = int(os.environ.get("N_DRAWS", "60"))
SEED = 20260921
os.makedirs("results", exist_ok=True)
T0 = time.time()

ez = pd.read_csv("data/expression_zones.csv")
m = cobra.io.read_sbml_model("models/yeast9_cncm_i745.xml")
print(f"[{time.time()-T0:.0f}s] strain model loaded ({len(m.reactions)} rxns)", flush=True)

rng = np.random.default_rng(SEED)
zones = ["duodenum", "ileum", "colon"]
ZONE_O2 = {"duodenum": 2.0, "ileum": 5.0, "colon": 0.0}

# ---------------------------------------------------------------- 1. E-Flux method comparison
cmp_rows = []
for z in zones:
    sub = ez[ez.zone == z]
    expr = dict(zip(sub.systematic_id, np.power(2.0, sub.log2_ratio)))
    for meth in ["canonical", "sqrt_geom"]:
        with m:
            applied, mapped = E.apply_eflux(m, expr, method=meth)
            G.set_medium(m, carbon=("glucose", G.GUT_GLUCOSE), oxygen=max(ZONE_O2[z], 0.1))
            mu = m.slim_optimize()
        cmp_rows.append(dict(zone=z, method=meth, n_constrained=len(applied),
                             n_gpr_mapped=mapped, mu=float(mu) if mu == mu else 0.0))
pd.DataFrame(cmp_rows).to_csv("results/layer3_method_comparison.csv", index=False)
print(pd.DataFrame(cmp_rows).to_string(index=False), flush=True)

# ---------------------------------------------------------------- 2. zone ensembles
from cobra.flux_analysis import pfba
flux_store = {}
for z in zones:
    sub = ez[ez.zone == z]
    mu_l2 = dict(zip(sub.systematic_id, sub.log2_ratio))
    sd_l2 = dict(zip(sub.systematic_id, sub.sd_log2))
    mat, mus = [], []
    for k, expr in enumerate(E.expression_ensemble(mu_l2, sd_l2, N_DRAWS, rng)):
        with m:
            E.apply_eflux(m, expr, method="canonical")
            G.set_medium(m, carbon=("glucose", G.GUT_GLUCOSE), oxygen=max(ZONE_O2[z], 0.1))
            try:
                s = pfba(m)
                mat.append(s.fluxes.values)
                mus.append(float(s.fluxes["r_2111"]))
            except Exception:
                continue
        if (k + 1) % 20 == 0:
            print(f"    {z} {k+1}/{N_DRAWS} ({time.time()-T0:.0f}s)", flush=True)
    flux_store[z] = np.array(mat)
    print(f"[{time.time()-T0:.0f}s] {z}: {len(mat)} feasible draws, "
          f"mu={np.mean(mus):.4f}+/-{np.std(mus):.4f}", flush=True)

rxn_ids = [r.id for r in m.reactions]
rxn_names = {r.id: r.name for r in m.reactions}
summ = []
for z, arr in flux_store.items():
    if arr.size == 0:
        continue
    lo, med, hi = np.percentile(arr, [2.5, 50, 97.5], axis=0)
    for i, rid in enumerate(rxn_ids):
        summ.append(dict(zone=z, reaction=rid, name=rxn_names[rid],
                         median=round(float(med[i]), 6), lo95=round(float(lo[i]), 6),
                         hi95=round(float(hi[i]), 6)))
pd.DataFrame(summ).to_csv("results/layer3_zone_fluxes.csv", index=False)

# ---------------------------------------------------------------- 3. FDR-controlled contrasts
from scipy import stats
contrasts = [("ileum", "colon"), ("duodenum", "colon"), ("duodenum", "ileum")]
rows = []
for a, b in contrasts:
    A, Bm = flux_store[a], flux_store[b]
    if A.size == 0 or Bm.size == 0:
        continue
    t, p = stats.ttest_ind(A, Bm, axis=0, equal_var=False)
    d = A.mean(axis=0) - Bm.mean(axis=0)
    pool = np.sqrt((A.var(axis=0, ddof=1) + Bm.var(axis=0, ddof=1)) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(pool > 0, d / pool, 0.0)
    ok = np.isfinite(p)
    padj = np.full_like(p, np.nan)
    pv = p[ok]
    order = np.argsort(pv)
    n = len(pv)
    adj = np.empty(n)
    adj[order] = np.minimum.accumulate((pv[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    padj[ok] = np.clip(adj, 0, 1)
    for i, rid in enumerate(rxn_ids):
        rows.append(dict(contrast=f"{a}_vs_{b}", reaction=rid, name=rxn_names[rid],
                         mean_diff=round(float(d[i]), 6), cohens_d=round(float(g[i]), 4),
                         p=float(p[i]) if np.isfinite(p[i]) else np.nan,
                         p_adj=float(padj[i]) if np.isfinite(padj[i]) else np.nan))
con = pd.DataFrame(rows)
con.to_csv("results/layer3_zone_contrasts.csv", index=False)
sig = con[(con.p_adj < 0.05) & (con.mean_diff.abs() > 1e-6)]
print(f"[{time.time()-T0:.0f}s] significant reactions (FDR<0.05): "
      f"{sig.groupby('contrast').size().to_dict()}", flush=True)

# ---------------------------------------------------------------- 4. stomach bioenergetics
bio = []
for k_leak in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
    bio += B.growth_vs_pH(m, G.set_medium, k_leak,
                          pH_grid=np.round(np.arange(1.5, 7.6, 0.25), 2),
                          carbon=("glucose", G.GUT_GLUCOSE), oxygen=0.1)
bdf = pd.DataFrame(bio)
bdf.to_csv("results/layer3_ph_bioenergetics.csv", index=False)
surv = {float(k): B.survival_pH(g.to_dict("records")) for k, g in bdf.groupby("k_leak")}
print("survival pH by leak coefficient:", {k: round(v, 2) for k, v in surv.items()}, flush=True)
print(f"[{time.time()-T0:.0f}s] LAYER3 DONE", flush=True)

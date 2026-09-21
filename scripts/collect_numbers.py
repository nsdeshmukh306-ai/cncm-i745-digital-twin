"""Collect every number the manuscript quotes into tables/NUMBERS.json.

Nothing in the paper is hand-typed: the DOCX builder reads only this file, so
a claim in the text cannot drift from the result that produced it.
"""
import os, json, glob
import numpy as np, pandas as pd

R, T = "results", "tables"
os.makedirs(T, exist_ok=True)
N = {}


def csv(name):
    p = os.path.join(R, name)
    return pd.read_csv(p) if os.path.exists(p) else None


# ---------------------------------------------------------------- Layer 1
gs = csv("layer1_genome_stats.csv")
calls = csv("layer1_marker_calls.csv")
intro = csv("layer1_introgression.csv")
SB6 = ["Sb_unique28", "Sb_PY0001", "Sb_EDRL", "Sb_CLPY01", "Sb_ATCC_MYA797", "Sb_strain17"]
CTRL = ["ACT1", "PGK1", "TDH3", "PMA1", "TPS1", "GPD1", "GLR1", "TRR1"]
if gs is not None:
    s288c = gs[gs.assembly == "Sc_S288C"].iloc[0]
    sb = gs[gs.assembly.isin(SB6)]
    N["l1_n_assemblies_screened"] = int(len(gs))
    N["l1_n_boulardii_qualifying"] = len(SB6)
    N["l1_s288c_seqs"] = int(s288c.n_seqs)
    N["l1_s288c_bp"] = int(s288c.total_bp)
    N["l1_s288c_n50"] = int(s288c.n50)
    N["l1_s288c_gc"] = float(s288c.gc_pct)
    N["l1_sb_bp_min"] = int(sb.total_bp.min()); N["l1_sb_bp_max"] = int(sb.total_bp.max())
    N["l1_sb_gc_min"] = float(sb.gc_pct.min()); N["l1_sb_gc_max"] = float(sb.gc_pct.max())
    N["l1_sb_n50_min"] = int(sb.n50.min()); N["l1_sb_n50_max"] = int(sb.n50.max())
if calls is not None:
    N["l1_n_query_genes"] = int(calls.gene.nunique())
    ctl = calls[(calls.assembly == "Sc_S288C")]
    N["l1_control_recovery"] = int((ctl.call.isin(["present", "present_ambiguous"])).sum())
    N["l1_control_total"] = int(len(ctl))
    kc = calls[(calls.assembly == "Sb_KCTC13826BP") & (calls.gene.isin(CTRL))]
    N["l1_kctc_core_pid_min"] = round(float(kc.best_pident.min()), 1)
    N["l1_kctc_core_pid_max"] = round(float(kc.best_pident.max()), 1)
    N["l1_kctc_core_pid_median"] = round(float(kc.best_pident.median()), 1)
    N["l1_kctc_core_below95"] = int((kc.best_pident < 95).sum())
    N["l1_n_core_controls"] = int(len(CTRL))
    ok = calls[calls.assembly.isin(SB6) & calls.gene.isin(CTRL)]
    N["l1_sb_core_pid_min"] = round(float(ok.best_pident.min()), 1)
    N["l1_sb_core_pid_median"] = round(float(ok.best_pident.median()), 1)
    N["l1_sb_core_below95"] = int((ok.best_pident < 95).sum())
    piv = calls.pivot_table(index="gene", columns="assembly", values="call", aggfunc="first")[SB6]
    lost = piv[piv.isin(["absent", "degraded"]).all(axis=1)].index.tolist()
    N["l1_genes_absent_all6"] = sorted(lost)
    N["l1_n_absent_all6"] = len(lost)
    kh = pd.read_csv("data/khatri2017_absent_genes.csv")
    testable = sorted(set(kh.gene) & set(piv.index))
    N["l1_khatri_testable"] = len(testable)
    N["l1_khatri_confirmed"] = int(sum(g in lost for g in testable))
    N["l1_khatri_unconfirmed"] = sorted(g for g in testable if g not in lost)
    gal = [g for g in piv.index if g.startswith("GAL")]
    N["l1_gal_genes"] = sorted(gal)
    N["l1_gal_all_present"] = bool(piv.loc[gal].isin(["present", "present_ambiguous"]).all().all())
if intro is not None:
    c4 = intro[intro.chromosome == "chrIV"]
    N["l1_introgressed_genes"] = int(c4.zb_protein.nunique())
    N["l1_introgression_copies"] = int(c4.groupby("zb_protein").size().max())
    N["l1_introgression_start"] = int(c4.start.min()); N["l1_introgression_end"] = int(c4.end.max())
    N["l1_introgression_pid_min"] = round(float(c4.identity_to_Zbailii_pct.min()), 2)
    N["l1_introgression_pid_max"] = round(float(c4.identity_to_Zbailii_pct.max()), 2)
    N["l1_introgression_sc_pid_max"] = round(float(c4.identity_in_S288C_pct.max()), 1)

# ---------------------------------------------------------------- Layer 2
ed = csv("layer2_gpr_edits.csv")
if ed is not None:
    N["l2_n_gpr_edits"] = int(len(ed))
    N["l2_n_deleted"] = int((ed.consequence == "reaction_deleted").sum())
    N["l2_n_rewritten_only"] = int((ed.consequence == "gpr_rewritten_only").sum())
    N["l2_deleted_reactions"] = ed[ed.consequence == "reaction_deleted"].reaction_name.tolist()
    gl = set()
    for g in ed.genes_removed.astype(str):
        gl.update(x.strip() for x in g.split(";") if x.strip())
    N["l2_n_absent_in_model"] = len(gl)
    N["l2_absent_in_model"] = sorted(gl)
for ph in ("base", "strain"):
    ms = csv(f"layer2_model_summary_{ph}.csv")
    if ms is not None:
        r0 = ms.iloc[0]
        N[f"l2_{ph}_reactions"] = int(r0.reactions)
        N[f"l2_{ph}_active_reactions"] = int(r0.active_reactions)
        N[f"l2_{ph}_metabolites"] = int(r0.metabolites)
        N[f"l2_{ph}_genes"] = int(r0.genes)
pb, ps = csv("layer2_phenotype_base.csv"), csv("layer2_phenotype_strain.csv")
if pb is not None and ps is not None:
    w = pd.concat([pb, ps]).pivot_table(index="substrate", columns="model", values="mu")
    w["delta"] = w.CNCM_I745 - w.Yeast9_base
    N["l2_phenotype"] = {k: dict(base=round(float(v.Yeast9_base), 5),
                                 strain=round(float(v.CNCM_I745), 5),
                                 delta=round(float(v.delta), 5)) for k, v in w.iterrows()}
    N["l2_n_substrates"] = int(len(w))
    N["l2_n_differing"] = int((w.delta.abs() > 1e-6).sum())
    N["l2_differing"] = sorted(w.index[w.delta.abs() > 1e-6].tolist())
eb, es = csv("layer2_essentiality_base.csv"), csv("layer2_essentiality_strain.csv")
if eb is not None and es is not None:
    N["l2_ess_base"] = {k: int(v) for k, v in eb.classification.value_counts().items()}
    N["l2_ess_strain"] = {k: int(v) for k, v in es.classification.value_counts().items()}
    A = set(eb.query("classification=='essential'").gene)
    B = set(es.query("classification=='essential'").gene)
    N["l2_ess_gained"] = sorted(B - A); N["l2_ess_lost"] = sorted(A - B)
    N["l2_ess_n_changed"] = len(A ^ B)
    N["l2_wt_growth"] = round(float(eb.wt_growth.iloc[0]), 5)
cb, cs = csv("layer2_capacity_base.csv"), csv("layer2_capacity_strain.csv")
if cb is not None and cs is not None:
    cap = pd.concat([cb, cs]).pivot_table(index=["substrate", "v_per_paralogue"],
                                          columns="model", values="mu").reset_index()
    cap["ratio"] = cap.CNCM_I745 / cap.Yeast9_base.replace(0, np.nan)
    N["l2_capacity"] = {}
    for s, g in cap.groupby("substrate"):
        b = g[(g.ratio < 0.999) & g.ratio.notna()]
        N["l2_capacity"][s] = dict(
            min_ratio=round(float(g.ratio.min()), 3) if g.ratio.notna().any() else None,
            binding_v_lo=float(b.v_per_paralogue.min()) if len(b) else None,
            binding_v_hi=float(b.v_per_paralogue.max()) if len(b) else None,
            max_deficit_pct=round(100 * (1 - float(b.ratio.min())), 1) if len(b) else 0.0,
            deficit_at_1_pct=round(100 * (1 - float(
                g.loc[np.isclose(g.v_per_paralogue, 1.0), "ratio"].iloc[0])), 1)
            if (np.isclose(g.v_per_paralogue, 1.0)).any() else None)
tc = csv("layer2_transporter_capacity.csv")
if tc is not None:
    for rid, nm in [("r_1135", "galactose"), ("r_1166", "glucose"), ("r_1227", "maltose")]:
        row = tc[tc.reaction == rid]
        if len(row):
            N[f"l2_paralogues_{nm}"] = [int(row.n_paralogues.iloc[0]), int(row.n_retained.iloc[0])]

pm = csv("layer2_phenotype_scores.csv")
if pm is not None:
    N["l2_phenotype_scores"] = {r.model: dict(n=int(r.n), TP=int(r.TP), TN=int(r.TN),
                                              FP=int(r.FP), FN=int(r.FN),
                                              accuracy=float(r.accuracy),
                                              mcc=float(r.mcc))
                                for _, r in pm.iterrows()}
f_mc = "results/layer2_mcnemar.json"
if os.path.exists(f_mc):
    N["l2_mcnemar"] = json.load(open(f_mc))
pmx = csv("layer2_phenotype_matrix.csv")
if pmx is not None:
    N["l2_panel_n"] = int(pmx.substrate.nunique())
    N["l2_panel_representable"] = int(pmx[pmx.model == "CNCM_I745"].representable.sum())

# ---------------------------------------------------------------- Layer 3
mc = csv("layer3_method_comparison.csv")
if mc is not None:
    N["l3_n_reactions_constrained"] = int(mc.n_constrained.max())
    N["l3_n_gpr_mapped"] = int(mc.n_gpr_mapped.max())
    N["l3_zone_mu"] = {r.zone: round(float(r.mu), 5)
                       for r in mc[mc.method == "canonical"].itertuples()}
ez = pd.read_csv("data/expression_zones.csv") if os.path.exists("data/expression_zones.csv") else None
if ez is not None:
    N["l3_genes_per_zone"] = int(ez.groupby("zone").size().min())
zc = csv("layer3_zone_contrasts.csv")
if zc is not None:
    sig = zc[(zc.p_adj < 0.05) & (zc.mean_diff.abs() > 1e-6)]
    N["l3_n_tested_per_contrast"] = int(len(zc) // zc.contrast.nunique())
    N["l3_significant"] = {k: int(v) for k, v in sig.groupby("contrast").size().items()}
ph3 = csv("layer3_ph_bioenergetics.csv")
if ph3 is not None:
    at2 = ph3[np.isclose(ph3.pH, 2.0)]
    N["l3_pump_cost_pH2"] = round(float(at2.pump_cost_kJ_per_mol.iloc[0]), 2)
    N["l3_pump_cost_pH7"] = round(float(ph3[np.isclose(ph3.pH, 7.0)].pump_cost_kJ_per_mol.iloc[0]), 2)
    N["l3_ngam_basal"] = round(float(ph3.ngam_base.iloc[0]), 3)
    N["l3_ph2_ngam"] = {float(r.k_leak): round(float(r.ngam_extra), 3) for r in at2.itertuples()}
    N["l3_ph2_mu"] = {float(r.k_leak): round(float(r.mu), 5) for r in at2.itertuples()}
    dead = at2[at2.mu < 1e-6]
    N["l3_kleak_lethal_at_pH2"] = float(dead.k_leak.min()) if len(dead) else None
    N["l3_kleak_grid"] = sorted(float(x) for x in ph3.k_leak.unique())
sp = csv("layer3_survival_pH.csv")
if sp is not None:
    N["l3_survival_pH"] = {float(r.k_leak): float(r.survival_pH) for r in sp.itertuples()}

# ---------------------------------------------------------------- Layer 4
p4 = csv("layer4_posterior_summary.csv")
if p4 is not None:
    N["l4_posterior"] = {r.quantity: dict(median=float(r["median"]), lo95=float(r.lo95),
                                          hi95=float(r.hi95)) for _, r in p4.iterrows()}
s4 = csv("layer4_sobol.csv")
if s4 is not None:
    top = s4.sort_values("ST", ascending=False).iloc[0]
    N["l4_sobol_top"] = dict(parameter=top.parameter, S1=round(float(top.S1), 3),
                             ST=round(float(top.ST), 3))
    N["l4_sobol"] = [dict(parameter=r.parameter, S1=round(float(r.S1), 4),
                          ST=round(float(r.ST), 4)) for _, r in s4.iterrows()]
f4 = "results/layer4_nfkb_prior_predictive.json"
if os.path.exists(f4):
    N["l4_nfkb_prior_predictive"] = json.load(open(f4))["suppression_prior_predictive"]

# ---------------------------------------------------------------- Layer 5
cv = csv("layer5_cv_metrics.csv")
if cv is not None:
    N["l5_cv"] = {r.model: dict(r2=round(float(r.r2_mean), 4), r2_sd=round(float(r.r2_std), 4),
                                mae=round(float(r.mae_mean), 6))
                  for _, r in cv.iterrows()}
    best = cv.sort_values("r2_mean", ascending=False).iloc[0]
    N["l5_best_model"] = best.model
sb5 = "results/layer5_speed_benchmark.json"
if os.path.exists(sb5):
    d = json.load(open(sb5))
    N["l5_speed"] = dict(surrogate_us=round(d["seconds_per_surrogate_prediction"] * 1e6, 2),
                         lp_ms=round(d["seconds_per_lp_solve"] * 1e3, 2),
                         speedup=round(d["speedup"], 1),
                         coverage=round(d.get("ensemble_95_coverage",
                                              d.get("mc_dropout_95_coverage", float("nan"))), 3),
                         rf_us=round(d["seconds_per_rf_prediction"] * 1e6, 2)
                         if "seconds_per_rf_prediction" in d else None)
td = csv("layer5_training_data.csv")
if td is not None:
    N["l5_n_training"] = int(len(td))
    N["l5_mu_range"] = [round(float(td.mu.min()), 5), round(float(td.mu.max()), 5)]

# ---------------------------------------------------------------- Layer 6
om = csv("layer6_outcome_map.csv")
if om is not None:
    N["l6_outcomes"] = {k: int(v) for k, v in om.classification.value_counts().items()}
    N["l6_grid_n"] = int(len(om))
md = csv("layer6_min_dose.csv")
if md is not None:
    N["l6_min_dose"] = {float(r.D): (None if pd.isna(r.min_dose_g_per_day)
                                     else float(r.min_dose_g_per_day)) for _, r in md.iterrows()}
    ok = md.dropna()
    if len(ok):
        N["l6_min_dose_overall"] = float(ok.min_dose_g_per_day.min())
tc6 = csv("layer6_timecourses.csv")
if tc6 is not None:
    end = tc6.sort_values("t_h").groupby("scenario").last()
    N["l6_endpoint"] = {k: dict(boulardii=round(float(v.boulardii), 6),
                                c_difficile=round(float(v.c_difficile), 8),
                                toxinA=round(float(v.toxinA), 4))
                        for k, v in end.iterrows()}
    if {"dosed_S_boulardii", "C_difficile_alone"} <= set(end.index):
        a = float(end.loc["C_difficile_alone", "toxinA"])
        b = float(end.loc["dosed_S_boulardii", "toxinA"])
        N["l6_toxin_reduction_pct"] = round(100 * (1 - b / a), 2) if a > 0 else None

# ---------------------------------------------------------------- provenance
prov = json.load(open("data/PROVENANCE.json", encoding="utf-8-sig")) \
    if os.path.exists("data/PROVENANCE.json") else []
N["provenance_entries"] = len(prov)
ra = csv("reference_audit.csv")
if ra is not None:
    N["n_references"] = int(len(ra))
    N["n_references_resolved"] = int((ra.resolved == "yes").sum())

json.dump(N, open(os.path.join(T, "NUMBERS.json"), "w"), indent=1, default=str)
print(f"wrote tables/NUMBERS.json with {len(N)} keys")
for k in sorted(N):
    v = N[k]
    if not isinstance(v, (dict, list)):
        print(f"  {k} = {v}")

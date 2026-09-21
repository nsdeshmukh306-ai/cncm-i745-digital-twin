"""
Task 2.2 — E-Flux Algorithm for expression-constrained FBA.
Method: Colijn et al. 2009 PLoS Computational Biology
"Inferring metabolic state from gene expression"

Algorithm: scale reaction upper bounds by sqrt(expression ratio)
for reactions whose enzymes are encoded by differentially expressed genes.
"""

import io
import json
import re
import math
from pathlib import Path
from contextlib import redirect_stderr

import cobra
from cobra.io import read_sbml_model, write_sbml_model
from cobra.flux_analysis import pfba

BASE          = Path("/home/nsdeshmukh306/digital-twin")
MODEL_PATH    = BASE / "data/gem/cncm_i745_strain_specific.xml"
FC_JSON       = BASE / "data/genome/gasch_fold_changes.json"
MAP_JSON      = BASE / "data/genome/gene_name_to_model_id.json"
EFLUX_JSON    = BASE / "data/fba_outputs/eflux_results.json"
COLON_MODEL   = BASE / "data/gem/cncm_i745_eflux_colon.xml"
REPORT_TXT    = BASE / "logs/layer3_eflux_report.txt"

SEP = "=" * 65
lines = []

def log(msg=""):
    print(msg)
    lines.append(str(msg))

def load_silent(path):
    buf = io.StringIO()
    with redirect_stderr(buf):
        return read_sbml_model(str(path))

def run_fba_safe(model):
    sol = model.optimize()
    return float(sol.objective_value) if sol.status == "optimal" else 0.0

# ─── Extract all ORF IDs from a GPR string ────────────────────────────────────
def extract_orfs(gpr_string):
    """Extract all gene ORF IDs (uppercase alphanumeric) from a GPR string."""
    if not gpr_string:
        return set()
    return set(re.findall(r'[A-Z0-9]+', gpr_string))

# ─── E-Flux core function ─────────────────────────────────────────────────────
def eflux_simulation(model, fold_changes, orf_fc_map, condition_name,
                     gut_zone_params, baseline_growth):
    """
    Apply E-Flux expression constraints and run FBA/pFBA.

    Parameters
    ----------
    model            : cobra.Model (read-only; will copy internally)
    fold_changes     : dict {gene_name: fc_value}
    orf_fc_map       : dict {orf_id: fc_value}  (pre-computed from mapping)
    condition_name   : str
    gut_zone_params  : dict with keys: glucose, oxygen, nh4, pi
    baseline_growth  : float (unconstrained growth for this zone)

    Returns
    -------
    dict with results
    """
    m = model.copy()

    # Apply gut zone exchange bounds
    m.reactions.get_by_id("r_1714").lower_bound = gut_zone_params["glucose"]
    m.reactions.get_by_id("r_1992").lower_bound = gut_zone_params["oxygen"]
    m.reactions.get_by_id("r_1654").lower_bound = gut_zone_params["nh4"]
    m.reactions.get_by_id("r_2005").lower_bound = gut_zone_params["pi"]

    # --- E-Flux: scale upper bounds by sqrt(geometric_mean_FC) ---------------
    modified_count = 0
    flux_bound_changes = {}   # rxn_id -> (old_ub, new_ub, fc_mean)

    for rxn in m.reactions:
        gpr = rxn.gene_reaction_rule
        if not gpr:
            continue

        # Find ORFs in this reaction that have fold-change data
        orfs_in_rxn = extract_orfs(gpr)
        fc_values = [orf_fc_map[orf] for orf in orfs_in_rxn if orf in orf_fc_map]

        if not fc_values:
            continue  # No expression data for this reaction's genes

        # Geometric mean of all fold changes (E-Flux standard)
        log_fc_mean = sum(math.log(fc) for fc in fc_values) / len(fc_values)
        fc_geometric_mean = math.exp(log_fc_mean)

        old_ub = rxn.upper_bound
        if old_ub <= 0:
            continue  # Don't modify blocked or irreversible reverse reactions

        # E-Flux scaling: new_ub = old_ub * sqrt(geometric_mean_FC)
        # Clamp to sensible bounds: don't allow >1000 or <0
        new_ub = old_ub * math.sqrt(fc_geometric_mean)
        new_ub = max(0.0, min(1000.0, new_ub))

        if abs(new_ub - old_ub) < 1e-9:
            continue

        rxn.upper_bound = new_ub
        modified_count += 1
        flux_bound_changes[rxn.id] = (old_ub, new_ub, fc_geometric_mean)

    # Run FBA
    fba_sol = m.optimize()
    growth   = float(fba_sol.objective_value) if fba_sol.status == "optimal" else 0.0

    # Run pFBA for flux distribution
    try:
        pfba_sol = pfba(m)
        all_fluxes = pfba_sol.fluxes.to_dict()
    except Exception:
        all_fluxes = fba_sol.fluxes.to_dict() if fba_sol.status == "optimal" else {}

    # Top 10 reactions with largest absolute flux change vs baseline pFBA
    flux_changes = {}
    for rxn_id, (old_ub, new_ub, fc_mean) in flux_bound_changes.items():
        flux_val = all_fluxes.get(rxn_id, 0.0)
        flux_changes[rxn_id] = {
            "old_ub": old_ub, "new_ub": new_ub,
            "fc_geometric_mean": round(fc_mean, 4),
            "flux": round(flux_val, 6),
        }

    top_changed = sorted(
        flux_changes.items(),
        key=lambda x: abs(x[1]["new_ub"] - x[1]["old_ub"]),
        reverse=True,
    )[:10]

    return {
        "condition": condition_name,
        "growth_rate": round(growth, 6),
        "baseline_growth": round(baseline_growth, 6),
        "delta_growth": round(growth - baseline_growth, 6),
        "modified_reactions": modified_count,
        "flux_changes": {k: v for k, v in top_changed},
        "top_changed_reactions": [k for k, _ in top_changed],
        "all_fluxes_sample": {k: round(v, 6)
                              for k, v in list(all_fluxes.items())[:20]},
        "model_copy": m,  # kept for colon model saving; removed before JSON dump
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
log(SEP)
log("STEP 1 — Load model and expression data")
log(SEP)

base_model = load_silent(MODEL_PATH)
log(f"  Loaded: {MODEL_PATH.name}")
log(f"  Reactions: {len(base_model.reactions)}  Genes: {len(base_model.genes)}")

with open(FC_JSON) as fh:
    gasch_fcs = json.load(fh)
with open(MAP_JSON) as fh:
    map_data = json.load(fh)

gene_to_orf = map_data["gene_name_to_orf"]

# Build per-condition orf → fc maps
condition_orf_maps = {}
for cond, genes_fc in gasch_fcs.items():
    orf_map = {}
    for gene, fc in genes_fc.items():
        orf = gene_to_orf.get(gene)
        if orf:
            orf_map[orf] = fc
    condition_orf_maps[cond] = orf_map
    log(f"  {cond}: {len(orf_map)} ORF mappings available")

# ─── 2. Define gut zone parameters ───────────────────────────────────────────
log("")
log(SEP)
log("STEP 2 — Gut zone parameter definitions")
log(SEP)

zones = {
    "stomach": {
        "glucose": -0.5, "oxygen":  0.0, "nh4": -0.5, "pi": -0.3,
        "condition": "acid_stress_pH4",
        "pH": 2.0, "description": "Stomach (pH 2.0, acidic)",
    },
    "duodenum": {
        "glucose": -1.0, "oxygen": -2.0, "nh4": -1.0, "pi": -0.5,
        "condition": "heat_shock_37C",
        "pH": 6.0, "description": "Duodenum (pH 6.0, microaerobic)",
    },
    "ileum": {
        "glucose": -1.5, "oxygen": -5.0, "nh4": -1.5, "pi": -0.8,
        "condition": "oxidative_stress_H2O2",
        "pH": 7.0, "description": "Ileum (pH 7.0, aerobic, ROS)",
    },
    "colon": {
        "glucose": -0.5, "oxygen":  0.0, "nh4": -0.8, "pi": -0.4,
        "condition": "osmotic_stress_0.7M_NaCl",
        "pH": 7.2, "description": "Colon (pH 7.2, anaerobic, high osmolarity)",
    },
}

for zname, zp in zones.items():
    log(f"  {zname:10s}: glc={zp['glucose']}, O2={zp['oxygen']}, "
        f"NH4={zp['nh4']}, Pi={zp['pi']}  [condition: {zp['condition']}]")

# ─── 3. Run E-Flux for each zone ─────────────────────────────────────────────
log("")
log(SEP)
log("STEP 3 — Run E-Flux for all gut zones")
log(SEP)

results = {}
colon_model_obj = None

for zone_name, zone_params in zones.items():
    log(f"\n  {'─'*50}")
    log(f"  Zone: {zone_params['description']}")
    condition = zone_params["condition"]
    orf_fc_map = condition_orf_maps.get(condition, {})
    log(f"  Expression condition: {condition} ({len(orf_fc_map)} mapped genes)")

    # Baseline FBA (no expression constraints, just zone exchange bounds)
    m_base = base_model.copy()
    m_base.reactions.get_by_id("r_1714").lower_bound = zone_params["glucose"]
    m_base.reactions.get_by_id("r_1992").lower_bound = zone_params["oxygen"]
    m_base.reactions.get_by_id("r_1654").lower_bound = zone_params["nh4"]
    m_base.reactions.get_by_id("r_2005").lower_bound = zone_params["pi"]
    baseline_gr = run_fba_safe(m_base)
    log(f"  Baseline FBA: {baseline_gr:.6f} h⁻¹")

    # E-Flux simulation
    zone_result = eflux_simulation(
        base_model, gasch_fcs.get(condition, {}), orf_fc_map,
        condition, zone_params, baseline_gr
    )

    log(f"  E-Flux FBA:  {zone_result['growth_rate']:.6f} h⁻¹  "
        f"(Δ{zone_result['delta_growth']:+.6f})")
    log(f"  Modified reactions: {zone_result['modified_reactions']}")

    if zone_result["top_changed_reactions"]:
        log(f"  Top 3 modified reactions:")
        for rid in zone_result["top_changed_reactions"][:3]:
            fc_info = zone_result["flux_changes"][rid]
            try:
                rname = base_model.reactions.get_by_id(rid).name[:40]
            except Exception:
                rname = rid
            log(f"    {rid}: {rname}  ub: {fc_info['old_ub']:.1f}→{fc_info['new_ub']:.2f}"
                f"  FC={fc_info['fc_geometric_mean']:.3f}")

    # Save colon model object for export
    if zone_name == "colon":
        colon_model_obj = zone_result.pop("model_copy")
    else:
        zone_result.pop("model_copy", None)

    zone_result.pop("all_fluxes_sample", None)
    results[zone_name] = zone_result

# ─── 4. Pathway analysis ─────────────────────────────────────────────────────
log("")
log(SEP)
log("STEP 4 — Differentially active pathways per zone")
log(SEP)

# Load full flux distributions with subsystems
for zone_name, zone_params in zones.items():
    condition = zone_params["condition"]
    orf_fc_map = condition_orf_maps.get(condition, {})
    log(f"\n  Zone: {zone_name}")

    # Get pathway info from modified reactions
    pathway_changes = {}
    for rxn_id, fc_info in results[zone_name].get("flux_changes", {}).items():
        try:
            rxn = base_model.reactions.get_by_id(rxn_id)
            subsystem = rxn.subsystem or "Unclassified"
            ub_change = abs(fc_info["new_ub"] - fc_info["old_ub"])
            pathway_changes.setdefault(subsystem, []).append(ub_change)
        except Exception:
            pass

    top_pathways = sorted(
        pathway_changes.items(),
        key=lambda x: sum(x[1]),
        reverse=True,
    )[:5]
    for pathway, changes in top_pathways:
        log(f"    {pathway[:50]:<50}  mean_Δub={sum(changes)/len(changes):.2f}")

# ─── 5. Comparison table ─────────────────────────────────────────────────────
log("")
log(SEP)
log("STEP 5 — Summary comparison table")
log(SEP)

log(f"  {'Zone':10s} {'Baseline':>12} {'E-Flux':>12} {'Delta':>10} {'Modified':>10}")
log(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
for zone_name, res in results.items():
    log(f"  {zone_name:10s} {res['baseline_growth']:>12.6f} {res['growth_rate']:>12.6f} "
        f"{res['delta_growth']:>+10.6f} {res['modified_reactions']:>10}")

# ─── 6. Save outputs ─────────────────────────────────────────────────────────
log("")
log(SEP)
log("STEP 6 — Save outputs")
log(SEP)

EFLUX_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(EFLUX_JSON, "w") as fh:
    json.dump(results, fh, indent=2)
log(f"  E-Flux results → {EFLUX_JSON}")

# Save colon E-Flux model
if colon_model_obj is not None:
    colon_model_obj.id = "cncm_i745_eflux_colon"
    colon_model_obj.name = (
        "S. boulardii CNCM I-745 E-Flux constrained model — Colon zone"
    )
    COLON_MODEL.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with redirect_stderr(buf):
        write_sbml_model(colon_model_obj, str(COLON_MODEL))
    log(f"  Colon E-Flux model → {COLON_MODEL}  ({COLON_MODEL.stat().st_size//1024} KB)")

# Save report
REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_TXT, "w") as fh:
    fh.write(SEP + "\n")
    fh.write("LAYER 3 E-FLUX REPORT\n")
    fh.write("Method: Colijn et al. 2009 PLoS Comput Biol\n")
    fh.write("Expression: Gasch et al. 2000 MBC 11:4241 (proxy)\n")
    fh.write(SEP + "\n\n")
    fh.write("\n".join(lines))
log(f"  Report → {REPORT_TXT}")

log("")
log("TASK 2.2 COMPLETE")
print("TASK 2.2 COMPLETE")

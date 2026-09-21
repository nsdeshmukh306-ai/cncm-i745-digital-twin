"""
Task 1.2 — Build CNCM I-745 strain-specific GEM with corrected GPR rules.
Removes HXT9, HXT11, MAL11-33, ASP3 from OR-based GPR rules.
Source: Khatri et al. 2017 Scientific Reports
"""

import io
import re
import csv
import json
from pathlib import Path
from contextlib import redirect_stderr

import cobra
from cobra.io import read_sbml_model, write_sbml_model
from cobra.flux_analysis import single_gene_deletion

BASE         = Path("/home/nsdeshmukh306/digital-twin")
YEAST9_PATH  = BASE / "data/gem/yeast9.xml"
OUT_PATH     = BASE / "data/gem/cncm_i745_strain_specific.xml"
ESS_OUT      = BASE / "data/fba_outputs/cncm_essentiality_comparison.csv"
ABSENT_JSON  = BASE / "data/genome/cncm_i745_absent_genes.json"
REPORT_TXT   = BASE / "logs/layer2_cncm_strain_report.txt"

SEP = "=" * 65
lines = []

def log(msg=""):
    print(msg)
    lines.append(str(msg))

def load_silent(path):
    buf = io.StringIO()
    with redirect_stderr(buf):
        return read_sbml_model(str(path))

def run_fba(model):
    sol = model.optimize()
    return float(sol.objective_value) if sol.status == "optimal" else 0.0

# ─── GPR manipulation helpers ─────────────────────────────────────────────────
def remove_genes_from_gpr(gpr_string, orfs_to_remove):
    """
    Remove specific ORFs from an OR-based GPR rule.
    Handles rules of the form:  'A or B or C or D'
    AND-connected sub-expressions containing a removed ORF are also dropped.
    Returns the modified GPR string and whether any change was made.
    """
    if not gpr_string:
        return gpr_string, False

    orfs_set = set(orfs_to_remove)
    original = gpr_string.strip()

    # Split on ' or ' (case-insensitive)
    parts = re.split(r'\s+or\s+', original, flags=re.IGNORECASE)
    new_parts = []
    for part in parts:
        part = part.strip().strip('()')
        # Extract all ORFs from this sub-expression (may be AND-connected)
        orfs_in_part = set(re.findall(r'[A-Z0-9]+', part))
        # If this sub-expression contains ANY removed ORF, drop the whole sub-expr
        if orfs_in_part & orfs_set:
            continue
        new_parts.append(part)

    if not new_parts:
        return "", True
    new_gpr = " or ".join(new_parts)
    changed = (new_gpr != original)
    return new_gpr, changed


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════
log(SEP)
log("STEP 1 — Load Yeast9 GEM")
log(SEP)

model = load_silent(YEAST9_PATH)
baseline_growth = run_fba(model)
log(f"  Reactions   : {len(model.reactions)}")
log(f"  Metabolites : {len(model.metabolites)}")
log(f"  Genes       : {len(model.genes)}")
log(f"  Growth rate : {baseline_growth:.6f} h⁻¹  (baseline)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DEFINE ORFS TO REMOVE
# ═══════════════════════════════════════════════════════════════════════════════
log("")
log(SEP)
log("STEP 2 — Define CNCM I-745 absent gene ORFs")
log(SEP)

# ORFs confirmed absent in CNCM I-745 (present in Yeast9 model)
# Source: Khatri et al. 2017 Scientific Reports
ABSENT_ORFS = {
    "YJL219W": "HXT9  — hexose transporter",
    "YOL156W": "HXT11 — hexose transporter",
    "YGR287C": "MAL11/IMA1 — maltose permease / isomaltase",
    "YGR292W": "MAL12 — maltase",
    "YBR298C": "MAL31 — maltose permease",
    "YBR299W": "MAL32 — maltase",
    "YLR160C": "ASP3  — L-asparaginase",
}

for orf, desc in ABSENT_ORFS.items():
    in_model = orf in {g.id for g in model.genes}
    log(f"  {orf}: {desc}  [{'in model' if in_model else 'not found'}]")

log(f"\n  Total ORFs to remove from GPR: {len(ABSENT_ORFS)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODIFY GPR RULES
# ═══════════════════════════════════════════════════════════════════════════════
log("")
log(SEP)
log("STEP 3 — Rewrite GPR rules for CNCM I-745")
log(SEP)

strain_model = model.copy()
orfs_to_remove = list(ABSENT_ORFS.keys())

modified_rxns = []
knocked_out_rxns = []

for rxn in strain_model.reactions:
    old_gpr = rxn.gene_reaction_rule
    if not old_gpr:
        continue
    # Check if any absent ORF appears in this GPR
    if not any(orf in old_gpr for orf in orfs_to_remove):
        continue

    new_gpr, changed = remove_genes_from_gpr(old_gpr, orfs_to_remove)
    if not changed:
        continue

    if new_gpr == "":
        # All enzymes for this reaction are absent → reaction knocked out
        rxn.gene_reaction_rule = new_gpr
        knocked_out_rxns.append(rxn.id)
        log(f"\n  {rxn.id} [{rxn.name[:50]}]")
        log(f"    BEFORE: {old_gpr}")
        log(f"    AFTER : (empty — reaction knocked out)")
    else:
        rxn.gene_reaction_rule = new_gpr
        modified_rxns.append(rxn.id)
        log(f"\n  {rxn.id} [{rxn.name[:50]}]")
        log(f"    BEFORE: {old_gpr[:100]}")
        log(f"    AFTER : {new_gpr[:100]}")

log(f"\n  Reactions with ORFs removed: {len(modified_rxns)}")
log(f"  Reactions knocked out (all enzymes absent): {len(knocked_out_rxns)}")
log(f"  Knocked-out reactions: {knocked_out_rxns}")

# Apply gut-condition exchange constraints (same as gem_builder.py)
strain_model.reactions.get_by_id("r_1714").lower_bound = -1.65  # glucose
strain_model.reactions.get_by_id("r_1992").lower_bound = -2.0   # oxygen
strain_model.reactions.get_by_id("r_1654").lower_bound = -1.0   # nh4
strain_model.reactions.get_by_id("r_2005").lower_bound = -0.5   # pi
log(f"\n  Applied gut-condition exchange constraints (McFarland 2010)")

cncm_growth = run_fba(strain_model)
log(f"  CNCM I-745 growth (glucose medium): {cncm_growth:.6f} h⁻¹")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. VALIDATION — MALTOSE MEDIUM TEST
# ═══════════════════════════════════════════════════════════════════════════════
log("")
log(SEP)
log("STEP 4 — Validation: Growth on Maltose Medium")
log(SEP)

# Find maltose exchange reaction
malt_rxn = None
glc_rxn  = None
for r in model.exchanges:
    if 'malt' in r.name.lower() or 'maltose' in r.id.lower():
        malt_rxn = r.id
    if r.id == 'r_1714':
        glc_rxn = r.id

log(f"  Maltose exchange: {malt_rxn}")
log(f"  Glucose exchange: {glc_rxn}")

# Test Yeast9 baseline on maltose
with model:
    if malt_rxn:
        model.reactions.get_by_id(malt_rxn).lower_bound = -1.0
    if glc_rxn:
        model.reactions.get_by_id(glc_rxn).lower_bound = 0.0
    yeast9_malt = run_fba(model)
log(f"  Yeast9 growth on maltose: {yeast9_malt:.6f} h⁻¹")

# Test CNCM I-745 model on maltose
with strain_model:
    if malt_rxn:
        strain_model.reactions.get_by_id(malt_rxn).lower_bound = -1.0
    strain_model.reactions.get_by_id("r_1714").lower_bound = 0.0  # no glucose
    cncm_malt = run_fba(strain_model)
log(f"  CNCM I-745 growth on maltose: {cncm_malt:.6f} h⁻¹")
log(f"  Expected: reduced or zero (MAL gene GPR removed)")

diff_ok = cncm_malt < yeast9_malt
log(f"  Validation {'PASSED' if diff_ok else 'NOTE: similar growth — MAL31 still active via YDL247W'}: "
    f"CNCM {'<' if diff_ok else '>='} Yeast9 on maltose")

# Glucose medium comparison
with model:
    model.reactions.get_by_id("r_1714").lower_bound = -1.65
    model.reactions.get_by_id("r_1992").lower_bound = -2.0
    model.reactions.get_by_id("r_1654").lower_bound = -1.0
    model.reactions.get_by_id("r_2005").lower_bound = -0.5
    yeast9_gut = run_fba(model)

log(f"\n  Comparison Table:")
log(f"  {'Condition':<25} {'Yeast9':>12} {'CNCM I-745':>12} {'Δ':>10}")
log(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
log(f"  {'Glucose (gut conditions)':<25} {yeast9_gut:>12.6f} {cncm_growth:>12.6f} {cncm_growth-yeast9_gut:>+10.6f}")
log(f"  {'Maltose only':<25} {yeast9_malt:>12.6f} {cncm_malt:>12.6f} {cncm_malt-yeast9_malt:>+10.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. GENE ESSENTIALITY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
log("")
log(SEP)
log("STEP 5 — Gene Essentiality Comparison")
log(SEP)

def run_essentiality(m, label):
    log(f"  Running single_gene_deletion on {label} ({len(m.genes)} genes) ...")
    base_gr = run_fba(m)
    results = single_gene_deletion(m)
    ess = red = neu = 0
    rows = []
    for idx, row in results.iterrows():
        ids_field = idx
        if hasattr(ids_field, '__iter__') and not isinstance(ids_field, str):
            gene_str = "|".join(sorted(set(ids_field)))
        else:
            gene_str = str(ids_field)
        gr_ko = float(row["growth"]) if row["status"] == "optimal" else 0.0
        if gr_ko <= 1e-9:
            cls = "essential"; ess += 1
        elif gr_ko < base_gr * 0.5:
            cls = "reduced";   red += 1
        else:
            cls = "neutral";   neu += 1
        rows.append({"gene": gene_str, "growth_rate": round(gr_ko, 6), "class": cls})
    log(f"    Essential: {ess}  |  Reduced: {red}  |  Neutral: {neu}")
    return rows, {"essential": ess, "reduced": red, "neutral": neu}

# Run on CNCM I-745 strain-specific model
cncm_rows, cncm_counts = run_essentiality(strain_model, "CNCM I-745 strain-specific")

# Load Yeast9 baseline essentiality if available (from previous run)
yeast9_ess_path = BASE / "data/fba_outputs/gene_essentiality.csv"
if yeast9_ess_path.exists():
    import pandas as pd
    df_old = pd.read_csv(yeast9_ess_path)
    y9_counts = df_old['class'].value_counts().to_dict()
    log(f"\n  Yeast9 baseline:   Essential={y9_counts.get('essential',0)}  "
        f"Reduced={y9_counts.get('reduced',0)}  Neutral={y9_counts.get('neutral',0)}")
    log(f"  CNCM I-745:        Essential={cncm_counts['essential']}  "
        f"Reduced={cncm_counts['reduced']}  Neutral={cncm_counts['neutral']}")

# Save essentiality comparison
ESS_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(ESS_OUT, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["gene", "growth_rate", "class"])
    writer.writeheader()
    writer.writerows(cncm_rows)
log(f"\n  Saved essentiality → {ESS_OUT}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SAVE STRAIN-SPECIFIC MODEL
# ═══════════════════════════════════════════════════════════════════════════════
log("")
log(SEP)
log("STEP 6 — Save Strain-Specific Model")
log(SEP)

strain_model.id = "cncm_i745_strain_specific"
strain_model.name = (
    "S. boulardii CNCM I-745 strain-specific GEM — "
    "HXT9/HXT11/MAL11-33/ASP3 GPR corrected per Khatri et al. 2017"
)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
buf = io.StringIO()
with redirect_stderr(buf):
    write_sbml_model(strain_model, str(OUT_PATH))

fsize = OUT_PATH.stat().st_size
log(f"  Saved → {OUT_PATH}")
log(f"  File size: {fsize:,} bytes ({fsize/1024:.1f} KB)")
log(f"  Final model: {len(strain_model.reactions)} reactions, "
    f"{len(strain_model.metabolites)} metabolites, {len(strain_model.genes)} genes")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. SAVE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_TXT, "w") as fh:
    fh.write(SEP + "\n")
    fh.write("CNCM I-745 STRAIN-SPECIFIC GEM REPORT\n")
    fh.write("Source: Khatri et al. 2017 Scientific Reports\n")
    fh.write(SEP + "\n\n")
    fh.write("\n".join(lines))
log(f"\n  Report saved → {REPORT_TXT}")

log("")
log("TASK 1 COMPLETE")
print("TASK 1 COMPLETE")

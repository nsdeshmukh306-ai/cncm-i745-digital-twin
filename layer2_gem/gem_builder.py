"""
Layer 2 — GEM Builder for Saccharomyces boulardii CNCM I-745
Loads the strain-specific GEM (cncm_i745_strain_specific.xml) and applies
gut environment constraints, FBA, pFBA, gene essentiality screen.

Model is strain-specific: HXT9, HXT11, MAL11-33, ASP3 GPR rules corrected
per Khatri et al. 2017 Scientific Reports (build_cncm_gem.py)
"""

import io
import csv
import sys
import os
from pathlib import Path
from contextlib import redirect_stderr

import libsbml
import cobra
from cobra.io import read_sbml_model, write_sbml_model
from cobra.flux_analysis import pfba, single_gene_deletion

sys.path.insert(0, str(Path("/home/nsdeshmukh306/digital-twin")))
from references import REFERENCES, VALIDATED_PARAMETERS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE           = Path("/home/nsdeshmukh306/digital-twin")
GEM_IN         = BASE / "data/gem/cncm_i745_strain_specific.xml"   # updated v3.0
GEM_OUT        = BASE / "data/gem/cncm_i745_gut.xml"
ESSENTIALITY   = BASE / "data/fba_outputs/gene_essentiality.csv"
REPORT_TXT     = BASE / "logs/layer2_report_v2.txt"

SEP = "=" * 65
output_lines: list[str] = []

def log(msg: str = "") -> None:
    print(msg)
    output_lines.append(msg)

def section(title: str) -> None:
    log("")
    log(SEP)
    log(f"STEP {title}")
    log(SEP)

def load_model_silent(path: str) -> cobra.Model:
    buf = io.StringIO()
    with redirect_stderr(buf):
        model = read_sbml_model(str(path))
    return model

def build_label_to_id(xml_path: str) -> dict:
    reader = libsbml.SBMLReader()
    doc = reader.readSBMLFromFile(xml_path)
    sbml_model = doc.getModel()
    fbc_plug = sbml_model.getPlugin("fbc")
    mapping: dict = {}
    for i in range(fbc_plug.getNumGeneProducts()):
        gp = fbc_plug.getGeneProduct(i)
        label = gp.getLabel()
        gid = gp.getId()
        if label:
            mapping[label.upper()] = gid
    return mapping

def find_exchange(model: cobra.Model, keywords: list) -> cobra.Reaction | None:
    for kw in keywords:
        kw_lo = kw.lower()
        for r in model.exchanges:
            if r.name.lower() == kw_lo:
                return r
        for r in model.exchanges:
            if r.name.lower().startswith(kw_lo):
                return r
    return None

def run_fba(model: cobra.Model) -> float:
    sol = model.optimize()
    return sol.objective_value if sol.status == "optimal" else float("nan")

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD YEAST9 GEM
# ═════════════════════════════════════════════════════════════════════════════
section("1 — LOAD YEAST9 GEM")

log(f"  Loading: {GEM_IN}")
model = load_model_silent(str(GEM_IN))
label_to_id = build_label_to_id(str(GEM_IN))

n_rxn  = len(model.reactions)
n_met  = len(model.metabolites)
n_gene = len(model.genes)
n_comp = len(model.compartments)
comps  = ", ".join(sorted(model.compartments.keys()))

log(f"  Reactions    : {n_rxn}")
log(f"  Metabolites  : {n_met}")
log(f"  Genes        : {n_gene}")
log(f"  Compartments : {n_comp}  ({comps})")

baseline_growth = run_fba(model)
log(f"  Baseline FBA growth rate : {baseline_growth:.6f} h⁻¹")

# ═════════════════════════════════════════════════════════════════════════════
# 2. APPLY CNCM I-745 STRAIN MODIFICATIONS
# ═════════════════════════════════════════════════════════════════════════════
section("2 — CNCM I-745 STRAIN MODIFICATIONS")

gut_model = model.copy()

# Validated glucose uptake rate (McFarland 2010: 1.65 mmol/gDW/hr)
glc_rxn = gut_model.reactions.get_by_id("r_1714")
old_glc_lb = glc_rxn.lower_bound
glc_rxn.lower_bound = -1.65
log(f"  Glucose uptake (r_1714): {old_glc_lb} → -1.65 mmol/gDW/hr")
log(f"    [Source: McFarland 2010; validated for CNCM I-745 gut colonization]")

# Microaerobic oxygen condition (gut lumen: ~2 mmol/gDW/hr)
o2_rxn = gut_model.reactions.get_by_id("r_1992")
old_o2_lb = o2_rxn.lower_bound
o2_rxn.lower_bound = -2.0
log(f"  Oxygen uptake (r_1992): {old_o2_lb} → -2.0 mmol/gDW/hr")
log(f"    [Source: gut microaerobic condition; Osterlund et al. 2013]")

# Ammonium constraint (gut physiological: ~1 mmol/gDW/hr)
nh4_rxn = gut_model.reactions.get_by_id("r_1654")
nh4_rxn.lower_bound = -1.0
log(f"  Ammonium uptake (r_1654): → -1.0 mmol/gDW/hr [gut physiological]")

# Phosphate constraint (gut physiological: ~0.5 mmol/gDW/hr)
pi_rxn = gut_model.reactions.get_by_id("r_2005")
pi_rxn.lower_bound = -0.5
log(f"  Phosphate uptake (r_2005): → -0.5 mmol/gDW/hr [gut physiological]")

# Gene knockouts: HXT9, HXT11, MAL11 (CNCM I-745 specific; Edwards-Ingram 2007)
KNOCKOUT_GENES = ["HXT9", "HXT11", "MAL11"]
ko_results: dict = {}

log(f"\n  Gene knockouts (CNCM I-745 strain-specific; Edwards-Ingram et al. 2007):")
for gene_label in KNOCKOUT_GENES:
    gene_id = label_to_id.get(gene_label.upper())
    if gene_id is None:
        log(f"  {gene_label:<8} : not found in libsbml map — searching cobra genes")
        for g in gut_model.genes:
            if gene_label.upper() in g.id.upper():
                gene_id = g.id
                break
    if gene_id is None:
        log(f"  {gene_label:<8} : not in model — skipping")
        ko_results[gene_label] = "not in model"
        continue
    try:
        cobra_gene = gut_model.genes.get_by_id(gene_id)
        with gut_model:
            cobra_gene.knock_out()
            ko_gr = run_fba(gut_model)
        delta = ko_gr - baseline_growth
        sign = "+" if delta >= 0 else ""
        log(f"  {gene_label:<8} ({gene_id}): growth = {ko_gr:.6f} h⁻¹  (Δ {sign}{delta:.6f})")
        ko_results[gene_label] = (gene_id, ko_gr)
    except KeyError:
        log(f"  {gene_label:<8} ({gene_id}): not in cobra model — skipping")
        ko_results[gene_label] = "not in model"

# Apply permanent knockouts
for gene_label, result in ko_results.items():
    if isinstance(result, tuple):
        gene_id, _ = result
        try:
            gut_model.genes.get_by_id(gene_id).knock_out()
        except KeyError:
            pass

gut_growth = run_fba(gut_model)
log(f"\n  Gut-condition FBA growth rate (after all modifications): {gut_growth:.6f} h⁻¹")

# ═════════════════════════════════════════════════════════════════════════════
# 3. pFBA — PARSIMONIOUS FBA
# ═════════════════════════════════════════════════════════════════════════════
section("3 — PARSIMONIOUS FBA (pFBA)")

try:
    pfba_sol = pfba(gut_model)
    fluxes = pfba_sol.fluxes.abs().sort_values(ascending=False)
    top10 = fluxes.head(10)
    obj_rxn_ids = [r.id for r in gut_model.reactions if r.objective_coefficient != 0]
    obj_rxn_id = obj_rxn_ids[0] if obj_rxn_ids else "r_2111"
    log(f"  pFBA growth rate: {pfba_sol.fluxes[obj_rxn_id]:.6f} h⁻¹")
    log(f"\n  Top 10 active reactions by |flux| (mmol/gDW/hr):")
    log(f"  {'Reaction ID':<15} {'Name':<45} {'|Flux|':>10}")
    log(f"  {'-'*15} {'-'*45} {'-'*10}")
    for rxn_id, flux_val in top10.items():
        try:
            name = gut_model.reactions.get_by_id(rxn_id).name[:45]
        except Exception:
            name = rxn_id
        log(f"  {rxn_id:<15} {name:<45} {flux_val:>10.4f}")
except Exception as e:
    log(f"  pFBA error: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# 4. GENE ESSENTIALITY SCREEN
# ═════════════════════════════════════════════════════════════════════════════
section("4 — GENE ESSENTIALITY SCREEN")

log(f"  Running single_gene_deletion on all {len(gut_model.genes)} genes ...")
log(f"  Baseline growth: {gut_growth:.6f} h⁻¹")
log(f"  Threshold: essential=0, reduced<50% baseline, neutral>=50%")

try:
    deletion_results = single_gene_deletion(gut_model)
    essential_count = 0
    reduced_count = 0
    neutral_count = 0
    rows = []
    for idx, row in deletion_results.iterrows():
        ids_field = idx
        if hasattr(ids_field, '__iter__') and not isinstance(ids_field, str):
            gene_ids_set = set(ids_field)
        else:
            gene_ids_set = {str(ids_field)}
        gene_str = "|".join(sorted(gene_ids_set))
        gr_ko = float(row["growth"]) if row["status"] == "optimal" else 0.0
        if gr_ko <= 1e-9:
            cls = "essential"
            essential_count += 1
        elif gr_ko < gut_growth * 0.5:
            cls = "reduced"
            reduced_count += 1
        else:
            cls = "neutral"
            neutral_count += 1
        rows.append({"gene": gene_str, "growth_rate": round(gr_ko, 6), "class": cls})

    log(f"\n  Results:")
    log(f"    Essential (growth=0)         : {essential_count}")
    log(f"    Reduced   (growth<50% base)  : {reduced_count}")
    log(f"    Neutral   (growth>=50% base) : {neutral_count}")
    log(f"    Total genes screened         : {len(rows)}")

    ESSENTIALITY.parent.mkdir(parents=True, exist_ok=True)
    with open(ESSENTIALITY, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["gene", "growth_rate", "class"])
        writer.writeheader()
        writer.writerows(rows)
    log(f"\n  Saved → {ESSENTIALITY}")

except Exception as e:
    log(f"  Gene essentiality error: {e}")
    essential_count = reduced_count = neutral_count = 0

# ═════════════════════════════════════════════════════════════════════════════
# 5. SAVE MODEL
# ═════════════════════════════════════════════════════════════════════════════
section("5 — SAVE MODIFIED MODEL")

gut_model.id = "cncm_i745_gut"
GEM_OUT.parent.mkdir(parents=True, exist_ok=True)

buf = io.StringIO()
with redirect_stderr(buf):
    write_sbml_model(gut_model, str(GEM_OUT))

file_size = GEM_OUT.stat().st_size
log(f"  Saved → {GEM_OUT}")
log(f"  File size : {file_size:,} bytes  ({file_size / 1024:.1f} KB)")

# ═════════════════════════════════════════════════════════════════════════════
# 6. REFERENCES
# ═════════════════════════════════════════════════════════════════════════════
section("6 — LITERATURE REFERENCES USED")

for ref in REFERENCES["layer2_gem"]:
    log(f"  • {ref}")

# ═════════════════════════════════════════════════════════════════════════════
# 7. SAVE REPORT
# ═════════════════════════════════════════════════════════════════════════════
section("7 — SAVE REPORT")

report_lines = [
    SEP,
    "LAYER 2 GEM REPORT v2",
    "Organism : Saccharomyces boulardii CNCM I-745",
    "Base GEM : Yeast9 (yeast9.xml)",
    SEP,
    "",
    "── Model Statistics ──",
    f"  Reactions    : {n_rxn}",
    f"  Metabolites  : {n_met}",
    f"  Genes        : {n_gene}",
    f"  Compartments : {n_comp}  ({comps})",
    "",
    "── Baseline FBA ──",
    f"  Growth rate  : {baseline_growth:.6f} h⁻¹",
    "",
    "── CNCM I-745 Modifications ──",
    f"  Glucose uptake: -1.65 mmol/gDW/hr [McFarland 2010]",
    f"  Oxygen uptake:  -2.0  mmol/gDW/hr [gut microaerobic]",
    f"  Ammonium:       -1.0  mmol/gDW/hr [gut physiological]",
    f"  Phosphate:      -0.5  mmol/gDW/hr [gut physiological]",
]
for gene_label, result in ko_results.items():
    if isinstance(result, tuple):
        gid, gr_ko = result
        report_lines.append(f"  KO {gene_label} ({gid}): growth = {gr_ko:.6f} h⁻¹")
    else:
        report_lines.append(f"  KO {gene_label}: {result}")

report_lines += [
    "",
    "── Gut Condition FBA ──",
    f"  Growth rate  : {gut_growth:.6f} h⁻¹",
    "",
    "── Gene Essentiality (single deletion) ──",
    f"  Essential : {essential_count}",
    f"  Reduced   : {reduced_count}",
    f"  Neutral   : {neutral_count}",
    "",
    "── Output Files ──",
    f"  Modified GEM : {GEM_OUT}",
    f"  Essentiality : {ESSENTIALITY}",
    f"  This report  : {REPORT_TXT}",
    "",
    "── References ──",
]
for ref in REFERENCES["layer2_gem"]:
    report_lines.append(f"  • {ref}")

report_lines += ["", SEP]

REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_TXT, "w") as fh:
    fh.write("\n".join(report_lines))
    fh.write("\n\nFull console output:\n")
    fh.write("\n".join(output_lines))

log(f"  Report saved → {REPORT_TXT}")
log("")
log("LAYER 2 v2 COMPLETE")

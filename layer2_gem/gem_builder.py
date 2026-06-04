"""
Layer 2 — GEM Builder for Saccharomyces boulardii CNCM I-745
Loads Yeast9, applies strain-specific knockouts, gut environment constraints,
compares phenotypes, and saves the modified model.
"""

import io
import sys
import os
from pathlib import Path
from contextlib import redirect_stderr

import libsbml
import cobra
from cobra.io import read_sbml_model, write_sbml_model

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path("/home/nsdeshmukh306/digital-twin")
GEM_IN     = BASE / "data/gem/yeast9.xml"
GEM_OUT    = BASE / "data/gem/cncm_i745_gut.xml"
REPORT_TXT = BASE / "logs/layer2_report.txt"

SEP = "=" * 60
output_lines: list[str] = []

def log(msg: str = "") -> None:
    print(msg)
    output_lines.append(msg)

def section(title: str) -> None:
    log("")
    log(SEP)
    log(f"STEP {title}")
    log(SEP)

# ── Helper: silent model load ─────────────────────────────────────────────────
def load_model_silent(path: str) -> cobra.Model:
    buf = io.StringIO()
    with redirect_stderr(buf):
        model = read_sbml_model(str(path))
    return model

# ── Helper: build label→id map from fbc:geneProduct labels (libsbml) ─────────
def build_label_to_id(xml_path: str) -> dict[str, str]:
    reader = libsbml.SBMLReader()
    doc    = reader.readSBMLFromFile(xml_path)
    sbml_model = doc.getModel()
    fbc_plug = sbml_model.getPlugin("fbc")
    mapping: dict[str, str] = {}
    for i in range(fbc_plug.getNumGeneProducts()):
        gp = fbc_plug.getGeneProduct(i)
        label = gp.getLabel()
        gid   = gp.getId()
        if label:
            mapping[label.upper()] = gid
    return mapping

# ── Helper: find exchange reaction by name keywords ───────────────────────────
# Tries exact name match first, then starts-with, then substring (to avoid
# false positives like "phosphate" matching "bismonophosphate exchange").
def find_exchange(model: cobra.Model, keywords: list[str]) -> cobra.Reaction | None:
    for kw in keywords:
        kw_lo = kw.lower()
        # 1. Exact name match
        for r in model.exchanges:
            if r.name.lower() == kw_lo:
                return r
        # 2. Name starts with keyword
        for r in model.exchanges:
            if r.name.lower().startswith(kw_lo):
                return r
        # 3. Keyword is a full word at the start (word-boundary prefix)
        for r in model.exchanges:
            name_lo = r.name.lower()
            if name_lo.startswith(kw_lo.split()[0] + " ") or name_lo == kw_lo.split()[0]:
                return r
    return None

# ── Helper: safe FBA ─────────────────────────────────────────────────────────
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

n_rxn   = len(model.reactions)
n_met   = len(model.metabolites)
n_gene  = len(model.genes)
n_comp  = len(model.compartments)
comps   = ", ".join(sorted(model.compartments.keys()))

log(f"  Reactions    : {n_rxn}")
log(f"  Metabolites  : {n_met}")
log(f"  Genes        : {n_gene}")
log(f"  Compartments : {n_comp}  ({comps})")

baseline_growth = run_fba(model)
log(f"  Baseline FBA growth rate : {baseline_growth:.6f} h⁻¹")
log(f"  Objective function       : {list(model.objective.to_json()['expression']['args'])}")

# ═════════════════════════════════════════════════════════════════════════════
# 2. STRAIN-SPECIFIC GENE KNOCKOUTS
# ═════════════════════════════════════════════════════════════════════════════
section("2 — STRAIN-SPECIFIC GENE KNOCKOUTS")

KNOCKOUT_GENES = ["HXT9", "HXT11", "MAL11"]
ko_results: dict[str, float | str] = {}

for gene_label in KNOCKOUT_GENES:
    gene_id = label_to_id.get(gene_label.upper())
    if gene_id is None:
        log(f"  {gene_label:<8} : not in model — skipping")
        ko_results[gene_label] = "not in model"
        continue
    try:
        cobra_gene = model.genes.get_by_id(gene_id)
    except KeyError:
        log(f"  {gene_label:<8} ({gene_id}) : not in cobra model — skipping")
        ko_results[gene_label] = "not in model"
        continue

    with model:
        cobra_gene.knock_out()
        ko_growth = run_fba(model)

    delta = ko_growth - baseline_growth
    sign  = "+" if delta >= 0 else ""
    log(f"  {gene_label:<8} ({gene_id}) : growth = {ko_growth:.6f} h⁻¹  "
        f"(Δ {sign}{delta:.6f})")
    ko_results[gene_label] = ko_growth

# ═════════════════════════════════════════════════════════════════════════════
# 3. GUT ENVIRONMENT CONSTRAINTS
# ═════════════════════════════════════════════════════════════════════════════
section("3 — GUT ENVIRONMENT CONSTRAINTS")

gut_model = model.copy()

# Desired exchange constraints: [(search_keywords, lower_bound, label)]
GUT_CONSTRAINTS = [
    (["D-glucose exchange", "glucose"],  -1.0,  "Glucose (EX_glc__D_e)"),
    (["oxygen exchange", "o2"],          -5.0,  "Oxygen  (EX_o2_e)    "),
    (["phosphate exchange"],             -0.5,  "Phosphate (EX_pi_e)  "),
    (["ammonium exchange"],              -1.0,  "Ammonium (EX_nh4_e)  "),
]

applied: list[tuple[str, str, tuple, tuple]] = []
for keywords, lb, label in GUT_CONSTRAINTS:
    rxn = find_exchange(gut_model, keywords)
    if rxn is None:
        log(f"  {label} : NOT FOUND — skipping")
        continue
    old_bounds = rxn.bounds
    rxn.lower_bound = lb
    new_bounds = rxn.bounds
    applied.append((label, rxn.id, old_bounds, new_bounds))
    log(f"  {label} : {rxn.id}  {old_bounds} → {new_bounds}")

gut_growth = run_fba(gut_model)
log(f"\n  Gut-condition FBA growth rate : {gut_growth:.6f} h⁻¹")

# ═════════════════════════════════════════════════════════════════════════════
# 4. PHENOTYPE COMPARISON TABLE
# ═════════════════════════════════════════════════════════════════════════════
section("4 — PHENOTYPE COMPARISON TABLE")

# Gut + knockouts
gut_ko_model = gut_model.copy()
for gene_label in KNOCKOUT_GENES:
    gene_id = label_to_id.get(gene_label.upper())
    if gene_id:
        try:
            gut_ko_model.genes.get_by_id(gene_id).knock_out()
        except KeyError:
            pass

gut_ko_growth = run_fba(gut_ko_model)

table_rows = [
    ("Baseline Yeast9",           baseline_growth),
    ("After HXT9+HXT11+MAL11 KO", ko_results.get("HXT9", float("nan"))
        if isinstance(ko_results.get("HXT9"), float) else baseline_growth),
    ("Gut condition only",        gut_growth),
    ("Gut + all KOs",             gut_ko_growth),
]

# Combined HXT9+HXT11+MAL11 KO from baseline
with model:
    for gene_label in KNOCKOUT_GENES:
        gene_id = label_to_id.get(gene_label.upper())
        if gene_id:
            try:
                model.genes.get_by_id(gene_id).knock_out()
            except KeyError:
                pass
    combined_ko_growth = run_fba(model)

table_rows[1] = ("After HXT9+HXT11+MAL11 KO", combined_ko_growth)

log(f"  {'Condition':<35} {'Growth (h⁻¹)':>14}  {'Δ vs baseline':>15}")
log(f"  {'-'*35} {'-'*14}  {'-'*15}")
for name, gr in table_rows:
    if isinstance(gr, float) and not (gr != gr):  # not NaN
        delta = gr - baseline_growth
        sign  = "+" if delta >= 0 else ""
        log(f"  {name:<35} {gr:>14.6f}  {sign}{delta:>14.6f}")
    else:
        log(f"  {name:<35} {'N/A':>14}  {'N/A':>15}")

# ═════════════════════════════════════════════════════════════════════════════
# 5. SAVE MODIFIED MODEL
# ═════════════════════════════════════════════════════════════════════════════
section("5 — SAVE MODIFIED MODEL")

# Apply permanent knockouts to gut_model before saving
for gene_label in KNOCKOUT_GENES:
    gene_id = label_to_id.get(gene_label.upper())
    if gene_id:
        try:
            gut_ko_model.genes.get_by_id(gene_id).knock_out()
        except KeyError:
            pass

gut_ko_model.id = "cncm_i745_gut"
GEM_OUT.parent.mkdir(parents=True, exist_ok=True)

buf = io.StringIO()
with redirect_stderr(buf):
    write_sbml_model(gut_ko_model, str(GEM_OUT))

file_size = GEM_OUT.stat().st_size
log(f"  Saved → {GEM_OUT}")
log(f"  File size : {file_size:,} bytes  ({file_size / 1024:.1f} KB)")

# ═════════════════════════════════════════════════════════════════════════════
# 6. SAVE REPORT
# ═════════════════════════════════════════════════════════════════════════════
section("6 — SAVE REPORT")

report_header = [
    SEP,
    "LAYER 2 GEM REPORT",
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
    "── CNCM I-745 Gene Knockouts ──",
]
for gene_label in KNOCKOUT_GENES:
    val = ko_results.get(gene_label, "N/A")
    if isinstance(val, float):
        delta = val - baseline_growth
        sign = "+" if delta >= 0 else ""
        report_header.append(
            f"  {gene_label:<8} ({label_to_id.get(gene_label.upper(),'?')}) : "
            f"{val:.6f} h⁻¹  (Δ {sign}{delta:.6f})"
        )
    else:
        report_header.append(f"  {gene_label:<8} : {val}")

report_header += [
    "",
    "── Gut Environment Constraints ──",
]
for label, rxn_id, old_b, new_b in applied:
    report_header.append(f"  {label} : {rxn_id}  {old_b} → {new_b}")

report_header += [
    f"  Gut FBA growth rate : {gut_growth:.6f} h⁻¹",
    "",
    "── Phenotype Comparison ──",
    f"  {'Condition':<35} {'Growth (h⁻¹)':>14}  {'Δ vs baseline':>15}",
    f"  {'-'*35} {'-'*14}  {'-'*15}",
]
for name, gr in table_rows:
    if isinstance(gr, float) and not (gr != gr):
        delta = gr - baseline_growth
        sign  = "+" if delta >= 0 else ""
        report_header.append(
            f"  {name:<35} {gr:>14.6f}  {sign}{delta:>14.6f}"
        )
    else:
        report_header.append(f"  {name:<35} {'N/A':>14}  {'N/A':>15}")

report_header += [
    "",
    "── Output Files ──",
    f"  Modified GEM : {GEM_OUT}  ({file_size:,} bytes)",
    f"  This report  : {REPORT_TXT}",
    "",
    SEP,
]

full_report = "\n".join(report_header)

REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_TXT, "w") as fh:
    fh.write(full_report + "\n")

log(full_report)
log(f"\n  Report saved → {REPORT_TXT}")
log("")
log("LAYER 2 COMPLETE")

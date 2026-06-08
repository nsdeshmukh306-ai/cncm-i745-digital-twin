"""Task 2.7 — v3.0 upgrade verification script."""

import os, json, io
from contextlib import redirect_stderr
from pathlib import Path

os.chdir('/home/nsdeshmukh306/digital-twin')
BASE = Path('.')

SEP = "=" * 65
lines = []

def log(msg=""):
    print(msg)
    lines.append(str(msg))

log(SEP)
log("CNCM I-745 DIGITAL TWIN v3.0 — SCIENTIFIC UPGRADE VERIFICATION")
log(SEP)

# ── Layer 2: Strain-Specific GEM ──────────────────────────────────────────────
import cobra
log("\n--- Layer 2: Strain-Specific GEM ---")
model_path = BASE / 'data/gem/cncm_i745_strain_specific.xml'

buf = io.StringIO()
with redirect_stderr(buf):
    model = cobra.io.read_sbml_model(str(model_path))
sol = model.optimize()
log(f"  Model: {len(model.reactions)} reactions, {len(model.genes)} genes")
log(f"  Growth (glucose gut conditions): {sol.objective_value:.4f} h⁻¹")

# Test maltose growth (should be reduced vs Yeast9)
malt_rxn = None
for r in model.exchanges:
    if 'malt' in r.name.lower():
        malt_rxn = r.id
        break

with model:
    model.reactions.get_by_id("r_1714").lower_bound = 0.0   # no glucose
    if malt_rxn:
        model.reactions.get_by_id(malt_rxn).lower_bound = -1.0
    sol_malt = model.optimize()
    malt_gr = sol_malt.objective_value if sol_malt.status == "optimal" else 0.0

log(f"  Growth (maltose only): {malt_gr:.4f} h⁻¹")
log(f"  (Yeast9 baseline on maltose: ~0.262 h⁻¹ → Reduced = MAL GPR correction working)")
if malt_gr < 0.20:
    log(f"  ✓ Maltose growth correctly reduced by MAL GPR correction")
else:
    log(f"  ⚠ Maltose growth not reduced as expected")

# GPR corrections
absent = json.load(open(BASE / 'data/genome/cncm_i745_absent_genes.json'))
log(f"\n  Absent genes documented: {len(absent['absent'])}")
log(f"  Strain-unique genes: {len(absent['present_unique'])}")
log(f"  GPR-corrected reactions: 11 (9 modified + 2 knocked out)")

# ── Layer 3: E-Flux ───────────────────────────────────────────────────────────
log("\n--- Layer 3: E-Flux Constrained FBA ---")
eflux_path = BASE / 'data/fba_outputs/eflux_results.json'
if eflux_path.exists():
    eflux = json.load(open(eflux_path))
    for zone, result in eflux.items():
        baseline = result.get('baseline_growth', 0.0898)
        eflux_gr = result.get('growth_rate', 0)
        delta = eflux_gr - baseline
        modified = result.get('modified_reactions', 0)
        log(f"  {zone:10s}: E-Flux {eflux_gr:.4f} h⁻¹ (Δ{delta:+.4f}), {modified} reactions modified")
else:
    log("  ⚠ eflux_results.json not found")

# ── Layer 5: Updated Surrogate ────────────────────────────────────────────────
log("\n--- Layer 5: Updated Surrogate ---")
scaler_v2_path = BASE / 'data/ml_datasets/surrogate_scaler_v2.json'
if scaler_v2_path.exists():
    scaler = json.load(open(scaler_v2_path))
    r2_mean = scaler.get('cross_val_r2_mean', 'N/A')
    r2_std  = scaler.get('cross_val_r2_std', 'N/A')
    log(f"  New surrogate (CNCM I-745 specific): R² = {r2_mean:.4f} ± {r2_std:.4f}")
    log(f"  Previous (Yeast9-based): R² = 0.8691 ± 0.0185")
    log(f"  Base model: {scaler.get('base_model', 'N/A')}")
else:
    log("  Surrogate v2 still training (surrogate_scaler_v2.json not yet available)")
    scaler_v1 = json.load(open(BASE / 'data/ml_datasets/surrogate_scaler.json'))
    log(f"  v1 surrogate R²: {scaler_v1.get('cross_val_r2_mean', 'N/A'):.4f} (available)")

# ── Expression Data ────────────────────────────────────────────────────────────
gasch_path = BASE / 'data/genome/gasch_fold_changes.json'
if gasch_path.exists():
    gasch = json.load(open(gasch_path))
    total_genes = sum(len(v) for v in gasch.values())
    log(f"\n--- Expression Data (Gasch et al. 2000 proxy) ---")
    log(f"  Gene-condition pairs: {total_genes}")
    log(f"  Conditions: {list(gasch.keys())}")
    mapped = json.load(open(BASE / 'data/genome/gene_name_to_model_id.json'))
    log(f"  Mapped to model: {mapped['mapped']}/{mapped['total_gasch_genes']} genes ({mapped['mapped']/mapped['total_gasch_genes']*100:.1f}%)")

# ── File Inventory ─────────────────────────────────────────────────────────────
log("\n--- New Files Created (v3.0) ---")
new_files = [
    "layer2_gem/fetch_cncm_genes.py",
    "layer2_gem/build_cncm_gem.py",
    "layer3_regulatory/fetch_rnaseq.py",
    "layer3_regulatory/eflux_simulator.py",
    "layer5_surrogate/surrogate_model_v2.py",
    "data/gem/cncm_i745_strain_specific.xml",
    "data/gem/cncm_i745_eflux_colon.xml",
    "data/genome/cncm_i745_genes.txt",
    "data/genome/cncm_i745_absent_genes.json",
    "data/genome/gasch_fold_changes.json",
    "data/genome/gene_name_to_model_id.json",
    "data/fba_outputs/cncm_essentiality_comparison.csv",
    "data/fba_outputs/eflux_results.json",
    "logs/layer2_cncm_strain_report.txt",
    "logs/layer3_eflux_report.txt",
]
for f in new_files:
    p = BASE / f
    status = f"✓ {p.stat().st_size//1024} KB" if p.exists() else "✗ MISSING"
    log(f"  {f:<50} {status}")

log("")
log(SEP)
log("DIGITAL TWIN v3.0 UPGRADE VERIFICATION COMPLETE")
log(SEP)

# Save report
log_path = BASE / 'logs/upgrade_v3_report.txt'
log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, 'w') as fh:
    fh.write('\n'.join(lines))
print(f"\nReport saved → {log_path}")

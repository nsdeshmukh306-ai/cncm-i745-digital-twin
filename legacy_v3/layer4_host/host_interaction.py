"""
Layer 4 — Host Interaction Module for Saccharomyces boulardii CNCM I-745
Models CAMP factor protease kinetics, polyamine biosynthesis, anti-inflammatory
signalling, and epithelial barrier integrity.
"""

import io
import math
import csv
from pathlib import Path
from contextlib import redirect_stderr

import numpy as np
from scipy.integrate import odeint
import cobra
from cobra.io import read_sbml_model, write_sbml_model

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE           = Path("/home/nsdeshmukh306/digital-twin")
GEM_IN         = BASE / "data/gem/cncm_i745_regulated.xml"
OUT_PROTEASE   = BASE / "data/fba_outputs/protease_kinetics.csv"
OUT_INFLAM     = BASE / "data/fba_outputs/inflammatory_signaling.csv"
REPORT_TXT     = BASE / "logs/layer4_report.txt"

SEP  = "=" * 60
SEP2 = "-" * 60
log_lines: list[str] = []

def log(msg: str = "") -> None:
    print(msg)
    log_lines.append(msg)

def section(title: str) -> None:
    log("")
    log(SEP)
    log(f"STEP {title}")
    log(SEP)

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def load_silent(path) -> cobra.Model:
    buf = io.StringIO()
    with redirect_stderr(buf):
        return read_sbml_model(str(path))

# ═════════════════════════════════════════════════════════════════════════════
# 1. PROTEASE SECRETION MODULE — CAMP factor cleavage of TcdA
# ═════════════════════════════════════════════════════════════════════════════
section("1 — PROTEASE SECRETION MODULE (CAMP factor / TcdA cleavage)")

S0   = 100.0   # nM  initial TcdA concentration
KM   = 15.0    # nM  Michaelis constant
VMAX = 0.8     # nM/min  maximum cleavage rate

def mm_ode(S, t):
    """Michaelis-Menten: dS/dt = -Vmax * S / (Km + S)"""
    return [-VMAX * S[0] / (KM + S[0])]

# Extend integration to 300 min to ensure we capture t90 analytically
# (analytical solution: Km*ln(S0/S) + (S0-S) = Vmax*t  →  t90 ~ 156 min)
t_long     = np.linspace(0, 300, 3001)   # 0–300 min for threshold detection
sol_long   = odeint(mm_ode, [S0], t_long)
S_long     = sol_long[:, 0]

# Clip to 120 min for reporting / CSV
mask_120   = t_long <= 120.0
t_protease = t_long[mask_120]
S_t        = S_long[mask_120]

# Find time to 50% and 90% cleavage (search full 300-min window)
S_50 = S0 * 0.50
S_10 = S0 * 0.10   # 90% cleaved → 10% remaining

t50 = t90 = None
for i, s in enumerate(S_long):
    if t50 is None and s <= S_50:
        t50 = t_long[i]
    if t90 is None and s <= S_10:
        t90 = t_long[i]
        break

# Analytical fallback if still not found
if t90 is None:
    # Km*ln(S0/S) + (S0-S) = Vmax*t  →  solve for S=0.1*S0
    t90_analytic = (KM * math.log(S0 / (S0 * 0.1)) + S0 * 0.9) / VMAX
    t90_str = f"~{t90_analytic:.1f} min (extrapolated)"
else:
    t90_str = f"{t90:.2f} min"

S_final = S_t[-1]
pct_cleaved_120 = (S0 - S_final) / S0 * 100

log(f"  CAMP factor kinetics  (Vmax={VMAX} nM/min, Km={KM} nM)")
log(f"  Initial TcdA          : {S0:.1f} nM")
log(f"  Time to 50% cleavage  : {t50:.2f} min")
log(f"  Time to 90% cleavage  : {t90_str}  (beyond 120-min window)")
log(f"  TcdA remaining at 120 min : {S_final:.3f} nM  ({pct_cleaved_120:.1f}% cleaved)")

# Save time-course CSV (120-min window)
OUT_PROTEASE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PROTEASE, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["time_min", "TcdA_nM", "cleaved_nM", "pct_cleaved"])
    for t, s in zip(t_protease, S_t):
        writer.writerow([f"{t:.2f}", f"{s:.4f}",
                         f"{S0-s:.4f}", f"{(S0-s)/S0*100:.4f}"])
log(f"  Saved protease kinetics (120-min) → {OUT_PROTEASE}")

# ═════════════════════════════════════════════════════════════════════════════
# 2. POLYAMINE BIOSYNTHESIS MODULE
# ═════════════════════════════════════════════════════════════════════════════
section("2 — POLYAMINE BIOSYNTHESIS MODULE")

log(f"  Loading GEM: {GEM_IN}")
model = load_silent(GEM_IN)

POLYAMINE_RXNS = {
    "r_0817": "ornithine decarboxylase",
    "r_0816": "ornithine carbamoyltransferase",
    "r_0818": "ornithine transacetylase",
    "r_0819": "ornithine transaminase",
    "r_0118": "acetylornithinetransaminase",
    "r_1237": "ornithine transport",
    "r_1001": "spermidine synthase",
    "r_1002": "spermine synthase",
    "r_1250": "putrescine excretion",
    "r_1259": "spermidine excretion",
    "r_2024": "putrescine exchange",
    "r_2051": "spermidine exchange",
    "r_2052": "spermine exchange",
    "r_0929": "polyamine oxidase",
}

# Baseline FBA fluxes
baseline_sol = model.optimize()
log("  Baseline FBA polyamine fluxes (mmol/gDW/hr):")
log(f"  {'Reaction':<10} {'Flux':>10}  Name")
log(f"  {'-'*10} {'-'*10}  {'-'*35}")
for rid, rname in POLYAMINE_RXNS.items():
    try:
        f = baseline_sol.fluxes[rid]
        log(f"  {rid:<10} {f:>+10.6f}  {rname}")
    except KeyError:
        log(f"  {rid:<10} {'N/A':>10}  {rname}")

# Maximum polyamine flux (change objective to spermidine exchange)
with model:
    model.objective = "r_2051"   # spermidine exchange
    max_sol = model.optimize()
    max_spd_flux = max_sol.objective_value if max_sol.status == "optimal" else 0.0

# Total polyamine production rate at baseline
# Key synthetic reactions: ODC (r_0817), spermidine synthase (r_1001), spermine synthase (r_1002)
pa_baseline = sum(
    abs(baseline_sol.fluxes.get(r, 0.0))
    for r in ["r_0817", "r_1001", "r_1002"]
)
# Convert mmol/gDW/hr → nmol/gDW/hr  (1 mmol = 10^6 nmol, but FBA units often ~10^-3 scale)
# Standard: 1 mmol/gDW/hr = 1000 µmol/gDW/hr = 1,000,000 nmol/gDW/hr
pa_rate_nmol = pa_baseline * 1e6   # full SI conversion
# More practically: report in µmol/gDW/hr  (×1000) for readability
pa_rate_umol = pa_baseline * 1e3

log(f"\n  Maximum spermidine export capacity : {max_spd_flux:.4f} mmol/gDW/hr")
log(f"  Baseline total polyamine synth     : {pa_baseline:.6f} mmol/gDW/hr")
log(f"                                       {pa_rate_umol:.3f} µmol/gDW/hr")
log(f"                                       {pa_rate_nmol:.1f} nmol/gDW/hr")

# ── CO2 production as metabolic activity proxy (used in barrier module) ──────
co2_flux = abs(baseline_sol.fluxes.get("r_1672", 0.0))   # CO2 exchange
log(f"\n  CO2 production rate (metabolic proxy) : {co2_flux:.4f} mmol/gDW/hr")

# ═════════════════════════════════════════════════════════════════════════════
# 3. ANTI-INFLAMMATORY SIGNALLING MODULE
# ═════════════════════════════════════════════════════════════════════════════
section("3 — ANTI-INFLAMMATORY SIGNALLING MODULE (NF-κB ODE)")

# State: [NFkB, IL1b, TNFa, IL10]
def inflammatory_ode(y, t, Sb):
    NFkB, IL1b, TNFa, IL10 = y
    dNFkB = 0.30 * (1 - Sb * 0.7) - 0.20 * NFkB
    dIL1b = 0.40 * NFkB             - 0.30 * IL1b
    dTNFa = 0.35 * NFkB             - 0.25 * TNFa
    dIL10 = 0.20 * Sb               - 0.15 * IL10
    return [dNFkB, dIL1b, dTNFa, dIL10]

y0         = [0.0, 0.0, 0.0, 0.0]   # all start at 0 (naive immune state)
t_inflam   = np.linspace(0, 240, 2401)

sol_probiotic = odeint(inflammatory_ode, y0, t_inflam, args=(1.0,))  # Sb=1
sol_control   = odeint(inflammatory_ode, y0, t_inflam, args=(0.0,))  # Sb=0

# Steady-state values (last time point)
ss_p = sol_probiotic[-1]   # [NFkB, IL1b, TNFa, IL10]
ss_c = sol_control[-1]

# Analytical steady states (solve dY/dt = 0):
#   Sb=1: NFkB_ss=0.3*(0.3)/0.2=0.45, IL1b_ss=0.6, TNFa_ss=0.63, IL10_ss=1.333
#   Sb=0: NFkB_ss=1.5, IL1b_ss=2.0, TNFa_ss=2.1, IL10_ss=0

labels = ["NFkB", "IL-1β", "TNF-α", "IL-10"]
log("  Steady-state values after 240 min:")
log(f"  {'Variable':<10} {'Probiotic (Sb=1)':>18}  {'Control (Sb=0)':>16}  {'Δ (Sb1-Sb0)':>14}")
log(f"  {'-'*10} {'-'*18}  {'-'*16}  {'-'*14}")
for label, sp, sc in zip(labels, ss_p, ss_c):
    delta = sp - sc
    log(f"  {label:<10} {sp:>18.4f}  {sc:>16.4f}  {delta:>+14.4f}")

# % reductions
nfkb_reduction = (ss_c[0] - ss_p[0]) / ss_c[0] * 100 if ss_c[0] > 0 else 0
il1b_reduction = (ss_c[1] - ss_p[1]) / ss_c[1] * 100 if ss_c[1] > 0 else 0
tnfa_reduction = (ss_c[2] - ss_p[2]) / ss_c[2] * 100 if ss_c[2] > 0 else 0
log(f"\n  NF-κB suppression vs control : {nfkb_reduction:.1f}%")
log(f"  IL-1β  reduction vs control  : {il1b_reduction:.1f}%")
log(f"  TNF-α  reduction vs control  : {tnfa_reduction:.1f}%")
log(f"  IL-10  induction (Sb=1)      : {ss_p[3]:.4f} (control: {ss_c[3]:.4f})")

# Save time-course CSV
OUT_INFLAM.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_INFLAM, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["time_min",
                     "NFkB_Sb1", "IL1b_Sb1", "TNFa_Sb1", "IL10_Sb1",
                     "NFkB_Sb0", "IL1b_Sb0", "TNFa_Sb0", "IL10_Sb0"])
    for i, t in enumerate(t_inflam):
        row = [f"{t:.2f}"]
        row += [f"{v:.6f}" for v in sol_probiotic[i]]
        row += [f"{v:.6f}" for v in sol_control[i]]
        writer.writerow(row)
log(f"\n  Saved signalling time-course → {OUT_INFLAM}")

# ═════════════════════════════════════════════════════════════════════════════
# 4. BARRIER INTEGRITY MODULE
# ═════════════════════════════════════════════════════════════════════════════
section("4 — BARRIER INTEGRITY MODULE")

# polyamine_flux: total baseline polyamine synthesis (mmol/gDW/hr)
polyamine_flux = pa_baseline   # ~1.6e-5 mmol/gDW/hr

# butyrate_proxy: CO2 production rate normalised to glucose uptake
# (captures overall fermentative metabolic output)
glucose_uptake  = abs(baseline_sol.fluxes.get("r_1714", 1.0))
butyrate_proxy  = co2_flux / (glucose_uptake * 6)   # fraction of theoretical CO2 yield

log(f"  Inputs:")
log(f"    polyamine_flux  = {polyamine_flux:.6f} mmol/gDW/hr")
log(f"    butyrate_proxy  = {butyrate_proxy:.4f}  (CO2/(6*glucose), normalised)")

claudin3  = sigmoid(polyamine_flux * 2.0)
occludin  = sigmoid(butyrate_proxy * 1.5)
zo1       = sigmoid((claudin3 + occludin) / 2.0)
barrier_score = (claudin3 + occludin + zo1) / 3.0

log(f"\n  Tight junction protein expression (0–1 scale):")
log(f"  {'Protein':<15} {'Expression':>12}  Formula")
log(f"  {'-'*15} {'-'*12}  {'-'*40}")
log(f"  {'Claudin-3':<15} {claudin3:>12.4f}  sigmoid(polyamine_flux × 2.0)")
log(f"  {'Occludin':<15} {occludin:>12.4f}  sigmoid(butyrate_proxy × 1.5)")
log(f"  {'ZO-1':<15} {zo1:>12.4f}  sigmoid((claudin + occludin) / 2)")
log(f"\n  Barrier integrity score : {barrier_score:.4f}  (mean of three proteins)")

# Interpretation
if barrier_score >= 0.75:
    interp = "HIGH — robust epithelial protection"
elif barrier_score >= 0.50:
    interp = "MODERATE — partial barrier function"
else:
    interp = "LOW — compromised barrier"
log(f"  Interpretation          : {interp}")

# ═════════════════════════════════════════════════════════════════════════════
# 5. INTEGRATED HOST INTERACTION REPORT
# ═════════════════════════════════════════════════════════════════════════════
section("5 — INTEGRATED HOST INTERACTION REPORT")

log(f"  {'Metric':<45} {'Value':>18}  Unit")
log(f"  {'-'*45} {'-'*18}  {'-'*20}")
log(f"  {'Toxin neutralisation (TcdA cleaved at 120 min)':<45} {pct_cleaved_120:>17.1f}%  %")
log(f"  {'Time to 50% TcdA cleavage':<45} {t50:>17.2f}  min")
log(f"  {'Time to 90% TcdA cleavage':<45} {t90_str:>18}  min")
log(f"  {'Polyamine production (baseline)':<45} {pa_rate_umol:>17.4f}  µmol/gDW/hr")
log(f"  {'Max spermidine export capacity':<45} {max_spd_flux*1e3:>17.1f}  µmol/gDW/hr")
log(f"  {'NF-κB suppression vs control':<45} {nfkb_reduction:>17.1f}%  %")
log(f"  {'IL-1β reduction vs control':<45} {il1b_reduction:>17.1f}%  %")
log(f"  {'TNF-α reduction vs control':<45} {tnfa_reduction:>17.1f}%  %")
log(f"  {'IL-10 induction (probiotic vs control)':<45} {ss_p[3]-ss_c[3]:>17.4f}  rel. units")
log(f"  {'Claudin-3 expression':<45} {claudin3:>17.4f}  0–1")
log(f"  {'Occludin expression':<45} {occludin:>17.4f}  0–1")
log(f"  {'ZO-1 expression':<45} {zo1:>17.4f}  0–1")
log(f"  {'Barrier integrity score':<45} {barrier_score:>17.4f}  0–1")

# Save full report
REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
report_header = [
    SEP,
    "LAYER 4 HOST INTERACTION REPORT",
    "Organism : Saccharomyces boulardii CNCM I-745",
    "Input GEM: cncm_i745_regulated.xml",
    SEP,
    "",
    "── Module 1: CAMP Protease Kinetics ──",
    f"  Vmax={VMAX} nM/min, Km={KM} nM, [TcdA]₀={S0} nM",
    f"  t50%  = {t50:.2f} min",
    f"  t90%  = {t90_str}",
    f"  TcdA remaining @ 120 min = {S_final:.3f} nM ({pct_cleaved_120:.1f}% cleaved)",
    "",
    "── Module 2: Polyamine Biosynthesis ──",
    f"  Baseline ODC flux         = {baseline_sol.fluxes.get('r_0817',0):.6f} mmol/gDW/hr",
    f"  Baseline total PA synth   = {pa_baseline:.6f} mmol/gDW/hr ({pa_rate_umol:.4f} µmol/gDW/hr)",
    f"  Max spermidine export     = {max_spd_flux:.4f} mmol/gDW/hr",
    "",
    "── Module 3: Anti-inflammatory Signalling ──",
    f"  Sb=1  NFkB={ss_p[0]:.4f}  IL-1β={ss_p[1]:.4f}  TNF-α={ss_p[2]:.4f}  IL-10={ss_p[3]:.4f}",
    f"  Sb=0  NFkB={ss_c[0]:.4f}  IL-1β={ss_c[1]:.4f}  TNF-α={ss_c[2]:.4f}  IL-10={ss_c[3]:.4f}",
    f"  NF-κB suppression = {nfkb_reduction:.1f}%",
    f"  IL-1β  reduction  = {il1b_reduction:.1f}%",
    f"  TNF-α  reduction  = {tnfa_reduction:.1f}%",
    "",
    "── Module 4: Barrier Integrity ──",
    f"  Claudin-3 = {claudin3:.4f}",
    f"  Occludin  = {occludin:.4f}",
    f"  ZO-1      = {zo1:.4f}",
    f"  Score     = {barrier_score:.4f}  ({interp})",
    "",
    "── Output Files ──",
    f"  Protease kinetics CSV   : {OUT_PROTEASE}",
    f"  Inflammatory signals CSV: {OUT_INFLAM}",
    f"  This report             : {REPORT_TXT}",
    "",
    SEP,
    "",
]
with open(REPORT_TXT, "w") as fh:
    fh.write("\n".join(report_header))
    fh.write("\n\nFull console log:\n")
    fh.write("\n".join(log_lines))

log(f"\n  Report saved → {REPORT_TXT}")
log("")
log("LAYER 4 COMPLETE")

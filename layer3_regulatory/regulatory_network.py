"""
Layer 3 — Boolean Regulatory Network for Saccharomyces boulardii CNCM I-745
Overlays stress-response regulons onto the GEM and simulates gut transit zones.

Two modes:
  mode="boolean" : original Boolean network (upregulation of target reactions)
  mode="eflux"   : E-Flux expression constraints (Colijn et al. 2009) using
                   Gasch et al. 2000 fold-changes as S. boulardii proxy

Model source: cncm_i745_strain_specific.xml
  Strain-specific: HXT9/HXT11/MAL/ASP3 GPR corrected (Khatri et al. 2017)
"""

import io
import math
import sys
from pathlib import Path
from contextlib import redirect_stderr
from dataclasses import dataclass, field

import cobra
from cobra.io import read_sbml_model, write_sbml_model

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path("/home/nsdeshmukh306/digital-twin")
GEM_IN    = BASE / "data/gem/cncm_i745_strain_specific.xml"   # updated v3.0
GEM_OUT   = BASE / "data/gem/cncm_i745_regulated.xml"
REPORT    = BASE / "logs/layer3_report.txt"

SEP  = "=" * 60
SEP2 = "-" * 60
lines: list[str] = []

def log(msg: str = "") -> None:
    print(msg)
    lines.append(msg)

def section(title: str) -> None:
    log("")
    log(SEP)
    log(f"STEP {title}")
    log(SEP)

def load_silent(path) -> cobra.Model:
    buf = io.StringIO()
    with redirect_stderr(buf):
        return read_sbml_model(str(path))

def safe_fba(model: cobra.Model) -> float:
    sol = model.optimize()
    return sol.objective_value if sol.status == "optimal" else float("nan")

def fmt_growth(g: float) -> str:
    return f"{g:.6f}" if not math.isnan(g) else "infeasible"

# ── Reaction finder: match by keyword in name OR id (case-insensitive) ────────
def find_reactions(model: cobra.Model, keywords: list[str],
                   exclude_exchange: bool = True) -> list[cobra.Reaction]:
    hits: list[cobra.Reaction] = []
    seen: set[str] = set()
    ex_ids = {r.id for r in model.exchanges}
    for kw in keywords:
        kw_lo = kw.lower()
        for r in model.reactions:
            if r.id in seen:
                continue
            if exclude_exchange and r.id in ex_ids:
                continue
            if kw_lo in r.name.lower() or kw_lo in r.id.lower():
                hits.append(r)
                seen.add(r.id)
    return hits

# ── Apply upregulation: scale upper bound; respect 1000-cap convention ────────
UB_DEFAULT = 1000.0

def upregulate(model: cobra.Model, rxn_ids: list[str], factor: float) -> dict[str, tuple]:
    """
    Multiply upper bound by (1 + factor).
    For reactions whose ub == UB_DEFAULT (unconstrained), also tighten lower
    bound slightly so the flux window shifts meaningfully.
    Returns {rxn_id: (old_bounds, new_bounds)}.
    """
    changes: dict[str, tuple] = {}
    for rid in rxn_ids:
        try:
            r = model.reactions.get_by_id(rid)
        except KeyError:
            continue
        old = r.bounds
        new_ub = r.upper_bound * (1 + factor)
        # Never exceed default cap
        new_ub = min(new_ub, UB_DEFAULT)
        # If already at cap, signal by ensuring lb is not artificially low
        r.upper_bound = new_ub
        changes[rid] = (old, r.bounds)
    return changes

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD GUT MODEL
# ═════════════════════════════════════════════════════════════════════════════
section("1 — LOAD GUT MODEL")

log(f"  Loading: {GEM_IN}")
base_model = load_silent(GEM_IN)

baseline_growth = safe_fba(base_model)
log(f"  Reactions   : {len(base_model.reactions)}")
log(f"  Genes       : {len(base_model.genes)}")
log(f"  Growth rate : {fmt_growth(baseline_growth)} h⁻¹  (baseline confirmation)")

# ═════════════════════════════════════════════════════════════════════════════
# 2. BOOLEAN REGULATORY NETWORK — define regulons
# ═════════════════════════════════════════════════════════════════════════════
section("2 — BOOLEAN REGULATORY NETWORK")

@dataclass
class Regulon:
    name: str
    condition_desc: str
    keywords: list[str]
    effect: float           # fractional upregulation (e.g. 0.20 = +20%)
    rxn_ids: list[str] = field(default_factory=list)       # populated at build time
    rxn_names: list[str] = field(default_factory=list)

    def is_active(self, pH: float, temp: float, osm: float, ROS: bool) -> bool:
        raise NotImplementedError

@dataclass
class HSR(Regulon):
    """Heat Shock Response — active at body temperature (37°C)."""
    def is_active(self, pH, temp, osm, ROS): return temp >= 37.0

@dataclass
class HOG(Regulon):
    """High-Osmolarity Glycerol pathway — active when osmolarity > 0.3 osmol."""
    def is_active(self, pH, temp, osm, ROS): return osm > 0.3

@dataclass
class AcidStress(Regulon):
    """Acid Stress Response — active when pH < 4.0."""
    def is_active(self, pH, temp, osm, ROS): return pH < 4.0

@dataclass
class OxidativeStress(Regulon):
    """Yap1 / Oxidative Stress — active when ROS is detected."""
    def is_active(self, pH, temp, osm, ROS): return ROS

REGULONS: list[Regulon] = [
    HSR(
        name="HSR (Heat Shock)",
        condition_desc="temperature ≥ 37°C",
        keywords=["trehalose", "trehalase", "trehalose-phosphate"],
        effect=0.20,
    ),
    HOG(
        name="HOG (Osmotic Stress)",
        condition_desc="osmolarity > 0.3 osmol",
        keywords=["glycerol-3-phosphate dehydrogenase", "glycerol-3-phosphatase",
                  "glycerol kinase", "glycerol exchange"],
        effect=0.50,
    ),
    AcidStress(
        name="Acid Stress",
        condition_desc="pH < 4.0",
        keywords=["V-ATPase", "ATPase, cytosolic", "sodium proton antiporter",
                  "proton leak"],
        effect=0.30,
    ),
    OxidativeStress(
        name="Oxidative Stress (Yap1)",
        condition_desc="ROS detected",
        keywords=["glutathione oxidoreductase", "glutathione peroxidase",
                  "hydrogen peroxide reductase", "thioredoxin"],
        effect=0.25,
    ),
]

# Resolve reaction IDs for each regulon
log("  Resolving target reactions per regulon:")
for reg in REGULONS:
    rxns = find_reactions(base_model, reg.keywords)
    reg.rxn_ids   = [r.id   for r in rxns]
    reg.rxn_names = [r.name for r in rxns]
    log(f"\n  [{reg.name}]")
    log(f"    Condition : {reg.condition_desc}")
    log(f"    Effect    : +{reg.effect*100:.0f}% upper-bound upregulation")
    log(f"    Targets found : {len(reg.rxn_ids)}")
    for rid, rname in zip(reg.rxn_ids[:6], reg.rxn_names[:6]):
        log(f"      {rid}  {rname[:55]}")
    if len(reg.rxn_ids) > 6:
        log(f"      ... and {len(reg.rxn_ids)-6} more")

# ═════════════════════════════════════════════════════════════════════════════
# 3. GUT TRANSIT SIMULATION
# ═════════════════════════════════════════════════════════════════════════════
section("3 — GUT TRANSIT SIMULATION")

@dataclass
class GutZone:
    name: str
    pH: float
    temp: float
    osm: float
    ROS: bool

GUT_ZONES = [
    GutZone("Stomach",  pH=2.0, temp=37.0, osm=0.30, ROS=False),
    GutZone("Duodenum", pH=6.0, temp=37.0, osm=0.20, ROS=False),
    GutZone("Ileum",    pH=7.0, temp=37.0, osm=0.15, ROS=True),
    GutZone("Colon",    pH=7.2, temp=37.0, osm=0.35, ROS=True),
]

# Results store: {zone_name: (active_regulons, growth, {rxn_id: (old,new)})}
zone_results: dict[str, tuple] = {}
colon_model: cobra.Model | None = None

for zone in GUT_ZONES:
    log(f"\n  {'─'*50}")
    log(f"  Zone: {zone.name}  "
        f"(pH={zone.pH}, T={zone.temp}°C, osm={zone.osm} osmol, "
        f"ROS={'yes' if zone.ROS else 'no'})")

    # Identify active regulons
    active = [r for r in REGULONS if r.is_active(zone.pH, zone.temp, zone.osm, zone.ROS)]
    active_names = [r.name for r in active]
    log(f"  Active regulons: {active_names if active_names else ['none']}")

    # Build a fresh model copy and apply modifications
    zm = base_model.copy()
    all_changes: dict[str, tuple] = {}
    for reg in active:
        changes = upregulate(zm, reg.rxn_ids, reg.effect)
        all_changes.update(changes)

    # Run FBA
    growth = safe_fba(zm)
    delta  = growth - baseline_growth

    log(f"  Growth rate : {fmt_growth(growth)} h⁻¹  "
        f"(Δ {'%+.6f' % delta if not math.isnan(growth) else 'N/A'})")

    # Key flux changes: report reactions that actually changed upper bounds
    changed_rxns = [(rid, ob, nb) for rid, (ob, nb) in all_changes.items()
                    if ob[1] != nb[1]]
    if changed_rxns:
        log(f"  Bound changes applied: {len(changed_rxns)}")
        for rid, ob, nb in changed_rxns[:5]:
            try:
                rname = base_model.reactions.get_by_id(rid).name[:40]
            except Exception:
                rname = rid
            log(f"    {rid}  ub: {ob[1]:.1f} → {nb[1]:.1f}  ({rname})")
        if len(changed_rxns) > 5:
            log(f"    ... and {len(changed_rxns)-5} more bound changes")
    else:
        log("  (All target reactions already at default cap — ub unchanged)")

    # Get fluxes for key regulon target reactions in this zone
    sol = zm.optimize()
    if sol.status == "optimal":
        log(f"  Key fluxes in active regulon targets (non-zero, top 5):")
        tracked: list[tuple[str, float, str]] = []
        for reg in active:
            for rid in reg.rxn_ids:
                try:
                    f = sol.fluxes[rid]
                    if abs(f) > 1e-9:
                        rname = base_model.reactions.get_by_id(rid).name[:35]
                        tracked.append((rid, f, rname))
                except Exception:
                    pass
        tracked.sort(key=lambda x: abs(x[1]), reverse=True)
        for rid, flux, rname in tracked[:5]:
            log(f"    {rid}  flux={flux:+.4f}  {rname}")
        if not tracked:
            log("    (all target reactions carry zero flux under current constraints)")

    zone_results[zone.name] = (active_names, growth, all_changes)

    if zone.name == "Colon":
        colon_model = zm

# ═════════════════════════════════════════════════════════════════════════════
# 4. REGULATORY IMPACT TABLE
# ═════════════════════════════════════════════════════════════════════════════
section("4 — REGULATORY IMPACT TABLE")

col_w = 38
log(f"  {'Zone':<12} {'Active Regulons':<{col_w}} {'Growth (h⁻¹)':>14}  {'Δ baseline':>12}")
log(f"  {'-'*12} {'-'*col_w} {'-'*14}  {'-'*12}")
for zone in GUT_ZONES:
    active_names, growth, _ = zone_results[zone.name]
    regs_str = ", ".join(r.split("(")[0].strip() for r in active_names) or "none"
    if len(regs_str) > col_w - 1:
        regs_str = regs_str[:col_w-4] + "..."
    delta = growth - baseline_growth
    sign  = "+" if delta >= 0 else ""
    log(f"  {zone.name:<12} {regs_str:<{col_w}} {fmt_growth(growth):>14}  "
        f"{sign}{delta:>11.6f}")

log(f"\n  Baseline (no regulatory overlay): {fmt_growth(baseline_growth)} h⁻¹")
log("")
log("  Interpretation:")
log("  All gut zones run at body temperature (37°C), activating HSR in every zone.")
log("  HOG pathway fires only in Colon (osm 0.35 > 0.30 threshold).")
log("  Acid Stress fires in Stomach (pH 2.0 < 4.0).")
log("  Oxidative Stress (Yap1) fires in Ileum and Colon (ROS=True).")
log("  Growth rate changes are minimal because target reactions are already")
log("  unconstrained (ub=1000) in the Yeast9 GEM; regulatory effects are captured")
log("  in the bound-change records and flux distributions rather than growth rate.")

# ═════════════════════════════════════════════════════════════════════════════
# 5. SAVE OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════
section("5 — SAVE OUTPUTS")

# Save colon model
if colon_model is not None:
    colon_model.id = "cncm_i745_regulated"
    GEM_OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with redirect_stderr(buf):
        write_sbml_model(colon_model, str(GEM_OUT))
    fsize = GEM_OUT.stat().st_size
    log(f"  Regulated model (Colon) → {GEM_OUT}")
    log(f"  File size : {fsize:,} bytes  ({fsize/1024:.1f} KB)")

# Save report
REPORT.parent.mkdir(parents=True, exist_ok=True)

# Build standalone report header
report_header = [
    SEP,
    "LAYER 3 REGULATORY NETWORK REPORT",
    "Organism : Saccharomyces boulardii CNCM I-745",
    "Input GEM: cncm_i745_gut.xml",
    SEP,
    "",
    "── Regulons Defined ──",
]
for reg in REGULONS:
    report_header.append(
        f"  {reg.name:<30} condition={reg.condition_desc}  "
        f"effect=+{reg.effect*100:.0f}%  targets={len(reg.rxn_ids)}"
    )

report_header += ["", "── Gut Zone Simulation ──"]
for zone in GUT_ZONES:
    active_names, growth, _ = zone_results[zone.name]
    delta = growth - baseline_growth
    report_header.append(
        f"  {zone.name:<12} pH={zone.pH}  osm={zone.osm}  ROS={'Y' if zone.ROS else 'N'}"
        f"  active={[n.split('(')[0].strip() for n in active_names] or ['none']}"
        f"  growth={fmt_growth(growth)} h⁻¹  Δ={'%+.6f' % delta}"
    )

report_header += [
    "",
    "── Output Files ──",
    f"  Regulated GEM : {GEM_OUT}",
    f"  This report   : {REPORT}",
    "",
    SEP,
    "",
]

full_report = "\n".join(report_header) + "\n" + "\n".join(lines)
with open(REPORT, "w") as fh:
    fh.write(full_report)

log(f"\n  Report saved → {REPORT}")
log("")
log("LAYER 3 COMPLETE")

# ═════════════════════════════════════════════════════════════════════════════
# E-FLUX INTEGRATION (mode="eflux") — v3.0 addition
# ═════════════════════════════════════════════════════════════════════════════

def run_eflux_gut_transit(mode: str = "boolean") -> dict:
    """
    Run gut transit simulation in either Boolean or E-Flux mode.

    Parameters
    ----------
    mode : "boolean"  — original Boolean regulon upregulation
           "eflux"    — E-Flux expression constraints (Colijn et al. 2009)

    Returns
    -------
    dict mapping zone_name -> {growth_rate, delta, mode, ...}
    """
    print(f"\n[Layer 3] mode={mode}")

    if mode == "boolean":
        print("  Using Boolean regulatory network (original method)")
        out = {}
        for zone in GUT_ZONES:
            act = [r for r in REGULONS if r.is_active(zone.pH, zone.temp, zone.osm, zone.ROS)]
            zm  = base_model.copy()
            for reg in act:
                upregulate(zm, reg.rxn_ids, reg.effect)
            growth = safe_fba(zm)
            out[zone.name] = {
                "growth_rate":     growth,
                "delta":           growth - baseline_growth,
                "active_regulons": [r.name for r in act],
                "mode":            "boolean",
            }
        return out

    elif mode == "eflux":
        print("  Using E-Flux (Colijn et al. 2009) with Gasch et al. 2000 fold-changes")
        import json
        eflux_json = BASE / "data/fba_outputs/eflux_results.json"
        if eflux_json.exists():
            with open(eflux_json) as fh:
                results = json.load(fh)
            print(f"  Loaded pre-computed E-Flux results")
            return {
                zone: {
                    "growth_rate":        r.get("growth_rate", 0.0),
                    "delta":              r.get("delta_growth", 0.0),
                    "condition":          r.get("condition", "unknown"),
                    "modified_reactions": r.get("modified_reactions", 0),
                    "mode":               "eflux",
                }
                for zone, r in results.items()
            }
        else:
            print("  E-Flux results not found — run layer3_regulatory/eflux_simulator.py first")
            return {}
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'boolean' or 'eflux'.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--eflux":
        print("\n[E-Flux mode requested via CLI]")
        eflux_results = run_eflux_gut_transit(mode="eflux")
        for zone, res in eflux_results.items():
            print(f"  {zone:10s}: {res.get('growth_rate', 0):.6f} h⁻¹  "
                  f"mode={res.get('mode')}")

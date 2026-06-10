"""
A3 — Multi-condition comparative endpoints.

* ``POST /compare/gut_transit`` runs FBA + E-Flux for a list of gut zones and
  returns, per zone, the growth rate, the five most up- and down-regulated
  reactions (parsimonious flux vs the glucose-limited baseline), the modelled
  NF-κB level and the epithelial barrier-integrity score.
* ``POST /compare/carbon_sources`` compares growth and key flux distributions
  across alternative sole carbon sources (glucose, fructose, galactose,
  ethanol), reflecting the CNCM I-745 strain-specific carbohydrate phenotype.

NF-κB suppression follows the Layer-4 ODE (Thomas et al. 2019 report 60–75 %
suppression) scaled by zonal viability; barrier scoring follows the Layer-4
polyamine / metabolic-output sigmoid model.
"""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from cobra.exceptions import Infeasible
from cobra.flux_analysis import pfba

from api import state
from api.auth import require_api_key
from logging_setup import get_logger

logger = get_logger("api.compare")
router = APIRouter(prefix="/compare", tags=["compare"])

# Polyamine biosynthesis reactions used for barrier scoring (Layer 4).
_PA_RXNS = ["r_0817", "r_1001", "r_1002"]
_CO2_RXN = "r_1672"
_CTRL_NFKB = 1.5            # NF-κB steady state with no probiotic (Sb=0)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _barrier_from_fluxes(fluxes: dict, glucose_uptake: float) -> dict:
    polyamine_flux = sum(abs(fluxes.get(r, 0.0)) for r in _PA_RXNS)
    co2 = abs(fluxes.get(_CO2_RXN, 0.0))
    butyrate_proxy = co2 / (abs(glucose_uptake) * 6) if glucose_uptake else 0.0
    claudin3 = _sigmoid(polyamine_flux * 2.0)
    occludin = _sigmoid(butyrate_proxy * 1.5)
    zo1 = _sigmoid((claudin3 + occludin) / 2.0)
    score = (claudin3 + occludin + zo1) / 3.0
    return {"claudin3": round(claudin3, 4), "occludin": round(occludin, 4),
            "zo1": round(zo1, 4), "barrier_score": round(score, 4)}


def _nfkb_for_growth(growth: float) -> dict:
    """Zone NF-κB level: probiotic effectiveness scales with viability."""
    sb_eff = max(0.0, min(1.0, growth / state.BASELINE_GROWTH if state.BASELINE_GROWTH else 0.0))
    nfkb = 0.30 * (1 - sb_eff * 0.7) / 0.20
    suppression = (_CTRL_NFKB - nfkb) / _CTRL_NFKB * 100
    return {"nfkb_level": round(nfkb, 4),
            "nfkb_suppression_pct": round(suppression, 1),
            "viability_factor": round(sb_eff, 3)}


def _baseline_pfba() -> dict:
    model = state.get_gem()
    with model:
        model.reactions.get_by_id(state.RXN_GLUCOSE).lower_bound = -1.65
        model.reactions.get_by_id(state.RXN_OXYGEN).lower_bound = -2.0
        model.reactions.get_by_id(state.RXN_NH4).lower_bound = -1.0
        model.reactions.get_by_id(state.RXN_PI).lower_bound = -0.5
        try:
            return pfba(model).fluxes.to_dict()
        except Exception:
            sol = model.optimize()
            return sol.fluxes.to_dict() if sol.status == "optimal" else {}


# ── Gut transit comparison ─────────────────────────────────────────────────────
class GutTransitRequest(BaseModel):
    gut_zones: list[str] = Field(
        default_factory=lambda: ["stomach", "duodenum", "ileum", "colon"],
        description="Ordered list of gut zones to compare.")
    use_eflux: bool = Field(True, description="Apply E-Flux expression constraints.")


@router.post("/gut_transit", dependencies=[Depends(require_api_key)])
def gut_transit(req: GutTransitRequest) -> dict:
    """Compare FBA + E-Flux across an ordered gut-transit sequence."""
    model = state.get_gem()
    baseline = _baseline_pfba()
    zones_out = []
    for zone_name in req.gut_zones:
        key = zone_name.lower()
        if key not in state.GUT_ZONES:
            zones_out.append({"gut_zone": zone_name, "error": "unknown gut zone"})
            continue
        try:
            with model:
                params = state.apply_zone_bounds(model, key)
                eflux = state.apply_eflux_bounds(model, key) if req.use_eflux else {"applied": False}
                sol = model.optimize()
                growth = float(sol.objective_value) if sol.status == "optimal" else 0.0
                try:
                    fluxes = pfba(model).fluxes.to_dict() if growth > 0 else {}
                except Exception:
                    fluxes = sol.fluxes.to_dict() if sol.status == "optimal" else {}
        except Infeasible:
            growth, fluxes, params, eflux = 0.0, {}, state.GUT_ZONES[key], {"applied": False}

        # Up / down regulated reactions vs baseline (parsimonious flux deltas).
        deltas = []
        for rid, val in fluxes.items():
            d = val - baseline.get(rid, 0.0)
            if abs(d) > 1e-6:
                deltas.append((rid, d))
        deltas.sort(key=lambda kv: kv[1], reverse=True)
        up = [{"reaction": r, "name": _rname(model, r), "delta_flux": round(d, 6)}
              for r, d in deltas[:5]]
        down = [{"reaction": r, "name": _rname(model, r), "delta_flux": round(d, 6)}
                for r, d in sorted(deltas, key=lambda kv: kv[1])[:5]]

        zones_out.append({
            "gut_zone": key,
            "description": state.GUT_ZONES[key]["description"],
            "pH": state.GUT_ZONES[key]["pH"],
            "condition": state.GUT_ZONES[key]["condition"],
            "active_regulons": state.GUT_ZONES[key]["regulons"],
            "zone_params": params,
            "eflux": eflux,
            "growth_rate": round(growth, 6),
            "top_upregulated": up,
            "top_downregulated": down,
            **_nfkb_for_growth(growth),
            **_barrier_from_fluxes(fluxes, params.get("glucose", -1.65)),
        })
    logger.info("gut_transit compared %d zones", len(zones_out))
    return {"comparison": "gut_transit", "use_eflux": req.use_eflux,
            "baseline_growth": state.BASELINE_GROWTH, "zones": zones_out}


# ── Carbon source comparison ───────────────────────────────────────────────────
_CARBON_KEYWORDS = {
    "glucose": "d-glucose",
    "fructose": "d-fructose",
    "galactose": "d-galactose",
    "ethanol": "ethanol",
}
_KEY_FLUX_RXNS = {
    "biomass": None, "glucose_ex": state.RXN_GLUCOSE, "oxygen_ex": state.RXN_OXYGEN,
    "co2_ex": _CO2_RXN, "ethanol_ex": None,
}


class CarbonSourceRequest(BaseModel):
    carbon_sources: list[str] = Field(
        default_factory=lambda: ["glucose", "fructose", "galactose", "ethanol"],
        description="Carbon sources to compare (sole-source uptake).")
    uptake: float = Field(-10.0, le=0.0, description="Sole carbon-source uptake bound.")


@router.post("/carbon_sources", dependencies=[Depends(require_api_key)])
def carbon_sources(req: CarbonSourceRequest) -> dict:
    """Compare growth on alternative sole carbon sources."""
    model = state.get_gem()
    results = []
    for cs in req.carbon_sources:
        kw = _CARBON_KEYWORDS.get(cs.lower())
        ex = _find_carbon_exchange(model, kw) if kw else None
        if ex is None:
            results.append({"carbon_source": cs, "error": "exchange reaction not found"})
            continue
        try:
            with model:
                # Shut off glucose, open the chosen carbon source.
                model.reactions.get_by_id(state.RXN_GLUCOSE).lower_bound = 0.0
                ex.lower_bound = req.uptake
                model.reactions.get_by_id(state.RXN_OXYGEN).lower_bound = -2.0
                model.reactions.get_by_id(state.RXN_NH4).lower_bound = -1.0
                model.reactions.get_by_id(state.RXN_PI).lower_bound = -0.5
                sol = model.optimize()
                growth = float(sol.objective_value) if sol.status == "optimal" else 0.0
                key_fluxes = {
                    "co2_production": round(abs(sol.fluxes.get(_CO2_RXN, 0.0)), 4) if growth else 0.0,
                    "oxygen_uptake": round(sol.fluxes.get(state.RXN_OXYGEN, 0.0), 4) if growth else 0.0,
                } if sol.status == "optimal" else {}
        except Infeasible:
            growth, key_fluxes = 0.0, {}
        results.append({
            "carbon_source": cs, "exchange_reaction": ex.id, "exchange_name": ex.name,
            "uptake_bound": req.uptake, "growth_rate": round(growth, 6),
            "feasible": growth > 1e-9, "key_fluxes": key_fluxes,
        })
    logger.info("carbon_sources compared %d sources", len(results))
    return {"comparison": "carbon_sources", "uptake_bound": req.uptake, "results": results}


# ── helpers ────────────────────────────────────────────────────────────────────
def _rname(model, rid: str) -> str:
    try:
        return model.reactions.get_by_id(rid).name
    except KeyError:
        return rid


def _find_carbon_exchange(model, keyword: str):
    """Locate the exchange reaction for a carbon source by metabolite name."""
    keyword = keyword.lower()
    for r in model.exchanges:
        if keyword in r.name.lower():
            return r
    return None

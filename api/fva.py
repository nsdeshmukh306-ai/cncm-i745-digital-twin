"""
A2 — Flux Variability Analysis (FVA) endpoint.

Computes the feasible minimum and maximum flux of each requested reaction while
holding the biomass objective at a fraction of its optimum, using COBRApy's
``flux_variability_analysis``. The flux *span* (max − min) quantifies how
tightly each reaction is constrained at the given growth requirement.

Reference: Mahadevan R & Schilling CH (2003) "The effects of alternate optimal
solutions in constraint-based genome-scale metabolic models." *Metabolic
Engineering* 5(4), 264–276.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import cobra
from cobra.exceptions import Infeasible
from cobra.flux_analysis import flux_variability_analysis

from api import state
from api.auth import require_api_key
from logging_setup import get_logger

logger = get_logger("api.fva")
router = APIRouter(tags=["fba"])


class FVARequest(BaseModel):
    gut_zone: str = Field("none", description="Optional gut zone (stomach/duodenum/ileum/colon).")
    fraction_of_optimum: float = Field(
        0.9, ge=0.0, le=1.0,
        description="Biomass objective held at this fraction of its optimum.")
    loopless: bool = Field(True, description="Use loopless FVA (ll-FVA).")
    reaction_list: list[str] | None = Field(
        None, description="Reaction IDs to analyse. Empty → top 20 exchange reactions.")


@router.post("/fba/fva", dependencies=[Depends(require_api_key)])
def run_fva(req: FVARequest) -> dict:
    """Run flux variability analysis and return min/max/span per reaction."""
    model = state.get_gem()
    try:
        with model:
            zone = {}
            if req.gut_zone and req.gut_zone.lower() in state.GUT_ZONES:
                zone = state.apply_zone_bounds(model, req.gut_zone)

            # Resolve the reaction list (default: top 20 exchanges by |flux|).
            if req.reaction_list:
                rxn_objs, missing = [], []
                for rid in req.reaction_list:
                    try:
                        rxn_objs.append(model.reactions.get_by_id(rid))
                    except KeyError:
                        missing.append(rid)
            else:
                rxn_objs = [model.reactions.get_by_id(r)
                            for r in state.top_exchange_reactions(model, 20)]
                missing = []

            if not rxn_objs:
                raise HTTPException(422, {"error": "No valid reactions",
                                          "detail": "reaction_list matched no model reactions"})

            fva = flux_variability_analysis(
                model, reaction_list=rxn_objs,
                fraction_of_optimum=req.fraction_of_optimum,
                loopless=req.loopless,
            )
    except Infeasible:
        logger.warning("FVA infeasible for zone=%s", req.gut_zone)
        raise                                         # → global 422 handler
    except HTTPException:
        raise
    except Exception as exc:                         # pragma: no cover - solver edge cases
        logger.exception("FVA error")
        raise HTTPException(500, {"error": "FVA error", "detail": str(exc)})

    rows = []
    for rid, row in fva.iterrows():
        lo, hi = float(row["minimum"]), float(row["maximum"])
        try:
            name = model.reactions.get_by_id(rid).name
        except KeyError:
            name = rid
        rows.append({"reaction": rid, "name": name,
                     "minimum": round(lo, 6), "maximum": round(hi, 6),
                     "flux_span": round(hi - lo, 6)})
    rows.sort(key=lambda r: r["flux_span"], reverse=True)
    logger.info("FVA: %d reactions, zone=%s, foo=%.2f", len(rows), req.gut_zone,
                req.fraction_of_optimum)
    return {
        "method": "Flux Variability Analysis",
        "citation": "Mahadevan & Schilling 2003, Metab Eng 5(4):264-276",
        "gut_zone": req.gut_zone,
        "zone_params": zone,
        "fraction_of_optimum": req.fraction_of_optimum,
        "loopless": req.loopless,
        "n_reactions": len(rows),
        "missing_reactions": missing,
        "results": rows,
    }

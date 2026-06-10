"""
A4 — Phenotype Phase Plane (PhPP) endpoint.

Sweeps two reactions over their feasible flux ranges and records the optimal
biomass at each grid node, producing a 2-D surface suitable for heatmap /
contour plotting. The feasible range of each axis reaction is obtained with
flux variability analysis (fraction_of_optimum = 0), then each node fixes both
reaction fluxes and re-optimises biomass.

This reproduces the phenotypic phase-plane analysis of Edwards JS, Ibarra RU &
Palsson BØ (2001) *Nat Biotechnol* 19, 125 (COBRApy historically exposed this
as ``phenotypic_phase_plane`` / ``production_envelope``). Default axes are
glucose uptake (x) versus oxygen uptake (y).
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from cobra.exceptions import Infeasible
from cobra.flux_analysis import flux_variability_analysis

from api import state
from api.auth import require_api_key
from logging_setup import get_logger

logger = get_logger("api.ppp")
router = APIRouter(tags=["fba"])

# Sensible default sweep windows for the canonical nutrient exchanges.
_DEFAULT_RANGES = {
    state.RXN_GLUCOSE: (-20.0, 0.0),
    state.RXN_OXYGEN:  (-5.0, 0.0),
    state.RXN_NH4:     (-10.0, 0.0),
    state.RXN_PI:      (-10.0, 0.0),
}


class PhasePlaneRequest(BaseModel):
    x_axis_reaction: str = Field(state.RXN_GLUCOSE, description="Reaction ID for the x-axis.")
    y_axis_reaction: str = Field(state.RXN_OXYGEN,  description="Reaction ID for the y-axis.")
    n_points: int = Field(20, ge=5, le=30, description="Grid resolution per axis.")


def _axis_range(model, rid: str) -> tuple[float, float]:
    if rid in _DEFAULT_RANGES:
        return _DEFAULT_RANGES[rid]
    try:
        fva = flux_variability_analysis(model, [rid], fraction_of_optimum=0.0, loopless=False)
        lo, hi = float(fva.loc[rid, "minimum"]), float(fva.loc[rid, "maximum"])
        # Clamp pathological open bounds.
        return max(lo, -50.0), min(hi, 50.0)
    except Exception:
        rxn = model.reactions.get_by_id(rid)
        return max(rxn.lower_bound, -50.0), min(rxn.upper_bound, 50.0)


@router.post("/fba/phase_plane", dependencies=[Depends(require_api_key)])
def phase_plane(req: PhasePlaneRequest) -> dict:
    """Compute a biomass phenotype phase plane over two reactions."""
    model = state.get_gem()
    for rid in (req.x_axis_reaction, req.y_axis_reaction):
        if rid not in model.reactions:
            raise HTTPException(404, {"error": "reaction not found", "detail": rid})

    try:
        with model:
            xlo, xhi = _axis_range(model, req.x_axis_reaction)
            ylo, yhi = _axis_range(model, req.y_axis_reaction)
            xs = np.linspace(xlo, xhi, req.n_points)
            ys = np.linspace(ylo, yhi, req.n_points)
            xr = model.reactions.get_by_id(req.x_axis_reaction)
            yr = model.reactions.get_by_id(req.y_axis_reaction)

            grid = []          # rows indexed by y, columns by x
            for yv in ys:
                row = []
                for xv in xs:
                    with model:
                        xr.bounds = (float(xv), float(xv))
                        yr.bounds = (float(yv), float(yv))
                        sol = model.optimize()
                        row.append(round(float(sol.objective_value), 6)
                                   if sol.status == "optimal" else 0.0)
                grid.append(row)
    except Infeasible:
        raise                                         # → global 422 handler
    except HTTPException:
        raise
    except Exception as exc:                          # pragma: no cover
        logger.exception("phase plane error")
        raise HTTPException(500, {"error": "phase plane error", "detail": str(exc)})

    logger.info("phase_plane %s × %s (%d²)", req.x_axis_reaction,
                req.y_axis_reaction, req.n_points)
    return {
        "method": "Phenotype Phase Plane (Edwards, Ibarra & Palsson 2001)",
        "x_axis_reaction": req.x_axis_reaction,
        "y_axis_reaction": req.y_axis_reaction,
        "x_name": model.reactions.get_by_id(req.x_axis_reaction).name,
        "y_name": model.reactions.get_by_id(req.y_axis_reaction).name,
        "x_values": [round(float(v), 4) for v in xs],
        "y_values": [round(float(v), 4) for v in ys],
        "growth_grid": grid,
        "operating_point": {"x": -1.65, "y": -2.0},
        "n_points": req.n_points,
    }

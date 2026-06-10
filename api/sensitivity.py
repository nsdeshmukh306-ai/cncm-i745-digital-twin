"""
A1 — Global sensitivity analysis over the four nutrient inputs
(glucose, oxygen, NH4, Pi) of the CNCM I-745 digital twin.

Two complementary methods are exposed, both evaluated cheaply through the CNN
growth surrogate (``surrogate_model_v2.pt``) rather than full FBA:

* **Morris one-at-a-time (OAT) screening** — elementary-effects method giving
  ``mu_star`` (mean absolute effect, importance ranking) and ``sigma``
  (interaction / non-linearity).  Morris MD (1991) *Technometrics* 33, 161;
  Campolongo F et al. (2007) *Environ Model Softw* 22, 1509.
* **Sobol variance decomposition** — first-order (``S1``) and total-order
  (``ST``) indices.  Sobol IM (2001) *Math Comput Simul* 55, 271; Saltelli A
  et al. (2010) *Comput Phys Commun* 181, 259.

Both are implemented with SALib (Herman & Usher 2017, *JOSS* 2, 97).
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from SALib.sample.morris import sample as morris_sample
from SALib.analyze.morris import analyze as morris_analyze
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze.sobol import analyze as sobol_analyze

from api import state
from api.auth import require_api_key
from api.metrics import SURROGATE_PREDICTIONS_TOTAL
from logging_setup import get_logger

logger = get_logger("api.sensitivity")
router = APIRouter(prefix="/sensitivity", tags=["sensitivity"])

# Input ranges (mmol gDW⁻¹ hr⁻¹) — consistent with the hardened Pydantic models.
PARAM_NAMES  = ["glucose", "oxygen", "nh4", "pi"]
PARAM_BOUNDS = [[-20.0, 0.0], [-5.0, 0.0], [-10.0, 0.0], [-10.0, 0.0]]


def _problem() -> dict:
    return {"num_vars": 4, "names": PARAM_NAMES, "bounds": [list(b) for b in PARAM_BOUNDS]}


class MorrisRequest(BaseModel):
    trajectories: int = Field(20, ge=4, le=200,
                              description="Number of Morris trajectories (r).")
    num_levels:   int = Field(4,  ge=2, le=10, description="Grid levels (p).")
    gut_zone:     str = Field("none", description="Optional gut zone context label.")


class SobolRequest(BaseModel):
    base_samples: int = Field(256, ge=16, le=4096,
                              description="Base sample size N (total = N·(2·D+2)).")
    gut_zone:     str = Field("none", description="Optional gut zone context label.")


@router.post("/morris", dependencies=[Depends(require_api_key)])
def morris(req: MorrisRequest) -> dict:
    """Morris elementary-effects screening of the four nutrient inputs.

    Returns a ranking by ``mu_star`` (descending). Higher ``mu_star`` means the
    input more strongly governs predicted growth; large ``sigma`` relative to
    ``mu_star`` indicates non-linear or interaction effects.
    """
    problem = _problem()
    X = morris_sample(problem, N=req.trajectories, num_levels=req.num_levels)
    Y = state.surrogate_batch(X, version="v2")
    SURROGATE_PREDICTIONS_TOTAL.inc(len(X))
    Si = morris_analyze(problem, X, Y, num_levels=req.num_levels, print_to_console=False)

    ranking = sorted(
        (
            {"parameter": name,
             "morris_mu_star": round(float(mu), 6),
             "morris_mu": round(float(mu0), 6),
             "morris_sigma": round(float(sig), 6)}
            for name, mu, mu0, sig in zip(Si["names"], Si["mu_star"], Si["mu"], Si["sigma"])
        ),
        key=lambda d: d["morris_mu_star"], reverse=True,
    )
    logger.info("Morris OAT: %d evaluations, top=%s", len(X), ranking[0]["parameter"])
    return {
        "method": "Morris one-at-a-time (elementary effects)",
        "citation": "Morris 1991 Technometrics; Campolongo et al. 2007",
        "evaluator": "CNN surrogate v2 (surrogate_model_v2.pt)",
        "n_evaluations": int(len(X)),
        "trajectories": req.trajectories,
        "num_levels": req.num_levels,
        "gut_zone": req.gut_zone,
        "parameter_bounds": dict(zip(PARAM_NAMES, PARAM_BOUNDS)),
        "ranking": ranking,
    }


@router.post("/sobol", dependencies=[Depends(require_api_key)])
def sobol(req: SobolRequest) -> dict:
    """Sobol variance-based first-order (S1) and total-order (ST) indices.

    ``S1`` is the fraction of output variance attributable to an input alone;
    ``ST`` additionally captures its interactions with the other inputs.
    """
    problem = _problem()
    X = sobol_sample(problem, req.base_samples, calc_second_order=False)
    Y = state.surrogate_batch(X, version="v2")
    SURROGATE_PREDICTIONS_TOTAL.inc(len(X))
    Si = sobol_analyze(problem, Y, calc_second_order=False, print_to_console=False)

    indices = sorted(
        (
            {"parameter": name,
             "S1": round(float(s1), 6), "S1_conf": round(float(s1c), 6),
             "ST": round(float(st), 6), "ST_conf": round(float(stc), 6)}
            for name, s1, s1c, st, stc in zip(
                problem["names"], Si["S1"], Si["S1_conf"], Si["ST"], Si["ST_conf"])
        ),
        key=lambda d: d["ST"], reverse=True,
    )
    logger.info("Sobol: %d evaluations, top=%s", len(X), indices[0]["parameter"])
    return {
        "method": "Sobol variance decomposition (first + total order)",
        "citation": "Sobol 2001; Saltelli et al. 2010",
        "evaluator": "CNN surrogate v2 (surrogate_model_v2.pt)",
        "n_evaluations": int(len(X)),
        "base_samples": req.base_samples,
        "gut_zone": req.gut_zone,
        "parameter_bounds": dict(zip(PARAM_NAMES, PARAM_BOUNDS)),
        "indices": indices,
    }

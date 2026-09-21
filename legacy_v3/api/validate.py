"""
A7 — Validation endpoints (public GET).

* ``GET /validate/gem`` runs ``cobra.io.validate_sbml_model`` on the
  strain-specific SBML and returns structured error / warning counts.
* ``GET /validate/surrogate`` benchmarks the CNN surrogate against full FBA on a
  fresh random sample and returns live MAE, RMSE and R².

References: Hucka M et al. (2003) SBML, *Bioinformatics* 19, 524 (validation);
Ebrahim A et al. (2013) COBRApy, *BMC Syst Biol* 7, 74.
"""

from __future__ import annotations

import io
import math
import contextlib

import numpy as np
from fastapi import APIRouter, HTTPException
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from api import state
from logging_setup import get_logger

logger = get_logger("api.validate")
router = APIRouter(prefix="/validate", tags=["validation"])

_ERROR_KEYS = ("SBML_FATAL", "SBML_ERROR", "SBML_SCHEMA_ERROR", "COBRA_FATAL", "COBRA_ERROR")
_WARN_KEYS = ("SBML_WARNING", "COBRA_WARNING", "COBRA_CHECK")


@router.get("/gem")
def validate_gem() -> dict:
    """Validate the strain-specific SBML model and summarise issues."""
    if not state.GEM_PATH.exists():
        raise HTTPException(404, {"error": "GEM not found", "detail": state.GEM_PATH.name})
    from cobra.io import validate_sbml_model
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        model, errors = validate_sbml_model(str(state.GEM_PATH))

    errors = errors or {}
    err_count = sum(len(errors.get(k, [])) for k in _ERROR_KEYS)
    warn_count = sum(len(errors.get(k, [])) for k in _WARN_KEYS)
    flat = {k: v for k, v in errors.items() if v}
    logger.info("validate_gem: errors=%d warnings=%d valid=%s",
                err_count, warn_count, model is not None)
    return {
        "model_file": state.GEM_PATH.name,
        "valid": model is not None and err_count == 0,
        "loaded": model is not None,
        "error_count": err_count,
        "warning_count": warn_count,
        "total_messages": err_count + warn_count,
        "messages": {k: (v[:10] if isinstance(v, list) else v) for k, v in flat.items()},
        "reactions": len(model.reactions) if model else None,
        "metabolites": len(model.metabolites) if model else None,
        "genes": len(model.genes) if model else None,
    }


@router.get("/surrogate")
def validate_surrogate(n: int = 50, seed: int = 7) -> dict:
    """Benchmark surrogate vs full FBA on ``n`` fresh random nutrient samples."""
    n = max(5, min(n, 200))
    rng = np.random.default_rng(seed)
    glucose = rng.uniform(-20.0, -0.1, n)
    oxygen = rng.uniform(-5.0, 0.0, n)
    nh4 = rng.uniform(-10.0, -0.1, n)
    pi = rng.uniform(-2.0, -0.1, n)
    X = np.column_stack([glucose, oxygen, nh4, pi]).astype(np.float32)

    model = state.get_gem()
    fba_y = []
    with model:
        for g, o, a, p in X:
            with model:
                model.reactions.get_by_id(state.RXN_GLUCOSE).lower_bound = float(g)
                model.reactions.get_by_id(state.RXN_OXYGEN).lower_bound = float(o)
                model.reactions.get_by_id(state.RXN_NH4).lower_bound = float(a)
                model.reactions.get_by_id(state.RXN_PI).lower_bound = float(p)
                sol = model.optimize()
                fba_y.append(float(sol.objective_value) if sol.status == "optimal" else 0.0)
    fba_y = np.array(fba_y)

    try:
        surr_y = state.surrogate_batch(X, version="v2")
    except Exception as exc:
        raise HTTPException(500, {"error": "surrogate error", "detail": str(exc)})

    mae = float(mean_absolute_error(fba_y, surr_y))
    rmse = float(math.sqrt(mean_squared_error(fba_y, surr_y)))
    r2 = float(r2_score(fba_y, surr_y)) if np.ptp(fba_y) > 1e-9 else float("nan")
    logger.info("validate_surrogate: n=%d MAE=%.5f RMSE=%.5f R2=%.4f", n, mae, rmse, r2)
    return {
        "n_samples": n,
        "evaluator": "CNN surrogate v2 vs full FBA",
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 4) if not math.isnan(r2) else None,
        "fba_growth_mean": round(float(fba_y.mean()), 6),
        "surrogate_growth_mean": round(float(surr_y.mean()), 6),
    }

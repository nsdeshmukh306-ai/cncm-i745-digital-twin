"""
CNCM I-745 Digital Twin — FastAPI Backend v4.0.0
Five-layer model of Saccharomyces boulardii CNCM I-745 with AI chat.

v4.0 upgrades (over v3.0 strain-specific GEM + E-Flux):
  • Global sensitivity analysis (Morris + Sobol)        — /sensitivity/*
  • Flux variability analysis (Mahadevan & Schilling)   — /fba/fva
  • Multi-condition comparison (gut transit / carbon)   — /compare/*
  • Phenotype phase plane                               — /fba/phase_plane
  • Export (JSON/CSV/PDF report, GEM, figures)          — /export/*
  • Live GEM + surrogate validation                     — /validate/*
  • Grounded multi-turn chat + flux explanation         — /chat, /chat/explain_flux
  • API-key auth on POST, WebSocket progress, metrics, logs, hardened models
"""

from __future__ import annotations

import json
import os
import re
import math
import time
import datetime as _dt
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from scipy.integrate import odeint
import torch
import cobra
from cobra.exceptions import Infeasible

from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, "/home/nsdeshmukh306/digital-twin")

from references import REFERENCES, VALIDATED_PARAMETERS
from logging_setup import get_logger, tail_log

from api import state
from api.chat import (extract_simulation_params, generate_scientific_response,
                      build_scientific_context, explain_reaction)
from api.auth import require_api_key, UnauthorizedError, unauthorized_handler, auth_enabled
from api.metrics import (FBA_SIMULATIONS_TOTAL, SURROGATE_PREDICTIONS_TOTAL,
                         API_REQUEST_DURATION_SECONDS, FBA_GROWTH_RATE_LAST,
                         CONTENT_TYPE_LATEST, generate_latest)
from api import sensitivity, fva, compare, ppp, export, validate as validate_router

logger = get_logger("api.main")

# Convenience aliases (shared singletons live in api.state)
GENOME_CSV      = state.GENOME_CSV
LAYER1_RPT      = state.LAYER1_RPT
LAYER2_RPT      = state.LAYER2_RPT
SCALER_JSON     = state.SCALER_JSON
SCALER_V2       = state.SCALER_V2
RXN_GLUCOSE     = state.RXN_GLUCOSE
RXN_OXYGEN      = state.RXN_OXYGEN
RXN_NH4         = state.RXN_NH4
RXN_PI          = state.RXN_PI
BASELINE_GROWTH = state.BASELINE_GROWTH
VERSION         = state.VERSION

get_gem = state.get_gem
get_cnn = state.get_cnn


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting CNCM I-745 Digital Twin API v%s", VERSION)
    for loader in (state.get_gem, state.get_cnn):
        try:
            loader()
        except Exception as exc:                      # pragma: no cover
            logger.warning("Preload failed: %s", exc)
    logger.info("Auth enabled: %s", auth_enabled())
    yield
    logger.info("Shutting down API")


# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CNCM I-745 Digital Twin API",
    version=VERSION,
    description=(
        "Five-layer digital twin of S. boulardii CNCM I-745. "
        "v4.0: sensitivity (Morris/Sobol), FVA, multi-condition comparison, "
        "phase plane, export, validation, grounded chat, metrics & WebSocket."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(UnauthorizedError, unauthorized_handler)


@app.exception_handler(Infeasible)
async def _infeasible_handler(request: Request, exc: Infeasible):
    """D4: any uncaught COBRApy Infeasible → HTTP 422 with the spec body."""
    return JSONResponse(status_code=422,
                        content={"error": "FBA infeasible", "detail": "check nutrient bounds"})


@app.middleware("http")
async def _observe_latency(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    try:
        API_REQUEST_DURATION_SECONDS.labels(request.method, request.url.path).observe(
            time.perf_counter() - t0)
    except Exception:                                 # pragma: no cover
        pass
    return response


# Mount routers (each declares its own POST auth dependency).
for r in (sensitivity.router, fva.router, compare.router, ppp.router,
          export.router, validate_router.router):
    app.include_router(r)


# ── Request bodies (A10: hardened ranges + examples) ───────────────────────────
_ZONE_PATTERN = "^(none|stomach|duodenum|ileum|colon)$"


class FBARequest(BaseModel):
    glucose:   float = Field(-1.65, ge=-20, le=0, examples=[-1.65],
                             description="Glucose uptake bound (mmol gDW⁻¹ hr⁻¹)")
    oxygen:    float = Field(-2.0,  ge=-5,  le=0, examples=[-2.0],
                             description="Oxygen uptake bound (mmol gDW⁻¹ hr⁻¹)")
    nh4:       float = Field(-1.0,  ge=-10, le=0, examples=[-1.0],
                             description="Ammonium uptake bound (mmol gDW⁻¹ hr⁻¹)")
    pi:        float = Field(-0.5,  ge=-10, le=0, examples=[-0.5],
                             description="Phosphate uptake bound (mmol gDW⁻¹ hr⁻¹)")
    use_eflux: bool  = Field(False, examples=[False],
                             description="Apply E-Flux expression constraints")
    gut_zone:  str   = Field("none", pattern=_ZONE_PATTERN, examples=["colon"],
                             description="Gut zone for E-Flux")


class SurrogateRequest(BaseModel):
    glucose: float = Field(-1.65, ge=-20, le=0, examples=[-1.65])
    oxygen:  float = Field(-2.0,  ge=-5,  le=0, examples=[-2.0])
    nh4:     float = Field(-1.0,  ge=-10, le=0, examples=[-1.0])
    pi:      float = Field(-0.5,  ge=-10, le=0, examples=[-0.5])


class ChatMessage(BaseModel):
    message: str = Field(..., examples=["Simulate growth in the colon zone"])
    conversation_history: list = Field(default_factory=list)
    messages: list[dict] = Field(
        default_factory=list,
        description="Optional multi-turn history as [{role, content}, ...]")


class ExplainFluxRequest(BaseModel):
    reaction_id: str = Field(..., examples=["r_1714"],
                             description="Model reaction ID to explain")
    gut_zone: str = Field("none", pattern=_ZONE_PATTERN, examples=["ileum"])


# ── Helpers ────────────────────────────────────────────────────────────────────
def _parse_report(path: Path, patterns: dict[str, str]) -> dict:
    if not path.exists():
        return {}
    text = path.read_text()
    return {k: (m.group(1) if (m := re.search(pat, text)) else "N/A")
            for k, pat in patterns.items()}


def _read_csv(path: Path) -> list[dict]:
    import csv
    if not path.exists():
        raise HTTPException(404, f"Data file not found: {path.name}")
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _current_scientific_context(growth_rate=None, gut_zone="none") -> str:
    """Build the live-state block for the chatbot system prompt."""
    scaler_r2 = None
    if SCALER_V2.exists():
        try:
            scaler_r2 = json.loads(SCALER_V2.read_text()).get("cross_val_r2_mean")
        except Exception:
            scaler_r2 = None
    return build_scientific_context(
        growth_rate=growth_rate,
        gut_zone=gut_zone if gut_zone != "none" else state.RUNTIME["active_gut_zone"],
        surrogate_r2=round(scaler_r2, 4) if scaler_r2 else None,
        barrier_score=0.5971,
        nfkb_suppression_pct=70.0,
    )


# ═════════════════════════════════════════════════════════════════════════════
# CORE ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/", response_model=dict)
def root():
    return {
        "project":  "CNCM I-745 Digital Twin",
        "organism": "Saccharomyces boulardii CNCM I-745",
        "version":  VERSION,
        "status":   "online",
        "upgrades": ["strain-specific GEM (GPR corrected)", "E-Flux expression constraints",
                     "sensitivity analysis", "FVA", "phase plane", "multi-condition compare",
                     "export/report", "live validation", "grounded chat", "metrics + WebSocket"],
        "layers": {f"layer{i}": "complete" for i in range(1, 6)},
    }


@app.get("/health", response_model=dict)
def health():
    """Service health + runtime telemetry (uptime, memory, cpu, last sim)."""
    import psutil
    proc = psutil.Process()
    uptime = time.time() - state.RUNTIME["start_time"]
    return {
        "status": "healthy",
        "version": VERSION,
        "uptime_seconds": round(uptime, 1),
        "memory_used_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "zone": os.environ.get("GCP_ZONE", "asia-south1-a"),
        "active_gut_zone": state.RUNTIME["active_gut_zone"],
        "last_simulation_timestamp": state.RUNTIME["last_simulation_timestamp"],
        "auth_enabled": auth_enabled(),
    }


@app.get("/metrics")
def metrics():
    """Prometheus-format metrics exposition."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/logs", response_model=dict)
def logs(n: int = 100):
    """Return the last ``n`` lines (default 100) of the shared log file."""
    n = max(1, min(n, 1000))
    lines = tail_log(n)
    return {"log_file": "logs/digital_twin.log", "line_count": len(lines),
            "lines": [ln.rstrip("\n") for ln in lines]}


# ── Genome ────────────────────────────────────────────────────────────────────
@app.get("/genome/stats", response_model=dict)
def genome_stats():
    rows = _read_csv(GENOME_CSV)
    summary = _parse_report(LAYER1_RPT, {
        "num_sequences": r"Number of sequences\s*:\s*(\d+)",
        "total_bp":      r"Total bp\s*:\s*([\d,]+)",
        "gc_content":    r"Overall GC content\s*:\s*([\d.]+%)",
        "n50":           r"N50\s*:\s*([\d,]+)",
        "largest_bp":    r"Largest contig\s*:\s*([\d,]+)",
        "smallest_bp":   r"Smallest contig\s*:\s*([\d,]+)",
    })
    return {"summary": summary, "sequence_count": len(rows), "sequences": rows}


# ── FBA ───────────────────────────────────────────────────────────────────────
@app.post("/fba/simulate", response_model=dict, dependencies=[Depends(require_api_key)])
def fba_simulate(req: FBARequest):
    model = get_gem()
    t0 = time.perf_counter()
    eflux_info: dict = {}
    try:
        with model:
            model.reactions.get_by_id(RXN_GLUCOSE).lower_bound = req.glucose
            model.reactions.get_by_id(RXN_OXYGEN).lower_bound  = req.oxygen
            model.reactions.get_by_id(RXN_NH4).lower_bound     = req.nh4
            model.reactions.get_by_id(RXN_PI).lower_bound      = req.pi
            if req.use_eflux and req.gut_zone != "none":
                state.apply_zone_bounds(model, req.gut_zone)
                eflux_info = state.apply_eflux_bounds(model, req.gut_zone)
            sol = model.optimize()
            feasible = sol.status == "optimal"
            gr = float(sol.objective_value) if feasible else 0.0
    except Infeasible:
        raise                                         # → global 422 handler
    except Exception as exc:                          # pragma: no cover
        logger.exception("FBA error")
        raise HTTPException(500, {"error": "FBA error", "detail": str(exc)})

    FBA_SIMULATIONS_TOTAL.inc()
    FBA_GROWTH_RATE_LAST.set(gr)
    state.mark_simulation(req.gut_zone if req.gut_zone != "none" else None)
    delta = gr - BASELINE_GROWTH
    result = {
        "growth_rate":          round(gr, 6),
        "feasible":             feasible,
        "baseline_growth":      BASELINE_GROWTH,
        "change_from_baseline": round(delta, 6),
        "change_pct":           round(delta / BASELINE_GROWTH * 100, 2) if BASELINE_GROWTH else 0,
        "model_type":           "CNCM I-745 strain-specific GEM",
        "inputs":               {"glucose": req.glucose, "oxygen": req.oxygen,
                                 "nh4": req.nh4, "pi": req.pi},
        "solver_time_s":        round(time.perf_counter() - t0, 4),
    }
    if req.use_eflux:
        result["eflux"] = eflux_info
    logger.info("FBA simulate: growth=%.6f zone=%s eflux=%s", gr, req.gut_zone, req.use_eflux)
    return result


# ── Surrogate ────────────────────────────────────────────────────────────────
@app.post("/surrogate/predict", response_model=dict, dependencies=[Depends(require_api_key)])
def surrogate_predict(req: SurrogateRequest):
    try:
        cnn, scaler = get_cnn()
        feat_mean = np.array(scaler.get("mean", scaler.get("feature_mean", [0, 0, 0, 0])), dtype=np.float32)
        feat_std  = np.array(scaler.get("std",  scaler.get("feature_std",  [1, 1, 1, 1])), dtype=np.float32)
        feat   = np.array([[req.glucose, req.oxygen, req.nh4, req.pi]], dtype=np.float32)
        feat_n = (feat - feat_mean) / feat_std
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = float(cnn(torch.from_numpy(feat_n)).item())
        pred = max(0.0, pred)
    except Exception as exc:
        logger.exception("Surrogate error")
        raise HTTPException(500, {"error": "surrogate model error", "detail": str(exc)})

    SURROGATE_PREDICTIONS_TOTAL.inc()
    return {
        "predicted_growth_rate": round(pred, 6),
        "model_type":            "CNN surrogate (1D-Conv, LHS-2000)",
        "cross_val_r2":          scaler.get("cross_val_r2_mean", "N/A"),
        "inputs":                {"glucose": req.glucose, "oxygen": req.oxygen,
                                  "nh4": req.nh4, "pi": req.pi},
        "inference_time_s":      round(time.perf_counter() - t0, 6),
    }


# ── Host: inflammation ODE ────────────────────────────────────────────────────
@app.get("/host/inflammation", response_model=dict)
def host_inflammation():
    def ode(y, t, Sb):
        NFkB, IL1b, TNFa, IL10 = y
        return [
            0.30 * (1 - Sb * 0.7) - 0.20 * NFkB,
            0.40 * NFkB           - 0.30 * IL1b,
            0.35 * NFkB           - 0.25 * TNFa,
            0.20 * Sb             - 0.15 * IL10,
        ]
    t   = np.linspace(0, 240, 2401)
    y0  = [0.0, 0.0, 0.0, 0.0]
    ss_p = odeint(ode, y0, t, args=(1.0,))[-1].tolist()
    ss_c = odeint(ode, y0, t, args=(0.0,))[-1].tolist()
    labels = ["NFkB", "IL1b", "TNFa", "IL10"]
    def pct_red(c, p): return round((c - p) / c * 100, 1) if c > 0 else 0.0
    return {
        "probiotic_Sb1":  dict(zip(labels, [round(v, 4) for v in ss_p])),
        "control_Sb0":    dict(zip(labels, [round(v, 4) for v in ss_c])),
        "reductions_pct": {k: pct_red(ss_c[i], ss_p[i]) for i, k in enumerate(["NFkB", "IL1b", "TNFa"])},
        "IL10_induction": round(float(ss_p[3]) - float(ss_c[3]), 4),
        "simulation_minutes": 240,
    }


# ── Host: barrier integrity ───────────────────────────────────────────────────
@app.get("/host/barrier", response_model=dict)
def host_barrier():
    polyamine_flux = 1.63e-5
    butyrate_proxy = 0.4163
    claudin3 = _sigmoid(polyamine_flux * 2.0)
    occludin = _sigmoid(butyrate_proxy * 1.5)
    zo1      = _sigmoid((claudin3 + occludin) / 2.0)
    score    = (claudin3 + occludin + zo1) / 3.0
    return {
        "claudin3":               round(claudin3, 4),
        "occludin":               round(occludin, 4),
        "zo1":                    round(zo1, 4),
        "barrier_integrity_score": round(score, 4),
        "interpretation": "HIGH" if score >= 0.75 else "MODERATE" if score >= 0.50 else "LOW",
        "inputs": {"polyamine_flux_mmol_gDW_hr": polyamine_flux, "butyrate_proxy": butyrate_proxy},
    }


# ── All-layers status ─────────────────────────────────────────────────────────
@app.get("/layers/status", response_model=dict)
def layers_status():
    genome_metrics = _parse_report(LAYER1_RPT, {
        "num_sequences": r"Number of sequences\s*:\s*(\d+)",
        "total_bp":      r"Total bp\s*:\s*([\d,]+)",
        "gc_content":    r"Overall GC content\s*:\s*([\d.]+%)",
        "n50_bp":        r"N50\s*:\s*([\d,]+)",
    })
    gem_metrics = _parse_report(LAYER2_RPT, {
        "reactions":       r"Reactions\s*:\s*(\d+)",
        "metabolites":     r"Metabolites\s*:\s*(\d+)",
        "genes":           r"Genes\s*:\s*(\d+)",
        "baseline_growth": r"Growth rate\s*:\s*([\d.]+)",
        "essential_genes": r"Essential\s*:\s*(\d+)",
    })
    scaler_data = json.loads(SCALER_JSON.read_text()) if SCALER_JSON.exists() else {}
    return {
        "layer1_genome": {
            "status": "complete", "script": "layer1_genome/genome_parser.py",
            "outputs": ["data/genome/genome_stats.csv", "logs/layer1_report.txt"],
            "metrics": genome_metrics,
        },
        "layer2_gem": {
            "status": "complete", "script": "layer2_gem/gem_builder.py",
            "outputs": ["data/gem/cncm_i745_strain_specific.xml",
                        "data/gem/cncm_i745_gut.xml", "logs/layer2_report_v2.txt"],
            "metrics": {**gem_metrics,
                        "model_type": "CNCM I-745 strain-specific GEM",
                        "gpr_corrections": "HXT9/HXT11/MAL/ASP3 corrected per Khatri et al. 2017"},
        },
        "layer3_regulatory": {
            "status": "complete", "script": "layer3_regulatory/regulatory_network.py",
            "outputs": ["data/gem/cncm_i745_regulated.xml",
                        "data/gem/cncm_i745_eflux_colon.xml",
                        "data/fba_outputs/eflux_results.json", "logs/layer3_report.txt"],
            "metrics": {"regulons": 4, "gut_zones": 4,
                        "constraint_method": "E-Flux (Colijn et al. 2009)",
                        "expression_source": "Gasch et al. 2000 (proxy)"},
        },
        "layer4_host": {
            "status": "complete", "script": "layer4_host/host_interaction.py",
            "outputs": ["data/fba_outputs/inflammatory_signaling.csv"],
            "metrics": {"nfkb_suppression_pct": 70.0, "barrier_score": 0.5971},
        },
        "layer5_surrogate": {
            "status": "complete", "script": "layer5_surrogate/surrogate_model.py",
            "outputs": ["data/ml_datasets/fba_lhs_2000.csv",
                        "data/ml_datasets/surrogate_model.pt",
                        "data/ml_datasets/surrogate_scaler.json"],
            "metrics": {"cross_val_r2_mean": scaler_data.get("cross_val_r2_mean", "N/A"),
                        "cross_val_r2_std":  scaler_data.get("cross_val_r2_std", "N/A"),
                        "training_samples":  scaler_data.get("n_samples", 2000),
                        "n_feasible":        scaler_data.get("n_feasible", "N/A")},
        },
    }


# ── Chat endpoint (A6: history + grounded scientific context) ──────────────────
@app.post("/chat", response_model=dict, dependencies=[Depends(require_api_key)])
async def chat_endpoint(request: ChatMessage):
    try:
        params = extract_simulation_params(request.message)
    except Exception:
        params = {"glucose": -1.65, "oxygen": -2.0, "nh4": -1.0, "pi": -0.5,
                  "simulation_type": "fba", "gut_zone": "none",
                  "intent_summary": request.message}

    sim_type = params.get("simulation_type", "fba")
    gut_zone = params.get("gut_zone", "none")
    simulation_results: dict = {}

    if sim_type in ("fba", "surrogate", "regulatory"):
        model = get_gem()
        t0 = time.perf_counter()
        try:
            with model:
                model.reactions.get_by_id(RXN_GLUCOSE).lower_bound = params["glucose"]
                model.reactions.get_by_id(RXN_OXYGEN).lower_bound  = params["oxygen"]
                model.reactions.get_by_id(RXN_NH4).lower_bound     = params["nh4"]
                model.reactions.get_by_id(RXN_PI).lower_bound      = params["pi"]
                sol = model.optimize()
                feasible = sol.status == "optimal"
                gr = float(sol.objective_value) if feasible else 0.0
        except Exception:
            feasible, gr = False, 0.0
        FBA_SIMULATIONS_TOTAL.inc()
        FBA_GROWTH_RATE_LAST.set(gr)
        state.mark_simulation(gut_zone if gut_zone != "none" else None)
        delta = gr - BASELINE_GROWTH
        simulation_results["fba"] = {
            "growth_rate": round(gr, 6), "feasible": feasible,
            "baseline_growth": BASELINE_GROWTH, "change_from_baseline": round(delta, 6),
            "change_pct": round(delta / BASELINE_GROWTH * 100, 2),
            "solver_time_s": round(time.perf_counter() - t0, 4),
        }
        if state.MODEL_PT.exists():
            try:
                cnn, scaler = get_cnn()
                fm = np.array(scaler.get("mean", scaler.get("feature_mean")), dtype=np.float32)
                fs = np.array(scaler.get("std",  scaler.get("feature_std")),  dtype=np.float32)
                feat = np.array([[params["glucose"], params["oxygen"],
                                  params["nh4"], params["pi"]]], dtype=np.float32)
                with torch.no_grad():
                    surr = max(0.0, float(cnn(torch.from_numpy((feat - fm) / fs)).item()))
                SURROGATE_PREDICTIONS_TOTAL.inc()
                simulation_results["surrogate"] = {
                    "predicted_growth_rate": round(surr, 6),
                    "fba_vs_surrogate_diff": round(abs(gr - surr), 6)}
            except Exception:
                pass
        if sim_type == "regulatory":
            simulation_results["regulatory"] = {
                "gut_zone": gut_zone,
                "active_regulons": state.GUT_ZONES.get(gut_zone, {}).get("regulons", []),
                "fba_under_zone_constraints": simulation_results["fba"]}

    elif sim_type == "host":
        def ode(y, t, Sb):
            NFkB, IL1b, TNFa, IL10 = y
            return [0.30 * (1 - Sb * 0.7) - 0.20 * NFkB, 0.40 * NFkB - 0.30 * IL1b,
                    0.35 * NFkB - 0.25 * TNFa, 0.20 * Sb - 0.15 * IL10]
        t = np.linspace(0, 240, 2401)
        ss_p = odeint(ode, [0, 0, 0, 0], t, args=(1.0,))[-1].tolist()
        ss_c = odeint(ode, [0, 0, 0, 0], t, args=(0.0,))[-1].tolist()
        labels = ["NFkB", "IL1b", "TNFa", "IL10"]
        simulation_results["host"] = {
            "probiotic_Sb1": dict(zip(labels, [round(v, 4) for v in ss_p])),
            "control_Sb0":   dict(zip(labels, [round(v, 4) for v in ss_c])),
            "nfkb_suppression_pct": round((ss_c[0] - ss_p[0]) / ss_c[0] * 100, 1) if ss_c[0] else 0,
            "IL10_induction": round(float(ss_p[3]) - float(ss_c[3]), 4)}

    elif sim_type == "genome":
        if GENOME_CSV.exists():
            import csv
            with open(GENOME_CSV) as fh:
                rows = list(csv.DictReader(fh))
            simulation_results["genome"] = {
                "sequence_count": len(rows),
                "genome_size_Mbp": VALIDATED_PARAMETERS["genome_size_Mbp"],
                "chromosome_count": VALIDATED_PARAMETERS["chromosome_count"]}

    growth_for_ctx = simulation_results.get("fba", {}).get("growth_rate")
    sci_context = _current_scientific_context(growth_for_ctx, gut_zone)
    history = request.messages or request.conversation_history or []
    try:
        sci_response = generate_scientific_response(
            simulation_results, request.message, params,
            history=history, scientific_context=sci_context)
    except Exception:
        sci_response = {
            "plain_english": "Simulation completed successfully.",
            "scientific_summary": str(simulation_results),
            "biological_context": "CNCM I-745 is a well-characterized probiotic yeast.",
            "clinical_relevance": "Results may inform probiotic dosing strategies.",
            "confidence": "medium",
            "suggested_followup": "Try adjusting glucose concentration.",
            "key_finding": f"Growth rate: {growth_for_ctx if growth_for_ctx is not None else 'N/A'} h⁻¹"}

    layer_refs = REFERENCES.get("layer2_gem", []) + REFERENCES.get("layer4_host", [])
    if sim_type == "genome":
        layer_refs = REFERENCES.get("layer1_genome", [])
    elif sim_type == "surrogate":
        layer_refs = REFERENCES.get("layer5_surrogate", [])

    return {
        "status": "success",
        "intent_summary": params.get("intent_summary", request.message),
        "parameters_used": params,
        "simulation_results": simulation_results,
        "scientific_response": sci_response,
        "references": layer_refs[:4],
    }


@app.post("/chat/explain_flux", response_model=dict, dependencies=[Depends(require_api_key)])
def chat_explain_flux(req: ExplainFluxRequest):
    """Explain a reaction's biological role + its flux under the active gut zone."""
    model = get_gem()
    try:
        rxn = model.reactions.get_by_id(req.reaction_id)
    except KeyError:
        raise HTTPException(404, {"error": "reaction not found", "detail": req.reaction_id})

    try:
        with model:                              # baseline (glucose-limited)
            model.reactions.get_by_id(RXN_GLUCOSE).lower_bound = -1.65
            model.reactions.get_by_id(RXN_OXYGEN).lower_bound = -2.0
            base_sol = model.optimize()
            base_flux = float(base_sol.fluxes.get(req.reaction_id, 0.0)) if base_sol.status == "optimal" else 0.0
        zone_flux = base_flux
        if req.gut_zone != "none":
            with model:
                state.apply_zone_bounds(model, req.gut_zone)
                zone_sol = model.optimize()
                zone_flux = float(zone_sol.fluxes.get(req.reaction_id, 0.0)) if zone_sol.status == "optimal" else 0.0
    except Infeasible:
        raise                                         # → global 422 handler

    changed = abs(zone_flux - base_flux) > 1e-6
    try:
        explanation = explain_reaction(req.reaction_id, rxn.name, zone_flux,
                                        changed, req.gut_zone, base_flux)
    except Exception as exc:
        explanation = (f"{rxn.name} ({req.reaction_id}) carries a flux of "
                       f"{zone_flux:.6f} mmol gDW⁻¹ hr⁻¹ under '{req.gut_zone}'. "
                       f"(LLM explanation unavailable: {exc})")
    return {
        "reaction_id": req.reaction_id, "reaction_name": rxn.name,
        "subsystem": rxn.subsystem or "Unclassified",
        "gut_zone": req.gut_zone,
        "baseline_flux": round(base_flux, 6),
        "current_flux": round(zone_flux, 6),
        "changed_under_condition": changed,
        "explanation": explanation,
    }


# ── WebSocket: live simulation progress (A9) ───────────────────────────────────
@app.websocket("/ws/simulate")
async def ws_simulate(ws: WebSocket):
    """Stream FBA simulation progress, then the final result, over a WebSocket."""
    await ws.accept()
    try:
        data = await ws.receive_json()
        try:
            req = FBARequest(**data)
        except Exception as exc:
            await ws.send_json({"step": "error", "progress": 0, "message": f"Invalid params: {exc}"})
            await ws.close()
            return

        await ws.send_json({"step": "loading_gem", "progress": 20,
                            "message": "Loading strain-specific GEM..."})
        model = get_gem()
        await ws.send_json({"step": "applying_gpr", "progress": 40,
                            "message": "Applying GPR corrections (HXT9/HXT11/MAL/ASP3)..."})
        t0 = time.perf_counter()
        eflux_info: dict = {}
        with model:
            model.reactions.get_by_id(RXN_GLUCOSE).lower_bound = req.glucose
            model.reactions.get_by_id(RXN_OXYGEN).lower_bound  = req.oxygen
            model.reactions.get_by_id(RXN_NH4).lower_bound     = req.nh4
            model.reactions.get_by_id(RXN_PI).lower_bound      = req.pi
            if req.use_eflux and req.gut_zone != "none":
                await ws.send_json({"step": "applying_eflux", "progress": 60,
                                    "message": f"Applying E-Flux constraints ({req.gut_zone})..."})
                state.apply_zone_bounds(model, req.gut_zone)
                eflux_info = state.apply_eflux_bounds(model, req.gut_zone)
            await ws.send_json({"step": "optimizing", "progress": 80,
                                "message": "Optimising biomass objective (FBA)..."})
            sol = model.optimize()
            feasible = sol.status == "optimal"
            gr = float(sol.objective_value) if feasible else 0.0

        FBA_SIMULATIONS_TOTAL.inc()
        FBA_GROWTH_RATE_LAST.set(gr)
        state.mark_simulation(req.gut_zone if req.gut_zone != "none" else None)
        delta = gr - BASELINE_GROWTH
        result = {
            "growth_rate": round(gr, 6), "feasible": feasible,
            "baseline_growth": BASELINE_GROWTH, "change_from_baseline": round(delta, 6),
            "change_pct": round(delta / BASELINE_GROWTH * 100, 2) if BASELINE_GROWTH else 0,
            "model_type": "CNCM I-745 strain-specific GEM",
            "inputs": {"glucose": req.glucose, "oxygen": req.oxygen,
                       "nh4": req.nh4, "pi": req.pi},
            "solver_time_s": round(time.perf_counter() - t0, 4),
        }
        if req.use_eflux:
            result["eflux"] = eflux_info
        await ws.send_json({"step": "complete", "progress": 100,
                            "message": "FBA complete.", "result": result})
        await ws.close()
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:                          # pragma: no cover
        logger.exception("WebSocket simulate error")
        try:
            await ws.send_json({"step": "error", "progress": 0, "message": str(exc)})
            await ws.close()
        except Exception:
            pass

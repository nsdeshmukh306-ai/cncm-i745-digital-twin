"""
CNCM I-745 Digital Twin — FastAPI Backend v2.0.0
5-layer model with AI chat endpoint.
"""

from __future__ import annotations

import io
import json
import re
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from scipy.integrate import odeint
import torch
import torch.nn as nn
import cobra
from cobra.io import read_sbml_model

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.chat import extract_simulation_params, generate_scientific_response

import sys
sys.path.insert(0, "/home/nsdeshmukh306/digital-twin")
from references import REFERENCES, VALIDATED_PARAMETERS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path("/home/nsdeshmukh306/digital-twin")
GENOME_CSV   = BASE / "data/genome/genome_stats.csv"
LAYER1_RPT   = BASE / "logs/layer1_report.txt"
LAYER2_RPT   = BASE / "logs/layer2_report_v2.txt"
GEM_PATH     = BASE / "data/gem/cncm_i745_gut.xml"
MODEL_PT     = BASE / "data/ml_datasets/surrogate_model.pt"
SCALER_JSON  = BASE / "data/ml_datasets/surrogate_scaler.json"

RXN_GLUCOSE = "r_1714"
RXN_OXYGEN  = "r_1992"
RXN_NH4     = "r_1654"
RXN_PI      = "r_2005"

# ── Surrogate architecture ─────────────────────────────────────────────────────
SEQ_LEN = 32

class GrowthCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(64)
        self.fc1   = nn.Linear(64 * SEQ_LEN, 256)
        self.drop1 = nn.Dropout(0.3)
        self.fc2   = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.2)
        self.fc3   = nn.Linear(128, 1)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, 1, SEQ_LEN // x.size(1)).reshape(x.size(0), 1, SEQ_LEN)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x).squeeze(1)

# ── Application state ─────────────────────────────────────────────────────────
class _State:
    gem_model: cobra.Model | None = None
    cnn_model: GrowthCNN | None = None
    scaler: dict | None = None

_state = _State()

def _load_gem_silent() -> cobra.Model:
    buf = io.StringIO()
    import contextlib
    with contextlib.redirect_stderr(buf):
        return read_sbml_model(str(GEM_PATH))

def get_gem() -> cobra.Model:
    if _state.gem_model is None:
        if not GEM_PATH.exists():
            raise HTTPException(503, f"GEM file not found: {GEM_PATH.name}")
        _state.gem_model = _load_gem_silent()
    return _state.gem_model

def get_cnn() -> tuple[GrowthCNN, dict]:
    if _state.cnn_model is None:
        if not MODEL_PT.exists():
            raise HTTPException(503, f"Surrogate model not found: {MODEL_PT.name}")
        if not SCALER_JSON.exists():
            raise HTTPException(503, "Surrogate scaler not found.")
        with open(SCALER_JSON) as fh:
            scaler = json.load(fh)
        # Require new 4-feature model; reject old 3-feature scaler
        feature_names = scaler.get("feature_names", [])
        if len(feature_names) < 4 or "nh4" not in feature_names:
            raise HTTPException(503, "Surrogate model not yet upgraded to v2 (4 features). "
                                     "Run layer5_surrogate/surrogate_model.py to generate it.")
        m = GrowthCNN()
        m.load_state_dict(torch.load(MODEL_PT, map_location="cpu", weights_only=True))
        m.eval()
        _state.cnn_model = m
        _state.scaler = scaler
    return _state.cnn_model, _state.scaler

@asynccontextmanager
async def lifespan(app: FastAPI):
    for loader in (get_gem, get_cnn):
        try:
            loader()
        except Exception:
            pass
    yield

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CNCM I-745 Digital Twin API",
    version="2.0.0",
    description="5-layer computational model of Saccharomyces boulardii CNCM I-745 with AI chat",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request bodies ─────────────────────────────────────────────────────────────
class FBARequest(BaseModel):
    glucose:   float = Field(-1.65, le=0,  description="Glucose uptake bound (mmol/gDW/hr)")
    oxygen:    float = Field(-2.0,  le=0,  description="Oxygen uptake bound (mmol/gDW/hr)")
    nh4:       float = Field(-1.0,  le=0,  description="Ammonium uptake bound (mmol/gDW/hr)")
    pi:        float = Field(-0.5,  le=0,  description="Phosphate uptake bound (mmol/gDW/hr)")

class SurrogateRequest(BaseModel):
    glucose:   float = Field(-1.65, le=0)
    oxygen:    float = Field(-2.0,  le=0)
    nh4:       float = Field(-1.0,  le=0)
    pi:        float = Field(-0.5,  le=0)

class ChatMessage(BaseModel):
    message: str
    conversation_history: list = Field(default_factory=list)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _parse_report(path: Path, patterns: dict[str, str]) -> dict:
    if not path.exists():
        return {}
    text = path.read_text()
    result = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        result[key] = m.group(1) if m else "N/A"
    return result

def _read_csv(path: Path) -> list[dict]:
    import csv
    if not path.exists():
        raise HTTPException(404, f"Data file not found: {path.name}")
    with open(path) as fh:
        return list(csv.DictReader(fh))

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

BASELINE_GROWTH = 0.089786

# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "project":   "CNCM I-745 Digital Twin",
        "organism":  "Saccharomyces boulardii CNCM I-745",
        "version":   "2.0.0",
        "status":    "online",
        "layers": {
            "layer1_genome":     "complete",
            "layer2_gem":        "complete",
            "layer3_regulatory": "complete",
            "layer4_host":       "complete",
            "layer5_surrogate":  "complete",
        },
    }

@app.get("/health")
def health():
    return {"status": "healthy", "version": "2.0.0"}

# ── Genome ────────────────────────────────────────────────────────────────────
@app.get("/genome/stats")
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
    return {
        "summary":        summary,
        "sequence_count": len(rows),
        "sequences":      rows,
    }

# ── FBA ───────────────────────────────────────────────────────────────────────
@app.post("/fba/simulate")
def fba_simulate(req: FBARequest):
    model = get_gem()
    t0 = time.perf_counter()
    try:
        with model:
            model.reactions.get_by_id(RXN_GLUCOSE).lower_bound = req.glucose
            model.reactions.get_by_id(RXN_OXYGEN).lower_bound  = req.oxygen
            model.reactions.get_by_id(RXN_NH4).lower_bound     = req.nh4
            model.reactions.get_by_id(RXN_PI).lower_bound      = req.pi
            sol = model.optimize()
            feasible = sol.status == "optimal"
            gr = float(sol.objective_value) if feasible else 0.0
    except Exception as exc:
        raise HTTPException(500, f"FBA error: {exc}")
    delta = gr - BASELINE_GROWTH
    return {
        "growth_rate":        round(gr, 6),
        "feasible":           feasible,
        "baseline_growth":    BASELINE_GROWTH,
        "change_from_baseline": round(delta, 6),
        "change_pct":         round(delta / BASELINE_GROWTH * 100, 2) if BASELINE_GROWTH else 0,
        "inputs":             {"glucose": req.glucose, "oxygen": req.oxygen,
                               "nh4": req.nh4, "pi": req.pi},
        "solver_time_s":      round(time.perf_counter() - t0, 4),
    }

# ── Surrogate ────────────────────────────────────────────────────────────────
@app.post("/surrogate/predict")
def surrogate_predict(req: SurrogateRequest):
    cnn, scaler = get_cnn()
    feat_mean = np.array(scaler.get("mean", scaler.get("feature_mean", [0,0,0,0])), dtype=np.float32)
    feat_std  = np.array(scaler.get("std",  scaler.get("feature_std",  [1,1,1,1])), dtype=np.float32)
    feat   = np.array([[req.glucose, req.oxygen, req.nh4, req.pi]], dtype=np.float32)
    feat_n = (feat - feat_mean) / feat_std
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = float(cnn(torch.from_numpy(feat_n)).item())
    pred = max(0.0, pred)
    return {
        "predicted_growth_rate": round(pred, 6),
        "model_type":            "CNN surrogate (1D-Conv, LHS-2000)",
        "cross_val_r2":          scaler.get("cross_val_r2_mean", "N/A"),
        "inputs":                {"glucose": req.glucose, "oxygen": req.oxygen,
                                  "nh4": req.nh4, "pi": req.pi},
        "inference_time_s":      round(time.perf_counter() - t0, 6),
    }

# ── Host: inflammation ODE ────────────────────────────────────────────────────
@app.get("/host/inflammation")
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
        "reductions_pct": {k: pct_red(ss_c[i], ss_p[i]) for i, k in enumerate(["NFkB","IL1b","TNFa"])},
        "IL10_induction": round(float(ss_p[3]) - float(ss_c[3]), 4),
        "simulation_minutes": 240,
    }

# ── Host: barrier integrity ───────────────────────────────────────────────────
@app.get("/host/barrier")
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
@app.get("/layers/status")
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
    scaler_data = {}
    if SCALER_JSON.exists():
        with open(SCALER_JSON) as fh:
            scaler_data = json.load(fh)
    return {
        "layer1_genome": {
            "status": "complete",
            "script": "layer1_genome/genome_parser.py",
            "outputs": ["data/genome/genome_stats.csv", "logs/layer1_report.txt"],
            "metrics": genome_metrics,
        },
        "layer2_gem": {
            "status": "complete",
            "script": "layer2_gem/gem_builder.py",
            "outputs": ["data/gem/cncm_i745_gut.xml", "logs/layer2_report_v2.txt"],
            "metrics": gem_metrics,
        },
        "layer3_regulatory": {
            "status": "complete",
            "script": "layer3_regulatory/regulatory_network.py",
            "outputs": ["data/gem/cncm_i745_regulated.xml", "logs/layer3_report.txt"],
            "metrics": {"regulons": 4, "gut_zones": 4},
        },
        "layer4_host": {
            "status": "complete",
            "script": "layer4_host/host_interaction.py",
            "outputs": ["data/fba_outputs/inflammatory_signaling.csv"],
            "metrics": {
                "nfkb_suppression_pct": 70.0,
                "barrier_score": 0.5971,
            },
        },
        "layer5_surrogate": {
            "status": "complete",
            "script": "layer5_surrogate/surrogate_model.py",
            "outputs": ["data/ml_datasets/fba_lhs_2000.csv",
                        "data/ml_datasets/surrogate_model.pt",
                        "data/ml_datasets/surrogate_scaler.json"],
            "metrics": {
                "cross_val_r2_mean": scaler_data.get("cross_val_r2_mean", "N/A"),
                "cross_val_r2_std":  scaler_data.get("cross_val_r2_std", "N/A"),
                "training_samples":  scaler_data.get("n_samples", 2000),
                "n_feasible":        scaler_data.get("n_feasible", "N/A"),
            },
        },
    }

# ── Chat endpoint ─────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat_endpoint(request: ChatMessage):
    try:
        params = extract_simulation_params(request.message)
    except Exception as e:
        params = {
            "glucose": -1.65, "oxygen": -2.0, "nh4": -1.0, "pi": -0.5,
            "simulation_type": "fba", "gut_zone": "none",
            "intent_summary": request.message
        }

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
            feasible = False
            gr = 0.0
        delta = gr - BASELINE_GROWTH
        simulation_results["fba"] = {
            "growth_rate":          round(gr, 6),
            "feasible":             feasible,
            "baseline_growth":      BASELINE_GROWTH,
            "change_from_baseline": round(delta, 6),
            "change_pct":           round(delta / BASELINE_GROWTH * 100, 2),
            "solver_time_s":        round(time.perf_counter() - t0, 4),
        }

        if MODEL_PT.exists():
            try:
                cnn, scaler = get_cnn()
                feat_mean = np.array(scaler.get("mean", scaler.get("feature_mean")), dtype=np.float32)
                feat_std  = np.array(scaler.get("std",  scaler.get("feature_std")),  dtype=np.float32)
                feat = np.array([[params["glucose"], params["oxygen"],
                                  params["nh4"], params["pi"]]], dtype=np.float32)
                feat_n = (feat - feat_mean) / feat_std
                with torch.no_grad():
                    surr_pred = max(0.0, float(cnn(torch.from_numpy(feat_n)).item()))
                simulation_results["surrogate"] = {
                    "predicted_growth_rate": round(surr_pred, 6),
                    "fba_vs_surrogate_diff": round(abs(gr - surr_pred), 6),
                }
            except Exception:
                pass

        if sim_type == "regulatory":
            zone_regulons = {
                "stomach": ["HSR", "acid_response"],
                "duodenum": ["HOG", "aerobic_response"],
                "ileum": ["aerobic_response", "YAP1"],
                "colon": ["HSR", "HOG"],
                "none": [],
            }
            simulation_results["regulatory"] = {
                "gut_zone": gut_zone,
                "active_regulons": zone_regulons.get(gut_zone, []),
                "fba_under_zone_constraints": simulation_results["fba"],
            }

    elif sim_type == "host":
        def ode(y, t, Sb):
            NFkB, IL1b, TNFa, IL10 = y
            return [
                0.30 * (1 - Sb * 0.7) - 0.20 * NFkB,
                0.40 * NFkB           - 0.30 * IL1b,
                0.35 * NFkB           - 0.25 * TNFa,
                0.20 * Sb             - 0.15 * IL10,
            ]
        t  = np.linspace(0, 240, 2401)
        y0 = [0.0, 0.0, 0.0, 0.0]
        ss_p = odeint(ode, y0, t, args=(1.0,))[-1].tolist()
        ss_c = odeint(ode, y0, t, args=(0.0,))[-1].tolist()
        labels = ["NFkB", "IL1b", "TNFa", "IL10"]
        simulation_results["host"] = {
            "probiotic_Sb1": dict(zip(labels, [round(v, 4) for v in ss_p])),
            "control_Sb0":   dict(zip(labels, [round(v, 4) for v in ss_c])),
            "nfkb_suppression_pct": round((ss_c[0] - ss_p[0]) / ss_c[0] * 100, 1) if ss_c[0] else 0,
            "IL10_induction": round(float(ss_p[3]) - float(ss_c[3]), 4),
        }

    elif sim_type == "genome":
        if GENOME_CSV.exists():
            import csv
            with open(GENOME_CSV) as fh:
                rows = list(csv.DictReader(fh))
            simulation_results["genome"] = {
                "sequence_count": len(rows),
                "genome_size_Mbp": VALIDATED_PARAMETERS["genome_size_Mbp"],
                "chromosome_count": VALIDATED_PARAMETERS["chromosome_count"],
            }

    try:
        sci_response = generate_scientific_response(simulation_results, request.message, params)
    except Exception as e:
        sci_response = {
            "plain_english": "Simulation completed successfully.",
            "scientific_summary": str(simulation_results),
            "biological_context": "CNCM I-745 is a well-characterized probiotic yeast.",
            "clinical_relevance": "Results may inform probiotic dosing strategies.",
            "confidence": "medium",
            "suggested_followup": "Try adjusting glucose concentration.",
            "key_finding": f"Growth rate: {simulation_results.get('fba', {}).get('growth_rate', 'N/A')} h⁻¹",
        }

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

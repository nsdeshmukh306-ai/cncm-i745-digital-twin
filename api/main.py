"""
CNCM I-745 Digital Twin — FastAPI Backend  v1.0.0
All 5 layers exposed via REST endpoints.
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

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path("/home/nsdeshmukh306/digital-twin")
GENOME_CSV   = BASE / "data/genome/genome_stats.csv"
LAYER1_RPT   = BASE / "logs/layer1_report.txt"
LAYER2_RPT   = BASE / "logs/layer2_report.txt"
GEM_PATH     = BASE / "data/gem/cncm_i745_regulated.xml"
MODEL_PT     = BASE / "data/ml_datasets/surrogate_model.pt"
SCALER_JSON  = BASE / "data/ml_datasets/surrogate_scaler.json"

RXN_GLUCOSE = "r_1714"
RXN_OXYGEN  = "r_1992"

# ── Surrogate architecture (mirrors layer5 exactly) ───────────────────────────
REPEAT  = 8
SEQ_LEN = 3 * REPEAT   # 24

class GrowthCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(32 * SEQ_LEN, 128)
        self.drop  = nn.Dropout(0.2)
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, 1)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, 1, REPEAT).reshape(x.size(0), 1, SEQ_LEN)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
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
        m = GrowthCNN()
        m.load_state_dict(torch.load(MODEL_PT, map_location="cpu", weights_only=True))
        m.eval()
        _state.cnn_model = m
        with open(SCALER_JSON) as fh:
            _state.scaler = json.load(fh)
    return _state.cnn_model, _state.scaler

# ── Lifespan: warm up models at startup ───────────────────────────────────────
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
    version="1.0.0",
    description="5-layer computational model of Saccharomyces boulardii CNCM I-745",
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
    glucose:   float = Field(-1.0, le=0,  description="Glucose uptake bound (mmol/gDW/hr)")
    oxygen:    float = Field(-5.0, le=0,  description="Oxygen uptake bound (mmol/gDW/hr)")
    ph_factor: float = Field(1.0,  ge=0.1, le=2.0, description="pH scaling factor on exchange bounds")

class SurrogateRequest(BaseModel):
    glucose:   float = Field(-1.0, le=0)
    oxygen:    float = Field(-5.0, le=0)
    ph_factor: float = Field(1.0,  ge=0.1, le=2.0)

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

# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "project":   "CNCM I-745 Digital Twin",
        "organism":  "Saccharomyces boulardii CNCM I-745",
        "version":   "1.0.0",
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
    return {"status": "healthy"}

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
            for rxn in model.exchanges:
                if rxn.id not in (RXN_GLUCOSE, RXN_OXYGEN) and rxn.lower_bound < 0:
                    rxn.lower_bound = rxn.lower_bound * req.ph_factor
            sol = model.optimize()
            feasible = sol.status == "optimal"
            gr = float(sol.objective_value) if feasible else 0.0
    except Exception as exc:
        raise HTTPException(500, f"FBA error: {exc}")
    return {
        "growth_rate":     round(gr, 6),
        "feasible":        feasible,
        "objective_value": round(gr, 6),
        "baseline_growth": 0.085844,
        "delta_vs_baseline": round(gr - 0.085844, 6),
        "inputs":          {"glucose": req.glucose, "oxygen": req.oxygen, "ph_factor": req.ph_factor},
        "solver_time_s":   round(time.perf_counter() - t0, 4),
    }

# ── Surrogate ────────────────────────────────────────────────────────────────
@app.post("/surrogate/predict")
def surrogate_predict(req: SurrogateRequest):
    cnn, scaler = get_cnn()
    feat_mean = np.array(scaler["feature_mean"], dtype=np.float32)
    feat_std  = np.array(scaler["feature_std"],  dtype=np.float32)
    feat   = np.array([[req.glucose, req.oxygen, req.ph_factor]], dtype=np.float32)
    feat_n = (feat - feat_mean) / feat_std
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = float(cnn(torch.from_numpy(feat_n)).item())
    pred = max(0.0, pred)
    return {
        "predicted_growth_rate": round(pred, 6),
        "model_type":            "CNN surrogate (1D-Conv)",
        "inputs":                {"glucose": req.glucose, "oxygen": req.oxygen, "ph_factor": req.ph_factor},
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
    polyamine_flux = 1.63e-5   # mmol/gDW/hr (spermine synthase baseline)
    butyrate_proxy = 0.4163    # CO2/(6×glucose), normalised
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
    })
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
            "outputs": ["data/gem/cncm_i745_gut.xml", "logs/layer2_report.txt"],
            "metrics": gem_metrics,
        },
        "layer3_regulatory": {
            "status": "complete",
            "script": "layer3_regulatory/regulatory_network.py",
            "outputs": ["data/gem/cncm_i745_regulated.xml", "logs/layer3_report.txt"],
            "metrics": {"regulons": 4, "gut_zones": 4, "hsr_targets": 6,
                        "hog_targets": 5, "acid_targets": 5, "yap1_targets": 14},
        },
        "layer4_host": {
            "status": "complete",
            "script": "layer4_host/host_interaction.py",
            "outputs": ["data/fba_outputs/protease_kinetics.csv",
                        "data/fba_outputs/inflammatory_signaling.csv",
                        "logs/layer4_report.txt"],
            "metrics": {
                "toxin_cleaved_120min_pct": 75.1,
                "nfkb_suppression_pct": 70.0,
                "il1b_reduction_pct": 70.0,
                "barrier_score": 0.5971,
                "polyamine_rate_nmol_gDW_hr": 16.3,
            },
        },
        "layer5_surrogate": {
            "status": "complete",
            "script": "layer5_surrogate/surrogate_model.py",
            "outputs": ["data/ml_datasets/fba_training_data.csv",
                        "data/ml_datasets/surrogate_model.pt",
                        "data/ml_datasets/surrogate_scaler.json",
                        "logs/layer5_report.txt"],
            "metrics": {
                "r2_score": 0.9595,
                "rmse": 0.00925,
                "mae": 0.00628,
                "validation_mean_error_pct": 3.88,
                "training_samples": 150,
                "parameters": 119265,
            },
        },
    }

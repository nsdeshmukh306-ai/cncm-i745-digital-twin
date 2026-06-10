"""
Shared application state for the CNCM I-745 Digital Twin API (v4.0.0).

This module is the single source of truth for the heavy, cached resources used
across the FastAPI backend: the strain-specific genome-scale metabolic model
(GEM), the CNN growth surrogates (v1 and v2), the gut-zone constraint
definitions, and the runtime telemetry consumed by ``/health`` and ``/metrics``.

Centralising these here lets ``main.py`` and every router module
(``sensitivity``, ``fva``, ``compare``, ``ppp``, ``export``, ``validate`` …)
import the same singletons without circular imports or loading the GEM twice.

References
----------
Lu H et al. (2019) Yeast8 consensus model. *Nat Commun* 10, 3586.
Khatri I et al. (2017) S. boulardii genome / GPR corrections. *Sci Rep* 7, 371.
Colijn C et al. (2009) E-Flux expression-constrained FBA. *PLoS Comput Biol* 5, e1000489.
"""

from __future__ import annotations

import io
import json
import time
import contextlib
from pathlib import Path

import numpy as np
import cobra
from cobra.io import read_sbml_model
import torch
import torch.nn as nn

import sys
sys.path.insert(0, "/home/nsdeshmukh306/digital-twin")
from logging_setup import get_logger

logger = get_logger("api.state")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE         = Path("/home/nsdeshmukh306/digital-twin")
GENOME_CSV   = BASE / "data/genome/genome_stats.csv"
LAYER1_RPT   = BASE / "logs/layer1_report.txt"
LAYER2_RPT   = BASE / "logs/layer2_report_v2.txt"
GEM_PATH     = BASE / "data/gem/cncm_i745_strain_specific.xml"   # strain-specific
MODEL_PT     = BASE / "data/ml_datasets/surrogate_model.pt"      # v1 (used by /surrogate/predict)
SCALER_JSON  = BASE / "data/ml_datasets/surrogate_scaler.json"
MODEL_PT_V2  = BASE / "data/ml_datasets/surrogate_model_v2.pt"   # v2 strain-specific
SCALER_V2    = BASE / "data/ml_datasets/surrogate_scaler_v2.json"
EFLUX_JSON   = BASE / "data/fba_outputs/eflux_results.json"
FIGURES_DIR  = BASE / "figures"

# ── Exchange reaction identifiers (Yeast8/Yeast9 nomenclature) ─────────────────
RXN_GLUCOSE = "r_1714"
RXN_OXYGEN  = "r_1992"
RXN_NH4     = "r_1654"
RXN_PI      = "r_2005"

BASELINE_GROWTH = 0.089786              # h⁻¹, glucose-limited gut condition
SEQ_LEN         = 32                    # surrogate 1D-Conv sequence length
VERSION         = "4.0.0"

# ── Gut-zone constraint definitions (Layer 3; see eflux_simulator.py) ──────────
GUT_ZONES: dict[str, dict] = {
    "stomach":  {"glucose": -0.5, "oxygen":  0.0, "nh4": -0.5, "pi": -0.3,
                 "condition": "acid_stress_pH4",          "pH": 2.0,
                 "description": "Stomach (pH 2.0, anaerobic, acidic)",
                 "regulons": ["HSR", "acid_response"]},
    "duodenum": {"glucose": -1.0, "oxygen": -2.0, "nh4": -1.0, "pi": -0.5,
                 "condition": "heat_shock_37C",           "pH": 6.0,
                 "description": "Duodenum (pH 6.0, microaerobic, 37 °C)",
                 "regulons": ["HOG", "aerobic_response"]},
    "ileum":    {"glucose": -1.5, "oxygen": -5.0, "nh4": -1.5, "pi": -0.8,
                 "condition": "oxidative_stress_H2O2",     "pH": 7.0,
                 "description": "Ileum (pH 7.0, aerobic, ROS / H2O2)",
                 "regulons": ["aerobic_response", "YAP1"]},
    "colon":    {"glucose": -0.5, "oxygen":  0.0, "nh4": -0.8, "pi": -0.4,
                 "condition": "osmotic_stress_0.7M_NaCl",  "pH": 7.2,
                 "description": "Colon (pH 7.2, anaerobic, high osmolarity)",
                 "regulons": ["HSR", "HOG"]},
}

# ── Runtime telemetry (consumed by /health) ────────────────────────────────────
RUNTIME = {
    "start_time": time.time(),
    "active_gut_zone": "none",
    "last_simulation_timestamp": None,
}


def mark_simulation(gut_zone: str | None = None) -> None:
    """Record that a simulation just ran (updates /health telemetry)."""
    RUNTIME["last_simulation_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if gut_zone:
        RUNTIME["active_gut_zone"] = gut_zone


# ── Surrogate architecture (identical to layer5_surrogate training script) ─────
class GrowthCNN(nn.Module):
    """1-D convolutional surrogate mapping 4 nutrient bounds → growth rate."""

    def __init__(self) -> None:
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

    def forward(self, x):  # noqa: D401
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


# v2 strain-specific surrogate: nn.Sequential layout, SEQ_LEN_V2 = 128.
# (Architecture reconstructed from the saved surrogate_model_v2.pt state_dict:
#  conv.{0,3,6} Conv1d, conv.{1,4,7} BatchNorm1d, fc.0 Linear(64*128, 256) …)
SEQ_LEN_V2 = 128


class GrowthCNNv2(nn.Module):
    """1-D conv surrogate trained on the CNCM I-745 strain-specific GEM (v2)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * SEQ_LEN_V2, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):  # noqa: D401
        x = x.unsqueeze(1).repeat(1, 1, SEQ_LEN_V2 // x.size(1)).reshape(x.size(0), 1, SEQ_LEN_V2)
        x = self.conv(x).flatten(1)
        return self.fc(x).squeeze(1)


# ── Cached singletons ──────────────────────────────────────────────────────────
class _State:
    gem: cobra.Model | None = None
    cnn_v1: GrowthCNN | None = None
    scaler_v1: dict | None = None
    cnn_v2: GrowthCNNv2 | None = None
    scaler_v2: dict | None = None


_S = _State()


def _load_gem_silent(path: Path = GEM_PATH) -> cobra.Model:
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        return read_sbml_model(str(path))


def get_gem() -> cobra.Model:
    """Return the cached strain-specific GEM (loads on first use)."""
    if _S.gem is None:
        if not GEM_PATH.exists():
            raise FileNotFoundError(f"GEM file not found: {GEM_PATH.name}")
        logger.info("Loading strain-specific GEM: %s", GEM_PATH.name)
        _S.gem = _load_gem_silent()
        logger.info("GEM loaded: %d reactions, %d genes",
                    len(_S.gem.reactions), len(_S.gem.genes))
    return _S.gem


def get_cnn() -> tuple[GrowthCNN, dict]:
    """Return the v1 surrogate (4-feature). Preserves legacy /surrogate/predict."""
    if _S.cnn_v1 is None:
        if not MODEL_PT.exists() or not SCALER_JSON.exists():
            raise FileNotFoundError("Surrogate v1 model/scaler not found.")
        with open(SCALER_JSON) as fh:
            scaler = json.load(fh)
        feats = scaler.get("feature_names", [])
        if len(feats) < 4 or "nh4" not in feats:
            raise RuntimeError("Surrogate v1 not upgraded to 4 features.")
        m = GrowthCNN()
        m.load_state_dict(torch.load(MODEL_PT, map_location="cpu", weights_only=True))
        m.eval()
        _S.cnn_v1, _S.scaler_v1 = m, scaler
        logger.info("Surrogate v1 loaded (R²=%.4f)", scaler.get("cross_val_r2_mean", 0.0))
    return _S.cnn_v1, _S.scaler_v1


def get_cnn_v2() -> tuple[GrowthCNNv2, dict]:
    """Return the v2 strain-specific surrogate (used by sensitivity/validation)."""
    if _S.cnn_v2 is None:
        if not MODEL_PT_V2.exists() or not SCALER_V2.exists():
            # Fall back to v1 if v2 artefacts are missing.
            logger.warning("Surrogate v2 not found; falling back to v1.")
            return get_cnn()
        with open(SCALER_V2) as fh:
            scaler = json.load(fh)
        m = GrowthCNNv2()
        m.load_state_dict(torch.load(MODEL_PT_V2, map_location="cpu", weights_only=True))
        m.eval()
        _S.cnn_v2, _S.scaler_v2 = m, scaler
        logger.info("Surrogate v2 loaded (R²=%.4f)", scaler.get("cross_val_r2_mean", 0.0))
    return _S.cnn_v2, _S.scaler_v2


def _scaler_stats(scaler: dict) -> tuple[np.ndarray, np.ndarray]:
    mean = np.array(scaler.get("mean", scaler.get("feature_mean", [0, 0, 0, 0])), dtype=np.float32)
    std  = np.array(scaler.get("std",  scaler.get("feature_std",  [1, 1, 1, 1])), dtype=np.float32)
    return mean, std


def surrogate_batch(features, version: str = "v2") -> np.ndarray:
    """Deterministic batch prediction. ``features`` is (N, 4); returns (N,) ≥ 0."""
    cnn, scaler = get_cnn_v2() if version == "v2" else get_cnn()
    mean, std = _scaler_stats(scaler)
    X = np.asarray(features, dtype=np.float32).reshape(-1, 4)
    Xn = (X - mean) / std
    cnn.eval()
    with torch.no_grad():
        y = cnn(torch.from_numpy(Xn)).numpy().reshape(-1)
    return np.clip(y, 0.0, None)


def surrogate_mc(features, version: str = "v2", n_samples: int = 50) -> dict:
    """MC-Dropout prediction with 95 % CI (Gal & Ghahramani 2016).

    Returns mean, std and CI arrays aligned with the input rows.
    """
    cnn, scaler = get_cnn_v2() if version == "v2" else get_cnn()
    mean, std = _scaler_stats(scaler)
    X = np.asarray(features, dtype=np.float32).reshape(-1, 4)
    Xn = (X - mean) / std
    cnn.train()                                  # enable dropout stochasticity
    draws = []
    with torch.no_grad():
        for _ in range(n_samples):
            draws.append(cnn(torch.from_numpy(Xn)).numpy().reshape(-1))
    cnn.eval()
    arr = np.clip(np.stack(draws, axis=0), 0.0, None)   # (n_samples, N)
    mu = arr.mean(axis=0)
    sd = arr.std(axis=0)
    return {
        "mean": mu,
        "std": sd,
        "ci_lower": np.clip(mu - 1.96 * sd, 0.0, None),
        "ci_upper": mu + 1.96 * sd,
    }


# ── Gut-zone / E-Flux constraint application ───────────────────────────────────
def apply_zone_bounds(model: cobra.Model, gut_zone: str) -> dict:
    """Apply a gut zone's exchange uptake bounds in-place. Returns the params."""
    zone = GUT_ZONES.get(gut_zone.lower())
    if not zone:
        return {}
    model.reactions.get_by_id(RXN_GLUCOSE).lower_bound = zone["glucose"]
    model.reactions.get_by_id(RXN_OXYGEN).lower_bound  = zone["oxygen"]
    model.reactions.get_by_id(RXN_NH4).lower_bound     = zone["nh4"]
    model.reactions.get_by_id(RXN_PI).lower_bound      = zone["pi"]
    return zone


def apply_eflux_bounds(model: cobra.Model, gut_zone: str) -> dict:
    """Re-apply pre-computed E-Flux upper-bound scaling for a gut zone in-place.

    Uses ``data/fba_outputs/eflux_results.json`` produced by the Layer 3
    E-Flux simulator (Colijn et al. 2009). Safe no-op when data are missing.
    """
    if not EFLUX_JSON.exists():
        return {"applied": False, "reason": "eflux_results.json not found"}
    with open(EFLUX_JSON) as fh:
        eflux = json.load(fh)
    zone_key = gut_zone.lower()
    if zone_key not in eflux:
        return {"applied": False, "reason": f"zone '{zone_key}' not in E-Flux results"}
    zres = eflux[zone_key]
    modified = 0
    for rxn_id, info in zres.get("flux_changes", {}).items():
        try:
            model.reactions.get_by_id(rxn_id).upper_bound = info["new_ub"]
            modified += 1
        except Exception:
            pass
    return {
        "applied": True,
        "zone": zone_key,
        "condition": zres.get("condition", "unknown"),
        "modified_reactions": modified,
        "eflux_growth_rate": zres.get("growth_rate", 0.0),
    }


def top_exchange_reactions(model: cobra.Model, n: int = 20) -> list[str]:
    """Return up to ``n`` exchange reaction IDs by |flux| at the current optimum."""
    sol = model.optimize()
    if sol.status != "optimal":
        return [r.id for r in model.exchanges][:n]
    ex_ids = {r.id for r in model.exchanges}
    fluxes = [(rid, abs(v)) for rid, v in sol.fluxes.items() if rid in ex_ids]
    fluxes.sort(key=lambda kv: kv[1], reverse=True)
    return [rid for rid, _ in fluxes[:n]]

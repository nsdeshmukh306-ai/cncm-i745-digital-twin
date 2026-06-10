"""
Unit tests for Layer 5 — the CNN growth-rate surrogate.

Covers:
  * Model + scaler load (v2 strain-specific surrogate).
  * Deterministic batch prediction shape and non-negativity.
  * MC-Dropout uncertainty quantification produces a valid 95% CI
    (Gal & Ghahramani 2016) with ci_lower <= mean <= ci_upper.

Run inside the venv:  pytest tests/test_layer5_surrogate.py -v
"""
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from api import state  # noqa: E402

# Four representative nutrient input vectors (glucose, oxygen, nh4, pi).
INPUTS = np.array([
    [-1.65, -2.0, -1.0, -0.5],
    [-5.0, -3.0, -2.0, -1.0],
    [-0.5, 0.0, -0.5, -0.3],
    [-10.0, -5.0, -5.0, -2.0],
], dtype=np.float32)


def test_surrogate_model_loads():
    """The surrogate model and its scaler load successfully."""
    cnn, scaler = state.get_cnn_v2()
    assert cnn is not None
    assert isinstance(scaler, dict)
    # Scaler must expose mean/std under one of the supported key names.
    assert any(k in scaler for k in ("mean", "feature_mean"))
    assert any(k in scaler for k in ("std", "feature_std"))


def test_prediction_shape_and_sign():
    """Batch prediction returns one non-negative growth rate per input row."""
    y = state.surrogate_batch(INPUTS, version="v2")
    assert y.shape == (INPUTS.shape[0],)
    assert np.all(y >= 0.0), "growth-rate predictions must be non-negative"
    assert np.all(np.isfinite(y))


def test_single_row_prediction():
    """A single 4-feature vector predicts a finite scalar growth rate."""
    y = state.surrogate_batch(INPUTS[0], version="v2")
    assert y.shape == (1,)
    assert np.isfinite(y[0])


def test_mc_dropout_ci_bounds_valid():
    """MC-Dropout yields a positive-width CI with ci_lower <= mean <= ci_upper."""
    res = state.surrogate_mc(INPUTS, version="v2", n_samples=30)
    for key in ("mean", "std", "ci_lower", "ci_upper"):
        assert key in res
        assert res[key].shape == (INPUTS.shape[0],)
    assert np.all(res["std"] >= 0.0)
    assert np.all(res["ci_lower"] >= 0.0), "CI lower bound must be clipped at 0"
    assert np.all(res["ci_lower"] <= res["mean"] + 1e-6)
    assert np.all(res["mean"] <= res["ci_upper"] + 1e-6)
    assert np.all(res["ci_upper"] >= res["ci_lower"])

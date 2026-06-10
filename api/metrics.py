"""
Prometheus-compatible metrics for the CNCM I-745 Digital Twin API (v4.0.0).

Exposes counters, a request-duration histogram and a last-growth-rate gauge via
``prometheus_client``. Metric objects are module-level singletons so every
router can increment them; ``main.py`` serves them at ``GET /metrics`` using
``generate_latest`` and installs a middleware that observes request latency.

Reference: Prometheus client_python (https://github.com/prometheus/client_python).
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge, CONTENT_TYPE_LATEST, generate_latest

FBA_SIMULATIONS_TOTAL = Counter(
    "fba_simulations_total", "Total number of FBA simulations executed."
)
SURROGATE_PREDICTIONS_TOTAL = Counter(
    "surrogate_predictions_total", "Total number of CNN surrogate predictions served."
)
API_REQUEST_DURATION_SECONDS = Histogram(
    "api_request_duration_seconds", "API request latency in seconds.",
    labelnames=("method", "path"),
)
FBA_GROWTH_RATE_LAST = Gauge(
    "fba_growth_rate_last", "Growth rate (h^-1) from the most recent FBA simulation."
)

__all__ = [
    "FBA_SIMULATIONS_TOTAL",
    "SURROGATE_PREDICTIONS_TOTAL",
    "API_REQUEST_DURATION_SECONDS",
    "FBA_GROWTH_RATE_LAST",
    "CONTENT_TYPE_LATEST",
    "generate_latest",
]

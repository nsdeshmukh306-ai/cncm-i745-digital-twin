# API Reference

**CNCM I-745 Digital Twin API** — v4.0.0

Base URL (production): `http://34.14.186.73:8000`  ·  Interactive docs: `/docs`

This reference is generated from the live OpenAPI schema (`GET /openapi.json`) and augmented with examples. All `POST` endpoints accept an optional `X-API-Key` header; when `DT_API_KEY` is unset, auth is open. Out-of-range nutrient bounds return **422**; an infeasible FBA returns **422** `{"error":"FBA infeasible","detail":"check nutrient bounds"}`; model/solver errors return **500**.

---

## Authentication

`POST` endpoints depend on `require_api_key`. When the `DT_API_KEY` environment
variable is set on the server, requests must send a matching key:

```
X-API-Key: <your-key>
```

A missing or wrong key returns **401** `{"error": "Unauthorized"}`. When
`DT_API_KEY` is unset, authentication is open and the header is ignored.
`GET` endpoints are always public.

---

## Worked examples

### FBA simulation

```bash
curl -s -X POST http://34.14.186.73:8000/fba/simulate \
  -H 'Content-Type: application/json' \
  -d '{"glucose": -1.65, "oxygen": -2.0, "nh4": -1.0, "pi": -0.5}'
```

```json
{
  "growth_rate": 0.089786,
  "feasible": true,
  "baseline_growth": 0.089786,
  "change_from_baseline": 0.0,
  "change_pct": 0.0,
  "model_type": "CNCM I-745 strain-specific GEM",
  "inputs": {"glucose": -1.65, "oxygen": -2.0, "nh4": -1.0, "pi": -0.5},
  "solver_time_s": 0.05
}
```

To apply E-Flux expression constraints for a gut zone, add
`"use_eflux": true, "gut_zone": "colon"`.

### Surrogate prediction

```bash
curl -s -X POST http://34.14.186.73:8000/surrogate/predict \
  -H 'Content-Type: application/json' \
  -d '{"glucose": -1.65, "oxygen": -2.0, "nh4": -1.0, "pi": -0.5}'
```

```json
{
  "predicted_growth_rate": 0.126279,
  "model_type": "CNN surrogate (1D-Conv, LHS-2000)",
  "cross_val_r2": 0.8691,
  "inputs": {"glucose": -1.65, "oxygen": -2.0, "nh4": -1.0, "pi": -0.5},
  "inference_time_s": 0.0008
}
```

### Morris sensitivity

```bash
curl -s -X POST http://34.14.186.73:8000/sensitivity/morris \
  -H 'Content-Type: application/json' \
  -d '{"trajectories": 20, "num_levels": 4}'
```

Returns a `ranking` array of `{parameter, mu_star, sigma}` ordered by μ\*.

### WebSocket streaming (`/ws/simulate`)

Connect, send the same JSON body as `/fba/simulate`, then receive a sequence of
progress frames followed by the final result:

```json
{"step": "loading_gem",   "progress": 20,  "message": "Loading strain-specific GEM..."}
{"step": "applying_gpr",  "progress": 40,  "message": "Applying GPR corrections..."}
{"step": "optimizing",    "progress": 80,  "message": "Optimising biomass objective (FBA)..."}
{"step": "complete",      "progress": 100, "message": "FBA complete.", "result": { ... }}
```

### Error responses

| Status | When | Body |
|:---:|:---|:---|
| 401 | Missing/invalid API key (auth enabled) | `{"error": "Unauthorized"}` |
| 404 | Unknown reaction / missing data file | `{"error": "reaction not found", ...}` |
| 422 | Out-of-range input, or infeasible FBA | `{"error": "FBA infeasible", "detail": "check nutrient bounds"}` |
| 500 | Solver / model failure | `{"error": "FBA error", "detail": "..."}` |

---

## Endpoint catalogue


### `GET /`

### `POST /chat`

| Field | Type | Default | Constraint |
|:---|:---|:---|:---|
| `message` | string | `—` | — |
| `conversation_history` | array | `—` | — |
| `messages` | array | `—` | — |


### `POST /chat/explain_flux`

| Field | Type | Default | Constraint |
|:---|:---|:---|:---|
| `reaction_id` | string | `—` | — |
| `gut_zone` | string | `none` | `^(none|stomach|duodenum|ileum|colon)$` |


### `POST /compare/carbon_sources`

| Field | Type | Default | Constraint |
|:---|:---|:---|:---|
| `carbon_sources` | array | `—` | — |
| `uptake` | number | `-10.0` | ≤ 0.0 |


### `POST /compare/gut_transit`

| Field | Type | Default | Constraint |
|:---|:---|:---|:---|
| `gut_zones` | array | `—` | — |
| `use_eflux` | boolean | `True` | — |


### `GET /export/figures/{figure_id}`

Query parameters:

| Name | Type | Default |
|:---|:---|:---|
| `figure_id` | string | `—` |

### `GET /export/gem`

### `GET /export/report`

Query parameters:

| Name | Type | Default |
|:---|:---|:---|
| `format` | string | `json` |

### `POST /fba/fva`

| Field | Type | Default | Constraint |
|:---|:---|:---|:---|
| `gut_zone` | string | `none` | — |
| `fraction_of_optimum` | number | `0.9` | ≥ 0.0, ≤ 1.0 |
| `loopless` | boolean | `True` | — |
| `reaction_list` | array | `—` | — |


### `POST /fba/phase_plane`

| Field | Type | Default | Constraint |
|:---|:---|:---|:---|
| `x_axis_reaction` | string | `r_1714` | — |
| `y_axis_reaction` | string | `r_1992` | — |
| `n_points` | integer | `20` | ≥ 5.0, ≤ 30.0 |


### `POST /fba/simulate`

| Field | Type | Default | Constraint |
|:---|:---|:---|:---|
| `glucose` | number | `-1.65` | ≥ -20.0, ≤ 0.0 |
| `oxygen` | number | `-2.0` | ≥ -5.0, ≤ 0.0 |
| `nh4` | number | `-1.0` | ≥ -10.0, ≤ 0.0 |
| `pi` | number | `-0.5` | ≥ -10.0, ≤ 0.0 |
| `use_eflux` | boolean | `False` | — |
| `gut_zone` | string | `none` | `^(none|stomach|duodenum|ileum|colon)$` |


### `GET /genome/stats`

### `GET /health`

### `GET /host/barrier`

### `GET /host/inflammation`

### `GET /layers/status`

### `GET /logs`

Query parameters:

| Name | Type | Default |
|:---|:---|:---|
| `n` | integer | `100` |

### `GET /metrics`

### `POST /sensitivity/morris`

| Field | Type | Default | Constraint |
|:---|:---|:---|:---|
| `trajectories` | integer | `20` | ≥ 4.0, ≤ 200.0 |
| `num_levels` | integer | `4` | ≥ 2.0, ≤ 10.0 |
| `gut_zone` | string | `none` | — |


### `POST /sensitivity/sobol`

| Field | Type | Default | Constraint |
|:---|:---|:---|:---|
| `base_samples` | integer | `256` | ≥ 16.0, ≤ 4096.0 |
| `gut_zone` | string | `none` | — |


### `POST /surrogate/predict`

| Field | Type | Default | Constraint |
|:---|:---|:---|:---|
| `glucose` | number | `-1.65` | ≥ -20.0, ≤ 0.0 |
| `oxygen` | number | `-2.0` | ≥ -5.0, ≤ 0.0 |
| `nh4` | number | `-1.0` | ≥ -10.0, ≤ 0.0 |
| `pi` | number | `-0.5` | ≥ -10.0, ≤ 0.0 |


### `GET /validate/gem`

### `GET /validate/surrogate`

Query parameters:

| Name | Type | Default |
|:---|:---|:---|
| `n` | integer | `50` |
| `seed` | integer | `7` |

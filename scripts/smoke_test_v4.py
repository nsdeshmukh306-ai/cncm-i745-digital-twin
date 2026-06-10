"""Offline smoke test for the v4.0.0 API using Starlette TestClient.

Exercises every new + existing endpoint in-process (no live service touched).
Run inside the venv: python scripts/smoke_test_v4.py
"""
import os, sys, json, traceback
os.environ.pop("DT_API_KEY", None)          # auth open for the test
sys.path.insert(0, "/home/nsdeshmukh306/digital-twin")

from fastapi.testclient import TestClient
from api.main import app

results = []
def check(name, fn, expect=None):
    try:
        code, info = fn()
        ok = (code == expect) if expect is not None else (200 <= code < 300)
        results.append((name, code, ok, info))
        print(f"[{'OK ' if ok else 'ERR'}] {name:32s} -> {code}  {info}")
    except Exception as e:
        results.append((name, -1, False, str(e)))
        print(f"[EXC] {name:32s} -> {e}")
        traceback.print_exc()

with TestClient(app) as c:
    check("GET /health", lambda: (r := c.get("/health")).status_code and (r.status_code, f"v={r.json().get('version')} mem={r.json().get('memory_used_mb')}MB"))
    check("GET /", lambda: ((r := c.get("/")).status_code, r.json().get("version")))
    check("GET /metrics", lambda: ((r := c.get("/metrics")).status_code, r.text.split(chr(10))[0][:40]))
    check("GET /logs", lambda: ((r := c.get("/logs?n=5")).status_code, f"{r.json().get('line_count')} lines"))
    check("GET /genome/stats", lambda: ((r := c.get("/genome/stats")).status_code, f"{r.json().get('sequence_count')} seqs"))
    check("GET /layers/status", lambda: ((r := c.get("/layers/status")).status_code, list(r.json().keys())[:2]))
    check("GET /host/inflammation", lambda: ((r := c.get("/host/inflammation")).status_code, r.json().get("reductions_pct")))
    check("GET /host/barrier", lambda: ((r := c.get("/host/barrier")).status_code, r.json().get("barrier_integrity_score")))
    check("POST /fba/simulate", lambda: ((r := c.post("/fba/simulate", json={"glucose": -1.65, "oxygen": -2.0, "nh4": -1.0, "pi": -0.5})).status_code, f"growth={r.json().get('growth_rate')}"))
    check("POST /fba/simulate eflux", lambda: ((r := c.post("/fba/simulate", json={"glucose": -1.0, "oxygen": -2.0, "nh4": -1.0, "pi": -0.5, "use_eflux": True, "gut_zone": "duodenum"})).status_code, f"growth={r.json().get('growth_rate')}"))
    check("POST /fba/simulate badrange", lambda: ((r := c.post("/fba/simulate", json={"glucose": 5.0})).status_code, "rejected (422 expected)"), expect=422)
    check("POST /surrogate/predict", lambda: ((r := c.post("/surrogate/predict", json={"glucose": -1.65, "oxygen": -2.0, "nh4": -1.0, "pi": -0.5})).status_code, f"pred={r.json().get('predicted_growth_rate')}"))
    check("POST /sensitivity/morris", lambda: ((r := c.post("/sensitivity/morris", json={"trajectories": 8, "num_levels": 4})).status_code, [x["parameter"] for x in r.json().get("ranking", [])]))
    check("POST /sensitivity/sobol", lambda: ((r := c.post("/sensitivity/sobol", json={"base_samples": 16})).status_code, [x["parameter"] for x in r.json().get("indices", [])]))
    check("POST /fba/fva list", lambda: ((r := c.post("/fba/fva", json={"reaction_list": ["r_1714", "r_1992"], "fraction_of_optimum": 0.9, "loopless": False})).status_code, f"{r.json().get('n_reactions')} rxns"))
    check("POST /fba/phase_plane", lambda: ((r := c.post("/fba/phase_plane", json={"n_points": 6})).status_code, f"grid {len(r.json().get('growth_grid', []))}x{len(r.json().get('growth_grid', [[]])[0])}"))
    check("POST /compare/carbon_sources", lambda: ((r := c.post("/compare/carbon_sources", json={"carbon_sources": ["glucose", "fructose", "ethanol"]})).status_code, [(x.get("carbon_source"), x.get("growth_rate")) for x in r.json().get("results", [])]))
    check("POST /compare/gut_transit", lambda: ((r := c.post("/compare/gut_transit", json={"gut_zones": ["duodenum", "ileum"], "use_eflux": True})).status_code, [(z.get("gut_zone"), z.get("growth_rate"), z.get("barrier_score")) for z in r.json().get("zones", [])]))
    check("GET /validate/gem", lambda: ((r := c.get("/validate/gem")).status_code, f"errs={r.json().get('error_count')} warns={r.json().get('warning_count')}"))
    check("GET /validate/surrogate", lambda: ((r := c.get("/validate/surrogate?n=12")).status_code, f"MAE={r.json().get('mae')} R2={r.json().get('r2')}"))
    check("GET /export/report json", lambda: ((r := c.get("/export/report?format=json")).status_code, r.json().get("version")))
    check("GET /export/report csv", lambda: ((r := c.get("/export/report?format=csv")).status_code, f"{len(r.text)} bytes"))
    check("GET /export/report pdf", lambda: ((r := c.get("/export/report?format=pdf")).status_code, f"{len(r.content)} bytes pdf"))
    check("GET /export/gem", lambda: ((r := c.get("/export/gem")).status_code, f"{len(r.content)} bytes"))
    check("GET /export/figures/5", lambda: ((r := c.get("/export/figures/5")).status_code, f"{len(r.content)} bytes png"))

    # WebSocket
    try:
        with c.websocket_connect("/ws/simulate") as ws:
            ws.send_json({"glucose": -1.65, "oxygen": -2.0, "nh4": -1.0, "pi": -0.5})
            steps = []
            while True:
                msg = ws.receive_json()
                steps.append(msg["step"])
                if msg["step"] in ("complete", "error"):
                    final = msg
                    break
        ok = final["step"] == "complete"
        results.append(("WS /ws/simulate", 200 if ok else -1, ok, steps))
        print(f"[{'OK ' if ok else 'ERR'}] WS /ws/simulate              -> steps={steps} growth={final.get('result', {}).get('growth_rate')}")
    except Exception as e:
        results.append(("WS /ws/simulate", -1, False, str(e)))
        print(f"[EXC] WS /ws/simulate -> {e}")

n_ok = sum(1 for _, _, ok, _ in results if ok)
print(f"\n=== {n_ok}/{len(results)} endpoints OK ===")
sys.exit(0 if n_ok == len(results) else 1)

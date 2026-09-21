"""FastAPI service exposing the CNCM I-745 digital twin.

Run:  python -m uvicorn app.main:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs      Dashboard: http://127.0.0.1:8000/

Design notes
------------
* The genome-scale model costs a few hundred MB, so it is lazy-loaded on the
  first request that needs it and then cached for the process lifetime.
* /predict uses the trained surrogate and returns in microseconds; /fba solves
  the actual linear program. Both are exposed so the speed/accuracy trade-off
  is visible to the user rather than hidden.
* /query is keyword routing over the other endpoints. It is NOT a language
  model, and the response says so in the `method` field.
"""
from __future__ import annotations
import os, json, functools, time
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")

app = FastAPI(title="S. boulardii CNCM I-745 digital twin",
              version="4.0.0",
              description="Five-layer mechanistic model: comparative genomics, "
                          "strain-specific GEM, expression-constrained zone "
                          "metabolism, host-interaction kinetics, a machine-learning surrogate, "
                          "and two-species competition.")


# ----------------------------------------------------------------- helpers
def _csv(name: str) -> pd.DataFrame:
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        raise HTTPException(503, f"result file not built yet: {name}")
    return pd.read_csv(p)


def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


@functools.lru_cache(maxsize=2)
def get_model(which: str = "strain"):
    import cobra
    import sys
    sys.path.insert(0, ROOT)
    path = os.path.join(MODELS, "yeast9_cncm_i745.xml") if which == "strain" \
        else os.path.join(ROOT, "data", "yeast-GEM.xml")
    if not os.path.exists(path):
        raise HTTPException(503, f"model not built: {path}")
    return cobra.io.read_sbml_model(path)


@functools.lru_cache(maxsize=1)
def get_surrogate():
    """Load the MLP ensemble once; each member is a fitted sklearn pipeline."""
    import pickle
    p = os.path.join(MODELS, "surrogate_mlp_ensemble.pkl")
    if not os.path.exists(p):
        raise HTTPException(503, "surrogate not trained yet")
    with open(p, "rb") as fh:
        ck = pickle.load(fh)
    return ck["nets"], ck["keys"]


# ----------------------------------------------------------------- schemas
class FBARequest(BaseModel):
    carbon_source: str = Field("glucose", description="exchange key, e.g. glucose/galactose/maltose")
    carbon_uptake: float = Field(1.65, ge=0, le=50, description="mmol gDW-1 h-1")
    oxygen: float = Field(2.0, ge=0, le=50)
    nitrogen_source: str = Field("ammonium")
    nitrogen_uptake: float = Field(1000.0, ge=0, le=1000)
    model: str = Field("strain", pattern="^(strain|base)$")
    top_n: int = Field(15, ge=1, le=100)


class PredictRequest(BaseModel):
    glucose: float = Field(1.65, ge=0.1, le=10)
    oxygen: float = Field(2.0, ge=0.0, le=10)
    ammonium: float = Field(5.0, ge=0.1, le=10)
    phosphate: float = Field(1.0, ge=0.01, le=2)
    uncertainty: bool = True


class KineticsRequest(BaseModel):
    vmax: Optional[float] = None
    km: Optional[float] = None
    k_spont: Optional[float] = None
    t_end: float = Field(8.0, gt=0, le=48)
    use_posterior: bool = True


class CompetitionRequest(BaseModel):
    k_ox: float = Field(0.8, ge=0, le=5)
    D: float = Field(0.10, gt=0, le=1.0)
    S_in: float = Field(25.0, gt=0, le=100)
    k_prot: float = Field(1.2, ge=0, le=10)
    mu_y: float = Field(0.35, gt=0, le=1.0)
    t_end: float = Field(200.0, gt=0, le=1000)


class QueryRequest(BaseModel):
    question: str


# ----------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    built = {f: os.path.exists(os.path.join(RES, f)) for f in [
        "layer1_marker_calls.csv", "layer2_gpr_edits.csv", "layer3_zone_fluxes.csv",
        "layer4_posterior_summary.csv", "layer5_cv_metrics.csv", "layer6_outcome_map.csv"]}
    return dict(status="ok", results_built=built,
                model_present=os.path.exists(os.path.join(MODELS, "yeast9_cncm_i745.xml")),
                surrogate_present=os.path.exists(os.path.join(MODELS, "surrogate_mlp_ensemble.pkl")))


@app.get("/genome", summary="Layer 1: assembly statistics and marker calls")
def genome(gene: Optional[str] = None, assembly: Optional[str] = None):
    stats = _csv("layer1_genome_stats.csv")
    calls = _csv("layer1_marker_calls.csv")
    if gene:
        calls = calls[calls.gene.str.upper() == gene.upper()]
        if calls.empty:
            raise HTTPException(404, f"gene not in the marker panel: {gene}")
    if assembly:
        calls = calls[calls.assembly == assembly]
    sb = [a for a in calls.assembly.unique() if a.startswith("Sb_") and a != "Sb_KCTC13826BP"]
    piv = calls.pivot_table(index="gene", columns="assembly", values="call", aggfunc="first")
    lost = [g for g in piv.index
            if set(piv.columns) >= set(sb) and
            all(piv.loc[g, a] in ("absent", "degraded") for a in sb)]
    return dict(assemblies=_records(stats), n_calls=int(len(calls)),
                genes_absent_in_all_boulardii=sorted(lost),
                calls=_records(calls))


@app.get("/introgression", summary="Layer 1: Z. bailii introgression on chromosome IV")
def introgression():
    return dict(loci=_records(_csv("layer1_introgression.csv")))


@app.get("/model_summary", summary="Layer 2: model sizes and GPR corrections")
def model_summary():
    edits = _csv("layer2_gpr_edits.csv")
    out = dict(gpr_edits=_records(edits),
               n_edits=int(len(edits)),
               n_reactions_deleted=int((edits.consequence == "reaction_deleted").sum()))
    for ph in ("base", "strain"):
        f = os.path.join(RES, f"layer2_model_summary_{ph}.csv")
        if os.path.exists(f):
            out[f"summary_{ph}"] = _records(pd.read_csv(f))
    return out


@app.get("/phenotypes", summary="Layer 2: carbon and nitrogen source predictions")
def phenotypes():
    frames = [pd.read_csv(os.path.join(RES, f"layer2_phenotype_{p}.csv"))
              for p in ("base", "strain")
              if os.path.exists(os.path.join(RES, f"layer2_phenotype_{p}.csv"))]
    if not frames:
        raise HTTPException(503, "phenotype sweep not built yet")
    df = pd.concat(frames)
    wide = df.pivot_table(index=["role", "substrate"], columns="model",
                          values="mu").reset_index()
    if {"Yeast9_base", "CNCM_I745"} <= set(wide.columns):
        wide["delta"] = (wide["CNCM_I745"] - wide["Yeast9_base"]).round(6)
    return dict(predictions=_records(wide),
                panel=_records(pd.read_csv(os.path.join(ROOT, "data", "phenotype_panel.csv"))))


@app.post("/fba", summary="Layer 2: solve the linear program under custom nutrients")
def fba(req: FBARequest):
    import sys
    sys.path.insert(0, ROOT)
    from sbdtwin import gem as G
    from cobra.flux_analysis import pfba
    m = get_model(req.model)
    if req.carbon_source not in G.EX:
        raise HTTPException(400, f"unknown carbon source; options: {sorted(G.EX)}")
    t0 = time.perf_counter()
    with m:
        G.set_medium(m, carbon=(req.carbon_source, req.carbon_uptake),
                     nitrogen=(req.nitrogen_source, req.nitrogen_uptake),
                     oxygen=req.oxygen)
        mu = m.slim_optimize()
        fluxes = []
        if mu and mu == mu and mu > 1e-9:
            s = pfba(m)
            top = s.fluxes.abs().sort_values(ascending=False).head(req.top_n)
            fluxes = [dict(reaction=r, name=m.reactions.get_by_id(r).name,
                           flux=round(float(s.fluxes[r]), 6)) for r in top.index]
    return dict(growth_rate=round(float(mu), 6) if mu == mu else 0.0,
                solve_seconds=round(time.perf_counter() - t0, 4),
                model=req.model, top_fluxes=fluxes)


@app.get("/essentiality", summary="Layer 2: single-gene deletion screen")
def essentiality(model: str = "strain", classification: Optional[str] = None):
    f = os.path.join(RES, f"layer2_essentiality_{'strain' if model == 'strain' else 'base'}.csv")
    if not os.path.exists(f):
        raise HTTPException(503, "essentiality screen not built yet")
    d = pd.read_csv(f)
    counts = d.classification.value_counts().to_dict()
    if classification:
        d = d[d.classification == classification]
    return dict(counts=counts, n=int(len(d)), genes=_records(d.head(2000)))


@app.get("/zone/{zone}", summary="Layer 3: expression-constrained zone metabolism")
def zone(zone: str, top_n: int = 20):
    import sys
    sys.path.insert(0, ROOT)
    from sbdtwin import bioenergetics as B
    if zone not in B.ZONES:
        raise HTTPException(404, f"unknown zone; options: {sorted(B.ZONES)}")
    out = dict(zone=zone, conditions=B.ZONES[zone])
    if zone == "stomach":
        ph = _csv("layer3_ph_bioenergetics.csv")
        sub = ph[np.isclose(ph.pH, 2.0)]
        out["bioenergetics"] = _records(sub)
        out["note"] = ("modelled mechanistically: no acid-shock array exists in GSE18 and "
                       "the dominant effect of pH 2 is the ATP cost of the proton gradient")
        return out
    f = _csv("layer3_zone_fluxes.csv")
    f = f[f.zone == zone]
    if "median" in f.columns:
        # f["median"], not f.median -- the attribute is the DataFrame method
        f = f.reindex(f["median"].abs().sort_values(ascending=False).index)
    out["fluxes"] = _records(f.head(top_n))
    return out


@app.get("/zone_contrasts", summary="Layer 3: FDR-controlled between-zone contrasts")
def zone_contrasts(contrast: Optional[str] = None, fdr: float = 0.05, top_n: int = 50):
    c = _csv("layer3_zone_contrasts.csv")
    if contrast:
        c = c[c.contrast == contrast]
    sig = c[(c.p_adj < fdr) & (c.mean_diff.abs() > 1e-6)]
    sig = sig.reindex(sig.cohens_d.abs().sort_values(ascending=False).index)
    return dict(n_tested=int(len(c)), n_significant=int(len(sig)), fdr=fdr,
                contrasts=sorted(c.contrast.unique()), top=_records(sig.head(top_n)))


@app.get("/ph_response", summary="Layer 3: growth versus luminal pH")
def ph_response(k_leak: Optional[float] = None):
    d = _csv("layer3_ph_bioenergetics.csv")
    if k_leak is not None:
        d = d[np.isclose(d.k_leak, k_leak)]
        if d.empty:
            raise HTTPException(404, "k_leak not in the computed grid")
    return dict(curve=_records(d))


@app.post("/kinetics", summary="Layer 4: toxin A proteolysis")
def kinetics(req: KineticsRequest):
    import sys
    sys.path.insert(0, ROOT)
    from sbdtwin import kinetics as K
    if req.use_posterior and all(v is None for v in (req.vmax, req.km, req.k_spont)):
        s = _csv("layer4_posterior_summary.csv").set_index("quantity")
        vmax, km, ks = (float(s.loc[k, "median"]) for k in ("vmax", "km", "k_spont"))
        src = "posterior median"
    else:
        vmax = req.vmax if req.vmax is not None else 50.0
        km = req.km if req.km is not None else 50.0
        ks = req.k_spont if req.k_spont is not None else 0.15
        src = "user-supplied"
    t, s_t = K.simulate_tcda(vmax, km, ks, t_end=req.t_end, n=121)
    return dict(parameter_source=src, vmax=vmax, km=km, k_spont=ks,
                half_life_h=round(K.half_life(vmax, km, ks), 4),
                protease_share=round(K.protease_share(vmax, km, ks), 4),
                trajectory=[dict(t_h=round(float(a), 4), toxinA=round(float(b), 4))
                            for a, b in zip(t, s_t)])


@app.get("/kinetics/posterior", summary="Layer 4: posterior summary and trajectory band")
def kinetics_posterior():
    out = dict(summary=_records(_csv("layer4_posterior_summary.csv")))
    f = os.path.join(RES, "layer4_toxin_trajectory.csv")
    if os.path.exists(f):
        out["trajectory_band"] = _records(pd.read_csv(f))
    f = os.path.join(RES, "layer4_sobol.csv")
    if os.path.exists(f):
        out["nfkb_sobol"] = _records(pd.read_csv(f))
    f = os.path.join(RES, "layer4_nfkb_prior_predictive.json")
    if os.path.exists(f):
        out["nfkb_prior_predictive"] = json.load(open(f))
    return out


@app.post("/predict", summary="Layer 5: surrogate growth prediction (real time)")
def predict(req: PredictRequest):
    nets, keys = get_surrogate()
    x = np.array([[getattr(req, k) for k in keys]], dtype=float)
    t0 = time.perf_counter()
    draws = np.array([float(m.predict(x)[0]) for m in nets])
    dt = time.perf_counter() - t0
    mu, sd = float(draws.mean()), float(draws.std())
    out = dict(growth_rate=round(mu, 6), seconds=round(dt, 6),
               method=f"{len(nets)}-member MLP ensemble",
               inputs=dict(zip(keys, x[0].tolist())))
    if req.uncertainty:
        out.update(sd=round(sd, 6),
                   ci95=[round(mu - 1.96 * sd, 6), round(mu + 1.96 * sd, 6)])
    return out


@app.get("/surrogate/metrics", summary="Layer 5: cross-validated model comparison")
def surrogate_metrics():
    out = dict(cv=_records(_csv("layer5_cv_metrics.csv")))
    f = os.path.join(RES, "layer5_speed_benchmark.json")
    if os.path.exists(f):
        out["speed"] = json.load(open(f))
    return out


@app.post("/competition", summary="Layer 6: S. boulardii vs C. difficile")
def competition(req: CompetitionRequest):
    import sys
    sys.path.insert(0, ROOT)
    from sbdtwin import competition as C
    p = dict(k_ox=req.k_ox, D=req.D, S_in=req.S_in, k_prot=req.k_prot, mu_y=req.mu_y)
    t, y, pp = C.simulate(p, t_end=req.t_end)
    res = C.outcome(p)
    step = max(1, len(t) // 120)
    return dict(parameters=pp, outcome=res,
                timecourse=[dict(t_h=round(float(t[i]), 3),
                                 substrate=round(float(y[0][i]), 5),
                                 boulardii=round(float(y[1][i]), 6),
                                 c_difficile=round(float(y[2][i]), 6),
                                 toxinA=round(float(y[3][i]), 6))
                            for i in range(0, len(t), step)])


@app.get("/competition/map", summary="Layer 6: exclusion boundary")
def competition_map():
    return dict(grid=_records(_csv("layer6_outcome_map.csv")),
                boundary=_records(_csv("layer6_exclusion_boundary.csv")))


ROUTES = [
    (("gene", "absent", "hxt", "marker", "genome", "assembly", "chromosome"), "/genome"),
    (("introgression", "bailii", "hybrid"), "/introgression"),
    (("gpr", "correction", "reaction", "model size"), "/model_summary"),
    (("grow", "carbon", "galactose", "maltose", "trehalose", "phenotype", "substrate"), "/phenotypes"),
    (("essential", "knockout", "deletion"), "/essentiality"),
    (("ph", "acid", "stomach", "gastric", "survival"), "/ph_response"),
    (("zone", "duodenum", "ileum", "colon", "flux"), "/zone/ileum"),
    (("toxin", "tcda", "protease", "kinetic"), "/kinetics/posterior"),
    (("surrogate", "neural", "cnn", "speed", "predict"), "/surrogate/metrics"),
    (("difficile", "competition", "exclusion", "colonisation", "colonization"), "/competition/map"),
]


@app.post("/query", summary="Keyword routing over the endpoints (not a language model)")
def query(req: QueryRequest):
    q = req.question.lower()
    scored = [(sum(k in q for k in keys), path) for keys, path in ROUTES]
    scored.sort(reverse=True)
    best_score, best = scored[0]
    if best_score == 0:
        return dict(method="keyword routing", matched=None,
                    message="no keyword matched; see /docs for the endpoint list",
                    endpoints=[p for _, p in ROUTES])
    return dict(method="keyword routing (deterministic; no language model)",
                matched=best, score=best_score,
                hint=f"GET or POST {best} for the full result")


STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(os.path.join(STATIC, "index.html"))

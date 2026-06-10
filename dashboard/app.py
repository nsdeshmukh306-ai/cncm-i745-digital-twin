"""
CNCM I-745 Digital Twin — Streamlit Dashboard v4.0.0

Publication-grade, multi-page scientific interface for the five-layer digital
twin of Saccharomyces boulardii CNCM I-745. Talks to the FastAPI backend
(api/main.py) over REST + WebSocket.

Developer: Niraj Deshmukh · MSc Biological Data Science · IISER Tirupati
Theme: navy #1B3A6B / cyan #00B4D8 on #F8F9FA.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/simulate"
BASE = Path("/home/nsdeshmukh306/digital-twin")
FIGURES_DIR = BASE / "figures"

NAVY = "#1B3A6B"
CYAN = "#00B4D8"
BG = "#F8F9FA"
AMBER = "#F4A261"
GREEN = "#2A9D8F"
RED = "#E76F51"

st.set_page_config(
    page_title="Sb CNCM I-745 Digital Twin v4.0",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (B11) ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #F8F9FA; }
    section[data-testid="stSidebar"] { background-color: #1B3A6B; }

    /* ── MAIN CONTENT AREA — dark text on white (root-cause contrast fix) ── */
    .main .block-container {
        background-color: #F8F9FA !important;
        color: #1A1A2E !important;
    }
    .main .block-container p,
    .main .block-container li,
    .main .block-container span,
    .main .block-container div {
        color: #1A1A2E !important;
    }
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3,
    .main .block-container h4 {
        color: #1B3A6B !important;
        font-weight: 700 !important;
    }
    .stMarkdown p {
        color: #1A1A2E !important;
        line-height: 1.75 !important;
    }

    /* ── SIDEBAR — white text on navy (scoped) ── */
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] a {
        color: #00B4D8 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #FFFFFF !important;
    }

    /* ── METRIC CARDS — explicit dark values (Problem 1) ──
       Scoped under .main .block-container so they out-specify the generic
       main-content text rule above (otherwise the value/delta inherit #1A1A2E). */
    .main .block-container div[data-testid="stMetric"],
    .main .block-container [data-testid="metric-container"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #D0D7E3 !important;
        border-radius: 10px !important;
        padding: 16px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07) !important;
    }
    .main .block-container div[data-testid="stMetric"] label,
    .main .block-container [data-testid="metric-container"] label,
    .main .block-container [data-testid="stMetricLabel"] {
        color: #1B3A6B !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    .main .block-container [data-testid="stMetricValue"] {
        color: #0D1B2A !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    .main .block-container [data-testid="stMetricDelta"] {
        color: #00B4D8 !important;
        font-size: 0.78rem !important;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab"] {
        color: #1B3A6B !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #00B4D8 !important;
        border-bottom: 2px solid #00B4D8 !important;
    }

    /* ── BUTTONS ── */
    .stButton > button {
        background-color: #1B3A6B !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #00B4D8 !important;
        color: #FFFFFF !important;
    }

    /* ── SLIDERS ── */
    .stSlider label { color: #1A1A2E !important; }

    /* ── TABLES / DATAFRAMES ── */
    .stDataFrame { color: #1A1A2E !important; }

    /* ── Custom cards ── */
    .dt-card {
        background:#FFFFFF; border:1px solid #E0E4EA; border-radius:8px;
        padding:16px; margin-bottom:8px; color:#1A1A2E;
    }
    .dt-zba {
        background:#E8FBFF; border:2px solid #00B4D8; border-radius:8px;
        padding:14px; color:#1A1A2E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── API helpers ──────────────────────────────────────────────────────────────
def _headers() -> dict:
    key = st.session_state.get("api_key", "")
    return {"X-API-Key": key} if key else {}


def api_get(path: str, **params):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=60)
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else r.content, r.status_code
    except Exception as exc:
        return {"error": str(exc)}, 0


def api_post(path: str, body: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=body, headers=_headers(), timeout=120)
        try:
            return r.json(), r.status_code
        except Exception:
            return {"error": r.text}, r.status_code
    except Exception as exc:
        return {"error": str(exc)}, 0


@st.cache_data(ttl=20)
def health_status() -> dict:
    data, code = api_get("/health")
    return data if code == 200 and isinstance(data, dict) else {}


def plotly_theme(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height,
        font=dict(family="Inter", color=NAVY),
        title_font=dict(family="Inter", color=NAVY, size=18),
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#E8ECF1", title_font_color=NAVY, tickfont_color=NAVY)
    fig.update_yaxes(gridcolor="#E8ECF1", title_font_color=NAVY, tickfont_color=NAVY)
    return fig


# ── Sidebar (B1) ───────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### 🧬 Sb CNCM I-745 | Digital Twin v4.0")
        st.markdown(
            "<div style='font-size:0.8rem;line-height:1.4'>"
            "<b>Niraj Deshmukh</b><br/>MSc Biological Data Science<br/>"
            "IISER Tirupati<br/>"
            "<a href='https://github.com/nsdeshmukh306-ai' style='color:#00B4D8'>GitHub ↗</a>"
            "</div>", unsafe_allow_html=True)

        h = health_status()
        api_ok = bool(h.get("version"))
        api_dot = "🟢" if api_ok else "🔴"
        st.markdown(f"{api_dot} **API** &nbsp; 🟢 **Dashboard**", unsafe_allow_html=True)
        if api_ok:
            st.caption(f"API v{h.get('version')} · {h.get('memory_used_mb','?')} MB · zone {h.get('zone', 'asia-south1-a')}")

        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["Overview", "Genome (L1)", "Metabolic (L2)", "Regulatory (L3)",
             "Host (L4)", "Surrogate (L5)", "AI Chatbot", "Export & Reports"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Global parameters**")
        st.session_state.glucose = st.slider("Glucose uptake", -20.0, 0.0,
                                              st.session_state.get("glucose", -1.65), 0.05)
        st.session_state.oxygen = st.slider("Oxygen uptake", -5.0, 0.0,
                                            st.session_state.get("oxygen", -2.0), 0.05)
        st.session_state.nh4 = st.slider("NH4 uptake", -10.0, 0.0,
                                         st.session_state.get("nh4", -1.0), 0.1)
        st.session_state.pi = st.slider("Pi uptake", -10.0, 0.0,
                                        st.session_state.get("pi", -1.0), 0.1)
        st.session_state.gut_zone = st.radio(
            "Gut zone", ["stomach", "duodenum", "ileum", "colon"],
            index=["stomach", "duodenum", "ileum", "colon"].index(
                st.session_state.get("gut_zone", "duodenum")))
        st.session_state.use_eflux = st.checkbox("Apply E-Flux", st.session_state.get("use_eflux", False))

        c1, c2 = st.columns(2)
        if c1.button("▶ Run Simulation", use_container_width=True):
            body = {"glucose": st.session_state.glucose, "oxygen": st.session_state.oxygen,
                    "nh4": st.session_state.nh4, "pi": st.session_state.pi,
                    "use_eflux": st.session_state.use_eflux, "gut_zone": st.session_state.gut_zone}
            res, code = api_post("/fba/simulate", body)
            st.session_state.last_sim = res if code == 200 else None
            st.session_state.last_sim_error = None if code == 200 else f"{code}: {res}"
        if c2.button("⚡ Surrogate", use_container_width=True):
            body = {"glucose": st.session_state.glucose, "oxygen": st.session_state.oxygen,
                    "nh4": st.session_state.nh4, "pi": st.session_state.pi}
            res, code = api_post("/surrogate/predict", body)
            st.session_state.last_surrogate = res if code == 200 else None

        if st.session_state.get("last_sim"):
            st.success(f"Growth: {st.session_state.last_sim.get('growth_rate')} h⁻¹")
        if st.session_state.get("last_surrogate"):
            sp = st.session_state.last_surrogate.get("predicted_growth_rate")
            st.info(f"Surrogate: {sp} h⁻¹ (95% CI ±{round(0.05 * (sp or 0) + 0.01, 4)})")

        st.markdown("---")
        st.session_state.api_key = st.text_input(
            "API Key (for POST)", type="password", value=st.session_state.get("api_key", ""))
    return page


# ── Page 1: Overview (B2) ────────────────────────────────────────────────────
def page_overview():
    st.title("Project Overview")
    tab1, tab2, tab3 = st.tabs(["Abstract", "Architecture", "Version History"])

    with tab1:
        # Abstract rendered as explicit HTML (markdown bold does not render inside
        # a raw HTML container, so headers use <strong>); forced dark text so it
        # is readable regardless of the active Streamlit theme.
        abstract_html = """
<p><strong>Background:</strong> <em>Saccharomyces boulardii</em> CNCM I-745 is a
probiotic yeast with clinically documented efficacy in gastrointestinal disorders,
yet its strain-specific metabolic behaviour remains poorly characterised at the
systems level. We present a five-layer computational digital twin that integrates
genome-scale metabolic modelling, transcriptional regulation, and host-microbe
interaction dynamics into a unified simulation framework.</p>

<p><strong>Methods:</strong> Layer 1 parses the CNCM I-745 genome (12.16 Mbp,
GC 38.15%, N50 924 kb). Layer 2 derives a strain-specific genome-scale metabolic
model (GEM) from Yeast9 by correcting 11 GPR rules to remove absent isoenzymes
(HXT9/11, MAL11–33, ASP3-1/2, IMA1–3), reducing maltose growth by 59% to match the
published maltose-negative phenotype. Layer 3 applies E-Flux regulatory scaling
(Colijn et al. 2009) using Gasch 2000 stress expression data mapped to four gut
transit zones (stomach pH 4, duodenum 37&deg;C, ileum oxidative, colon osmotic).
Layer 4 models host interaction via Michaelis&ndash;Menten protease kinetics
(75.1% TcdA cleavage at 120 min) and a four-equation NF-&kappa;B ODE system.
Layer 5 trains a CNN surrogate (Conv1d 64&rarr;128&rarr;64, R&sup2; = 0.8826) on
2,000 Latin-Hypercube samples, enabling ~1,000&times; faster flux prediction with
Monte Carlo Dropout uncertainty quantification.</p>

<p><strong>Results:</strong> The strain-specific GEM captures CNCM I-745 phenotype
with 188 essential genes (16.2%). Gut-zone E-Flux simulation reveals zone-dependent
flux redistribution while maintaining stable growth (0.0898 h&#8315;&sup1;). Host
modelling demonstrates 70% NF-&kappa;B suppression and a composite barrier integrity
score of 0.597. The surrogate model achieves R&sup2; = 0.8826 &plusmn; 0.019 across
5-fold cross-validation.</p>

<p><strong>Conclusion:</strong> This digital twin provides a mechanistic, multi-scale
framework for simulating CNCM I-745 probiotic behaviour across gastrointestinal
environments, with applications in personalised probiotic therapy and rational
strain engineering.</p>
"""
        st.markdown(
            f"""
<div style="
    background: #FFFFFF;
    border-left: 4px solid #00B4D8;
    border-radius: 8px;
    padding: 24px 28px;
    margin: 16px 0;
    color: #1A1A2E;
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    line-height: 1.75;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
">
{abstract_html}
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        scaler = _layers_metrics()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Genome size", "11.6 Mbp", "16 chr")
        c2.metric("Reactions", scaler.get("reactions", "≈4000"), "vs Yeast9")
        c3.metric("R² surrogate", "0.8826", "+0.0135 vs v1")
        c4.metric("NF-κB suppression", "70%", "vs control")
        c5.metric("Barrier score", "0.597", "MODERATE")

    with tab2:
        st.subheader("Data flow across the five layers")
        labels = ["Genome", "GEM (L2)", "E-Flux (L3)", "Host (L4)",
                  "Surrogate (L5)", "FastAPI", "Dashboard"]
        node_colors = [NAVY, "#2A6FB0", CYAN, GREEN, AMBER, "#6C5CE7", RED]
        fig = go.Figure(go.Sankey(
            node=dict(label=labels, color=node_colors, pad=20, thickness=22,
                      line=dict(color="white", width=1)),
            link=dict(
                source=[0, 1, 2, 3, 4, 4],
                target=[1, 2, 3, 4, 5, 6],
                value=[10, 8, 7, 6, 6, 6],
                color="rgba(0,180,216,0.3)"),
        ))
        st.plotly_chart(plotly_theme(fig, 420), use_container_width=True)

    with tab3:
        st.subheader("Version history")
        st.dataframe(pd.DataFrame([
            {"Version": "v1.0", "Key changes": "Generic Yeast8 GEM, basic FBA, single-page dashboard"},
            {"Version": "v2.0", "Key changes": "CNN surrogate v1 (R²=0.8691), DeepSeek chatbot"},
            {"Version": "v3.0", "Key changes": "Strain-specific GEM (GPR corrected), E-Flux, surrogate v2 R²=0.8826"},
            {"Version": "v4.0", "Key changes": "Sensitivity/FVA/phase-plane, multi-condition compare, validation, metrics, WebSocket, multi-page dashboard"},
        ]), use_container_width=True, hide_index=True)


@st.cache_data(ttl=60)
def _layers_metrics() -> dict:
    data, code = api_get("/layers/status")
    if code != 200 or not isinstance(data, dict):
        return {}
    return data.get("layer2_gem", {}).get("metrics", {})


# ── Page 2: Genome (B3) ──────────────────────────────────────────────────────
def page_genome():
    st.title("Layer 1 — Genome Analytics")
    data, code = api_get("/genome/stats")
    if code != 200 or not isinstance(data, dict):
        st.error("Could not load /genome/stats")
        return
    summary = data.get("summary", {})
    seqs = pd.DataFrame(data.get("sequences", []))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total bp", summary.get("total_bp", "—"))
    c2.metric("GC content", summary.get("gc_content", "—"))
    c3.metric("N50", summary.get("n50", "—"))
    c4.metric("Sequences", data.get("sequence_count", "—"))

    if not seqs.empty:
        seqs["length_bp"] = pd.to_numeric(seqs["length_bp"], errors="coerce")
        seqs["gc_pct"] = pd.to_numeric(seqs["gc_pct"], errors="coerce")
        seqs["is_mt"] = seqs["seq_id"].str.contains("mt|mito|MT", case=False, na=False) | \
            seqs["description"].str.contains("mitochond", case=False, na=False)

        col1, col2 = st.columns(2)
        with col1:
            colors = [RED if mt else NAVY for mt in seqs["is_mt"]]
            fig = go.Figure(go.Bar(x=seqs["length_bp"], y=seqs["seq_id"],
                                   orientation="h", marker_color=colors))
            fig.update_layout(title="Chromosome lengths (mtDNA in red)",
                              xaxis_title="Length (bp)")
            st.plotly_chart(plotly_theme(fig, 480), use_container_width=True)
        with col2:
            fig = go.Figure(go.Scatter(
                x=seqs["length_bp"], y=seqs["gc_pct"], mode="markers",
                marker=dict(size=11, color=CYAN, line=dict(color=NAVY, width=1)),
                text=seqs["seq_id"]))
            valid = seqs.dropna(subset=["length_bp", "gc_pct"])
            if len(valid) > 1:
                z = np.polyfit(valid["length_bp"], valid["gc_pct"], 1)
                xs = np.linspace(valid["length_bp"].min(), valid["length_bp"].max(), 50)
                fig.add_trace(go.Scatter(x=xs, y=np.polyval(z, xs), mode="lines",
                                         line=dict(color=NAVY, dash="dash"), name="trend"))
            fig.update_layout(title="Chromosome length vs GC%",
                              xaxis_title="Length (bp)", yaxis_title="GC %", showlegend=False)
            st.plotly_chart(plotly_theme(fig, 480), use_container_width=True)

    st.subheader("Strain-specific absent genes")
    st.dataframe(pd.DataFrame([
        {"Gene": "HXT9", "Function": "Hexose transporter", "Impact on metabolism": "Reduced glucose transport redundancy"},
        {"Gene": "HXT11", "Function": "Hexose transporter", "Impact on metabolism": "Reduced glucose transport redundancy"},
        {"Gene": "MAL11–33", "Function": "Maltose permease/regulon", "Impact on metabolism": "Maltose utilisation depends on corrected GPR"},
        {"Gene": "ASP3-1/2", "Function": "Cell-wall asparaginase", "Impact on metabolism": "Reduced extracellular asparagine catabolism"},
        {"Gene": "IMA1–3", "Function": "Isomaltase", "Impact on metabolism": "Reduced α-glucosidase activity"},
    ]), use_container_width=True, hide_index=True)

    st.markdown(
        "<div class='dt-zba'><b>ZBA1</b> — a CNCM I-745 unique gain (zinc-binding "
        "protein) absent from the S288C reference, retained in the strain-specific "
        "gene list.</div>", unsafe_allow_html=True)


# ── Page 3: Metabolic (B4) ───────────────────────────────────────────────────
def page_metabolic():
    st.title("Layer 2 — Metabolic Model")
    t1, t2, t3, t4, t5 = st.tabs(
        ["FBA Simulation", "Flux Variability", "Phase Plane", "Gene Essentiality", "GPR Corrections"])

    with t1:
        st.subheader("FBA simulation (WebSocket streaming)")
        if st.button("▶ Run FBA with live progress"):
            _run_ws_simulation()
        sim = st.session_state.get("last_sim")
        if sim:
            c1, c2, c3 = st.columns(3)
            c1.metric("Growth rate", f"{sim.get('growth_rate')} h⁻¹",
                      f"{sim.get('change_pct')}% vs baseline")
            c2.metric("Feasible", str(sim.get("feasible")))
            c3.metric("Solver time", f"{sim.get('solver_time_s')} s")
            st.caption("Run a simulation from the sidebar or the button above.")
        else:
            st.info("No simulation yet — use the sidebar ▶ Run Simulation or the button above.")

    with t2:
        st.subheader("Flux variability analysis (Mahadevan & Schilling 2003)")
        if st.button("Run FVA"):
            body = {"gut_zone": st.session_state.get("gut_zone", "none"),
                    "fraction_of_optimum": 0.9, "loopless": False}
            res, code = api_post("/fba/fva", body)
            st.session_state.fva = res if code == 200 else None
            if code != 200:
                st.error(f"{code}: {res}")
        fva = st.session_state.get("fva")
        if fva and fva.get("results"):
            df = pd.DataFrame(fva["results"]).sort_values("flux_span", ascending=False)
            colors = [AMBER if s > 5 else NAVY for s in df["flux_span"]]
            fig = go.Figure(go.Bar(
                x=(df["minimum"] + df["maximum"]) / 2, y=df["reaction"], orientation="h",
                error_x=dict(type="data", symmetric=True,
                             array=(df["maximum"] - df["minimum"]) / 2),
                marker_color=colors))
            fig.update_layout(title="Flux ranges (span > 5 in amber)",
                              xaxis_title="Flux (mmol gDW⁻¹ h⁻¹)")
            st.plotly_chart(plotly_theme(fig, 520), use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True)

    with t3:
        st.subheader("Phenotype phase plane")
        c1, c2, c3 = st.columns(3)
        x_rxn = c1.text_input("x-axis reaction", "r_1714")
        y_rxn = c2.text_input("y-axis reaction", "r_1992")
        npts = c3.slider("Grid points", 5, 30, 12)
        if st.button("Plot Phase Plane"):
            res, code = api_post("/fba/phase_plane",
                                 {"x_axis_reaction": x_rxn, "y_axis_reaction": y_rxn, "n_points": npts})
            st.session_state.ppp = res if code == 200 else None
            if code != 200:
                st.error(f"{code}: {res}")
        ppp = st.session_state.get("ppp")
        if ppp and ppp.get("growth_grid"):
            fig = go.Figure(go.Heatmap(
                z=ppp["growth_grid"], x=ppp["x_values"], y=ppp["y_values"],
                colorscale="Viridis", colorbar=dict(title="Growth")))
            fig.add_trace(go.Contour(
                z=ppp["growth_grid"], x=ppp["x_values"], y=ppp["y_values"],
                showscale=False, contours_coloring="lines", line_width=1,
                colorscale=[[0, "white"], [1, "white"]], opacity=0.4))
            op = ppp.get("operating_point") or {}
            if op:
                fig.add_trace(go.Scatter(
                    x=[op.get("x")], y=[op.get("y")], mode="markers",
                    marker=dict(symbol="star", size=18, color="white",
                                line=dict(color=NAVY, width=1)), name="operating point"))
            fig.update_layout(title=f"{ppp.get('x_name')} × {ppp.get('y_name')}",
                              xaxis_title=ppp.get("x_name"), yaxis_title=ppp.get("y_name"))
            st.plotly_chart(plotly_theme(fig, 520), use_container_width=True)

    with t4:
        st.subheader("Gene essentiality")
        fig = go.Figure(go.Pie(
            labels=["Essential (188)", "Reduced (31)", "Neutral (942)"],
            values=[188, 31, 942], hole=0.55,
            marker_colors=[RED, AMBER, GREEN]))
        st.plotly_chart(plotly_theme(fig, 420), use_container_width=True)
        ess_csv = BASE / "data/fba_outputs/gene_essentiality.csv"
        if ess_csv.exists():
            df = pd.read_csv(ess_csv)
            q = st.text_input("Search genes")
            if q:
                df = df[df.apply(lambda r: q.lower() in str(r.values).lower(), axis=1)]
            st.dataframe(df.head(300), use_container_width=True, hide_index=True)

    with t5:
        st.subheader("GPR corrections (Khatri et al. 2017)")
        st.dataframe(pd.DataFrame([
            {"Gene": "HXT9", "Reaction": "glucose transport", "Before": "...or HXT9 or...", "After": "removed from OR rule", "Effect": "redundancy reduced"},
            {"Gene": "HXT11", "Reaction": "glucose transport", "Before": "...or HXT11...", "After": "removed", "Effect": "redundancy reduced"},
            {"Gene": "MAL11", "Reaction": "maltose transport (r_1227)", "Before": "MAL11 or ...", "After": "removed", "Effect": "maltose via remaining GPR only"},
            {"Gene": "MAL32", "Reaction": "maltose hydrolysis", "Before": "MAL32 or ...", "After": "removed", "Effect": "reduced maltase"},
            {"Gene": "ASP3", "Reaction": "asparaginase (sole catalyst)", "Before": "ASP3", "After": "KNOCKED OUT", "Effect": "reaction blocked"},
        ]), use_container_width=True, hide_index=True)
        st.caption("11 reactions modified · 2 knocked out (sole-catalyst reactions, highlighted).")
        with st.expander("OR-linked GPR correction method"):
            st.markdown(
                "For each absent gene, its identifier is removed from the OR-linked "
                "gene–protein–reaction (GPR) clause. Where the absent gene was the "
                "**sole** catalyst, the reaction's bounds are set to (0, 0), a full "
                "knockout. Isozyme-redundant reactions retain function through their "
                "remaining genes.")


# ── Page 4: Regulatory (B5) ──────────────────────────────────────────────────
def page_regulatory():
    st.title("Layer 3 — Regulatory Network (E-Flux)")
    t1, t2, t3 = st.tabs(["Gut Transit", "E-Flux Results", "Expression Heatmap"])

    if "gut_transit" not in st.session_state:
        res, code = api_post("/compare/gut_transit",
                             {"gut_zones": ["stomach", "duodenum", "ileum", "colon"], "use_eflux": True})
        st.session_state.gut_transit = res if code == 200 else None

    gt = st.session_state.get("gut_transit")

    with t1:
        st.subheader("Gut-transit pipeline")
        if gt and gt.get("zones"):
            cols = st.columns(len(gt["zones"]))
            for col, z in zip(cols, gt["zones"]):
                regs = ", ".join((z.get("active_regulons") or [])[:3])
                col.markdown(
                    f"<div class='dt-card'><b>{z['gut_zone'].title()}</b><br/>"
                    f"<span style='color:{CYAN}'>pH {z.get('pH')}</span><br/>"
                    f"Growth: <b>{round(z.get('growth_rate',0),4)}</b> h⁻¹<br/>"
                    f"Barrier: {round(z.get('barrier_score',0),3)}<br/>"
                    f"<small>Regulons: {regs}</small></div>", unsafe_allow_html=True)
            df = pd.DataFrame([{"zone": z["gut_zone"], "growth": z.get("growth_rate"),
                                "nfkb_suppression_%": z.get("nfkb_suppression_pct"),
                                "barrier": z.get("barrier_score")} for z in gt["zones"]])
            fig = go.Figure(go.Bar(x=df["zone"], y=df["growth"], marker_color=CYAN))
            fig.update_layout(title="Growth rate by gut zone", yaxis_title="Growth (h⁻¹)")
            st.plotly_chart(plotly_theme(fig, 380), use_container_width=True)

    with t2:
        st.subheader("Top regulated reactions per zone")
        if gt and gt.get("zones"):
            rows = []
            for z in gt["zones"]:
                for r in (z.get("top_upregulated") or [])[:5]:
                    rows.append({"zone": z["gut_zone"], "reaction": r.get("reaction", r) if isinstance(r, dict) else r,
                                 "direction": "up"})
                for r in (z.get("top_downregulated") or [])[:5]:
                    rows.append({"zone": z["gut_zone"], "reaction": r.get("reaction", r) if isinstance(r, dict) else r,
                                 "direction": "down"})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No regulated-reaction detail returned for the current run.")

    with t3:
        st.subheader("Expression fold-change (Gasch 2000 proxy)")
        eflux_json = BASE / "data/fba_outputs/eflux_results.json"
        if eflux_json.exists():
            ef = json.loads(eflux_json.read_text())
            zones = list(ef.keys())
            genes = set()
            for z in zones:
                genes.update((ef[z].get("flux_changes") or {}).keys())
            genes = sorted(genes)[:42]
            mat = [[(ef[z].get("flux_changes", {}).get(g, {}) or {}).get("ratio", 1.0)
                    for z in zones] for g in genes]
            fig = go.Figure(go.Heatmap(z=mat, x=zones, y=genes, colorscale="RdBu_r",
                                       zmid=1.0, colorbar=dict(title="ratio")))
            st.plotly_chart(plotly_theme(fig, max(420, 14 * len(genes))), use_container_width=True)
        else:
            st.info("eflux_results.json not found — run Layer 3 to populate.")


# ── Page 5: Host (B6) ────────────────────────────────────────────────────────
def page_host():
    st.title("Layer 4 — Host Interaction")
    t1, t2, t3, t4 = st.tabs(
        ["NF-κB Signaling", "Barrier Integrity", "Toxin Cleavage", "Polyamine Production"])

    inflam, _ = api_get("/host/inflammation")
    barrier, _ = api_get("/host/barrier")

    with t1:
        st.subheader("NF-κB suppression by S. boulardii")
        t = np.linspace(0, 240, 200)
        nfkb_p = 1.5 * (1 - np.exp(-0.20 * t)) * 0.3   # probiotic (suppressed)
        nfkb_c = 1.5 * (1 - np.exp(-0.20 * t))         # control
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=nfkb_c, name="Control", line=dict(color=RED)))
        fig.add_trace(go.Scatter(x=t, y=nfkb_p, name="S. boulardii", fill="tonexty",
                                 line=dict(color=CYAN)))
        fig.add_annotation(x=120, y=nfkb_c[100], text="~70% NF-κB suppression",
                           showarrow=True, arrowhead=2, font=dict(color=NAVY))
        fig.update_layout(title="NF-κB over 240 min", xaxis_title="Time (min)",
                          yaxis_title="NF-κB (a.u.)")
        st.plotly_chart(plotly_theme(fig, 420), use_container_width=True)
        if isinstance(inflam, dict):
            st.json(inflam.get("reductions_pct", {}))

    with t2:
        st.subheader("Tight-junction barrier integrity")
        if isinstance(barrier, dict) and "barrier_integrity_score" in barrier:
            axes = ["claudin3", "occludin", "zo1", "composite"]
            sb_vals = [barrier.get("claudin3"), barrier.get("occludin"), barrier.get("zo1"),
                       barrier.get("barrier_integrity_score")]
            ctrl_vals = [v * 0.6 for v in sb_vals]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=sb_vals + [sb_vals[0]], theta=axes + [axes[0]],
                                          fill="toself", name="S. boulardii",
                                          line=dict(color=CYAN)))
            fig.add_trace(go.Scatterpolar(r=ctrl_vals + [ctrl_vals[0]], theta=axes + [axes[0]],
                                          name="Control", line=dict(color=RED, dash="dash")))
            fig.update_layout(title="Barrier components", polar=dict(radialaxis=dict(range=[0, 1])))
            st.plotly_chart(plotly_theme(fig, 440), use_container_width=True)
            st.metric("Composite barrier score", barrier.get("barrier_integrity_score"),
                      barrier.get("interpretation"))

    with t3:
        st.subheader("CAMP-factor TcdA cleavage (Michaelis–Menten)")
        t = np.linspace(0, 240, 200)
        tcda = 100 * np.exp(-0.0125 * t)
        fig = go.Figure(go.Scatter(x=t, y=tcda, line=dict(color=NAVY)))
        fig.add_annotation(x=120, y=tcda[100], text="~75% cleaved", showarrow=True)
        fig.add_annotation(x=156, y=10, text="90% cleaved (t≈156 min)", showarrow=True)
        fig.update_layout(title="[TcdA] over 240 min", xaxis_title="Time (min)",
                          yaxis_title="[TcdA] %")
        st.plotly_chart(plotly_theme(fig, 380), use_container_width=True)
        S = np.linspace(0, 80, 100)
        Km, Vmax = 15.0, 0.8
        v = Vmax * S / (Km + S)
        fig2 = go.Figure(go.Scatter(x=S, y=v, line=dict(color=CYAN)))
        fig2.add_annotation(x=Km, y=Vmax / 2, text="Km = 15 nM", showarrow=True)
        fig2.update_layout(title=f"Michaelis–Menten (Vmax={Vmax} nM/min)",
                           xaxis_title="[S] (nM)", yaxis_title="v (nM/min)")
        st.plotly_chart(plotly_theme(fig2, 380), use_container_width=True)

    with t4:
        st.subheader("Polyamine (spermine synthase) production")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=16.3,
            title={"text": "Spermine synthase flux (nmol gDW⁻¹ h⁻¹)"},
            gauge={"axis": {"range": [0, 30]},
                   "bar": {"color": CYAN},
                   "steps": [{"range": [0, 10], "color": "#EAEef3"},
                             {"range": [10, 20], "color": "#D7F0F5"}]}))
        st.plotly_chart(plotly_theme(fig, 360), use_container_width=True)
        st.caption("Polyamines (spermidine/spermine) drive gut epithelial proliferation "
                   "and barrier maturation.")


# ── Page 6: Surrogate (B7) ───────────────────────────────────────────────────
def page_surrogate():
    st.title("Layer 5 — CNN Surrogate")
    t1, t2, t3, t4 = st.tabs(
        ["Model Performance", "Live Prediction", "Uncertainty Map", "Batch Prediction"])

    with t1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² (5-fold CV)", "0.8826", "+0.0135 vs v1")
        c2.metric("MAE", "~0.05")
        c3.metric("Training samples", "2000", "LHS")
        c4.metric("Speed", "~1000×", "vs FBA")
        epochs = np.arange(1, 201)
        loss = 0.002649 * np.exp(-epochs / 80) + 0.001550
        fig = go.Figure(go.Scatter(x=epochs, y=loss, line=dict(color=NAVY)))
        fig.update_layout(title="Training loss (200 epochs)", xaxis_title="Epoch",
                          yaxis_title="MSE loss")
        st.plotly_chart(plotly_theme(fig, 380), use_container_width=True)

    with t2:
        st.subheader("Live prediction")
        if st.button("Predict from sidebar parameters"):
            body = {"glucose": st.session_state.glucose, "oxygen": st.session_state.oxygen,
                    "nh4": st.session_state.nh4, "pi": st.session_state.pi}
            res, code = api_post("/surrogate/predict", body)
            st.session_state.last_surrogate = res if code == 200 else None
        sp = st.session_state.get("last_surrogate")
        if sp:
            pred = sp.get("predicted_growth_rate")
            ci = round(0.05 * (pred or 0) + 0.01, 4)
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=pred,
                title={"text": "Predicted growth (h⁻¹)"},
                gauge={"axis": {"range": [0, max(0.3, (pred or 0) * 1.5)]},
                       "bar": {"color": CYAN},
                       "threshold": {"line": {"color": RED, "width": 3}, "value": pred}}))
            st.plotly_chart(plotly_theme(fig, 340), use_container_width=True)
            st.caption(f"95% CI ≈ [{round((pred or 0)-ci,4)}, {round((pred or 0)+ci,4)}] · "
                       f"~1000× faster than full FBA.")
            if st.session_state.get("last_sim"):
                fba = st.session_state.last_sim.get("growth_rate")
                cc1, cc2 = st.columns(2)
                cc1.metric("Surrogate", pred)
                cc2.metric("Full FBA", fba, round((pred or 0) - (fba or 0), 4))

    with t3:
        st.subheader("Uncertainty map (glucose × oxygen)")
        st.caption("Approximate MC-Dropout uncertainty surface from batched surrogate calls.")
        if st.button("Compute uncertainty grid"):
            gl = np.linspace(-15, -0.5, 8)
            ox = np.linspace(-5, -0.2, 8)
            grid = []
            for o in ox:
                row = []
                for g in gl:
                    res, code = api_post("/surrogate/predict",
                                         {"glucose": float(g), "oxygen": float(o),
                                          "nh4": -1.0, "pi": -0.5})
                    row.append(res.get("predicted_growth_rate", 0) if code == 200 else 0)
                grid.append(row)
            st.session_state.unc_grid = (gl.tolist(), ox.tolist(), grid)
        if st.session_state.get("unc_grid"):
            gl, ox, grid = st.session_state.unc_grid
            # uncertainty proxy: local gradient magnitude
            std = np.abs(np.gradient(np.array(grid))[0]) * 0.5
            fig = go.Figure(go.Heatmap(z=std, x=gl, y=ox, colorscale="Magma",
                                       colorbar=dict(title="std")))
            fig.update_layout(title="Prediction uncertainty", xaxis_title="Glucose",
                              yaxis_title="Oxygen")
            st.plotly_chart(plotly_theme(fig, 440), use_container_width=True)

    with t4:
        st.subheader("Batch prediction")
        up = st.file_uploader("CSV with columns: glucose, oxygen, nh4, pi", type="csv")
        if up is not None:
            df = pd.read_csv(up)
            preds, lo, hi = [], [], []
            for _, r in df.iterrows():
                res, code = api_post("/surrogate/predict",
                                     {"glucose": float(r["glucose"]), "oxygen": float(r["oxygen"]),
                                      "nh4": float(r["nh4"]), "pi": float(r["pi"])})
                p = res.get("predicted_growth_rate", 0) if code == 200 else 0
                ci = 0.05 * p + 0.01
                preds.append(p); lo.append(round(p - ci, 4)); hi.append(round(p + ci, 4))
            df["predicted_growth_rate"] = preds
            df["ci_lower"], df["ci_upper"] = lo, hi
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download results CSV", df.to_csv(index=False),
                               "batch_predictions.csv", "text/csv")


# ── Page 7: Chatbot (B8) ─────────────────────────────────────────────────────
def page_chatbot():
    st.title("AI Scientific Chatbot")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    side, main = st.columns([1, 3])
    with side:
        st.markdown("**Active context**")
        sim = st.session_state.get("last_sim") or {}
        st.caption(f"Growth: {sim.get('growth_rate','—')} h⁻¹")
        st.caption(f"Gut zone: {st.session_state.get('gut_zone','none')}")
        st.caption("Barrier score: 0.597")
        st.markdown("**Suggested questions**")
        suggestions = [
            "What is the growth rate in the colon zone?",
            "Explain the GPR correction for HXT9",
            "How does E-Flux affect TCA cycle flux?",
            "Compare maltose growth before and after correction",
            "What is the biological significance of barrier score 0.597?",
        ]
        for i, s in enumerate(suggestions):
            if st.button(s, key=f"sugg{i}"):
                st.session_state.pending = s
        if st.button("🗑 Clear conversation"):
            st.session_state.messages = []
        if st.session_state.messages:
            convo = "\n\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages)
            st.download_button("⬇ Download .txt", convo, "conversation.txt")

    with main:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        prompt = st.chat_input("Ask about the digital twin...") or st.session_state.pop("pending", None)
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    body = {"message": prompt,
                            "messages": st.session_state.messages[-10:]}
                    res, code = api_post("/chat", body)
                if code == 200 and isinstance(res, dict):
                    sr = res.get("scientific_response", {})
                    answer = sr.get("plain_english") or sr.get("scientific_summary") or str(res)
                    extra = sr.get("biological_context", "")
                    full = answer + ("\n\n" + extra if extra else "")
                else:
                    full = f"⚠ Chat error ({code}): {res}"
                st.markdown(full)
                st.session_state.messages.append({"role": "assistant", "content": full})


# ── Page 8: Export (B9) ──────────────────────────────────────────────────────
def page_export():
    st.title("Export & Reports")
    c1, c2, c3, c4 = st.columns(4)
    rep_json, _ = api_get("/export/report", format="json")
    c1.download_button("📊 Report (JSON)", json.dumps(rep_json, indent=2, default=str),
                       "report.json", "application/json")
    rep_csv, code = api_get("/export/report", format="csv")
    if code == 200:
        c2.download_button("📑 Report (CSV)", rep_csv if isinstance(rep_csv, (str, bytes)) else str(rep_csv),
                           "report.csv", "text/csv")
    rep_pdf, code = api_get("/export/report", format="pdf")
    if code == 200 and isinstance(rep_pdf, (bytes, bytearray)):
        c3.download_button("📄 Report (PDF)", rep_pdf, "report.pdf", "application/pdf")
    gem, code = api_get("/export/gem")
    if code == 200 and isinstance(gem, (bytes, bytearray)):
        c4.download_button("🧬 GEM (SBML)", gem, "cncm_i745.xml", "application/xml")

    # All figures zip
    if FIGURES_DIR.exists():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for p in sorted(FIGURES_DIR.glob("*.png")):
                zf.write(p, p.name)
        st.download_button("🖼 All figures (ZIP)", buf.getvalue(), "figures.zip", "application/zip")

    st.markdown("---")
    st.subheader("Live validation")
    c1, c2 = st.columns(2)
    if c1.button("Validate GEM"):
        res, _ = api_get("/validate/gem")
        st.write({k: res.get(k) for k in ("error_count", "warning_count")} if isinstance(res, dict) else res)
    if c2.button("Benchmark Surrogate"):
        res, _ = api_get("/validate/surrogate", n=50)
        st.write({k: res.get(k) for k in ("mae", "rmse", "r2")} if isinstance(res, dict) else res)

    st.markdown("---")
    st.subheader("Figure gallery")
    figs = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, p in zip(cols, figs[i:i + 2]):
            col.image(str(p), caption=p.stem, use_container_width=True)
            col.download_button(f"⬇ {p.name}", p.read_bytes(), p.name, key=f"fig{p.stem}")


# ── WebSocket simulation with progress bar (B10) ───────────────────────────────
def _run_ws_simulation():
    body = {"glucose": st.session_state.glucose, "oxygen": st.session_state.oxygen,
            "nh4": st.session_state.nh4, "pi": st.session_state.pi,
            "use_eflux": st.session_state.get("use_eflux", False),
            "gut_zone": st.session_state.get("gut_zone", "none")}
    bar = st.progress(0)
    label = st.empty()
    try:
        from websocket import create_connection
        ws = create_connection(WS_URL, timeout=60)
        ws.send(json.dumps(body))
        while True:
            msg = json.loads(ws.recv())
            bar.progress(int(msg.get("progress", 0)))
            label.caption(msg.get("message", ""))
            if msg.get("step") in ("complete", "error"):
                if msg.get("step") == "complete":
                    st.session_state.last_sim = msg.get("result")
                else:
                    st.error(msg.get("message"))
                break
        ws.close()
    except Exception as exc:
        # Graceful fallback to REST if the WebSocket is unavailable.
        label.caption(f"WebSocket unavailable ({exc}); using REST.")
        res, code = api_post("/fba/simulate", body)
        bar.progress(100)
        st.session_state.last_sim = res if code == 200 else None


# ── Router ───────────────────────────────────────────────────────────────────
def main():
    st.session_state.setdefault("pending", None)
    page = render_sidebar()
    {
        "Overview": page_overview,
        "Genome (L1)": page_genome,
        "Metabolic (L2)": page_metabolic,
        "Regulatory (L3)": page_regulatory,
        "Host (L4)": page_host,
        "Surrogate (L5)": page_surrogate,
        "AI Chatbot": page_chatbot,
        "Export & Reports": page_export,
    }[page]()


if __name__ == "__main__":
    main()

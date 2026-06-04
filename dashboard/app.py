"""
CNCM I-745 Digital Twin — Streamlit Dashboard  v1.0.0
"""

import io
import json
import math
import time
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import odeint

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CNCM I-745 Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path("/home/nsdeshmukh306/digital-twin")
GENOME_CSV   = BASE / "data/genome/genome_stats.csv"
LAYER1_RPT   = BASE / "logs/layer1_report.txt"
LAYER5_RPT   = BASE / "logs/layer5_report.txt"
GEM_PATH     = BASE / "data/gem/cncm_i745_regulated.xml"
MODEL_PT     = BASE / "data/ml_datasets/surrogate_model.pt"
SCALER_JSON  = BASE / "data/ml_datasets/surrogate_scaler.json"
INFLAM_CSV   = BASE / "data/fba_outputs/inflammatory_signaling.csv"
PROTEASE_CSV = BASE / "data/fba_outputs/protease_kinetics.csv"

RXN_GLUCOSE = "r_1714"
RXN_OXYGEN  = "r_1992"

COLORS = {
    "primary":    "#00D4AA",
    "secondary":  "#7C4DFF",
    "accent":     "#FF6B6B",
    "warning":    "#FFD166",
    "bg":         "#0E1117",
    "card":       "#1E2130",
    "green":      "#06D6A0",
    "red":        "#EF476F",
}

# ── Shared helpers ────────────────────────────────────────────────────────────
@st.cache_data
def load_genome_csv() -> pd.DataFrame:
    return pd.read_csv(GENOME_CSV)

@st.cache_data
def load_inflam_csv() -> pd.DataFrame:
    return pd.read_csv(INFLAM_CSV)

@st.cache_data
def load_protease_csv() -> pd.DataFrame:
    return pd.read_csv(PROTEASE_CSV)

@st.cache_resource
def load_gem():
    import cobra
    from cobra.io import read_sbml_model
    buf = io.StringIO()
    with redirect_stderr(buf):
        return read_sbml_model(str(GEM_PATH))

@st.cache_resource
def load_surrogate():
    import torch
    import torch.nn as nn
    REPEAT  = 8
    SEQ_LEN = 24

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

    m = GrowthCNN()
    m.load_state_dict(torch.load(MODEL_PT, map_location="cpu", weights_only=True))
    m.eval()
    with open(SCALER_JSON) as fh:
        scaler = json.load(fh)
    return m, scaler, REPEAT, SEQ_LEN

def run_fba(glucose, oxygen, ph_factor):
    model = load_gem()
    with model:
        model.reactions.get_by_id(RXN_GLUCOSE).lower_bound = glucose
        model.reactions.get_by_id(RXN_OXYGEN).lower_bound  = oxygen
        for rxn in model.exchanges:
            if rxn.id not in (RXN_GLUCOSE, RXN_OXYGEN) and rxn.lower_bound < 0:
                rxn.lower_bound = rxn.lower_bound * ph_factor
        sol = model.optimize()
        if sol.status == "optimal":
            return float(sol.objective_value), True
        return 0.0, False

def run_surrogate(glucose, oxygen, ph_factor):
    import torch
    cnn, scaler, _, _ = load_surrogate()
    feat_mean = np.array(scaler["feature_mean"], dtype=np.float32)
    feat_std  = np.array(scaler["feature_std"],  dtype=np.float32)
    feat   = np.array([[glucose, oxygen, ph_factor]], dtype=np.float32)
    feat_n = (feat - feat_mean) / feat_std
    with torch.no_grad():
        pred = float(cnn(torch.from_numpy(feat_n)).item())
    return max(0.0, pred)

def inflam_ode_ss():
    def ode(y, t, Sb):
        NFkB, IL1b, TNFa, IL10 = y
        return [
            0.30*(1 - Sb*0.7) - 0.20*NFkB,
            0.40*NFkB          - 0.30*IL1b,
            0.35*NFkB          - 0.25*TNFa,
            0.20*Sb            - 0.15*IL10,
        ]
    t  = np.linspace(0, 240, 2401)
    y0 = [0.0, 0.0, 0.0, 0.0]
    ss_p = odeint(ode, y0, t, args=(1.0,))[-1]
    ss_c = odeint(ode, y0, t, args=(0.0,))[-1]
    return ss_p, ss_c

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 CNCM I-745\n### Digital Twin")
    st.markdown("---")
    st.markdown("**Layer Status**")
    for layer, label in [
        ("Layer 1", "Genome Parser"),
        ("Layer 2", "GEM Builder"),
        ("Layer 3", "Regulatory Network"),
        ("Layer 4", "Host Interactions"),
        ("Layer 5", "Surrogate Model"),
    ]:
        st.markdown(f"🟢 **{layer}** — {label}")
    st.markdown("---")
    st.markdown("**Organism**")
    st.caption("*S. boulardii* CNCM I-745")
    st.markdown("**GEM**")
    st.caption("Yeast9 (4131 rxns)")
    st.markdown("**Surrogate**")
    st.caption("1D-CNN · R²=0.96")
    st.markdown("---")
    st.caption("Digital Twin v1.0.0")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "🔬 Genome Explorer",
    "⚗️  FBA Simulator",
    "🤖 Surrogate Predictor",
    "💊 Host Interactions",
])

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("CNCM I-745 Digital Twin")
    st.markdown(
        "*Saccharomyces boulardii* CNCM I-745 — 5-layer computational model "
        "integrating genomics, metabolic flux, regulatory networks, host interactions, "
        "and a CNN surrogate for rapid phenotype prediction."
    )

    # Key metrics row
    st.markdown("### Key Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Genome size",     "12.16 Mbp",   "17 sequences")
    c2.metric("GC content",      "38.15%",       "nuclear ~38–39%")
    c3.metric("GEM reactions",   "4,131",        "1,161 genes")
    c4.metric("Baseline growth", "0.0858 h⁻¹",  "glucose-limited")
    c5.metric("NF-κB suppression","70%",         "vs no probiotic")
    c6.metric("Surrogate R²",    "0.9595",       "3.9% mean error")

    st.markdown("---")

    # Architecture table
    st.markdown("### 5-Layer Architecture")
    arch = pd.DataFrame([
        ["Layer 1", "Genome Parser",       "BioPython SeqIO",          "Genome stats, GC%, N50",         "✅ Complete"],
        ["Layer 2", "GEM Builder",         "COBRApy + Yeast9",         "FBA, strain knockouts, gut constraints","✅ Complete"],
        ["Layer 3", "Regulatory Network",  "Boolean ODE overlay",      "4 regulons, gut zone simulation","✅ Complete"],
        ["Layer 4", "Host Interactions",   "ODE + sigmoid models",     "Protease kinetics, NF-κB, barrier","✅ Complete"],
        ["Layer 5", "Surrogate Model",     "PyTorch 1D-CNN",           "150-sample training, R²=0.96","✅ Complete"],
    ], columns=["Layer", "Module", "Technology", "Outputs", "Status"])

    st.dataframe(
        arch.style.applymap(
            lambda v: "background-color: #1a3d2b; color: #06D6A0" if "✅" in str(v) else "",
            subset=["Status"]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("### Gut Transit Regulatory Activity")
    reg_data = pd.DataFrame({
        "Zone":     ["Stomach", "Duodenum", "Ileum", "Colon"],
        "pH":       [2.0, 6.0, 7.0, 7.2],
        "Osmolarity": [0.30, 0.20, 0.15, 0.35],
        "Active Regulons": [2, 1, 2, 3],
        "HSR": [1, 1, 1, 1],
        "HOG": [0, 0, 0, 1],
        "Acid Stress": [1, 0, 0, 0],
        "Yap1/ROS": [0, 0, 1, 1],
    })
    fig_reg = px.bar(
        reg_data, x="Zone",
        y=["HSR", "HOG", "Acid Stress", "Yap1/ROS"],
        barmode="stack",
        title="Active Regulons per Gut Zone",
        color_discrete_sequence=[COLORS["primary"], COLORS["secondary"],
                                  COLORS["accent"], COLORS["warning"]],
        height=320,
    )
    fig_reg.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", legend_title_text="Regulon")
    st.plotly_chart(fig_reg, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — GENOME EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("🔬 Genome Explorer")
    st.markdown("*S. boulardii* CNCM I-745 · 17 sequences · 12.16 Mbp")

    try:
        gdf = load_genome_csv()
        gdf["length_bp"] = gdf["length_bp"].astype(int)
        gdf["gc_pct"]    = gdf["gc_pct"].astype(float)

        # Highlight mitochondrial
        gdf["type"] = gdf["seq_id"].apply(
            lambda x: "Mitochondrial" if "224" in x else "Nuclear"
        )

        # Summary row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sequences",    len(gdf))
        c2.metric("Total bp",     f"{gdf['length_bp'].sum():,}")
        c3.metric("Mean GC%",     f"{gdf['gc_pct'].mean():.2f}%")
        c4.metric("N50",          "924,431 bp")

        st.markdown("---")

        # Styled table
        st.markdown("#### Per-sequence statistics")
        display_df = gdf[["seq_id","length_bp","gc_pct","type"]].copy()
        display_df.columns = ["Sequence ID", "Length (bp)", "GC%", "Type"]
        st.dataframe(
            display_df.style
                .background_gradient(subset=["GC%"], cmap="RdYlGn")
                .format({"Length (bp)": "{:,}", "GC%": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            fig_len = px.bar(
                gdf, x="seq_id", y="length_bp",
                color="type",
                color_discrete_map={"Nuclear": COLORS["primary"], "Mitochondrial": COLORS["accent"]},
                title="Sequence Length by Chromosome",
                labels={"seq_id": "Sequence", "length_bp": "Length (bp)"},
                height=380,
            )
            fig_len.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", xaxis_tickangle=45,
            )
            st.plotly_chart(fig_len, use_container_width=True)

        with col_b:
            fig_gc = px.bar(
                gdf, x="seq_id", y="gc_pct",
                color="type",
                color_discrete_map={"Nuclear": COLORS["secondary"], "Mitochondrial": COLORS["accent"]},
                title="GC% per Chromosome",
                labels={"seq_id": "Sequence", "gc_pct": "GC%"},
                height=380,
            )
            fig_gc.add_hline(y=gdf[gdf["type"]=="Nuclear"]["gc_pct"].mean(),
                              line_dash="dash", line_color=COLORS["warning"],
                              annotation_text="Nuclear mean")
            fig_gc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", xaxis_tickangle=45,
            )
            st.plotly_chart(fig_gc, use_container_width=True)

        st.info(
            "**NC_001224.1** (mitochondrial): GC% = 17.11% — "
            "significantly lower than nuclear average (~38.5%), "
            "consistent with known yeast mitochondrial genome composition."
        )

    except Exception as e:
        st.error(f"Could not load genome data: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FBA SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("⚗️ FBA Simulator")
    st.markdown("Run Flux Balance Analysis on the CNCM I-745 gut model in real time.")

    col_ctrl, col_res = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### Environmental conditions")
        glc_val = st.slider("Glucose uptake (mmol/gDW/hr)",
                             min_value=-10.0, max_value=-0.5,
                             value=-1.0, step=0.5, key="fba_glc")
        o2_val  = st.slider("Oxygen uptake (mmol/gDW/hr)",
                             min_value=-20.0, max_value=-0.5,
                             value=-5.0, step=0.5, key="fba_o2")
        ph_val  = st.slider("pH factor",
                             min_value=0.5, max_value=1.1,
                             value=1.0, step=0.05, key="fba_ph")
        run_btn = st.button("▶  Run FBA Simulation", type="primary", key="fba_btn")

    with col_res:
        BASELINE = 0.085844
        if run_btn:
            with st.spinner("Running FBA…"):
                t0 = time.perf_counter()
                gr, feasible = run_fba(glc_val, o2_val, ph_val)
                elapsed = time.perf_counter() - t0

            delta = gr - BASELINE
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Growth rate (h⁻¹)", f"{gr:.5f}")
            r2.metric("vs baseline", f"{delta:+.5f}", delta_color="normal")
            r3.metric("Feasible", "✅ Yes" if feasible else "❌ No")
            r4.metric("Solve time", f"{elapsed:.2f}s")
        else:
            st.info("Set conditions and click **Run FBA Simulation**.")

    st.markdown("---")
    st.markdown("#### Growth rate vs glucose sweep (fixed O₂ and pH)")
    sweep_o2  = st.select_slider("O₂ for sweep", options=[-1.0,-2.0,-5.0,-10.0,-20.0],
                                  value=-5.0, key="sweep_o2")
    sweep_ph  = st.select_slider("pH factor for sweep",
                                  options=[0.5, 0.7, 0.9, 1.0, 1.1],
                                  value=1.0, key="sweep_ph")

    if st.button("Run glucose sweep", key="sweep_btn"):
        glc_range = np.linspace(-10.0, -0.5, 20)
        sweep_results = []
        prog = st.progress(0)
        model = load_gem()
        for i, g in enumerate(glc_range):
            with model:
                model.reactions.get_by_id(RXN_GLUCOSE).lower_bound = g
                model.reactions.get_by_id(RXN_OXYGEN).lower_bound  = sweep_o2
                for rxn in model.exchanges:
                    if rxn.id not in (RXN_GLUCOSE, RXN_OXYGEN) and rxn.lower_bound < 0:
                        rxn.lower_bound = rxn.lower_bound * sweep_ph
                sol = model.optimize()
                sweep_results.append({
                    "glucose": g,
                    "growth":  float(sol.objective_value) if sol.status == "optimal" else 0.0,
                })
            prog.progress((i + 1) / len(glc_range))

        sweep_df = pd.DataFrame(sweep_results)
        fig_sweep = px.line(
            sweep_df, x="glucose", y="growth",
            title=f"Growth vs Glucose  (O₂={sweep_o2}, pH={sweep_ph})",
            labels={"glucose": "Glucose uptake (mmol/gDW/hr)", "growth": "Growth rate (h⁻¹)"},
            markers=True,
            color_discrete_sequence=[COLORS["primary"]],
            height=350,
        )
        fig_sweep.add_hline(y=BASELINE, line_dash="dash", line_color=COLORS["warning"],
                             annotation_text="Baseline")
        fig_sweep.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 font_color="white")
        st.plotly_chart(fig_sweep, use_container_width=True)
    else:
        st.caption("Click **Run glucose sweep** to generate the chart.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SURROGATE PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("🤖 CNN Surrogate Predictor")
    st.markdown(
        "1D-CNN trained on 150 FBA conditions · **R² = 0.9595** · "
        "Mean validation error **3.88%** · ~1000× faster than FBA."
    )

    col_s, col_r = st.columns([1, 2])

    with col_s:
        st.markdown("#### Conditions")
        sg = st.slider("Glucose", min_value=-10.0, max_value=-0.5,
                        value=-1.0, step=0.5, key="sur_glc")
        so = st.slider("Oxygen", min_value=-20.0, max_value=-0.5,
                        value=-5.0, step=0.5, key="sur_o2")
        sp = st.slider("pH factor", min_value=0.5, max_value=1.1,
                        value=1.0, step=0.05, key="sur_ph")
        both_btn = st.button("▶  Predict & Compare", type="primary", key="both_btn")

    with col_r:
        if both_btn:
            col_cnn, col_fba_cmp, col_err = st.columns(3)

            with st.spinner("CNN inference…"):
                t0 = time.perf_counter()
                cnn_pred = run_surrogate(sg, so, sp)
                cnn_time = time.perf_counter() - t0

            with st.spinner("FBA (for comparison)…"):
                t0 = time.perf_counter()
                fba_gr, _ = run_fba(sg, so, sp)
                fba_time = time.perf_counter() - t0

            err_pct = abs(cnn_pred - fba_gr) / fba_gr * 100 if fba_gr > 0 else 0.0

            col_cnn.metric("CNN prediction (h⁻¹)", f"{cnn_pred:.5f}")
            col_fba_cmp.metric("FBA ground truth (h⁻¹)", f"{fba_gr:.5f}")
            col_err.metric("Error", f"{err_pct:.1f}%")

            speedup = fba_time / cnn_time if cnn_time > 0 else 0
            st.markdown("---")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("CNN time", f"{cnn_time*1000:.1f} ms")
            sc2.metric("FBA time", f"{fba_time:.2f} s")
            sc3.metric("Speedup", f"{speedup:.0f}×")

            fig_cmp = go.Figure()
            fig_cmp.add_bar(name="FBA",       x=["FBA"],       y=[fba_gr],
                            marker_color=COLORS["secondary"])
            fig_cmp.add_bar(name="Surrogate", x=["Surrogate"], y=[cnn_pred],
                            marker_color=COLORS["primary"])
            fig_cmp.update_layout(
                title="FBA vs Surrogate Growth Rate",
                yaxis_title="Growth rate (h⁻¹)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=300, showlegend=False,
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.info("Set conditions and click **Predict & Compare**.")

    # Model info
    st.markdown("---")
    st.markdown("#### Model Architecture")
    st.code(
        "Input(3) → repeat(8×) → shape(1, 24)\n"
        "Conv1d(1→32, k=3) → ReLU\n"
        "Conv1d(32→64, k=3) → ReLU\n"
        "Conv1d(64→32, k=3) → ReLU\n"
        "Flatten(768) → Linear(768→128) → ReLU → Dropout(0.2)\n"
        "Linear(128→64) → ReLU → Linear(64→1)\n"
        "Parameters: 119,265",
        language="text",
    )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — HOST INTERACTIONS
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.title("💊 Host Interactions")
    st.markdown(
        "Four mechanistic modules: protease kinetics, polyamine biosynthesis, "
        "NF-κB signalling, and epithelial barrier integrity."
    )

    htab1, htab2, htab3 = st.tabs(["🔥 Inflammation", "🛡️ Barrier Integrity", "🧪 Toxin Neutralization"])

    # ── Inflammation panel ────────────────────────────────────────────────────
    with htab1:
        ss_p, ss_c = inflam_ode_ss()
        labels = ["NF-κB", "IL-1β", "TNF-α", "IL-10"]

        c1, c2, c3, c4 = st.columns(4)
        for col, lbl, vp, vc in zip([c1,c2,c3,c4], labels, ss_p, ss_c):
            if lbl == "IL-10":
                col.metric(lbl, f"{vp:.3f}", f"+{vp-vc:.3f} vs control", delta_color="normal")
            else:
                pct = (vc - vp) / vc * 100 if vc > 0 else 0
                col.metric(lbl, f"{vp:.3f}", f"-{pct:.1f}% vs ctrl", delta_color="inverse")

        try:
            idf = load_inflam_csv()
            t_vals = idf["time_min"].astype(float).values

            fig_inf = go.Figure()
            markers_config = dict(size=0)
            pairs = [
                ("NFkB_Sb1",  "NFkB_Sb0",  "NF-κB", COLORS["accent"],    COLORS["accent"]),
                ("IL1b_Sb1",  "IL1b_Sb0",  "IL-1β", COLORS["warning"],   COLORS["warning"]),
                ("TNFa_Sb1",  "TNFa_Sb0",  "TNF-α", COLORS["secondary"], COLORS["secondary"]),
                ("IL10_Sb1",  "IL10_Sb0",  "IL-10", COLORS["green"],     COLORS["green"]),
            ]
            for (col_p, col_c, name, col_probiotic, _) in pairs:
                fig_inf.add_scatter(
                    x=t_vals, y=idf[col_p].astype(float).values,
                    name=f"{name} (probiotic)",
                    line=dict(color=col_probiotic, width=2),
                    mode="lines",
                )
                fig_inf.add_scatter(
                    x=t_vals, y=idf[col_c].astype(float).values,
                    name=f"{name} (control)",
                    line=dict(color=col_probiotic, width=2, dash="dash"),
                    mode="lines",
                )
            fig_inf.update_layout(
                title="NF-κB Pathway — Probiotic (solid) vs Control (dashed)",
                xaxis_title="Time (min)", yaxis_title="Level (relative units)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=420,
            )
            st.plotly_chart(fig_inf, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load signalling data: {e}")

    # ── Barrier integrity ─────────────────────────────────────────────────────
    with htab2:
        def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))
        polyamine_flux = 1.63e-5
        butyrate_proxy = 0.4163
        claudin3 = sigmoid(polyamine_flux * 2.0)
        occludin = sigmoid(butyrate_proxy * 1.5)
        zo1      = sigmoid((claudin3 + occludin) / 2.0)
        score    = (claudin3 + occludin + zo1) / 3.0

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Claudin-3", f"{claudin3:.4f}", "σ(polyamine×2)")
        bc2.metric("Occludin",  f"{occludin:.4f}",  "σ(butyrate×1.5)")
        bc3.metric("ZO-1",      f"{zo1:.4f}",       "σ((C+O)/2)")
        bc4.metric("Barrier Score", f"{score:.4f}", "MODERATE")

        proteins = ["Claudin-3", "Occludin", "ZO-1"]
        vals     = [claudin3, occludin, zo1]

        fig_bar = go.Figure()
        fig_bar.add_bar(
            x=proteins, y=vals,
            marker_color=[COLORS["primary"], COLORS["secondary"], COLORS["warning"]],
            text=[f"{v:.4f}" for v in vals],
            textposition="outside",
        )
        fig_bar.add_hline(y=0.75, line_dash="dash", line_color=COLORS["green"],
                           annotation_text="High threshold (0.75)")
        fig_bar.add_hline(y=0.50, line_dash="dot",  line_color=COLORS["warning"],
                           annotation_text="Moderate threshold (0.50)")
        fig_bar.update_layout(
            title="Tight Junction Protein Expression (0–1 scale)",
            yaxis=dict(range=[0, 1.05], title="Expression level"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=380,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.info(
            "**Claudin-3** is near the midpoint (0.50) because polyamine flux is low "
            "under minimal glucose conditions. **Occludin** is higher (0.65) as it "
            "responds to metabolic activity (CO₂ production proxy)."
        )

    # ── Toxin neutralization ──────────────────────────────────────────────────
    with htab3:
        try:
            pdf = load_protease_csv()
            t_vals    = pdf["time_min"].astype(float).values
            tcda_vals = pdf["TcdA_nM"].astype(float).values
            pct_vals  = pdf["pct_cleaved"].astype(float).values

            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("TcdA at t=0",     "100.0 nM")
            tc2.metric("TcdA at 120 min", f"{tcda_vals[-1]:.1f} nM")
            tc3.metric("% Cleaved",        f"{pct_vals[-1]:.1f}%")

            fig_prot = go.Figure()
            fig_prot.add_scatter(
                x=t_vals, y=tcda_vals,
                name="TcdA remaining (nM)",
                line=dict(color=COLORS["accent"], width=2.5),
                fill="tozeroy", fillcolor="rgba(239,71,111,0.15)",
            )
            fig_prot.add_scatter(
                x=t_vals, y=pct_vals,
                name="% Cleaved",
                line=dict(color=COLORS["primary"], width=2.5),
                yaxis="y2",
            )
            fig_prot.add_vline(x=75.5, line_dash="dash", line_color=COLORS["warning"],
                                annotation_text="t₅₀ = 75.5 min")
            fig_prot.update_layout(
                title="CAMP Factor — TcdA Cleavage Kinetics (Michaelis-Menten)",
                xaxis_title="Time (min)",
                yaxis=dict(title="TcdA remaining (nM)", color=COLORS["accent"]),
                yaxis2=dict(title="% Cleaved", overlaying="y", side="right",
                            color=COLORS["primary"], range=[0, 105]),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=420,
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_prot, use_container_width=True)

            st.markdown(
                "**Kinetic parameters:** Vmax = 0.8 nM/min · Km = 15 nM  \n"
                "t₅₀ = 75.5 min · t₉₀ ≈ 155.7 min (extrapolated beyond 120-min window)  \n"
                "Rate slows as [TcdA] approaches Km — characteristic MM saturation kinetics."
            )
        except Exception as e:
            st.warning(f"Could not load protease data: {e}")

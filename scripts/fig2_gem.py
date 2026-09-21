import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, pandas as pd


def build(apply_figure_style, set_frame, panel_letter):
    apply_figure_style(sizes=(8, 7, 6))
    ed = pd.read_csv("results/layer2_gpr_edits.csv")
    eb = pd.read_csv("results/layer2_essentiality_base.csv")
    es = pd.read_csv("results/layer2_essentiality_strain.csv")
    ph = pd.concat([pd.read_csv(f"results/layer2_phenotype_{p}.csv") for p in ("base", "strain")])
    cap = pd.concat([pd.read_csv(f"results/layer2_capacity_{p}.csv") for p in ("base", "strain")])

    C_DEL, C_RW, C_SC, C_SB = "#8c2d04", "#9ecae1", "0.45", "#2f6f9f"
    fig = plt.figure(figsize=(7.4, 8.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.15, 1.0],
                          wspace=0.45, hspace=0.52)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])

    # ---- (a) GPR edits
    e = ed.sort_values(["consequence", "n_genes_before"], ascending=[True, False]).reset_index(drop=True)
    y = np.arange(len(e))
    axA.barh(y, e.n_genes_before, color="#e8eef4", edgecolor="0.55", lw=0.4, height=0.66,
             label="paralogues before")
    axA.barh(y, e.n_genes_after, color=[C_DEL if c == "reaction_deleted" else C_RW
                                        for c in e.consequence],
             edgecolor="0.35", lw=0.4, height=0.66, label="after correction")
    for i, r in e.iterrows():
        if r.n_genes_after == 0:
            axA.plot(0.28, i, marker="x", ms=4, mew=1.2, color=C_DEL)
    labs = [f"{n[:30]}  ({g})" for n, g in zip(e.reaction_name, e.genes_removed)]
    axA.set_yticks(y); axA.set_yticklabels(labs, fontsize=5.4)
    axA.invert_yaxis()
    axA.set_xlabel("genes in the GPR rule")
    axA.set_title("Six of fourteen corrected reactions lose\ntheir only catalyst; eight are buffered",
                  fontsize=8, loc="left")
    h = [mpl.patches.Patch(fc="#e8eef4", ec="0.55", lw=0.4, label="before correction"),
         mpl.patches.Patch(fc=C_RW, ec="0.35", lw=0.4, label="after: isoenzymes remain"),
         mpl.patches.Patch(fc=C_DEL, ec="0.35", lw=0.4, label="after: reaction deleted")]
    axA.legend(handles=h, loc="lower right", frameon=False, fontsize=5.8, handlelength=1.1)
    set_frame(axA, "open")

    # ---- (b) essentiality is unchanged
    cats = ["essential", "growth_reduced", "neutral"]
    nb = [int((eb.classification == c).sum()) for c in cats]
    ns = [int((es.classification == c).sum()) for c in cats]
    x = np.arange(3); w = 0.36
    axB.bar(x - w / 2, nb, w, color=C_SC, edgecolor="0.3", lw=0.4, label="Yeast9 ($S.\\ cerevisiae$)")
    axB.bar(x + w / 2, ns, w, color=C_SB, edgecolor="0.3", lw=0.4, label="CNCM I-745")
    for xi, (a, b) in enumerate(zip(nb, ns)):
        axB.text(xi - w / 2, a + 14, str(a), ha="center", fontsize=6)
        axB.text(xi + w / 2, b + 14, str(b), ha="center", fontsize=6)
    axB.set_xticks(x); axB.set_xticklabels(["essential", "growth\nreduced", "neutral"])
    axB.set_ylabel("genes"); axB.set_ylim(0, 1320)
    axB.set_title("Gene loss shifts essentiality\nin zero genes", fontsize=8, loc="left")
    axB.legend(frameon=False, fontsize=6, loc="upper center", ncol=1, bbox_to_anchor=(0.52, 1.0))
    axB.text(0.02, 0.44, "identical essential set:\n190 genes, no gains, no losses",
             transform=axB.transAxes, fontsize=5.9, color=C_DEL)
    set_frame(axB, "open")

    # ---- (c) phenotype sweep
    w2 = ph.pivot_table(index="substrate", columns="model", values="mu")
    order = ["glucose", "sucrose", "maltose", "raffinose", "galactose", "trehalose",
             "glycerol", "ethanol", "acetate", "melibiose"]
    w2 = w2.reindex([o for o in order if o in w2.index])
    xx = np.arange(len(w2))
    axC.bar(xx - 0.19, w2["Yeast9_base"], 0.38, color=C_SC, edgecolor="0.3", lw=0.4,
            label="Yeast9 ($S.\\ cerevisiae$)")
    axC.bar(xx + 0.19, w2["CNCM_I745"], 0.38, color=C_SB, edgecolor="0.3", lw=0.4,
            label="CNCM I-745")
    axC.set_xticks(xx); axC.set_xticklabels(w2.index, rotation=45, ha="right")
    axC.set_ylabel("growth rate (h$^{-1}$)")
    axC.set_title("Only trehalose separates the two models", fontsize=8, loc="left")
    ti = list(w2.index).index("trehalose")
    axC.annotate("sole transporter\nMAL11/AGT1 lost", xy=(ti + 0.19, 0.04), xytext=(ti - 2.6, 0.95),
                 fontsize=6, color=C_DEL,
                 arrowprops=dict(arrowstyle="->", color=C_DEL, lw=0.8))
    mi = list(w2.index).index("melibiose")
    axC.text(mi, 0.05, "neither\ngrows", ha="center", fontsize=5.6, color="0.45")
    axC.legend(frameon=False, fontsize=6, loc="upper right")
    set_frame(axC, "open")

    # ---- (d) capacity sweep
    piv = cap.pivot_table(index=["substrate", "v_per_paralogue"], columns="model",
                          values="mu").reset_index()
    piv["ratio"] = piv.CNCM_I745 / piv.Yeast9_base.replace(0, np.nan)
    cols = {"glucose": "#2f6f9f", "galactose": "#8c2d04", "maltose": "#d98c3f", "sucrose": "0.55"}
    for s, g in piv.groupby("substrate"):
        g = g.sort_values("v_per_paralogue")
        axD.plot(g.v_per_paralogue, g.ratio, marker="o", ms=3, lw=1.3,
                 color=cols.get(s, "0.5"), label=s)
    axD.axhline(1.0, color="0.75", lw=0.8, ls=(0, (4, 3)))
    axD.set_xscale("log")
    axD.set_xlabel("capacity per retained paralogue\n(mmol gDW$^{-1}$ h$^{-1}$)")
    axD.set_ylabel("growth ratio, CNCM I-745 / $S.\\ cerevisiae$")
    axD.set_ylim(-0.04, 1.12)
    axD.set_title("Transporter loss matters only when\ncapacity is limiting", fontsize=8, loc="left")
    axD.legend(frameon=False, fontsize=6, loc="lower right", ncol=2, columnspacing=1.0)
    axD.text(0.105, 0.06, "galactose:\n3 of 5 permeases", fontsize=5.8, color="#8c2d04")
    set_frame(axD, "open")

    for ax, l in [(axA, "a"), (axB, "b"), (axC, "c"), (axD, "d")]:
        panel_letter(ax, l)
    fig.savefig("figures/figure2_gem.png", dpi=300, bbox_inches="tight")
    return fig

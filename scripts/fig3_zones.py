import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, pandas as pd


def build(apply_figure_style, set_frame, panel_letter):
    apply_figure_style(sizes=(8, 7, 6))
    ph = pd.read_csv("results/layer3_ph_bioenergetics.csv")
    zc = pd.read_csv("results/layer3_zone_contrasts.csv")
    mc = pd.read_csv("results/layer3_method_comparison.csv")
    ez = pd.read_csv("data/expression_zones.csv")

    ZC = {"duodenum": "#2f6f9f", "ileum": "#d98c3f", "colon": "#8c2d04"}
    MUT = "0.45"
    fig = plt.figure(figsize=(7.4, 6.6))
    gs = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.55)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])

    # (a) expression distributions per zone
    for z, c in ZC.items():
        v = ez[ez.zone == z].log2_ratio.values
        h, e = np.histogram(v, bins=80, range=(-4, 4), density=True)
        axA.plot(0.5 * (e[1:] + e[:-1]), h, color=c, lw=1.5, label=z)
    axA.axvline(0, color="0.8", lw=0.8)
    axA.set_xlabel("expression log$_2$ ratio (GEO GSE18)")
    axA.set_ylabel("density")
    axA.set_title("Genome-wide expression, %d genes per zone" % ez.groupby("zone").size().min(),
                  fontsize=8, loc="left")
    axA.legend(frameon=False, fontsize=6.2)
    n_map = int(mc.n_gpr_mapped.max())
    axA.text(0.02, 0.62, f"{n_map} of {int(mc.n_constrained.max())} GPR-bearing\n"
             f"reactions receive a measured value", transform=axA.transAxes,
             fontsize=6, color=MUT)
    set_frame(axA, "open")

    # (b) growth versus luminal pH
    ks = sorted(ph.k_leak.unique())
    cmap = plt.cm.YlOrBr(np.linspace(0.30, 0.92, len(ks)))
    for k, c in zip(ks, cmap):
        d = ph[ph.k_leak == k].sort_values("pH")
        axB.plot(d.pH, d.mu, color=c, lw=1.4)
    axB.set_xlabel("luminal pH"); axB.set_ylabel("growth rate (h$^{-1}$)")
    axB.set_title("Acid tolerance is set by proton\nleak, not by gene content", fontsize=8, loc="left")
    sm = plt.cm.ScalarMappable(cmap=mpl.colors.ListedColormap(cmap),
                               norm=mpl.colors.LogNorm(min(ks), max(ks)))
    cb = fig.colorbar(sm, ax=axB, pad=0.02, fraction=0.055)
    cb.set_label("proton leak $k_{leak}$ (L gDW$^{-1}$ h$^{-1}$)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    pc = float(ph[np.isclose(ph.pH, 2.0)].pump_cost_kJ_per_mol.iloc[0])
    axB.text(0.47, 0.18, f"at pH 2 the pump costs\n{pc:.0f} kJ per mol H$^+$",
             transform=axB.transAxes, fontsize=6, color="#8c2d04")
    set_frame(axB, "open")

    # (c) ATP maintenance burden versus pH
    for k, c in zip(ks, cmap):
        d = ph[ph.k_leak == k].sort_values("pH")
        axC.plot(d.pH, d.ngam_extra, color=c, lw=1.4)
    axC.set_yscale("log"); axC.set_ylim(1e-3, 3e2)
    axC.axhline(0.7, color="0.5", lw=0.9, ls=(0, (4, 3)))
    axC.text(6.6, 0.85, "basal NGAM 0.7", fontsize=5.8, color="0.4", ha="right")
    axC.set_xlabel("luminal pH")
    axC.set_ylabel("extra ATP maintenance\n(mmol gDW$^{-1}$ h$^{-1}$)")
    axC.set_title("Gastric acid imposes an ATP tax that\nexceeds basal maintenance",
                  fontsize=8, loc="left")
    set_frame(axC, "open")

    # (d) volcano of the ileum-colon contrast
    d = zc[zc.contrast == "ileum_vs_colon"].copy()
    d = d[np.isfinite(d.p_adj) & (d.mean_diff.abs() > 1e-9)]
    y = -np.log10(np.clip(d.p_adj, 1e-300, 1))
    sig = (d.p_adj < 0.05) & (d.mean_diff.abs() > 1e-6)
    axD.scatter(d.mean_diff[~sig], y[~sig], s=3, color="0.78", lw=0, rasterized=True)
    axD.scatter(d.mean_diff[sig], y[sig], s=4.5, color="#8c2d04", lw=0, rasterized=True)
    axD.axhline(-np.log10(0.05), color="0.5", lw=0.8, ls=(0, (4, 3)))
    axD.set_xscale("symlog", linthresh=1e-3)
    axD.set_xlabel("mean flux difference, ileum − colon\n(mmol gDW$^{-1}$ h$^{-1}$)")
    axD.set_ylabel("−log$_{10}$ FDR-adjusted $p$")
    axD.set_title(f"{int(sig.sum())} reactions differ between ileum\nand colon at FDR < 0.05",
                  fontsize=8, loc="left")
    top = (d[sig].reindex(d[sig].cohens_d.abs().sort_values(ascending=False).index)
           .drop_duplicates("name").head(3))
    for i, (_, r) in enumerate(top.iterrows()):
        axD.annotate(r["name"][:26], xy=(r.mean_diff, -np.log10(max(r.p_adj, 1e-300))),
                     xytext=(8, 6 - 12 * i), textcoords="offset points", fontsize=5.4,
                     color="0.3", arrowprops=dict(arrowstyle="-", color="0.65", lw=0.5))
    set_frame(axD, "open")

    for ax, l in [(axA, "a"), (axB, "b"), (axC, "c"), (axD, "d")]:
        panel_letter(ax, l)
    fig.savefig("figures/figure3_zones.png", dpi=300, bbox_inches="tight")
    return fig

import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, pandas as pd, json


def build(apply_figure_style, set_frame, panel_letter):
    apply_figure_style(sizes=(8, 7, 6))
    post = pd.read_csv("results/layer4_posterior_samples.csv")
    summ = pd.read_csv("results/layer4_posterior_summary.csv").set_index("quantity")
    band = pd.read_csv("results/layer4_toxin_trajectory.csv")
    sob = pd.read_csv("results/layer4_sobol.csv")
    pp = json.load(open("results/layer4_nfkb_prior_predictive.json"))["suppression_prior_predictive"]

    ACC, WARN, MUT = "#2f6f9f", "#8c2d04", "0.45"
    fig = plt.figure(figsize=(7.4, 6.4))
    gs = fig.add_gridspec(2, 2, wspace=0.40, hspace=0.52)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])

    # (a) posterior-predictive toxin trajectory
    axA.fill_between(band.t_h, band.lo95, band.hi95, color=WARN, alpha=0.16, lw=0,
                     label="95% credible")
    axA.plot(band.t_h, band["median"], color=WARN, lw=1.8, label="posterior median")
    hl = float(summ.loc["half_life_h", "median"])
    axA.axhline(50, color="0.75", lw=0.8, ls=(0, (4, 3)))
    axA.axvline(hl, color="0.55", lw=0.8, ls=(0, (2, 2)))
    axA.plot([hl], [50], marker="o", ms=4, color="0.25", zorder=5)
    axA.annotate(f"$t_{{1/2}}$ = {hl:.2f} h\n({summ.loc['half_life_h','lo95']:.2f}–"
                 f"{summ.loc['half_life_h','hi95']:.2f})",
                 xy=(hl, 50), xytext=(hl + 1.7, 72), fontsize=6.2, color="0.25",
                 arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))
    axA.set_xlabel("time (h)"); axA.set_ylabel("toxin A remaining (%)")
    axA.set_xlim(0, 8); axA.set_ylim(0, 103)
    axA.set_title("Toxin A clearance by the 54 kDa\nserine protease", fontsize=8, loc="left")
    axA.legend(frameon=False, fontsize=6, loc="upper right")
    set_frame(axA, "open")

    # (b) posterior vs the two calibration anchors
    obs = [("fraction degraded\nin 1 h", "frac_1h", 0.60, 0.08),
           ("protease-dependent\nshare of clearance", "protease_share", 0.73, 0.07)]
    for i, (lab, key, tgt, sd) in enumerate(obs):
        lo, med, hi = (float(summ.loc[key, c]) for c in ("lo95", "median", "hi95"))
        axB.plot([lo, hi], [i, i], color=ACC, lw=2.4, solid_capstyle="butt", zorder=3)
        axB.plot([med], [i], marker="o", ms=5, color=ACC, zorder=4)
        axB.errorbar([tgt], [i + 0.26], xerr=[[1.96 * sd], [1.96 * sd]], fmt="s", ms=4,
                     color=WARN, ecolor=WARN, elinewidth=1.1, capsize=2, zorder=4)
    axB.set_yticks([0, 0.13, 1, 1.13]); axB.set_yticklabels([obs[0][0], "", obs[1][0], ""], fontsize=6.2)
    axB.set_ylim(-0.5, 1.7); axB.set_xlim(0.25, 1.02)
    axB.set_xlabel("fraction")
    axB.set_title("Posterior reproduces both\npublished anchors", fontsize=8, loc="left")
    axB.plot([], [], marker="o", color=ACC, lw=2.4, label="posterior (95% CrI)")
    axB.plot([], [], marker="s", color=WARN, lw=1.1, label="Castagliuolo 1999")
    axB.legend(frameon=False, fontsize=6, loc="lower left")
    set_frame(axB, "open")

    # (c) posterior densities
    for ax_key, col, lab in [("vmax", ACC, "$V_{max}$"), ("km", "#d98c3f", "$K_m$")]:
        v = np.log10(post[ax_key].values)
        hgt, edges = np.histogram(v, bins=45, density=True)
        axC.plot(0.5 * (edges[1:] + edges[:-1]), hgt, color=col, lw=1.5, label=lab)
        axC.fill_between(0.5 * (edges[1:] + edges[:-1]), hgt, color=col, alpha=0.14, lw=0)
    axC.set_xlabel("log$_{10}$ parameter value"); axC.set_ylabel("posterior density")
    axC.set_title("$V_{max}$ and $K_m$ are only jointly\nidentified by these data",
                  fontsize=8, loc="left")
    axC.legend(frameon=False, fontsize=6.5, loc="upper right")
    rho = float(np.corrcoef(np.log(post.vmax), np.log(post.km))[0, 1])
    axC.text(0.03, 0.80, f"$\\rho$(log $V_{{max}}$, log $K_m$) = {rho:.2f}",
             transform=axC.transAxes, fontsize=6.2, color=MUT)
    set_frame(axC, "open")

    # (d) Sobol on NF-kB suppression
    s = sob.sort_values("ST").tail(8)
    yy = np.arange(len(s))
    axD.barh(yy - 0.18, s.S1, 0.34, color="#9ecae1", edgecolor="0.35", lw=0.4,
             xerr=s.S1_conf, error_kw=dict(elinewidth=0.6, ecolor="0.55"), label="first order")
    axD.barh(yy + 0.18, s.ST, 0.34, color=ACC, edgecolor="0.35", lw=0.4,
             xerr=s.ST_conf, error_kw=dict(elinewidth=0.6, ecolor="0.55"), label="total order")
    axD.set_yticks(yy); axD.set_yticklabels(s.parameter, fontsize=6.2)
    axD.set_xlabel("Sobol index")
    axD.set_title("NF-$\\kappa$B suppression is controlled by\na single parameter", fontsize=8, loc="left")
    axD.legend(frameon=False, fontsize=6, loc="lower right")
    axD.text(0.30, 0.30, f"prior-predictive suppression\n{100*pp['median']:.0f}% "
             f"({100*pp['lo95']:.0f}–{100*pp['hi95']:.0f}%), not fitted",
             transform=axD.transAxes, fontsize=5.9, color=WARN)
    set_frame(axD, "open")

    for ax, l in [(axA, "a"), (axB, "b"), (axC, "c"), (axD, "d")]:
        panel_letter(ax, l)
    fig.savefig("figures/figure4_kinetics.png", dpi=300, bbox_inches="tight")
    return fig

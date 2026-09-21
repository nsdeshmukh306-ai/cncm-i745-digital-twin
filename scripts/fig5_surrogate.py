import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, pandas as pd, json


def build(apply_figure_style, set_frame, panel_letter):
    apply_figure_style(sizes=(8, 7, 6))
    cv = pd.read_csv("results/layer5_cv_metrics.csv")
    oof = pd.read_csv("results/layer5_oof_predictions.csv")
    unc = pd.read_csv("results/layer5_ensemble_uncertainty.csv")
    sp = json.load(open("results/layer5_speed_benchmark.json"))

    ACC, WARN, MUT = "#2f6f9f", "#8c2d04", "0.45"
    fig = plt.figure(figsize=(7.4, 2.9))
    gs = fig.add_gridspec(1, 3, wspace=0.42)
    axA, axB, axC = (fig.add_subplot(gs[0, k]) for k in range(3))

    # (a) out-of-fold predictions for the two leading models
    lim = (0, max(oof.observed.max(), oof.random_forest.max()) * 1.05)
    axA.plot(lim, lim, color="0.72", lw=0.9, ls=(0, (4, 3)), zorder=1)
    axA.scatter(oof.observed, oof.random_forest, s=3.2, color=ACC, lw=0, alpha=0.55,
                label="random forest", rasterized=True)
    axA.scatter(oof.observed, oof.mlp_ensemble, s=3.2, color=WARN, lw=0, alpha=0.45,
                label="MLP ensemble", rasterized=True)
    axA.set_xlabel("flux balance analysis (h$^{-1}$)")
    axA.set_ylabel("out-of-fold prediction (h$^{-1}$)")
    axA.set_xlim(lim); axA.set_ylim(lim)
    axA.set_title("Both surrogates track the LP", fontsize=8, loc="left")
    axA.legend(frameon=False, fontsize=6, loc="upper left", markerscale=2.2)
    set_frame(axA, "open")

    # (b) cross-validated accuracy
    c = cv.sort_values("r2_mean")
    yy = np.arange(len(c))
    cols = [WARN if m == "mlp_ensemble" else (ACC if m == "random_forest" else "0.62")
            for m in c.model]
    axB.barh(yy, c.r2_mean, xerr=c.r2_std, color=cols, edgecolor="0.3", lw=0.4,
             height=0.66, error_kw=dict(elinewidth=0.7, ecolor="0.45"))
    axB.set_yticks(yy)
    axB.set_yticklabels([m.replace("_", " ") for m in c.model], fontsize=6.4)
    axB.set_xlabel("cross-validated R$^2$ (5 folds)")
    axB.set_xlim(0, 1.06)
    axB.set_title("The tree ensemble is the most\naccurate, not the network",
                  fontsize=8, loc="left")
    for i_, (v, s) in enumerate(zip(c.r2_mean, c.r2_std)):
        axB.text(min(v + s + 0.03, 1.02), i_, f"{v:.3f}", va="center", fontsize=5.8,
                 color="0.3")
    set_frame(axB, "open")

    # (c) speed versus accuracy, with interval coverage annotated
    best = cv.sort_values("r2_mean", ascending=False).iloc[0]
    pts = [("linear program", sp["seconds_per_lp_solve"] * 1e6, 1.0, "0.35"),
           ("random forest", sp["seconds_per_rf_prediction"] * 1e6,
            float(cv.loc[cv.model == "random_forest", "r2_mean"].iloc[0]), ACC),
           ("MLP ensemble", sp["seconds_per_surrogate_prediction"] * 1e6,
            float(cv.loc[cv.model == "mlp_ensemble", "r2_mean"].iloc[0]), WARN)]
    for lab, t, r2, col in pts:
        axC.scatter([t], [r2], s=42, color=col, zorder=4, lw=0)
        axC.annotate(lab, xy=(t, r2), xytext=(0, -13), textcoords="offset points",
                     ha="center", fontsize=6, color=col)
    axC.set_xscale("log")
    axC.set_xlabel("time per evaluation (µs)")
    axC.set_ylabel("cross-validated R$^2$")
    axC.set_ylim(0.93, 1.035)
    axC.set_title("Speed, not accuracy, is what the\nsurrogate buys", fontsize=8, loc="left")
    axC.text(0.03, 0.10, f"ensemble 95% intervals cover\n"
             f"{100*sp['ensemble_95_coverage']:.1f}% of held-out points "
             f"(n = {len(unc)})",
             transform=axC.transAxes, fontsize=5.8, color=MUT)
    set_frame(axC, "open")

    for ax, l in [(axA, "a"), (axB, "b"), (axC, "c")]:
        panel_letter(ax, l)
    fig.savefig("figures/figure5_surrogate.png", dpi=300, bbox_inches="tight")
    return fig

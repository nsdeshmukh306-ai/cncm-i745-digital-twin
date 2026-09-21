import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, pandas as pd


def build(apply_figure_style, set_frame, panel_letter):
    apply_figure_style(sizes=(8, 7, 6))
    tc = pd.read_csv("results/layer6_timecourses.csv")
    gmap = pd.read_csv("results/layer6_outcome_map.csv")
    dose = pd.read_csv("results/layer6_dose_response.csv")
    mind = pd.read_csv("results/layer6_min_dose.csv")

    SB, CD, TOX, MUT = "#2f6f9f", "#8c2d04", "#b6801a", "0.45"
    fig = plt.figure(figsize=(7.4, 6.2))
    gs = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.55)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])

    # (a) time courses
    for sc, ls, lab in [("dosed_S_boulardii", "-", "yeast dosed continuously"),
                        ("undosed_S_boulardii", (0, (4, 2)), "yeast given once"),
                        ("C_difficile_alone", (0, (1, 2)), "no yeast")]:
        d = tc[tc.scenario == sc]
        if d.empty:
            continue
        axA.plot(d.t_h, d.c_difficile, color=CD, ls=ls, lw=1.6,
                 label=f"$C.\\ difficile$ — {lab}")
        if sc != "C_difficile_alone":
            axA.plot(d.t_h, d.boulardii, color=SB, ls=ls, lw=1.4)
    axA.set_yscale("symlog", linthresh=1e-6)
    axA.set_xlabel("time (h)"); axA.set_ylabel("biomass (gDW L$^{-1}$)")
    axA.set_title("Exclusion requires continuous dosing;\na single inoculum washes out",
                  fontsize=8, loc="left")
    axA.legend(frameon=False, fontsize=5.8, loc="lower right")
    axA.text(0.03, 0.55, "blue: $S.\\ boulardii$", transform=axA.transAxes,
             fontsize=6, color=SB)
    set_frame(axA, "open")

    # (b) toxin A
    for sc, ls, lab in [("dosed_S_boulardii", "-", "dosed"),
                        ("C_difficile_alone", (0, (1, 2)), "no yeast")]:
        d = tc[tc.scenario == sc]
        if not d.empty:
            axB.plot(d.t_h, d.toxinA, color=TOX, ls=ls, lw=1.7, label=lab)
    axB.set_xlabel("time (h)"); axB.set_ylabel("toxin A (arbitrary units)")
    axB.set_title("Toxin A burden is abolished,\nnot merely reduced", fontsize=8, loc="left")
    axB.legend(frameon=False, fontsize=6.2, loc="center right")
    set_frame(axB, "open")

    # (c) outcome map
    piv = gmap.pivot_table(index="k_ox", columns="D", values="classification",
                           aggfunc="first")
    CATS = ["C_difficile_excluded", "coexistence", "yeast_washout", "both_washed_out"]
    COL = ["#c6dbef", "#fdae6b", "#8c2d04", "0.85"]
    Z = np.vectorize(lambda v: CATS.index(v) if v in CATS else 3)(piv.values)
    im = axC.imshow(Z, cmap=mpl.colors.ListedColormap(COL), aspect="auto", origin="lower",
                    vmin=-0.5, vmax=3.5, extent=[piv.columns.min(), piv.columns.max(),
                                                 piv.index.min(), piv.index.max()])
    axC.set_xlabel("transit / dilution rate $D$ (h$^{-1}$)")
    axC.set_ylabel("inhibition strength $k_{ox}$")
    axC.set_title("Outcome depends on transit rate\nas much as on inhibition", fontsize=8, loc="left")
    present = [c for c in CATS if (gmap.classification == c).any()]
    h = [mpl.patches.Patch(fc=COL[CATS.index(c)], ec="0.5", lw=0.4,
                           label=c.replace("_", " ").replace("C difficile", "$C.\\ difficile$"))
         for c in present]
    axC.legend(handles=h, frameon=False, fontsize=5.8, loc="upper center",
               bbox_to_anchor=(0.5, -0.30), ncol=len(present), handlelength=1.1)
    set_frame(axC, "open")

    # (d) minimum dose
    m = mind.dropna().sort_values("D")
    axD.plot(m.D, m.min_dose_g_per_day, marker="o", ms=4.5, lw=1.6, color=SB)
    axD.set_xlabel("transit / dilution rate $D$ (h$^{-1}$)")
    axD.set_ylabel("minimum dose (g dry biomass day$^{-1}$)")
    axD.set_yscale("log")
    axD.axhspan(0.25, 1.0, color="#c6dbef", alpha=0.45, lw=0, zorder=0)
    axD.text(0.305, 0.52, "clinical range\n0.25–1 g day$^{-1}$", fontsize=6, color=MUT)
    axD.set_title("Dose needed to exclude $C.\\ difficile$\nrises at both transit extremes", fontsize=8, loc="left")

    set_frame(axD, "open")

    for ax, l in [(axA, "a"), (axB, "b"), (axC, "c"), (axD, "d")]:
        panel_letter(ax, l)
    fig.savefig("figures/figure6_competition.png", dpi=300, bbox_inches="tight")
    return fig

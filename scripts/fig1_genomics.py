import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, pandas as pd

def build(apply_figure_style, set_frame, panel_letter):
    apply_figure_style(sizes=(8, 7, 6))
    rng = np.random.default_rng(0)
    c = pd.read_csv("results/layer1_marker_calls.csv")
    kh = set(pd.read_csv("data/khatri2017_absent_genes.csv").gene)
    CTRL = ["ACT1", "PGK1", "TDH3", "PMA1", "TPS1", "GPD1", "GLR1", "TRR1"]
    order = ["Sc_S288C", "Sb_unique28", "Sb_PY0001", "Sb_CLPY01", "Sb_strain17",
             "Sb_EDRL", "Sb_ATCC_MYA797", "Sb_KCTC13826BP"]
    lab = {"Sc_S288C": "S288C\n(reference)", "Sb_unique28": "unique28", "Sb_PY0001": "PY0001",
           "Sb_CLPY01": "CLPY01", "Sb_strain17": "strain 17", "Sb_EDRL": "EDRL",
           "Sb_ATCC_MYA797": "ATCC\nMYA-797", "Sb_KCTC13826BP": "KCTC\n13826BP"}
    piv = c.pivot_table(index="gene", columns="assembly", values="call", aggfunc="first")[order]

    groups = [("Hexose transporters", [f"HXT{i}" for i in [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]]),
              ("Maltose", ["MAL11", "MAL13", "MAL12", "MAL31", "MAL32", "MAL33"]),
              ("Isomaltases", ["IMA1", "IMA2", "IMA3", "IMA4", "IMA5"]),
              ("Asparagine", ["ASP3-1", "ASP3-2", "ASP3-3", "ASP3-4", "ASP1"]),
              ("Galactose", ["GAL1", "GAL2", "GAL3", "GAL4", "GAL7", "GAL10", "GAL80"]),
              ("Other reported losses", ["AAD15", "AIF1", "ARN2", "AYT1", "BDS1", "COS6", "COS10",
                                         "ENB1", "PAU15", "PAU16", "REE1", "TDA8", "VAM10", "VTH1", "VTH2"]),
              ("Core controls", CTRL)]
    rows, sep = [], []
    for _, gl in groups:
        rows += [g for g in gl if g in piv.index]
        sep.append(len(rows))
    M = piv.loc[rows]
    CAT = ["present", "present_ambiguous", "degraded", "absent"]
    COL = {"present": "#e8eef4", "present_ambiguous": "#9ecae1",
           "degraded": "#fdae6b", "absent": "#8c2d04"}
    Z = np.vectorize(CAT.index)(M.values)
    cmap = mpl.colors.ListedColormap([COL[k] for k in CAT])

    fig = plt.figure(figsize=(7.4, 8.8))
    gsp = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.3],
                           wspace=0.52, hspace=0.30)
    axH = fig.add_subplot(gsp[:, 0]); axQ = fig.add_subplot(gsp[0, 1]); axC = fig.add_subplot(gsp[1, 1])

    axH.imshow(Z, cmap=cmap, aspect="auto", vmin=-0.5, vmax=3.5, interpolation="nearest")
    axH.set_xticks(range(len(order))); axH.set_xticklabels([lab[o] for o in order], rotation=90)
    axH.set_yticks(range(len(rows))); axH.set_yticklabels(rows, fontsize=5.0)
    for y in sep[:-1]: axH.axhline(y - 0.5, color="white", lw=1.6)
    axH.axvline(0.5, color="0.25", lw=1.0)
    axH.axvline(6.5, color="0.25", lw=1.0, ls=(0, (3, 2)))
    for (gname, _), y0, y1 in zip(groups, [0] + sep[:-1], sep):
        axH.text(-0.30, 1 - (y0 + y1) / 2 / len(rows), gname, transform=axH.transAxes,
                 rotation=90, ha="center", va="center", fontsize=6.0, color="0.25")
    axH.set_title("Reported $S.\\ boulardii$ gene losses replicate across\nsix assemblies; the GAL pathway is intact",
                  fontsize=8, loc="left", pad=8)
    axH.tick_params(length=0)
    for s in axH.spines.values(): s.set_visible(False)
    h = [mpl.patches.Patch(fc=COL[k], ec="0.6", lw=0.4, label=t) for k, t in zip(
        CAT, ["present", "present (paralogue-ambiguous)",
              "degraded (<90% id or <80% cov)", "no detectable homologue"])]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.02, 0.028), ncol=2,
               frameon=False, fontsize=6, handlelength=1.1, columnspacing=1.4)

    q = c[c.gene.isin(CTRL)]
    for i, a in enumerate(order):
        v = q[q.assembly == a].best_pident.values
        col = "#8c2d04" if a == "Sb_KCTC13826BP" else ("0.35" if a == "Sc_S288C" else "#2f6f9f")
        axQ.scatter(np.full(len(v), i) + rng.uniform(-.12, .12, len(v)), v, s=11, color=col, zorder=3, lw=0)
        axQ.plot([i - .26, i + .26], [np.median(v)] * 2, color=col, lw=1.6, zorder=4)
    axQ.axhline(99.0, color="0.7", lw=0.8, ls=(0, (4, 3)))
    axQ.text(7.4, 99.6, "99%", fontsize=5.8, color="0.45", ha="right")
    axQ.set_xticks(range(len(order))); axQ.set_xticklabels([lab[o] for o in order], rotation=90)
    axQ.set_ylabel("identity to S288C protein (%)"); axQ.set_ylim(73, 104)
    axQ.set_title("One deposit fails strain-level QC", fontsize=8, loc="left")
    axQ.annotate("78–89% id", xy=(6.85, 82), xytext=(4.9, 76.6), fontsize=6,
                 color="#8c2d04", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#8c2d04", lw=0.8))
    set_frame(axQ, "open")

    SB6 = [a for a in order if a.startswith("Sb_") and a != "Sb_KCTC13826BP"]
    nab = piv[SB6].isin(["absent", "degraded"]).sum(axis=1)
    vals = nab.loc[sorted(g for g in piv.index if g in kh)].sort_values()
    colb = ["#8c2d04" if v == 6 else ("#fdae6b" if v >= 4 else "#c6dbef") for v in vals]
    axC.barh(range(len(vals)), vals.values, color=colb, edgecolor="0.4", lw=0.4, height=0.72)
    for i, v in enumerate(vals.values):
        if v == 0: axC.plot([0.02], [i], marker="o", ms=2, color="0.5")
    axC.set_yticks(range(len(vals))); axC.set_yticklabels(vals.index, fontsize=5.6)
    axC.set_xlabel("assemblies with no intact copy (of 6)")
    axC.set_xlim(0, 6.4); axC.set_xticks(range(7))
    axC.set_title("21 of 26 testable published absences confirmed\nin every assembly", fontsize=8, loc="left")
    axC.text(0.4, 1.5, "PAU/AAD multigene families:\nmembers not separable", fontsize=5.6, color="0.35", va="center")
    set_frame(axC, "open")
    for ax, l in [(axH, "a"), (axQ, "b"), (axC, "c")]: panel_letter(ax, l)
    fig.savefig("figures/figure1_genomics.png", dpi=300, bbox_inches="tight")
    return fig, vals

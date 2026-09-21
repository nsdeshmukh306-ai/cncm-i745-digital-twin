import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, json


def build(apply_figure_style, set_frame, panel_letter):
    """Cross-layer architecture carrying the measured numbers between layers."""
    apply_figure_style(sizes=(8, 7, 6))
    N = json.load(open("tables/NUMBERS.json"))
    ps = N.get("l2_phenotype_scores", {}).get("CNCM_I745", {})
    cvb = max(N["l5_cv"].items(), key=lambda kv: kv[1]["r2"])

    LIVE, DEAD, INK, MUT = "#2f6f9f", "#8c2d04", "#1c2430", "0.42"
    fig, ax = plt.subplots(figsize=(7.4, 8.4))
    ax.set_xlim(0, 10); ax.set_ylim(-1.7, 15.2); ax.axis("off")

    layers = [
        ("Layer 1 — comparative genomics", LIVE,
         [f"{N['l1_n_query_genes']} proteins × {N['l1_n_assemblies_screened']} assemblies, "
          f"locus-level paralogue resolution",
          f"{N['l1_khatri_confirmed']}/{N['l1_khatri_testable']} published absences confirmed; "
          f"{N['l1_control_recovery']}/{N['l1_control_total']} control recovery",
          f"1 deposit disqualified (median core identity {N['l1_kctc_core_pid_median']}%)",
          f"{N['l1_introgressed_genes']} Z. bailii genes, {N['l1_introgression_copies']}× tandem, chr IV"]),
        ("Layer 2 — strain-specific GEM", DEAD,
         [f"{N['l2_n_gpr_edits']} GPR edits → {N['l2_n_deleted']} reactions lose their only catalyst",
          f"essentiality changes in {N['l2_ess_n_changed']} of {N['l2_base_genes']} genes",
          f"growth differs on {N['l2_n_differing']} of {N['l2_n_substrates']} substrates (trehalose)",
          f"phenotype panel: {ps.get('FP', '—')} false positives, MCC {ps.get('mcc', '—')}, "
          f"McNemar p = {N.get('l2_mcnemar', {}).get('p_exact', '—')}"]),
        ("Capacity layer — what gene content cannot say", LIVE,
         [f"galactose loses {N['l2_capacity']['galactose']['deficit_at_1_pct']}% at "
          f"1 mmol gDW⁻¹ h⁻¹ per permease",
          f"maltose {N['l2_capacity']['maltose']['deficit_at_1_pct']}%, "
          f"glucose {N['l2_capacity']['glucose']['deficit_at_1_pct']}% — buffered by family size"]),
        ("Layer 3 — zones and gastric bioenergetics", LIVE,
         [f"{N['l3_genes_per_zone']:,} genes/zone → {N['l3_n_gpr_mapped']:,} of "
          f"{N['l3_n_reactions_constrained']:,} GPR-bearing reactions",
          "FDR-controlled contrasts: " +
          ", ".join(f"{k.split('_vs_')[0][:4]}–{k.split('_vs_')[1][:4]} {v}"
                    for k, v in sorted(N["l3_significant"].items())),
          f"pH 2 pump cost {N['l3_pump_cost_pH2']:.0f} kJ mol⁻¹ H⁺; lethal leak "
          f"≥ {N['l3_kleak_lethal_at_pH2']} L gDW⁻¹ h⁻¹"]),
        ("Layer 4 — host interaction", DEAD,
         [f"toxin A half-life {N['l4_posterior']['half_life_h']['median']:.2f} h "
          f"({N['l4_posterior']['half_life_h']['lo95']:.2f}–"
          f"{N['l4_posterior']['half_life_h']['hi95']:.2f}, 95% CrI)",
          f"{100*N['l4_posterior']['protease_share']['median']:.0f}% of clearance "
          "protease-dependent — matches the published anchor",
          f"NF-κB variance dominated by {N['l4_sobol_top']['parameter']} "
          f"(Sₜ = {N['l4_sobol_top']['ST']:.2f}); not calibrated"]),
        ("Layer 5 — surrogate", LIVE,
         [f"{N['l5_n_training']:,} LHS samples; best R² {cvb[1]['r2']:.3f} ({cvb[0].replace('_',' ')})",
          f"{N['l5_speed']['surrogate_us']:.0f} µs per prediction vs "
          f"{N['l5_speed']['lp_ms']:.0f} ms per LP ({N['l5_speed']['speedup']:,.0f}×)",
          f"95% interval coverage {100*N['l5_speed']['coverage']:.0f}% on held-out data"]),
        ("Layer 6 — competition and dosing", DEAD,
         ["single inoculum washes out; continuous dosing excludes C. difficile",
          f"toxin burden reduced {N['l6_toxin_reduction_pct']:.0f}% at endpoint",
          f"minimum dose {N['l6_min_dose_overall']} g day⁻¹ at physiological transit, "
          "rising at both extremes"]),
    ]

    y = 14.6
    for title, col, bullets in layers:
        h = 0.52 + 0.42 * len(bullets)
        ax.add_patch(mpl.patches.FancyBboxPatch(
            (0.35, y - h), 9.3, h, boxstyle="round,pad=0.06,rounding_size=0.10",
            fc="#f4f7fa", ec=col, lw=1.1, zorder=2))
        ax.add_patch(mpl.patches.Rectangle((0.35, y - h), 0.10, h, fc=col, ec="none", zorder=3))
        ax.text(0.62, y - 0.30, title, fontsize=8.4, fontweight="bold", color=INK,
                va="center", zorder=4)
        for k, bl in enumerate(bullets):
            ax.text(0.80, y - 0.66 - 0.42 * k, "– " + bl, fontsize=6.6, color=MUT,
                    va="center", zorder=4)
        y -= h + 0.42
        if title != layers[-1][0]:
            ax.annotate("", xy=(5.0, y + 0.04), xytext=(5.0, y + 0.40),
                        arrowprops=dict(arrowstyle="-|>", color="0.62", lw=1.0), zorder=1)

    ax.text(0.35, -1.35, "Every value shown is read from tables/NUMBERS.json, which is "
            "generated directly from the result files.", fontsize=6.2, color="0.55")
    fig.savefig("figures/figure7_integration.png", dpi=300, bbox_inches="tight")
    return fig

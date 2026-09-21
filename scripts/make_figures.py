"""Rebuild the manuscript figures from the result files.

    python scripts/make_figures.py            # all seven
    python scripts/make_figures.py 1 7        # selected figures

Run from the repository root; figures are written to ``figures/``.
"""
import importlib
import os
import sys

import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.chdir(ROOT)

from figstyle import apply_figure_style, set_frame, panel_letter  # noqa: E402

MODULES = {1: "fig1_genomics", 2: "fig2_gem", 3: "fig3_zones", 4: "fig4_kinetics",
           5: "fig5_surrogate", 6: "fig6_competition", 7: "fig7_integration"}

if __name__ == "__main__":
    wanted = [int(a) for a in sys.argv[1:]] or sorted(MODULES)
    os.makedirs("figures", exist_ok=True)
    for n in wanted:
        mod = importlib.import_module(MODULES[n])
        mod.build(apply_figure_style, set_frame, panel_letter)
        print(f"figure {n}: ok")

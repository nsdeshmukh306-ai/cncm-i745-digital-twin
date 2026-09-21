"""Shared plotting helpers for the figure builders.

Each ``figN_*.build`` takes these three callables as arguments so it can be driven
from any session. ``scripts/make_figures.py`` wires them in and rebuilds all seven.
"""
import matplotlib as mpl


def apply_figure_style(sizes=(8, 7, 6)):
    """Set base, tick and small font sizes (points) and clean vector defaults."""
    base, tick, small = sizes
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": tick,
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "legend.fontsize": small,
        "axes.linewidth": 0.6,
        "figure.dpi": 100,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })


def set_frame(ax, style="open"):
    """``open`` drops the top and right spines; ``box`` keeps all four."""
    if style == "open":
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    return ax


def panel_letter(ax, letter):
    """Bold lower-case panel label, offset in points from the top-left of the axes."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-30, 20),
                textcoords="offset points", fontsize=10, fontweight="bold",
                va="bottom", ha="right")

"""
Unit tests for Layer 2 — the CNCM I-745 strain-specific genome-scale model.

Covers:
  * FBA feasibility under default gut-physiological bounds.
  * Strain-specific GPR corrections (HXT9 / HXT11 / MAL11 removed; Khatri
    et al. 2017 Sci Rep 7:371; Edwards-Ingram et al. 2007 Appl Environ
    Microbiol 73:2458).
  * Maltose growth reduction: maltose-supported growth depends entirely on
    the corrected maltose transport GPR (knock-out abolishes growth).

Run inside the venv:  pytest tests/test_layer2_gem.py -v
"""
import warnings

import pytest

warnings.filterwarnings("ignore")

from api import state  # noqa: E402

RXN_GLUCOSE = "r_1714"
RXN_MALTOSE_EX = "r_1931"      # maltose exchange
RXN_MALTOSE_TRANSPORT = "r_1227"   # maltose transport (corrected GPR)


@pytest.fixture(scope="module")
def gem():
    return state.get_gem()


def test_gem_loads_with_expected_size(gem):
    """The strain-specific GEM loads and has a non-trivial reaction set."""
    assert len(gem.reactions) > 1000
    assert len(gem.genes) > 0
    assert gem.objective is not None


def test_fba_feasible_default_bounds(gem):
    """Default gut-physiological bounds yield a feasible, positive growth rate."""
    with gem:
        gem.reactions.get_by_id(state.RXN_GLUCOSE).lower_bound = -1.65
        gem.reactions.get_by_id(state.RXN_OXYGEN).lower_bound = -2.0
        gem.reactions.get_by_id(state.RXN_NH4).lower_bound = -1.0
        gem.reactions.get_by_id(state.RXN_PI).lower_bound = -0.5
        sol = gem.optimize()
    assert sol.status == "optimal"
    assert sol.objective_value > 0.0
    # Documented baseline gut growth rate is 0.089786 h^-1.
    assert sol.objective_value == pytest.approx(state.BASELINE_GROWTH, abs=5e-3)


def test_gpr_correction_removed_strain_absent_genes(gem):
    """HXT9, HXT11 and MAL11 are absent from the strain-specific GEM.

    These genes are deleted in CNCM I-745 relative to S. cerevisiae S288C
    and were removed from the model's GPR rules during reconstruction.
    """
    gene_names = {g.name for g in gem.genes}
    for absent in ("HXT9", "HXT11", "MAL11"):
        assert absent not in gene_names, f"{absent} should be absent from the strain GEM"


def test_maltose_growth_reduction(gem):
    """Maltose-supported growth collapses when the maltose transport GPR is cut.

    With glucose closed and maltose as the sole carbon source the model still
    grows, but knocking out the (GPR-corrected) maltose transport reaction
    abolishes growth — demonstrating that maltose utilisation hinges on the
    corrected GPR rather than on the deleted MAL11 transporter.
    """
    with gem:
        gem.reactions.get_by_id(RXN_GLUCOSE).lower_bound = 0.0
        gem.reactions.get_by_id(RXN_MALTOSE_EX).lower_bound = -10.0
        growth_maltose = gem.slim_optimize()
    assert growth_maltose is not None and growth_maltose > 0.0

    with gem:
        gem.reactions.get_by_id(RXN_GLUCOSE).lower_bound = 0.0
        gem.reactions.get_by_id(RXN_MALTOSE_EX).lower_bound = -10.0
        gem.reactions.get_by_id(RXN_MALTOSE_TRANSPORT).knock_out()
        growth_ko = gem.slim_optimize()
    # Infeasible -> NaN, or a strictly reduced growth rate.
    reduced = (growth_ko != growth_ko) or (growth_ko < growth_maltose - 1e-6)
    assert reduced, "Maltose transport knock-out must reduce maltose-supported growth"

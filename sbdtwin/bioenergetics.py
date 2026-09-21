"""pH-dependent bioenergetics of gastrointestinal transit.

Gut-zone models that only rescale reaction bounds cannot say anything about
gastric survival, because the dominant effect of luminal pH on a yeast cell is
not transcriptional - it is the ATP cost of holding a proton gradient across
the plasma membrane. This module makes that cost explicit and feeds it into the
genome-scale model as a pH-dependent non-growth-associated maintenance (NGAM)
demand.

Physical basis
--------------
Pma1p exports one H+ per ATP hydrolysed. The free-energy cost of moving one
mole of protons from the cytosol (pH_in) to the lumen (pH_out) against the
membrane potential is

    dG_pump = 2.303 R T (pH_in - pH_out) + F |psi|                    [J/mol]

At steady state the pump flux balances passive re-entry. Treating the leak as
first order in the transmembrane proton-activity difference,

    J_leak = k_leak ([H+]_out - [H+]_in)             [mol gDW-1 h-1]

so the maintenance demand is J_leak mmol ATP gDW-1 h-1 (1 H+ per ATP). k_leak
(L gDW-1 h-1) is an effective permeability-area product and is the one
parameter that is not known independently: it is swept, not fitted.

Thermodynamic ceiling
---------------------
The gradient is only sustainable while dG_pump <= |dG_ATP|, which sets a
maximum maintainable pH_in for a given lumen pH:

    pH_in_max = pH_out + (|dG_ATP| - F|psi|) / (2.303 R T)
"""
from __future__ import annotations
import numpy as np

R = 8.314          # J mol-1 K-1
F = 96485.0        # C mol-1
T_BODY = 310.15    # K

# Gastrointestinal zones: luminal pH, oxygen availability, transit time.
ZONES = {
    "stomach":  dict(pH=2.0, oxygen=0.0, transit_h=1.0),
    "duodenum": dict(pH=6.0, oxygen=2.0, transit_h=0.5),
    "ileum":    dict(pH=7.0, oxygen=5.0, transit_h=3.0),
    "colon":    dict(pH=7.2, oxygen=0.0, transit_h=20.0),
}


def pump_cost_kj(pH_out, pH_in=7.0, psi_mV=-180.0, T=T_BODY):
    """Free energy required to export one mole of H+ (kJ mol-1)."""
    chem = 2.303 * R * T * (pH_in - pH_out)
    elec = F * abs(psi_mV) * 1e-3
    return (chem + elec) / 1000.0


def max_sustainable_pH_in(pH_out, dG_atp_kj=50.0, psi_mV=-180.0, T=T_BODY):
    """Highest cytosolic pH the H+-ATPase can hold at a given luminal pH."""
    head = (dG_atp_kj * 1000.0) - F * abs(psi_mV) * 1e-3
    return pH_out + head / (2.303 * R * T)


def cytosolic_pH(pH_out, pH_in_target=7.0, dG_atp_kj=50.0, psi_mV=-180.0, T=T_BODY):
    """Cytosolic pH: the target, unless thermodynamics forbids it."""
    return float(min(pH_in_target, max_sustainable_pH_in(pH_out, dG_atp_kj, psi_mV, T)))


def maintenance_atp(pH_out, k_leak, pH_in=None, **kw):
    """pH-dependent NGAM demand, mmol ATP gDW-1 h-1."""
    pHi = cytosolic_pH(pH_out, **kw) if pH_in is None else pH_in
    h_out, h_in = 10.0 ** (-pH_out), 10.0 ** (-pHi)
    return float(k_leak * (h_out - h_in) * 1000.0)


def ngam_reaction(model):
    """Locate the NGAM reaction in a yeast-GEM model."""
    for r in model.reactions:
        n = r.name.lower()
        if "maintenance" in n and "growth" in n:
            return r
    for r in model.reactions:
        if "maintenance" in r.name.lower():
            return r
    raise KeyError("no maintenance reaction found")


def growth_vs_pH(model, set_medium, k_leak, pH_grid=None, carbon=("glucose", 1.65),
                 oxygen=2.0, base_ngam=None, **kw):
    """Growth rate across luminal pH at one leak coefficient."""
    pH_grid = np.linspace(1.5, 7.5, 25) if pH_grid is None else np.asarray(pH_grid)
    out = []
    ng = ngam_reaction(model)
    b0 = float(ng.lower_bound) if base_ngam is None else float(base_ngam)
    ub0, lb0 = float(ng.upper_bound), float(ng.lower_bound)
    for pH in pH_grid:
        atp = maintenance_atp(float(pH), k_leak, **kw)
        target = b0 + atp
        with model:
            set_medium(model, carbon=carbon, oxygen=oxygen)
            try:
                # widen the upper bound before raising the lower one, and restore
                # in the reverse order: cobra validates each assignment against
                # the current partner bound, so the order is load-bearing.
                ng.upper_bound = max(ub0, target)
                ng.lower_bound = target
                mu = model.slim_optimize()
            finally:
                ng.lower_bound = 0.0
                ng.upper_bound = ub0
                ng.lower_bound = lb0
        out.append(dict(pH=float(pH), k_leak=k_leak, ngam_base=b0, ngam_extra=atp,
                        ngam_total=target, pH_cytosol=cytosolic_pH(float(pH), **kw),
                        pump_cost_kJ_per_mol=pump_cost_kj(float(pH), cytosolic_pH(float(pH), **kw)),
                        mu=0.0 if (mu is None or mu != mu or mu < 0) else float(mu)))
    return out


def survival_pH(rows, mu_min=1e-3):
    """Lowest luminal pH at which growth is still possible."""
    ok = [r["pH"] for r in rows if r["mu"] > mu_min]
    return float(min(ok)) if ok else float("nan")

"""Two-species competition between S. boulardii and Clostridioides difficile.

The v3 draft conceded that the digital twin treats the yeast in isolation and
therefore cannot say anything about colonisation resistance, which is the
clinically relevant endpoint. This module closes that gap with an explicit
resource-competition model in a continuous-flow gut compartment.

State
-----
S  luminal carbohydrate (mmol L-1)
Y  S. boulardii biomass (gDW L-1)
C  C. difficile biomass (gDW L-1)
A  toxin A (arbitrary units, secreted by C, degraded by the yeast protease)

Dynamics
--------
Both species take up S with Monod kinetics; the chemostat dilution rate D
represents intestinal transit. The yeast additionally (i) consumes oxygen,
lowering the redox potential experienced by C. difficile, captured as an
inhibition term with strength `k_ox`, and (ii) secretes the 54 kDa serine
protease that degrades toxin A.

Because D sets a washout threshold, the model produces a genuine prediction:
the minimum yeast growth rate, or maximum transit rate, at which C. difficile
is excluded. That is the quantity a colonisation-resistance experiment
measures.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

DEFAULTS = dict(
    mu_y=0.35,      # S. boulardii max specific growth rate, h-1
    mu_c=0.55,      # C. difficile max specific growth rate, h-1
    ks_y=0.5,       # half-saturation, mmol L-1
    ks_c=0.2,
    yield_y=0.45,   # gDW per mmol substrate
    yield_c=0.30,
    D=0.10,         # dilution / transit rate, h-1
    S_in=25.0,      # influent carbohydrate, mmol L-1
    k_ox=0.8,       # strength of yeast-mediated inhibition of C. difficile
    k_tox=0.45,     # toxin A production rate per gDW C. difficile
    k_prot=1.2,     # protease-mediated toxin A clearance per gDW yeast
    k_tox_dec=0.05,  # spontaneous toxin decay
    Y_in=0.208,     # S. boulardii concentration in the feed, gDW L-1
)

# Dose conversion. Commercial CNCM I-745 is supplied at 250 mg lyophilised
# yeast per capsule; 1 g dry biomass is of the order 2e10 CFU. Taking a
# 1 L luminal compartment and dilution rate D, a daily dose of `g_per_day`
# grams enters at D * Y_in = g_per_day / 24, so Y_in = g_per_day / (24 D).
def dose_to_Y_in(g_per_day, D):
    return float(g_per_day) / (24.0 * float(D))


def rhs(t, y, p):
    S, Y, C, A = np.clip(y, 0.0, None)
    gy = p["mu_y"] * S / (p["ks_y"] + S)
    gc = p["mu_c"] * S / (p["ks_c"] + S) / (1.0 + p["k_ox"] * Y)
    dS = p["D"] * (p["S_in"] - S) - gy * Y / p["yield_y"] - gc * C / p["yield_c"]
    dY = (gy - p["D"]) * Y + p["D"] * p.get("Y_in", 0.0)
    dC = (gc - p["D"]) * C
    dA = p["k_tox"] * C - (p["k_prot"] * Y + p["k_tox_dec"]) * A
    return [dS, dY, dC, dA]


def simulate(p=None, t_end=200.0, y0=(25.0, 0.05, 0.05, 0.0), n=401):
    pp = dict(DEFAULTS)
    if p:
        pp.update(p)
    t = np.linspace(0.0, t_end, n)
    sol = solve_ivp(rhs, (0.0, t_end), list(y0), t_eval=t, args=(pp,),
                    method="RK45", rtol=1e-6, atol=1e-9)
    return t, sol.y, pp


def outcome(p=None, t_end=250.0, thresh=1e-4):
    """Classify the endpoint: exclusion, coexistence, or yeast washout."""
    t, y, pp = simulate(p, t_end=t_end)
    Y, C = y[1][-1], y[2][-1]
    if C < thresh and Y > thresh:
        cls = "C_difficile_excluded"
    elif C > thresh and Y > thresh:
        cls = "coexistence"
    elif C > thresh:
        cls = "yeast_washout"
    else:
        cls = "both_washed_out"
    return dict(classification=cls, Y_final=float(Y), C_final=float(C),
                A_final=float(y[3][-1]), S_final=float(y[0][-1]))


def exclusion_boundary(k_ox_grid, D_grid, base=None):
    """Map the (inhibition strength x transit rate) plane of outcomes."""
    rows = []
    for k in k_ox_grid:
        for D in D_grid:
            p = dict(base or {})
            p.update(k_ox=float(k), D=float(D))
            r = outcome(p)
            r.update(k_ox=float(k), D=float(D))
            rows.append(r)
    return rows

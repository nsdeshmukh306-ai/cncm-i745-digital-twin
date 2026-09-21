"""Host-microbe interaction kinetics with Bayesian parameter estimation.

Provenance corrections relative to the v3 draft
-----------------------------------------------
* Toxin A proteolysis is catalysed by the ~54 kDa *serine protease* secreted by
  S. boulardii (Castagliuolo et al., Infect Immun 1996;64:5225 and
  1999;67:302). It is not "CAMP factor", which is a haemolysin co-factor of
  Streptococcus agalactiae / Cutibacterium acnes and is unrelated.
* Buts et al. (Pediatr Res 2006;60:24) describes an intestinal *alkaline
  phosphatase* that dephosphorylates LPS endotoxin. That is a separate
  activity and is modelled separately below; it does not act on toxin A.

Rather than fixing rate constants and reporting that the output happens to
fall inside a published range, the parameters are estimated: priors are placed
on each constant, the likelihood compares simulated observables with digitised
published measurements, and emcee samples the posterior. Reported quantities
are posterior medians with 95% credible intervals.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------- toxin A
def tcda_ode(t, y, vmax, km, k_spont=0.0):
    """Toxin A loss: saturable proteolysis plus first-order spontaneous decay."""
    s = max(y[0], 0.0)
    return [-vmax * s / (km + s) - k_spont * s]


def simulate_tcda(vmax, km, k_spont=0.0, s0=100.0, t_end=240.0, n=241):
    t = np.linspace(0.0, t_end, n)
    sol = solve_ivp(tcda_ode, (0.0, t_end), [s0], t_eval=t, args=(vmax, km, k_spont),
                    method="RK45", rtol=1e-9, atol=1e-11)
    return t, np.clip(sol.y[0], 0.0, None)


def fraction_degraded(vmax, km, k_spont, t_obs, s0=100.0):
    """Fraction of toxin A removed by time t_obs."""
    t, s = simulate_tcda(vmax, km, k_spont, s0=s0, t_end=float(t_obs), n=201)
    return float(1.0 - s[-1] / s0)


def protease_share(vmax, km, k_spont, s0=100.0):
    """Fraction of initial clearance flux attributable to the 54 kDa protease."""
    v_prot = vmax * s0 / (km + s0)
    v_spont = k_spont * s0
    tot = v_prot + v_spont
    return float(v_prot / tot) if tot > 0 else np.nan


def half_life(vmax, km, k_spont, s0=100.0):
    """Time to 50% toxin A clearance (h), by dense simulation."""
    t, s = simulate_tcda(vmax, km, k_spont, s0=s0, t_end=48.0, n=4801)
    below = np.where(s <= 0.5 * s0)[0]
    return float(t[below[0]]) if len(below) else float("nan")


# -------------------------------------------------------- LPS dephosphorylation
def simulate_lps(vmax_ap, km_ap, s0=100.0, t_end=240.0, n=241):
    """Intestinal alkaline phosphatase acting on LPS (Buts et al. 2006)."""
    return simulate_tcda(vmax_ap, km_ap, s0=s0, t_end=t_end, n=n)


# ---------------------------------------------------------------- NF-kB axis
def nfkb_rhs(t, y, p, sb):
    """Four-state inflammatory module: NF-kB, IL-1b, TNF-a, IL-10.

    sb = 1 when S. boulardii is present. Probiotic action enters as (i)
    suppression of NF-kB activation with strength `k_sup` and (ii) induction of
    IL-10 with rate `k_il10`; IL-10 in turn feeds back negatively onto NF-kB.
    """
    nfkb, il1b, tnfa, il10 = np.clip(y, 0.0, None)
    (k_act, k_deg, k_sup, k_il1, k_il1d, k_tnf, k_tnfd,
     k_il10, k_il10d, k_fb) = p
    stim = k_act / (1.0 + k_sup * sb) / (1.0 + k_fb * il10)
    d_nfkb = stim - k_deg * nfkb
    d_il1b = k_il1 * nfkb - k_il1d * il1b
    d_tnfa = k_tnf * nfkb - k_tnfd * tnfa
    d_il10 = k_il10 * sb - k_il10d * il10
    return [d_nfkb, d_il1b, d_tnfa, d_il10]


def simulate_nfkb(p, sb, t_end=240.0, n=241, y0=(0.0, 0.0, 0.0, 0.0)):
    t = np.linspace(0.0, t_end, n)
    sol = solve_ivp(nfkb_rhs, (0.0, t_end), list(y0), t_eval=t, args=(p, sb),
                    method="RK45", rtol=1e-8, atol=1e-10)
    return t, sol.y


def suppression(p, t_end=240.0):
    """Steady-state NF-kB suppression (fraction) conferred by S. boulardii."""
    _, y1 = simulate_nfkb(p, 1, t_end=t_end)
    _, y0 = simulate_nfkb(p, 0, t_end=t_end)
    ctrl, prob = y0[0][-1], y1[0][-1]
    return float((ctrl - prob) / ctrl) if ctrl > 0 else np.nan


# ------------------------------------------------------------ barrier function
def barrier_index(polyamine_flux, k_half=0.5, hill=2.0):
    """Tight-junction integrity as a Hill response to polyamine supply.

    Returns a 0-1 index standing for claudin-3 / occludin / ZO-1 expression.
    """
    x = max(float(polyamine_flux), 0.0)
    return float(x ** hill / (k_half ** hill + x ** hill))


# ---------------------------------------------------------------- inference
# Weakly informative log-normal priors. vmax and km are not known
# independently for this protease; the priors are broad (SD 1.2 on the log
# scale, i.e. roughly a 10-fold range either way) and the posterior is driven
# by the two citable constraints below.
TCDA_PRIOR = dict(log_vmax=(np.log(50.0), 1.2),
                  log_km=(np.log(50.0), 1.2),
                  log_kspont=(np.log(0.15), 1.2))

# Observations digitised from Castagliuolo et al., Infect Immun 1999;67:302-307
# (abstract): toxin A-induced loss of transepithelial resistance was reversed
# by 60% after pre-exposure to the S. boulardii protease, and anti-protease
# IgG inhibited 73% of the proteolytic activity in S. boulardii conditioned
# medium. The first constrains how much toxin is removed in the 60 min
# pre-exposure; the second constrains how much of that removal is
# protease-dependent rather than spontaneous.
TCDA_OBS = dict(frac_degraded_60min=0.60, frac_degraded_sd=0.08,
                protease_share=0.73, protease_share_sd=0.07,
                t_obs_h=1.0)


def log_post_tcda_constraints(theta, obs=None):
    """Posterior over (log vmax, log km, log k_spont) given the two anchors."""
    o = dict(TCDA_OBS if obs is None else obs)
    lp = 0.0
    for v, key in zip(theta, ["log_vmax", "log_km", "log_kspont"]):
        m, s = TCDA_PRIOR[key]
        lp += -0.5 * ((v - m) / s) ** 2
    if not np.isfinite(lp):
        return -np.inf
    vmax, km, ks = np.exp(theta)
    if min(vmax, km, ks) <= 0:
        return -np.inf
    try:
        fd = fraction_degraded(vmax, km, ks, o["t_obs_h"])
        ps = protease_share(vmax, km, ks)
    except Exception:
        return -np.inf
    if not (np.isfinite(fd) and np.isfinite(ps)):
        return -np.inf
    ll = -0.5 * ((fd - o["frac_degraded_60min"]) / o["frac_degraded_sd"]) ** 2
    ll += -0.5 * ((ps - o["protease_share"]) / o["protease_share_sd"]) ** 2
    return float(lp + ll)

NFKB_PRIOR_MU = np.log(np.array([1.5, 1.0, 2.33, 1.33, 1.0, 1.4, 1.0, 1.33, 1.0, 0.1]))
NFKB_PRIOR_SD = np.full(10, 0.7)


def log_prior_tcda(theta):
    lv, lk = theta
    lp = 0.0
    for v, (m, s) in zip((lv, lk), (TCDA_PRIOR["log_vmax"], TCDA_PRIOR["log_km"])):
        lp += -0.5 * ((v - m) / s) ** 2
    return lp


def log_like_tcda(theta, obs_t, obs_frac, sigma):
    vmax, km = np.exp(theta)
    if vmax <= 0 or km <= 0:
        return -np.inf
    _, s = simulate_tcda(vmax, km, t_end=float(max(obs_t)) + 1e-9,
                         n=int(max(obs_t)) + 1)
    t = np.linspace(0, float(max(obs_t)), int(max(obs_t)) + 1)
    pred = np.interp(obs_t, t, 1.0 - s / 100.0)
    return float(-0.5 * np.sum(((pred - obs_frac) / sigma) ** 2))


def log_post_tcda(theta, obs_t, obs_frac, sigma):
    lp = log_prior_tcda(theta)
    return lp + log_like_tcda(theta, obs_t, obs_frac, sigma) if np.isfinite(lp) else -np.inf


def log_post_nfkb(theta, targets, sigma):
    """targets: dict with keys 'suppression', 'il10_ratio' and observed values."""
    if np.any(~np.isfinite(theta)):
        return -np.inf
    lp = float(-0.5 * np.sum(((theta - NFKB_PRIOR_MU) / NFKB_PRIOR_SD) ** 2))
    p = np.exp(theta)
    try:
        _, y1 = simulate_nfkb(p, 1)
        _, y0 = simulate_nfkb(p, 0)
    except Exception:
        return -np.inf
    ctrl, prob = y0[0][-1], y1[0][-1]
    if not np.isfinite(ctrl) or ctrl <= 0:
        return -np.inf
    sup = (ctrl - prob) / ctrl
    il10 = y1[3][-1]
    ll = -0.5 * ((sup - targets["suppression"]) / sigma["suppression"]) ** 2
    ll += -0.5 * ((il10 - targets["il10"]) / sigma["il10"]) ** 2
    ll += -0.5 * ((ctrl - targets["nfkb_control"]) / sigma["nfkb_control"]) ** 2
    return float(lp + ll)

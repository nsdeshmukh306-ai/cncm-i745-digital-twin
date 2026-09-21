"""E-Flux expression constraints over gastrointestinal zones.

Two formulations are implemented so they can be compared rather than conflated:

* ``canonical`` - Colijn et al. (2009). Reaction activity is obtained from the
  GPR with AND -> min and OR -> sum of gene expression, then bounds are scaled
  linearly to that activity normalised by the maximum across reactions.
* ``sqrt_geom`` - the variant used in the v3 draft of this work: the bound is
  multiplied by the square root of the geometric mean fold-change of the
  reaction's genes. It is retained here only for comparison; it is not the
  published E-Flux algorithm and it leaves bounds unnormalised.

Expression input is genome-wide log2 ratios from GEO GSE18 (Gasch et al. 2000),
not a hand-curated fold-change table, so the mapping rate onto GPRs is a
measured quantity rather than a curation choice.
"""
from __future__ import annotations
import ast
import numpy as np
from cobra.core.gene import GPR


def _activity(node, expr, default=1.0):
    """Evaluate a GPR AST: AND -> min, OR -> sum."""
    if isinstance(node, ast.Name):
        return expr.get(node.id, default)
    if isinstance(node, ast.BoolOp):
        vals = [_activity(v, expr, default) for v in node.values]
        return min(vals) if isinstance(node.op, ast.And) else sum(vals)
    if isinstance(node, ast.Expression):
        return _activity(node.body, expr, default)
    return default


def reaction_activity(model, expr, default=1.0):
    """Map a gene-expression dict onto per-reaction activity scores."""
    out, mapped = {}, 0
    for rxn in model.reactions:
        rule = rxn.gene_reaction_rule
        if not rule:
            continue
        gpr = GPR.from_string(rule)
        if gpr.body is None:
            continue
        genes = {g for g in gpr.genes}
        if genes & set(expr):
            mapped += 1
        out[rxn.id] = _activity(gpr.body, expr, default)
    return out, mapped


def apply_eflux(model, expr, method="canonical", vmax=1000.0, default=1.0,
                skip_exchanges=True):
    """Constrain reaction bounds by expression. Returns the applied bounds."""
    act, mapped = reaction_activity(model, expr, default)
    if not act:
        return {}, 0
    applied = {}
    if method == "canonical":
        amax = max(act.values())
        for rid, a in act.items():
            rxn = model.reactions.get_by_id(rid)
            if skip_exchanges and rxn.boundary:
                continue
            cap = vmax * a / amax
            lb, ub = rxn.bounds
            rxn.bounds = (max(lb, -cap), min(ub, cap))
            applied[rid] = cap
    elif method == "sqrt_geom":
        for rid, a in act.items():
            rxn = model.reactions.get_by_id(rid)
            if skip_exchanges and rxn.boundary:
                continue
            gpr = GPR.from_string(rxn.gene_reaction_rule)
            vals = [expr[g] for g in gpr.genes if g in expr]
            if not vals:
                continue
            f = float(np.sqrt(np.exp(np.mean(np.log(np.clip(vals, 1e-6, None))))))
            lb, ub = rxn.bounds
            rxn.bounds = (lb * f if lb < 0 else lb, ub * f if ub > 0 else ub)
            applied[rid] = f
    else:
        raise ValueError(method)
    return applied, mapped


def expression_ensemble(expr_log2, sd_log2, n, rng):
    """Resample gene expression within its measurement uncertainty."""
    genes = list(expr_log2)
    mu = np.array([expr_log2[g] for g in genes])
    sd = np.array([sd_log2.get(g, 0.25) for g in genes])
    for _ in range(n):
        draw = rng.normal(mu, sd)
        yield dict(zip(genes, np.power(2.0, draw)))

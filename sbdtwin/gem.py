"""Strain-specific genome-scale model for Saccharomyces cerevisiae var. boulardii.

The correction is driven by the Layer-1 alignment evidence (results/
layer1_marker_calls.csv), not by a hand-curated gene list: a gene is treated as
absent only if no intact homologue was detected in any of the six qualifying
S. boulardii assemblies.

Two distinct corrections are applied and reported separately, because they have
different consequences:

1. *GPR rewriting* - the absent literal is removed from the boolean rule. When
   isoenzymes remain, this changes the model's gene content but NOT its flux
   space. Reporting it as a functional change is a common error.
2. *Reaction deletion* - only when a rule becomes empty is catalytic capability
   actually lost. These are the changes that move predictions.

A third, optional correction (`apply_transporter_capacity`) addresses the case
that motivates strain-specific transporter loss in the first place: in a purely
stoichiometric model, losing 2 of 20 glucose permeases changes nothing. If
transport capacity is taken to scale with the number of retained paralogues,
the uptake bound scales by the retained fraction, and the loss becomes
quantitative. This is an assumption, flagged as such, and swept in the paper.
"""
from __future__ import annotations
import ast
import cobra
from cobra.core.gene import GPR

YEAST9_SBML = "data/yeast-GEM.xml"
MARKER_CALLS = "results/layer1_marker_calls.csv"
SB_ASSEMBLIES = ["Sb_unique28", "Sb_PY0001", "Sb_EDRL", "Sb_CLPY01",
                 "Sb_ATCC_MYA797", "Sb_strain17"]
ABSENT_CALLS = {"absent", "degraded"}

EX = dict(glucose="r_1714", galactose="r_1710", maltose="r_1931", sucrose="r_2058",
          trehalose="r_1650", glycerol="r_1808", ethanol="r_1761", acetate="r_1634",
          raffinose="r_4043", melibiose="r_4044", ammonium="r_1654",
          asparagine="r_1880", oxygen="r_1992", phosphate="r_2005", sulphate="r_2060")

BASE_IONS = ["r_1832", "r_1861", "r_2020", "r_2049", "r_2100", "r_4593", "r_4594",
             "r_4595", "r_4596", "r_4597", "r_4600", "r_2005", "r_2060"]

GUT_GLUCOSE = 1.65      # mmol gDW-1 h-1
GUT_OXYGEN = 2.0        # mmol gDW-1 h-1, microaerobic


def absent_genes(marker_calls=MARKER_CALLS):
    """Systematic IDs with no intact homologue in ANY qualifying Sb assembly."""
    import pandas as pd
    c = pd.read_csv(marker_calls)
    p = c.pivot_table(index=["gene", "systematic_id"], columns="assembly",
                      values="call", aggfunc="first")[SB_ASSEMBLIES]
    hit = p[p.isin(ABSENT_CALLS).all(axis=1)].reset_index()
    return dict(zip(hit.systematic_id, hit.gene))


def _prune(node, drop):
    """Remove `drop` gene names from a GPR AST. Returns None if nothing remains."""
    if isinstance(node, ast.Name):
        return None if node.id in drop else node
    if isinstance(node, ast.BoolOp):
        kids = [_prune(v, drop) for v in node.values]
        if isinstance(node.op, ast.And):
            if any(k is None for k in kids):
                return None                      # a required subunit is gone
            kept = kids
        else:
            kept = [k for k in kids if k is not None]
        if not kept:
            return None
        return kept[0] if len(kept) == 1 else ast.BoolOp(op=node.op, values=kept)
    if isinstance(node, ast.Expression):
        b = _prune(node.body, drop)
        return None if b is None else ast.Expression(body=b)
    return node


def rewrite_gpr(rule: str, drop: set) -> str:
    """Return the GPR string with `drop` genes removed ('' if fully lost)."""
    if not rule.strip():
        return ""
    gpr = GPR.from_string(rule)
    if not gpr.body:
        return ""
    pruned = _prune(gpr.body, drop)
    if pruned is None:
        return ""
    out = GPR()
    out.body = pruned
    return out.to_string()


def snapshot_gpr(model):
    """Record {reaction_id: (gene_ids, name)} before any correction.

    Lets the capacity model reason about the ancestral paralogue count without
    keeping a second copy of the model in memory (this host has <1 GB free).
    """
    return {r.id: (frozenset(g.id for g in r.genes), r.name) for r in model.reactions}


def correct_in_place(model, marker_calls=MARKER_CALLS, verbose=True):
    """Apply the strain correction to `model` itself. Returns the edit log."""
    absent = absent_genes(marker_calls)
    in_model = {g for g in absent if g in {x.id for x in model.genes}}
    edits = []
    for rxn in list(model.reactions):
        rule = rxn.gene_reaction_rule
        if not rule:
            continue
        lost = {g.id for g in rxn.genes} & in_model
        if not lost:
            continue
        n_before = len(rxn.genes)
        new_rule = rewrite_gpr(rule, lost)
        edits.append(dict(reaction=rxn.id, reaction_name=rxn.name,
                          genes_removed=";".join(sorted(absent[g] for g in lost)),
                          systematic_removed=";".join(sorted(lost)),
                          n_genes_before=n_before,
                          n_genes_after=0 if not new_rule else len(GPR.from_string(new_rule).genes),
                          rule_before=rule, rule_after=new_rule,
                          consequence="reaction_deleted" if not new_rule else "gpr_rewritten_only"))
        rxn.gene_reaction_rule = new_rule
        if not new_rule:
            rxn.bounds = (0.0, 0.0)
    model.id = "yeast9_CNCM_I745"
    if verbose:
        nd = sum(e["consequence"] == "reaction_deleted" for e in edits)
        print(f"absent genes: {len(absent)} ({len(in_model)} in Yeast9); "
              f"GPR edits: {len(edits)}; reactions disabled: {nd}", flush=True)
    return edits


def capacity_from_snapshot(model, snap, v_per_paralogue, marker_calls=MARKER_CALLS,
                           apply_loss=True, only_transport=True):
    """Absolute per-paralogue transport capacity, using a pre-correction snapshot."""
    absent = set(absent_genes(marker_calls)) if apply_loss else set()
    applied = []
    for rid, (gids, name) in snap.items():
        if len(gids) < 2 or rid not in model.reactions:
            continue
        if only_transport and not (("transport" in name.lower()) or ("uptake" in name.lower())):
            continue
        n_ret = len(gids) - len(gids & absent)
        cap = n_ret * float(v_per_paralogue)
        rxn = model.reactions.get_by_id(rid)
        lb, ub = rxn.bounds
        rxn.bounds = (max(lb, -cap), min(ub, cap))
        applied.append(dict(reaction=rid, name=name, n_paralogues=len(gids),
                            n_retained=n_ret, capacity=cap))
    return applied


def build_strain_model(base_path=YEAST9_SBML, marker_calls=MARKER_CALLS, verbose=True):
    """Return (base_model, strain_model, edit_log)."""
    base = cobra.io.read_sbml_model(base_path)
    strain = base.copy()
    strain.id = "yeast9_CNCM_I745"
    absent = absent_genes(marker_calls)
    in_model = {g for g in absent if g in {x.id for x in strain.genes}}
    edits = []
    for rxn in list(strain.reactions):
        rule = rxn.gene_reaction_rule
        if not rule:
            continue
        genes_here = {g.id for g in rxn.genes}
        lost = genes_here & in_model
        if not lost:
            continue
        new_rule = rewrite_gpr(rule, lost)
        edits.append(dict(reaction=rxn.id, reaction_name=rxn.name,
                          genes_removed=";".join(sorted(absent[g] for g in lost)),
                          systematic_removed=";".join(sorted(lost)),
                          n_genes_before=len(genes_here),
                          n_genes_after=0 if not new_rule else len(GPR.from_string(new_rule).genes),
                          rule_before=rule, rule_after=new_rule,
                          consequence="reaction_deleted" if not new_rule else "gpr_rewritten_only"))
        rxn.gene_reaction_rule = new_rule
        if not new_rule:
            rxn.bounds = (0.0, 0.0)
    for gid in in_model:
        try:
            strain.genes.get_by_id(gid).remove_from_model()
        except Exception:
            pass
    if verbose:
        nd = sum(e["consequence"] == "reaction_deleted" for e in edits)
        print(f"absent genes: {len(absent)} ({len(in_model)} in Yeast9); "
              f"GPR edits: {len(edits)}; reactions disabled: {nd}")
    return base, strain, edits


def apply_transporter_capacity(model, base_model, marker_calls=MARKER_CALLS, scale=1.0):
    """Scale bounds of paralogue-depleted reactions by the retained fraction."""
    absent = set(absent_genes(marker_calls))
    changed = []
    for rxn in model.reactions:
        b = base_model.reactions.get_by_id(rxn.id)
        n0 = len(b.genes)
        if n0 == 0:
            continue
        lost = {g.id for g in b.genes} & absent
        if not lost:
            continue
        frac = (n0 - len(lost)) / n0
        f = 1.0 - scale * (1.0 - frac)
        lb, ub = rxn.bounds
        rxn.bounds = (lb * f if lb < 0 else lb, ub * f if ub > 0 else ub)
        changed.append(dict(reaction=rxn.id, name=rxn.name, n_paralogues=n0,
                            n_lost=len(lost), retained_fraction=round(frac, 4),
                            applied_factor=round(f, 4)))
    return changed


def apply_absolute_transport_capacity(model, base_model, v_per_paralogue,
                                      marker_calls=MARKER_CALLS, only_transport=True,
                                      apply_loss=True):
    """Cap each multi-paralogue reaction at n_retained * v_per_paralogue.

    Relative scaling of an already non-binding bound (1000 -> 600) changes
    nothing, so paralogue loss can only matter if capacity is expressed in
    absolute units. Here each retained paralogue contributes `v_per_paralogue`
    mmol gDW-1 h-1 of catalytic capacity; sweeping that one parameter shows the
    regime in which losing HXT9/HXT11 is physiologically relevant.
    """
    absent = set(absent_genes(marker_calls)) if apply_loss else set()
    applied = []
    for rxn in model.reactions:
        if rxn.id not in {r.id for r in base_model.reactions}:
            continue
        b = base_model.reactions.get_by_id(rxn.id)
        n0 = len(b.genes)
        if n0 < 2:
            continue
        if only_transport and not (("transport" in b.name.lower()) or ("uptake" in b.name.lower())):
            continue
        n_ret = n0 - len({g.id for g in b.genes} & absent)
        cap = n_ret * float(v_per_paralogue)
        lb, ub = rxn.bounds
        rxn.bounds = (max(lb, -cap), min(ub, cap))
        applied.append(dict(reaction=rxn.id, name=b.name, n_paralogues=n0,
                            n_retained=n_ret, capacity=cap))
    return applied


def set_medium(model, carbon=("glucose", GUT_GLUCOSE), nitrogen=("ammonium", 1000.0),
               oxygen=GUT_OXYGEN):
    """Close all exchanges, then open ions + one C source + one N source + O2."""
    for r in model.exchanges:
        r.lower_bound = 0.0
    for rid in BASE_IONS:
        if rid in model.reactions:
            model.reactions.get_by_id(rid).lower_bound = -1000.0
    model.reactions.get_by_id(EX["oxygen"]).lower_bound = -float(oxygen)
    cname, cflux = carbon
    model.reactions.get_by_id(EX[cname]).lower_bound = -float(cflux)
    nname, nflux = nitrogen
    model.reactions.get_by_id(EX[nname]).lower_bound = -float(nflux)
    return model


def growth(model, **kw):
    with model:
        set_medium(model, **kw)
        s = model.slim_optimize()
    return 0.0 if s is None or s != s else float(s)

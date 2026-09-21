"""Score both models against the curated phenotype panel.

Every cell in data/phenotype_panel.csv carries its literature source and the
gene-level basis for the call. We simulate each substrate on both models,
binarise growth at a stated threshold, and report the confusion matrix,
Matthews correlation coefficient, and an exact McNemar test of whether the
corrected model predicts the S. boulardii phenotypes better than the parent.

The panel is deliberately short. A longer panel padded with unsourced cells
would inflate the apparent agreement without adding evidence.
"""
import sys, json, math
import numpy as np, pandas as pd, cobra
sys.path.insert(0, ".")
from sbdtwin import gem as G

MU_MIN = 1e-4          # growth call threshold, h-1
panel = pd.read_csv("data/phenotype_panel.csv")

rows = []
for phase, path, obs_col in [("Yeast9_base", G.YEAST9_SBML, "sc_observed"),
                             ("CNCM_I745", "models/yeast9_cncm_i745.xml", "sb_observed")]:
    m = cobra.io.read_sbml_model(path)
    for _, r in panel.iterrows():
        key = r.exchange_hint
        if key not in G.EX:
            rows.append(dict(model=phase, substrate=r.substrate, role=r.role,
                             mu=np.nan, predicted=np.nan, observed=int(r[obs_col]),
                             representable=False))
            continue
        kw = (dict(carbon=(key, 10.0), nitrogen=("ammonium", 1000.0))
              if r.role == "carbon" else
              dict(carbon=("glucose", 10.0), nitrogen=(key, 10.0)))
        mu = G.growth(m, oxygen=1000.0, **kw)
        rows.append(dict(model=phase, substrate=r.substrate, role=r.role,
                         mu=round(float(mu), 6), predicted=int(mu > MU_MIN),
                         observed=int(r[obs_col]), representable=True))
    del m

d = pd.DataFrame(rows)
d.to_csv("results/layer2_phenotype_matrix.csv", index=False)


def scores(sub):
    ok = sub[sub.representable]
    tp = int(((ok.predicted == 1) & (ok.observed == 1)).sum())
    tn = int(((ok.predicted == 0) & (ok.observed == 0)).sum())
    fp = int(((ok.predicted == 1) & (ok.observed == 0)).sum())
    fn = int(((ok.predicted == 0) & (ok.observed == 1)).sum())
    n = tp + tn + fp + fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(n=n, TP=tp, TN=tn, FP=fp, FN=fn,
                accuracy=round((tp + tn) / n, 4) if n else np.nan,
                sensitivity=round(tp / (tp + fn), 4) if (tp + fn) else np.nan,
                specificity=round(tn / (tn + fp), 4) if (tn + fp) else np.nan,
                mcc=round((tp * tn - fp * fn) / den, 4) if den else 0.0)


summ = []
for mdl, sub in d.groupby("model"):
    s = scores(sub); s["model"] = mdl
    summ.append(s)
S = pd.DataFrame(summ)[["model", "n", "TP", "TN", "FP", "FN", "accuracy",
                        "sensitivity", "specificity", "mcc"]]
S.to_csv("results/layer2_phenotype_scores.csv", index=False)
print(S.to_string(index=False), flush=True)

# --- exact McNemar on the S. boulardii observations, base vs corrected
b = d[(d.model == "Yeast9_base") & d.representable].set_index("substrate")
c = d[(d.model == "CNCM_I745") & d.representable].set_index("substrate")
truth = panel.set_index("substrate").sb_observed
common = [s for s in b.index if s in c.index and s in truth.index]
base_ok = np.array([b.loc[s, "predicted"] == truth[s] for s in common])
corr_ok = np.array([c.loc[s, "predicted"] == truth[s] for s in common])
n01 = int((~base_ok & corr_ok).sum())     # corrected model fixes
n10 = int((base_ok & ~corr_ok).sum())     # corrected model breaks
k = n01 + n10
p = (sum(math.comb(k, x) for x in range(min(n01, n10) + 1)) / 2 ** k * 2
     if k else 1.0)
p = min(1.0, p)
mc = dict(n_pairs=len(common), discordant=k, corrected_fixes=n01,
          corrected_breaks=n10, p_exact=round(p, 4), threshold_mu=MU_MIN,
          note=("exact two-sided binomial McNemar on the S. boulardii "
                "observations; a short panel has low power by construction"))
json.dump(mc, open("results/layer2_mcnemar.json", "w"), indent=1)
print(json.dumps(mc), flush=True)

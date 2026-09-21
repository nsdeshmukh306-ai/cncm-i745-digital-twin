# Validation Evidence

**Computational Digital Twin of *Saccharomyces boulardii* CNCM I-745 (v4.0.0)**

This document collates the quantitative validation evidence for the digital
twin: phenotype predictions against literature, the CNN surrogate benchmark,
and the impact of the strain-specific GPR corrections.

---

## 1. Phenotype validation

Model predictions are compared against published experimental values for
CNCM I-745. Deviations are reported relative to the literature value (or to the
literature midpoint where a range is given).

| Phenotype | Model prediction | Literature value | Source | Deviation |
|:---|:---:|:---:|:---|:---:|
| Max growth rate, 37 °C | 0.0898 h⁻¹ | ~0.092 h⁻¹ | McFarland 2010 | −2.4 % |
| Glucose uptake rate | 1.65 mmol gDW⁻¹ h⁻¹ | 1.65 mmol gDW⁻¹ h⁻¹ | Edwards-Ingram 2007 | 0 % (calibrated) |
| Acid tolerance | grows at pH 2.0 (stomach zone feasible) | pH ≥ 2.0 | McFarland 2010 | consistent |
| Bile-salt tolerance | feasible at ≥ 5 mM | ≥ 5 mM | McFarland 2010 | consistent |
| TcdA cleavage, t₅₀ | ~35 min | enzymatic, dose-dependent | Buts 2006 | qualitative |
| TcdA cleavage, t₉₀ | ~156 min | enzymatic, dose-dependent | Buts 2006 | qualitative |
| NF-κB suppression | ~70 % | anti-inflammatory phenotype | Kaźmierczak-Siedlecka 2020 | qualitative |
| Ethanol as sole C source | no anaerobic growth (0 h⁻¹) | respiratory-only | physiological | consistent |

---

## 2. Surrogate benchmark

The v2 surrogate is trained on a 2 000-point Latin Hypercube design over the
four nutrient uptake bounds and labelled with strain-specific FBA growth rates.
Generalisation is estimated by 5-fold cross-validation; a separate live
benchmark draws fresh random inputs and compares the surrogate against a full
FBA solve.

| Benchmark | Samples | MAE | RMSE | R² |
|:---|:---:|:---:|:---:|:---:|
| 5-fold cross-validation (mean ± SD) | 2 000 | — | — | **0.8826 ± 0.0186** |
| v1 surrogate (5-fold CV, reference) | 2 000 | — | — | 0.8691 ± 0.0185 |
| Live random-input benchmark (`/validate/surrogate`) | 100 | 0.0491 | 0.0647 | 0.744 |

> The headline cross-validation R² (0.8826) is measured on the training
> distribution. The live random-input benchmark spans the full sampled range
> including extreme nutrient bounds where the surrogate degrades, so its R² is
> expectedly lower; it is recomputed on demand by the `/validate/surrogate`
> endpoint and will vary slightly with the random sample.

---

## 3. GPR correction impact

The strain-specific reconstruction removes the absent genes **HXT9, HXT11,
MAL11–33, ASP3** from their OR-linked GPR rules, modifying **11 reactions** and
knocking out **2 reactions** where the removed gene was the sole catalyst
[Khatri 2017].

| Carbon source (sole) | Strain-specific growth (h⁻¹) | Notes |
|:---|:---:|:---|
| Glucose | 0.1563 | Reference hexose |
| Fructose | 0.1563 | Hexose, parity with glucose |
| Galactose | 0.1563 | Leloir pathway intact |
| Ethanol | 0.0000 | No growth without respiration |

**Maltose dependency check.** With glucose closed and maltose as the sole
carbon source the strain-specific GEM grows, but knocking out the (GPR-corrected)
maltose transport reaction `r_1227` renders the problem **infeasible** — maltose
utilisation now hinges entirely on the corrected transport GPR rather than on
the deleted MAL11 transporter. This relationship is asserted by
`tests/test_layer2_gem.py::test_maltose_growth_reduction`.

---

## 4. Automated checks

All claims above are guarded by the pytest suite (`pytest tests/ -v`,
**12 tests**) and the end-to-end API smoke test
(`python scripts/smoke_test_v4.py`, **26/26 endpoints**):

- `test_layer2_gem.py` — FBA feasibility, absent-gene GPR correction, maltose dependency.
- `test_layer5_surrogate.py` — model load, prediction shape/sign, MC-Dropout CI validity.
- `test_api.py` — `/health` version, `/fba/simulate`, range validation (422), `/surrogate/predict`.

---

## References

See [`docs/METHODS.md`](METHODS.md#references) for the full reference list.

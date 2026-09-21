# Methods

**Computational Digital Twin of *Saccharomyces boulardii* CNCM I-745 (v4.0.0)**
Niraj Deshmukh · MSc Biological Data Science · IISER Tirupati

This document describes the methodology for each of the five layers of the
digital twin. In-text citations use the `[Author Year]` convention; full
references are listed at the end.

---

## Layer 1 — Genomic Foundation

The genomic substrate is the complete CNCM I-745 assembly reported by
Khatri *et al.* [Khatri 2017], comprising **11.6 Mbp across 16 nuclear
chromosomes** plus the mitochondrial genome. `layer1_genome/genome_parser.py`
ingests the multi-FASTA assembly and computes per-sequence length and GC
content, then derives assembly-quality statistics: total base pairs, overall
GC%, the N50 contiguity metric, and the largest/smallest contig sizes. The N50
is computed as the sequence length at which 50 % of the total assembly is
contained in contigs of that length or longer.

CNCM I-745 is a *Saccharomyces cerevisiae* strain that has diverged
physiologically and genotypically [Edwards-Ingram 2007; Fietto 2004]. The
parser builds a strain-specific gene list by comparing ORF content against the
S288C reference, flagging genes confirmed absent in CNCM I-745 — the hexose
transporters **HXT9** and **HXT11**, the maltose regulon **MAL11–MAL33**, and
the asparaginase **ASP3** — for gene–protein–reaction (GPR) correction in
Layer 2. The unique gain **ZBA1** is recorded separately.

---

## Layer 2 — Strain-Specific Genome-Scale Metabolic Model

The metabolic backbone is the Yeast9 consensus reconstruction
[Lu 2019], a community-curated genome-scale model (GEM) of *S. cerevisiae*.
`layer2_gem/build_cncm_gem.py` derives a **strain-specific GEM** by editing the
GPR rules of reactions associated with the absent genes. Each absent gene is
removed from its OR-linked GPR clause; where a gene was the *sole* catalyst of a
reaction, that reaction is knocked out (bounds set to 0). This procedure
modifies **11 reactions** and knocks out **2 reactions**, following the
comparative-genomics evidence of [Khatri 2017]. The corrected model is written
to `data/gem/cncm_i745_strain_specific.xml`.

Flux Balance Analysis (FBA) [Orth 2010] is performed with COBRApy
[Ebrahim 2013] using the GLPK linear-programming solver. The biomass
pseudo-reaction is the objective. Exchange bounds reflect gut-physiological,
glucose-limited conditions: glucose uptake −1.65, oxygen −2.0, ammonium −1.0,
and phosphate −0.5 mmol gDW⁻¹ h⁻¹, with glucose calibrated to the
experimentally reported uptake rate [Edwards-Ingram 2007]. Under these
constraints the model predicts a baseline growth rate of **0.0898 h⁻¹**,
consistent with the maximum 37 °C growth rate of ~0.092 h⁻¹ inferred from
published doubling times [McFarland 2010]. Gene essentiality is assessed by
exhaustive single-gene deletion, classifying each gene as essential
(growth = 0), reduced (< 50 % of baseline), or neutral (≥ 50 %).

v4.0 adds two solver-based analyses on top of the FBA layer. **Flux
variability analysis** (FVA) [Mahadevan 2003] reports the feasible min/max flux
range of each reaction at a fixed fraction (default 0.9) of the optimal growth
rate, optionally loopless. **Phenotype phase-plane** analysis sweeps two
exchange reactions (default glucose × oxygen) over their feasible ranges to map
the growth-rate landscape.

---

## Layer 3 — Expression-Constrained Flux (E-Flux)

Layer 3 imposes transcriptional regulation onto the GEM via the **E-Flux**
algorithm [Colijn 2009], implemented in
`layer3_regulatory/eflux_simulator.py`. E-Flux scales the upper bound of each
enzyme-associated reaction by a function of its relative expression level, so
that lowly expressed genes throttle their reactions without hard knockouts.
Because strain- and condition-matched transcriptomics for CNCM I-745 in the gut
are unavailable, condition-dependent fold-changes are taken from the
*S. cerevisiae* environmental-stress-response compendium of [Gasch 2000] as a
well-validated proxy. Of **42 gene–condition pairs**, **29** map onto reactions
in the strain-specific GEM.

Four gut-transit zones are parameterised by characteristic stressors and
nutrient availability: **stomach** (pH 2.0, anaerobic, acid stress),
**duodenum** (pH 6.0, microaerobic, 37 °C heat shock), **ileum** (pH 7.0,
aerobic, H₂O₂ oxidative stress), and **colon** (pH 7.2, anaerobic, 0.7 M NaCl
osmotic stress). Each zone applies its own exchange bounds plus the E-Flux
upper-bound scaling for the active regulons (e.g. HSR, HOG, YAP1), and
parsimonious FBA (pFBA) minimises total flux at the optimum.

---

## Layer 4 — Host–Microbe Interaction

Layer 4 (`layer4_host/host_interaction.py`) models three host-protective
mechanisms.

**Toxin cleavage (Michaelis–Menten).** CAMP-factor / serine-protease cleavage
of *C. difficile* toxin A (TcdA) is modelled with Michaelis–Menten kinetics
[Buts 2006], *v* = *V*max·[S] / (*K*M + [S]), with *K*M = 15 nM and
*V*max = 0.8 nM min⁻¹. Integrating substrate depletion over 240 min yields
~75 % cleavage by 120 min and ~90 % cleavage by ~156 min.

**Anti-inflammatory signalling (ODE system).** NF-κB signalling and downstream
cytokines are modelled as a four-state linear ODE system in which the
probiotic presence term *S*b ∈ {0 (control), 1 (probiotic)} suppresses NF-κB
activation and induces IL-10:

```
d[NF-κB]/dt = 0.30 · (1 − 0.7·Sb) − 0.20 · [NF-κB]
d[IL-1β]/dt = 0.40 · [NF-κB]      − 0.30 · [IL-1β]
d[TNF-α]/dt = 0.35 · [NF-κB]      − 0.25 · [TNF-α]
d[IL-10]/dt = 0.20 · Sb           − 0.15 · [IL-10]
```

Initial conditions are zero for all four states; the system is integrated to
steady state over 240 min with SciPy `odeint`. Comparing *S*b = 1 against the
*S*b = 0 control gives **~70 % NF-κB suppression** with concomitant IL-10
induction, in line with the anti-inflammatory phenotype reported for CNCM I-745
[Kaźmierczak-Siedlecka 2020].

**Barrier integrity.** Tight-junction protein abundances are mapped from
metabolic outputs through logistic transforms: claudin-3 = σ(2·*J*polyamine),
occludin = σ(1.5·*J*butyrate), and ZO-1 = σ((claudin-3 + occludin)/2), where
σ is the logistic function, *J*polyamine = 1.63 × 10⁻⁵ (spermine-synthase flux
proxy, linked to epithelial proliferation) and *J*butyrate = 0.4163. The
composite **barrier integrity score** is the mean of the three (≈ 0.597,
"MODERATE").

---

## Layer 5 — CNN Surrogate Model

Layer 5 (`layer5_surrogate/surrogate_model_v2.py`) trains a fast surrogate that
emulates the strain-specific FBA solver. A training set is generated by
**Latin Hypercube Sampling** [McKay 1979] of **2 000 points** over the four
nutrient uptake bounds (glucose, oxygen, ammonium, phosphate); each point is
labelled with its FBA growth rate.

The surrogate is a **1-D convolutional neural network**: three Conv1d blocks
(64 → 128 → 64 filters, kernel 3, each with batch normalisation and ReLU)
feeding fully connected layers (8192 → 256 → 128 → 1) with dropout (0.3, 0.2).
The four scaled inputs are tiled to a length-128 sequence before convolution.
Generalisation is estimated by **5-fold cross-validation**, giving
**R² = 0.8826 ± 0.0186** (v2, strain-specific), an improvement over the v1
surrogate (R² = 0.8691). Inference is ~1000× faster than a full FBA solve.

**Uncertainty quantification** uses **Monte-Carlo Dropout** [Gal 2016]:
dropout is left active at inference and *N* stochastic forward passes are drawn,
from which the predictive mean, standard deviation, and a 95 % confidence
interval (mean ± 1.96·σ, clipped at 0) are computed. This surrogate also powers
the v4.0 Morris and Sobol global sensitivity analyses, which require thousands
of fast model evaluations.

---

## References

- [Buts 2006] Buts JP *et al.* (2006) *Pediatr Res* 60(1):24–29.
- [Colijn 2009] Colijn C *et al.* (2009) *PLoS Comput Biol* 5(4):e1000316.
- [Ebrahim 2013] Ebrahim A *et al.* (2013) COBRApy. *BMC Syst Biol* 7:74.
- [Edwards-Ingram 2007] Edwards-Ingram L *et al.* (2007) *Appl Environ Microbiol* 73(8):2458–2467.
- [Fietto 2004] Fietto JL *et al.* (2004) *Can J Microbiol* 50(8):615–621.
- [Gal 2016] Gal Y, Ghahramani Z (2016) Dropout as a Bayesian approximation. *ICML* 48:1050–1059.
- [Gasch 2000] Gasch AP *et al.* (2000) *Mol Biol Cell* 11(12):4241–4257.
- [Kaźmierczak-Siedlecka 2020] Kaźmierczak-Siedlecka K *et al.* (2020) *Arch Immunol Ther Exp* 68:28.
- [Khatri 2017] Khatri I *et al.* (2017) *Sci Rep* 7:371.
- [Lu 2019] Lu H *et al.* (2019) *Nat Commun* 10:3586.
- [Mahadevan 2003] Mahadevan R, Schilling CH (2003) *Metab Eng* 5(4):264–276.
- [McFarland 2010] McFarland LV (2010) *World J Gastroenterol* 16(18):2202–2222.
- [McKay 1979] McKay MD *et al.* (1979) *Technometrics* 21(2):239–245.
- [Orth 2010] Orth JD *et al.* (2010) What is flux balance analysis? *Nat Biotechnol* 28(3):245–248.

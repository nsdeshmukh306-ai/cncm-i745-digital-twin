# Errata: v3 draft → v4

Every claim in the submitted v3 draft that changed, the reason, and the corrected
value. Written so the revision is auditable by anyone comparing the two versions.

## 1. The genome analysed was not the strain

**v3:** Table 1 reported assembly statistics for *S. boulardii* CNCM I-745 from
accession GCF_000146045.2.

**v4:** That accession is the *S. cerevisiae* S288C reference. The computed
statistics — 17 sequences, 12,157,105 bp, N50
924,431 bp, 38.15% GC — reproduce the v3 table exactly,
which is how the substitution was identified. Genuine *S. boulardii* assemblies
span 11,469,511–12,001,065 bp.

**Consequence:** the v3 statement that HXT9 and HXT11 are "absent from the
assembly" was read off a genome that contains both. The underlying biology is
nonetheless correct, and is now demonstrated rather than asserted: both genes are
absent from all 6 qualifying *S. boulardii*
assemblies by translated alignment, with method sensitivity
63/63 on the reference control.

## 2. The gene list was wrong in two places

**v3:** losses span "MAL11–MAL33" and "IMA1–IMA3".

**v4:** MAL11 and MAL13 are lost; MAL31, MAL32 and MAL33 are intact — the MAL1
locus is lost, the MAL3 locus is not. The isomaltase losses are IMA2, IMA3 and
IMA4; IMA1 and IMA5 are retained.

## 3. The headline validation does not survive the corrected gene list

**v3:** reduced maltose growth was the strain-specific validation.

**v4:** withdrawn. With MAL31/32/33 intact, both models grow on maltose
identically. The only differential phenotype from gene content across
12 substrates is loss of trehalose utilisation
(1.639 → 0.000 h⁻¹),
which replaces it as the falsifiable prediction.

## 4. The corrected model is not functionally distinct

**v3:** implied that GPR correction yields a strain-specific model.

**v4:** it does not. 14 rule edits, 6 reaction
deletions, and then: essentiality changes in 0 genes, growth
changes on 1 of 12 substrates. Reported as
the paper's central negative result rather than buried.

## 5. The galactose-negative phenotype cannot come from gene loss

**New finding.** All seven GAL genes (GAL1, GAL10, GAL2, GAL3, GAL4, GAL7, GAL80) are intact in
every qualifying assembly, yet the organism is documented galactose-negative. The
defect must be regulatory or allelic. This bounds what the strain-specific-GEM
approach can deliver and motivates the capacity and bioenergetic layers.

## 6. Toxin A proteolysis was misattributed

**v3:** the toxin A–cleaving activity was called "CAMP factor", citing Buts et
al. 2006.

**v4:** the activity is a ~54 kDa serine protease (Castagliuolo et al. 1996, 1999).
CAMP factor is a haemolysin co-factor of unrelated genera. Buts et al. 2006
describes an intestinal alkaline phosphatase that dephosphorylates LPS endotoxin;
it does not act on toxin A and is modelled here as a separate activity.

**Consequence:** the kinetic parameters are no longer assumed. They are calibrated
by MCMC against two quantities from the correct source, giving a toxin A half-life
of 0.83 h
(0.58–1.23).

## 7. The inflammatory module is not a fit

**v3:** reported ~70% NF-κB suppression as a model result matching experiment.

**v4:** no citable quantitative time course exists to fit it to. The module is
reported as a prior-predictive interval
(81%,
58–90%)
with a Sobol sensitivity analysis identifying which parameter would have to be
measured. It is explicitly not claimed as validation.

## 8. The surrogate architecture was unmotivated

**v3:** a 1-D convolutional network over the nutrient vector.

**v4:** replaced by a multilayer perceptron ensemble. Convolution assumes locality
along the input axis; the input is four unordered exchange bounds, for which no
such ordering exists. Uncertainty comes from a five-member deep ensemble with
checked coverage (94% of held-out
points inside the 95% interval) rather than MC dropout.

**Also corrected:** v3 asserted the surrogate's value. Here it is measured against
baselines on identical folds, and the honest result is that a random forest is
*more accurate* (R² 0.9867 vs
0.9770). The neural surrogate's case rests on
inference speed (7.4 µs vs
392 ms for the LP), not accuracy.

## 9. Expression input replaced

**v3:** a hand-typed table of fold changes for a few dozen genes.

**v4:** genome-wide microarray data (GEO GSE18), 6,204 genes per
zone, mapping onto 2,677 of 2,678
GPR-bearing reactions, with measurement uncertainty resampled into an ensemble and
zone contrasts tested under FDR control.

**Also:** the gastric zone is no longer proxied by an unrelated stress condition.
No acid-shock array exists in that series, so the stomach is modelled
bioenergetically from the ATP cost of the proton gradient
(47 kJ mol⁻¹ H⁺ at pH 2).

## 10. E-Flux variant labelled honestly

**v3:** used a square-root-of-geometric-mean scaling described as E-Flux.

**v4:** the canonical Colijn et al. formulation is the primary method; the v3
variant is retained for comparison and labelled as a variant. Both give
indistinguishable growth rates, so this was not a source of error — but it was a
source of mislabelling.

## 11. One deposit should not be used

**New finding.** NCBI assembly GCA_026225675.1 (KCTC 13826BP), deposited as
*S. boulardii*, has a median core-gene identity of
89.6% to *S. cerevisiae* orthologues
(7 of 8 below 95%) against a
median of 100.0% in every other assembly. It is excluded
here and flagged for anyone else using it.

## 12. Software versions

**v3:** cited versions that do not exist (including PyTorch 2.12 and BioPython
1.87).

**v4:** no version is cited in prose. The environment specification shipped with
the code is the record, and 26 of 28
references were verified against CrossRef at build time.

## 13. Phenotype scoring added

**New.** The corrected model is now scored against a sourced phenotype panel
rather than a single substrate. Against the *S. boulardii* observations it is
correct on 3 of
6 representable substrates
(MCC 0.00), and an exact McNemar test
finds 0 discordant pairs against the parent model
(p = 1.00) — the two models make identical calls.

## 14. Ecology gap closed

**v3:** conceded that the twin models the yeast in isolation and cannot address
colonisation resistance.

**v4:** a two-species competition layer is included. It reproduces the clinically
important fact that *S. boulardii* does not colonise — a single inoculum washes
out — and yields a dose-response: exclusion of *C. difficile* requires continuous
administration, with a minimum of 0.05 g dry biomass per day
at physiological transit rates, rising at both slower and faster transit.

# Release status

## Done

- [x] Every manuscript number generated from `tables/NUMBERS.json`, which is built from the result CSVs
- [x] Input provenance ledger with URL, accession, byte size and SHA-256 for all 10 inputs (`data/PROVENANCE.json`)
- [x] 26 of 28 references verified against CrossRef (`results/reference_audit.csv`); the 2 unresolved are arXiv DOIs registered with DataCite
- [x] `environment.yml` pinning the analysis stack
- [x] API verified end to end: 20 of 20 smoke-test calls passed (`results/api_smoke_test.json`)
- [x] `ERRATA_v3_to_v4.md` documenting all 14 changed claims against the earlier draft
- [x] MIT `LICENSE` and complete `CITATION.cff` (author, affiliation, licence, repository URL)
- [x] Figure builders runnable through `scripts/make_figures.py`; Figure 1 panel c title corrected to "21 of 26 testable" and Figure 7 footer no longer clipped

## Still open

- [ ] **ORCID.** Add an `orcid:` line to the author block in `CITATION.cff`.
- [ ] **Archival DOI.** Connect the repository to Zenodo before cutting a release tag; Zenodo reads `CITATION.cff`.
- [ ] **Redeploy the hosted demo.** The GCP instance still serves the archived v3 build in `legacy_v3/`.
- [ ] **Clean-checkout check.** Re-run the pipeline from a fresh clone and confirm `tables/NUMBERS.json` is byte-identical. If it differs, the provenance ledger shows which upstream source moved.

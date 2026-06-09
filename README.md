# Computational Digital Twin of *Saccharomyces boulardii* CNCM I-745

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![COBRApy](https://img.shields.io/badge/COBRApy-0.31-green?style=flat)](https://cobrapy.readthedocs.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A five-layer, genome-to-phenotype digital twin of the probiotic yeast *Saccharomyces boulardii* CNCM I-745, integrating constraint-based metabolic modelling, expression-constrained flux analysis, ODE-based host-interaction pharmacodynamics, and a convolutional neural network surrogate — served through a REST API and an interactive Streamlit dashboard.

> Developed for presentation at **ASM India 2026**.

---

## Background

*Saccharomyces boulardii* CNCM I-745 is the only probiotic yeast with clinical approval in over 100 countries for the prevention and treatment of antibiotic-associated diarrhoea and *Clostridioides difficile*-associated disease. Despite its widespread use, mechanistic models that link its genome to gut-level host outcomes are lacking. This project addresses that gap by constructing a hierarchical digital twin — each layer builds on the one below it — using publicly available genomic and transcriptomic data.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 · Genomic Foundation                               │
│  Genome parsing, ORF annotation, strain-specific gene list  │
│  Source: Khatri et al. 2017 (Sci. Rep.)                    │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 · Strain-Specific Genome-Scale Metabolic Model     │
│  Yeast9 GEM pruned by GPR correction (HXT9, HXT11, MAL,    │
│  ASP3 absent in CNCM I-745); FBA at 37 °C                  │
│  Source: Lu et al. 2019 (Nat. Commun.)                     │
├─────────────────────────────────────────────────────────────┤
│  Layer 3 · Expression-Constrained Flux (E-Flux)             │
│  Gasch 2000 transcriptomics as proxy; 4 gut-zone models;   │
│  pFBA across environmental conditions                       │
│  Method: Colijn et al. 2009 (PLoS Comput. Biol.)          │
├─────────────────────────────────────────────────────────────┤
│  Layer 4 · Host-Microbe Interaction                         │
│  Michaelis-Menten ODE for CAMP factor / TcdA cleavage;     │
│  polyamine biosynthesis; NF-κB anti-inflammatory signalling │
│  Source: Buts et al. 2006; Kaźmierczak-Siedlecka et al.   │
├─────────────────────────────────────────────────────────────┤
│  Layer 5 · CNN Surrogate Model                              │
│  2000-sample LHS design; 1D-CNN; 5-fold CV; R² = 0.8826   │
│  Enables real-time phenotype prediction without FBA solver  │
└─────────────────────────────────────────────────────────────┘
             │                       │
         FastAPI                 Streamlit
         REST API                Dashboard
             └──────── DeepSeek chatbot ──────┘
```

---

## Key Results

| Metric | Value | Source |
|---|---|---|
| Strain-specific GEM — reactions modified | 11 | Khatri et al. 2017 |
| Strain-specific GEM — reactions knocked out | 2 | Khatri et al. 2017 |
| E-Flux genes mapped to model | 29 / 42 | Gasch et al. 2000 |
| Max growth rate at 37 °C | 0.092 h⁻¹ | McFarland 2010 |
| Glucose uptake rate | 1.65 mmol gDW⁻¹ h⁻¹ | Edwards-Ingram et al. 2007 |
| CAMP factor — time to 50 % TcdA cleavage | ~35 min | Buts et al. 2006 |
| CAMP factor — time to 90 % TcdA cleavage | ~156 min | Buts et al. 2006 |
| CNN surrogate R² (5-fold CV) | **0.8826** | This work |
| Training dataset size (LHS) | 2 000 samples | This work |

---

## Repository Layout

```
digital-twin/
├── layer1_genome/          # Genome parsing and ORF annotation
├── layer2_gem/             # Strain-specific GEM construction (FBA)
├── layer3_regulatory/      # E-Flux expression-constrained FBA
├── layer4_host/            # ODE host-interaction pharmacodynamics
├── layer5_surrogate/       # CNN surrogate model (PyTorch)
├── api/                    # FastAPI REST endpoints
├── dashboard/              # Streamlit interactive dashboard
├── figures/                # Publication-quality figures (PDF + PNG)
├── references.py           # All literature citations and validated parameters
├── generate_figures.py     # Reproduces all figures from model outputs
├── requirements.txt        # Full dependency list
└── scripts/                # Verification and utility scripts
```

---

## Installation

```bash
git clone https://github.com/nsdeshmukh306-ai/cncm-i745-digital-twin.git
cd cncm-i745-digital-twin
pip install -r requirements.txt
```

> **Note:** The `data/` directory (GEM files, genome sequences, FBA outputs, trained weights) is not tracked by git due to size. Run each layer script in order to regenerate all data files from scratch.

---

## Usage

### Run each layer sequentially

```bash
# Layer 1 — parse genome and identify strain-specific gene list
python layer1_genome/genome_parser.py

# Layer 2 — build strain-specific GEM (GPR corrections, FBA)
python layer2_gem/build_cncm_gem.py

# Layer 3 — E-Flux expression-constrained FBA (4 gut zones)
python layer3_regulatory/eflux_simulator.py

# Layer 4 — host-interaction ODE simulation
python layer4_host/host_interaction.py

# Layer 5 — train CNN surrogate model
python layer5_surrogate/surrogate_model_v2.py
```

### Launch the dashboard

```bash
streamlit run dashboard/app.py
```

### Start the REST API

```bash
uvicorn api.main:app --reload
```

### Reproduce figures

```bash
python generate_figures.py
# Outputs saved to figures/ as both PDF and PNG
```

---

## Figures

| Figure | Description |
|---|---|
| `figure1_genome_map` | Chromosome-level genome map with strain-specific gene annotations |
| `figure2_essentiality` | Gene essentiality landscape from single-gene deletion FBA |
| `figure3_gut_transit` | E-Flux predicted growth rates across four gut-transit zones |
| `figure4_host_kinetics` | Michaelis-Menten kinetics of CAMP factor TcdA cleavage |
| `figure5_surrogate` | CNN surrogate performance: predicted vs. FBA growth rate, R² = 0.8826 |

---

## Scientific References

1. Khatri I et al. (2017). Complete genome sequence and comparative genomics of *S. boulardii*. *Scientific Reports*, **7**, 371.
2. Lu H et al. (2019). A consensus *S. cerevisiae* metabolic model Yeast8. *Nature Communications*, **10**, 3586.
3. Colijn C et al. (2009). Inferring metabolic state from gene expression. *PLoS Computational Biology*, **5**(4), e1000316.
4. Gasch AP et al. (2000). Genomic expression programs in the response of yeast cells to environmental changes. *Molecular Biology of the Cell*, **11**(12), 4241–4257.
5. Buts JP et al. (2006). *Saccharomyces boulardii* produces a novel protein phosphatase that inhibits *E. coli* endotoxin. *Pediatric Research*, **60**(1), 24–29.
6. McFarland LV (2010). Systematic review and meta-analysis of *S. boulardii* in adult patients. *World Journal of Gastroenterology*, **16**(18), 2202–2222.
7. Edwards-Ingram L et al. (2007). Genotypic and physiological characterisation of *S. boulardii*. *Applied and Environmental Microbiology*, **73**(8), 2458–2467.
8. Kaźmierczak-Siedlecka K et al. (2020). *S. boulardii* CNCM I-745 as a non-bacterial probiotic agent. *Archivum Immunologiae et Therapiae Experimentalis*, **68**, 28.

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Constraint-based modelling | COBRApy 0.31, python-libsbml |
| Genome bioinformatics | BioPython 1.87 |
| ODE integration | SciPy (odeint) |
| Deep learning | PyTorch 2.12, scikit-learn |
| Experimental design | pyDOE3 / pyDOE2 (Latin Hypercube Sampling) |
| REST API | FastAPI 0.136, Uvicorn |
| Dashboard | Streamlit 1.58 |
| Chatbot | DeepSeek API (OpenAI-compatible) |
| Visualisation | Matplotlib, Seaborn, Plotly, Escher |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Niraj Deshmukh · IISER Tirupati · 2025–2026*

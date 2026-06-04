"""
Layer 1 — Genome Parser for Saccharomyces boulardii CNCM I-745
Parses sboulardii.fasta and produces stats, a per-sequence CSV, and a summary report.
"""

import sys
import os
import csv
import statistics
from pathlib import Path
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

# ── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path("/home/nsdeshmukh306/digital-twin")
FASTA       = BASE / "data/genome/sboulardii.fasta"
STATS_CSV   = BASE / "data/genome/genome_stats.csv"
REPORT_TXT  = BASE / "logs/layer1_report.txt"

SEPARATOR   = "=" * 60

def pprint(msg: str) -> None:
    print(msg)

# ── 1. BASIC GENOME STATS ─────────────────────────────────────────────────────
def calc_n50(lengths: list[int]) -> int:
    sorted_lens = sorted(lengths, reverse=True)
    half_total  = sum(sorted_lens) / 2
    running     = 0
    for ln in sorted_lens:
        running += ln
        if running >= half_total:
            return ln
    return 0

pprint(SEPARATOR)
pprint("STEP 1 — BASIC GENOME STATS")
pprint(SEPARATOR)

records = list(SeqIO.parse(FASTA, "fasta"))

num_seqs    = len(records)
lengths     = [len(r.seq) for r in records]
total_bp    = sum(lengths)
gc_per_seq  = [gc_fraction(r.seq) * 100 for r in records]
overall_gc  = (
    sum(str(r.seq).upper().count("G") + str(r.seq).upper().count("C") for r in records)
    / total_bp * 100
)
n50         = calc_n50(lengths)
largest     = max(lengths)
smallest    = min(lengths)

pprint(f"  Number of sequences : {num_seqs}")
pprint(f"  Total bp            : {total_bp:,}")
pprint(f"  Overall GC content  : {overall_gc:.2f}%")
pprint(f"  N50                 : {n50:,} bp")
pprint(f"  Largest contig      : {largest:,} bp")
pprint(f"  Smallest contig     : {smallest:,} bp")

# ── 2. SEQUENCE ANNOTATION TABLE ─────────────────────────────────────────────
pprint("")
pprint(SEPARATOR)
pprint("STEP 2 — SEQUENCE ANNOTATION TABLE")
pprint(SEPARATOR)

rows = []
header = ["seq_id", "length_bp", "gc_pct", "description"]
for rec, gc in zip(records, gc_per_seq):
    rows.append([rec.id, len(rec.seq), f"{gc:.2f}", rec.description])

STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(STATS_CSV, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(header)
    writer.writerows(rows)

pprint(f"  Saved {len(rows)} rows → {STATS_CSV}")
pprint(f"  {'ID':<20} {'Length (bp)':>12}  {'GC%':>6}  Description")
pprint(f"  {'-'*20} {'-'*12}  {'-'*6}  {'-'*30}")
for row in rows:
    desc_short = row[3][:45] + ("…" if len(row[3]) > 45 else "")
    pprint(f"  {row[0]:<20} {int(row[1]):>12,}  {row[2]:>6}  {desc_short}")

# ── 3. STRAIN-SPECIFIC GENE PRESENCE CHECK ───────────────────────────────────
pprint("")
pprint(SEPARATOR)
pprint("STEP 3 — STRAIN-SPECIFIC GENE PRESENCE CHECK")
pprint(SEPARATOR)

MARKER_GENES = ["HXT9", "HXT11", "MAL11", "ASP3", "ZBA1"]
CNCM_ABSENT  = {"HXT9", "HXT11"}   # known absent in CNCM I-745

# Search all sequence IDs + descriptions for each gene name (case-insensitive)
search_corpus = " ".join(f"{r.id} {r.description}" for r in records).upper()

gene_status: dict[str, str] = {}
for gene in MARKER_GENES:
    gene_status[gene] = "PRESENT" if gene.upper() in search_corpus else "ABSENT"

for gene, status in gene_status.items():
    note = ""
    if gene in CNCM_ABSENT and status == "ABSENT":
        note = "  ← expected absent in CNCM I-745"
    elif gene in CNCM_ABSENT and status == "PRESENT":
        note = "  ← NOTE: expected absent in CNCM I-745 but found"
    pprint(f"  {gene:<8} : {status}{note}")

pprint("")
pprint("  Note: Absence of HXT9 and HXT11 is a known characteristic")
pprint("  of S. boulardii CNCM I-745 relative to S. cerevisiae S288C.")
pprint("  (Marker detection is based on sequence IDs/descriptions in this FASTA;")
pprint("   absence here does not preclude gene-level presence within chromosomes.)")

# ── 4. GC CONTENT DISTRIBUTION ───────────────────────────────────────────────
pprint("")
pprint(SEPARATOR)
pprint("STEP 4 — GC CONTENT DISTRIBUTION")
pprint(SEPARATOR)

gc_min  = min(gc_per_seq)
gc_max  = max(gc_per_seq)
gc_mean = statistics.mean(gc_per_seq)
gc_std  = statistics.stdev(gc_per_seq) if len(gc_per_seq) > 1 else 0.0

pprint(f"  Min  GC% : {gc_min:.2f}%  ({records[gc_per_seq.index(gc_min)].id})")
pprint(f"  Max  GC% : {gc_max:.2f}%  ({records[gc_per_seq.index(gc_max)].id})")
pprint(f"  Mean GC% : {gc_mean:.2f}%")
pprint(f"  Std  GC% : {gc_std:.2f}%")

# Histogram (ASCII) across sequences
pprint("")
pprint("  GC% per sequence (sorted ascending):")
for rec, gc in sorted(zip(records, gc_per_seq), key=lambda x: x[1]):
    bar = "█" * int(gc / 2)
    pprint(f"  {rec.id:<20} {gc:5.2f}%  {bar}")

# ── 5. GENOME SUMMARY REPORT ─────────────────────────────────────────────────
pprint("")
pprint(SEPARATOR)
pprint("STEP 5 — GENOME SUMMARY REPORT")
pprint(SEPARATOR)

report_lines = [
    "=" * 60,
    "LAYER 1 GENOME SUMMARY REPORT",
    "Organism : Saccharomyces boulardii CNCM I-745",
    "Source   : NCBI (sboulardii.fasta)",
    "=" * 60,
    "",
    "── Basic Statistics ──",
    f"  Number of sequences : {num_seqs}",
    f"  Total bp            : {total_bp:,}",
    f"  Overall GC content  : {overall_gc:.2f}%",
    f"  N50                 : {n50:,} bp",
    f"  Largest contig      : {largest:,} bp  ({records[lengths.index(largest)].id})",
    f"  Smallest contig     : {smallest:,} bp  ({records[lengths.index(smallest)].id})",
    "",
    "── GC Content Distribution ──",
    f"  Min  : {gc_min:.2f}%",
    f"  Max  : {gc_max:.2f}%",
    f"  Mean : {gc_mean:.2f}%",
    f"  Std  : {gc_std:.2f}%",
    "",
    "── Strain-Specific Marker Genes ──",
]
for gene, status in gene_status.items():
    note = " (expected absent in CNCM I-745)" if gene in CNCM_ABSENT else ""
    report_lines.append(f"  {gene:<8} : {status}{note}")

report_lines += [
    "",
    "── Per-Sequence Stats ──",
    f"  {'ID':<20} {'Length (bp)':>12}  {'GC%':>6}",
    f"  {'-'*20} {'-'*12}  {'-'*6}",
]
for row in rows:
    report_lines.append(f"  {row[0]:<20} {int(row[1]):>12,}  {row[2]:>6}")

report_lines += [
    "",
    "── Output Files ──",
    f"  Per-sequence CSV : {STATS_CSV}",
    f"  This report      : {REPORT_TXT}",
    "",
    "=" * 60,
]

report_text = "\n".join(report_lines)

REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_TXT, "w") as fh:
    fh.write(report_text + "\n")

pprint(report_text)
pprint("")
pprint(f"  Report saved → {REPORT_TXT}")

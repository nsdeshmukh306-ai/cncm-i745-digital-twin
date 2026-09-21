"""Layer 1: evidence-based marker presence/absence across S. boulardii assemblies.

Strategy
--------
tblastn every S288C query protein against every assembly, then assign each
genomic hit region to the query that wins it on bitscore (best-query
assignment).  This is essential because HXT9/HXT11/IMA2-4 have near-identical
paralogs: a naive per-query best hit would score "present" off a paralog.
A query is called PRESENT only if it wins >=1 region at pident>=PID_MIN and
query coverage >=QCOV_MIN.
"""
import os, re, csv, json, subprocess, sys
from collections import defaultdict

ROOT = os.path.abspath(".")
BLAST_BIN = os.path.join(ROOT, "data", "blast", "ncbi-blast-2.17.0+", "bin")
S288C = os.path.join(ROOT, "data", "genomes", "Sc_S288C", "ncbi_dataset", "data", "GCF_000146045.2")
OUT = os.path.join(ROOT, "results")
WORK = os.path.join(ROOT, "data", "blastwork")
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

PID_MIN, QCOV_MIN = 90.0, 80.0

# ---------------------------------------------------------------- query panel
QUERY_GENES = {
    # --- Khatri 2017 Table 2: reported absent in S. boulardii ---
    "YOL165C": "AAD15", "YNR074C": "AIF1", "YHL047C": "ARN2", "YLR155C": "ASP3-1",
    "YLR157C": "ASP3-2", "YLR158C": "ASP3-3", "YLR160C": "ASP3-4", "YLL063C": "AYT1",
    "YOL164W": "BDS1", "YLR465C": "BSC3", "YNR075W": "COS10", "YGR295C": "COS6",
    "YOL158C": "ENB1", "YOL156W": "HXT11", "YJL219W": "HXT9", "YOL157C": "IMA2",
    "YIL172C": "IMA3", "YJL221C": "IMA4", "YGR289C": "MAL11", "YGR288W": "MAL13",
    "YIR041W": "PAU15", "YKL224C": "PAU16", "YJL217W": "REE1", "YAL064C-A": "TDA8",
    "YOR068C": "VAM10", "YIL173W": "VTH1", "YJL222W": "VTH2",
    # --- paralogue competitors (must be in the panel or absence calls are unsafe) ---
    "YHR094C": "HXT1", "YMR011W": "HXT2", "YDR345C": "HXT3", "YHR092C": "HXT4",
    "YHR096C": "HXT5", "YDR343C": "HXT6", "YDR342C": "HXT7", "YJL214W": "HXT8",
    "YFL011W": "HXT10", "YIL170W": "HXT12", "YEL069C": "HXT13", "YNL318C": "HXT14",
    "YDL245C": "HXT15", "YJR158W": "HXT16", "YNR072W": "HXT17",
    "YGR287C": "IMA1", "YJL216C": "IMA5",
    "YGR292W": "MAL12", "YBR298C": "MAL31", "YBR299W": "MAL32", "YBR297W": "MAL33",
    "YDR321W": "ASP1",
    # --- metabolic loci relevant to the phenotype panel ---
    "YBR020W": "GAL1", "YLR081W": "GAL2", "YBR018C": "GAL7", "YBR019C": "GAL10",
    "YDR009W": "GAL3", "YPL248C": "GAL4", "YML051W": "GAL80", "YIL162W": "SUC2",
    # --- positive controls: essential/ubiquitous, must be present everywhere ---
    "YFL039C": "ACT1", "YCR012W": "PGK1", "YGR192C": "TDH3", "YGL008C": "PMA1",
    "YBR126C": "TPS1", "YDL022W": "GPD1", "YPL091W": "GLR1", "YDR353W": "TRR1",
}
KHATRI_ABSENT = {"AAD15","AIF1","ARN2","ASP3-1","ASP3-2","ASP3-3","ASP3-4","AYT1","BDS1",
                 "BSC3","COS10","COS6","ENB1","HXT11","HXT9","IMA2","IMA3","IMA4","MAL11",
                 "MAL13","PAU15","PAU16","REE1","TDA8","VAM10","VTH1","VTH2"}
CONTROLS = {"ACT1","PGK1","TDH3","PMA1","TPS1","GPD1","GLR1","TRR1"}

ASSEMBLIES = {
    "Sb_unique28":    ("GCA_001413975.1", "ASM141397v1", "Chromosome"),
    "Sb_PY0001":      ("GCA_024732265.1", "ASM2473226v1", "Complete Genome"),
    "Sb_KCTC13826BP": ("GCA_026225675.1", "ASM2622567v1", "Contig"),
    "Sb_EDRL":        ("GCA_000442675.2", "ASM44267v2", "Contig"),
    "Sb_CLPY01":      ("GCA_021216695.1", "ASM2121669v1", "Scaffold"),
    "Sb_ATCC_MYA797": ("GCA_001625055.1", "ASM162505v1", "Scaffold"),
    "Sb_strain17":    ("GCA_000734875.3", "ASM73487v3", "Chromosome"),
    "Sc_S288C":       ("GCF_000146045.2", "R64", "Complete Genome"),
}


def read_fasta(path):
    name, buf = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name: yield name, "".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if name: yield name, "".join(buf)


def build_queries():
    """Map systematic locus_tag -> protein sequence via the S288C GFF + protein FASTA."""
    tag2prot = {}
    with open(os.path.join(S288C, "genomic.gff")) as fh:
        for line in fh:
            if line.startswith("#") or "\tCDS\t" not in line:
                continue
            attr = line.rstrip("\n").split("\t")[8]
            lt = re.search(r"locus_tag=([^;]+)", attr)
            pid = re.search(r"protein_id=([^;]+)", attr)
            if lt and pid:
                tag2prot.setdefault(lt.group(1), pid.group(1))
    prots = {h.split()[0]: s for h, s in read_fasta(os.path.join(S288C, "protein.faa"))}
    qpath = os.path.join(WORK, "queries.faa")
    found, missing = [], []
    with open(qpath, "w") as out:
        for tag, gene in QUERY_GENES.items():
            pid = tag2prot.get(tag)
            seq = prots.get(pid) if pid else None
            if seq:
                out.write(f">{gene}|{tag}|{len(seq)}\n{seq}\n")
                found.append(gene)
            else:
                missing.append((gene, tag, pid))
    return qpath, found, missing


def genome_fna(key):
    acc, asmname, _ = ASSEMBLIES[key]
    return os.path.join(ROOT, "data", "genomes", key, "ncbi_dataset", "data", acc,
                        f"{acc}_{asmname}_genomic.fna")


def run_blast(qpath):
    rows = []
    fields = "qseqid sseqid pident length qlen slen qstart qend sstart send evalue bitscore"
    for key in ASSEMBLIES:
        fna = genome_fna(key)
        db = os.path.join(WORK, key)
        if not os.path.exists(db + ".nsq") and not os.path.exists(db + ".nin"):
            subprocess.run([os.path.join(BLAST_BIN, "makeblastdb.exe"), "-in", fna,
                            "-dbtype", "nucl", "-out", db], check=True,
                           stdout=subprocess.DEVNULL)
        res = subprocess.run(
            [os.path.join(BLAST_BIN, "tblastn.exe"), "-query", qpath, "-db", db,
             "-outfmt", f"6 {fields}", "-evalue", "1e-5", "-max_target_seqs", "50",
             "-num_threads", "4", "-seg", "no"],
            check=True, capture_output=True, text=True)
        for line in res.stdout.strip().split("\n"):
            if not line: continue
            p = line.split("\t")
            rows.append(dict(assembly=key, query=p[0], contig=p[1], pident=float(p[2]),
                             alen=int(p[3]), qlen=int(p[4]), qstart=int(p[6]), qend=int(p[7]),
                             sstart=int(p[8]), send=int(p[9]), evalue=float(p[10]),
                             bitscore=float(p[11])))
        print(f"  {key}: {sum(1 for r in rows if r['assembly']==key)} HSPs", flush=True)
    return rows


def best_query_assignment(rows):
    """Cluster HSPs into genomic regions; each region is won by one query."""
    calls = {}
    by_asm = defaultdict(list)
    for r in rows:
        by_asm[r["assembly"]].append(r)
    for asm, rs in by_asm.items():
        regions = defaultdict(list)          # (contig, 10kb bin) -> HSPs
        for r in rs:
            lo = min(r["sstart"], r["send"])
            regions[(r["contig"], lo // 10000)].append(r)
        # merge per query within region, then pick region winner
        winners = defaultdict(list)
        for reg, hsps in regions.items():
            agg = defaultdict(lambda: dict(bits=0.0, cov=0, best_pid=0.0))
            for h in hsps:
                a = agg[h["query"]]
                a["bits"] += h["bitscore"]
                a["cov"] += h["qend"] - h["qstart"] + 1
                a["best_pid"] = max(a["best_pid"], h["pident"])
                a["qlen"] = h["qlen"]
            win = max(agg.items(), key=lambda kv: kv[1]["bits"])
            q, a = win
            winners[q].append(dict(region=f"{reg[0]}:{reg[1]*10000}",
                                   qcov=100.0 * min(a["cov"], a["qlen"]) / a["qlen"],
                                   pident=a["best_pid"], bits=a["bits"]))
        for q in {r["query"] for r in rs}:
            w = winners.get(q, [])
            good = [x for x in w if x["pident"] >= PID_MIN and x["qcov"] >= QCOV_MIN]
            best = max(w, key=lambda x: x["bits"]) if w else None
            calls[(asm, q)] = dict(
                assembly=asm, query=q, n_regions_won=len(w), n_good=len(good),
                best_pident=round(best["pident"], 2) if best else 0.0,
                best_qcov=round(best["qcov"], 2) if best else 0.0,
                best_region=best["region"] if best else "",
                call="present" if good else ("absent" if not w else "degraded"))
    return calls


if __name__ == "__main__":
    qpath, found, missing = build_queries()
    print(f"queries built: {len(found)}; unresolved: {missing}", flush=True)
    rows = run_blast(qpath)
    with open(os.path.join(OUT, "layer1_tblastn_hsps.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    calls = best_query_assignment(rows)
    recs = []
    for (asm, q), c in calls.items():
        gene, tag, _ = q.split("|")
        c = dict(c); c["gene"], c["systematic_id"] = gene, tag
        c["khatri_absent"] = gene in KHATRI_ABSENT
        c["is_control"] = gene in CONTROLS
        recs.append(c)
    recs.sort(key=lambda r: (r["gene"], r["assembly"]))
    with open(os.path.join(OUT, "layer1_marker_calls.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gene","systematic_id","assembly","call","best_pident",
                                           "best_qcov","n_regions_won","n_good","best_region",
                                           "khatri_absent","is_control","query"])
        w.writeheader()
        for r in recs: w.writerow({k: r[k] for k in w.fieldnames})
    print(f"wrote {len(recs)} calls", flush=True)
    ctrl_fail = [r for r in recs if r["is_control"] and r["call"] != "present"]
    print("control failures:", ctrl_fail)

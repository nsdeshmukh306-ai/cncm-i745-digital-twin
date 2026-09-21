"""Layer 1b: locus-level presence/absence calls from the tblastn HSP table.

Per query x assembly we merge HSPs into candidate LOCI (same contig, HSPs
within LOCUS_GAP bp), score each locus by summed bitscore and query coverage,
then resolve competition between paralogous queries at overlapping loci.
Queries whose proteins are mutually near-identical (HXT9/HXT11, IMA2/3/4 ...)
cannot be separated by alignment; those loci are reported as
`present_ambiguous` rather than silently assigned to one member.
"""
import csv, os
from collections import defaultdict

PID_MIN, QCOV_MIN = 90.0, 80.0
LOCUS_GAP = 3000           # bp; merge HSPs of one query into one locus (yeast genes are short)
TIE = 0.98                 # bitscore ratio above which two queries are indistinguishable

rows = list(csv.DictReader(open("results/layer1_tblastn_hsps.csv")))
for r in rows:
    for k in ("pident", "bitscore", "evalue"): r[k] = float(r[k])
    for k in ("alen", "qlen", "qstart", "qend", "sstart", "send"): r[k] = int(r[k])
    r["lo"], r["hi"] = min(r["sstart"], r["send"]), max(r["sstart"], r["send"])

# ---------------------------------------------------------- 1. build loci
loci = []           # dict(assembly, query, contig, lo, hi, bits, qcov, pident)
grp = defaultdict(list)
for r in rows:
    grp[(r["assembly"], r["query"], r["contig"])].append(r)
for (asm, q, ctg), hs in grp.items():
    hs.sort(key=lambda h: h["lo"])
    cur = [hs[0]]
    for h in hs[1:]:
        if h["lo"] - cur[-1]["hi"] <= LOCUS_GAP:
            cur.append(h)
        else:
            loci.append((asm, q, ctg, cur)); cur = [h]
    loci.append((asm, q, ctg, cur))

def summarise(hs):
    qlen = hs[0]["qlen"]
    covered = set()
    for h in hs: covered.update(range(h["qstart"], h["qend"] + 1))
    bits = sum(h["bitscore"] for h in hs)
    pid = max(h["pident"] for h in hs)
    return bits, 100.0 * len(covered) / qlen, pid

L = []
for asm, q, ctg, hs in loci:
    bits, qcov, pid = summarise(hs)
    L.append(dict(assembly=asm, query=q, contig=ctg, lo=min(h["lo"] for h in hs),
                  hi=max(h["hi"] for h in hs), bits=bits, qcov=qcov, pident=pid))

# ------------------------------------------- 2. resolve competition per locus
by_ac = defaultdict(list)
for x in L: by_ac[(x["assembly"], x["contig"])].append(x)
for key, xs in by_ac.items():
    for x in xs:
        rivals = []
        for y in xs:
            if y is x: continue
            ov = min(y["hi"], x["hi"]) - max(y["lo"], x["lo"])
            if ov > 0.5 * min(y["hi"] - y["lo"], x["hi"] - x["lo"]):
                rivals.append(y)
        best_rival = max((y["bits"] for y in rivals), default=0.0)
        x["owned"] = x["bits"] >= best_rival
        x["ambiguous"] = bool(rivals) and (x["bits"] < best_rival) and (x["bits"] / best_rival >= TIE)

# ------------------------------------------------------- 3. call per query
QUAL = lambda x: x["pident"] >= PID_MIN and x["qcov"] >= QCOV_MIN
calls = {}
allq = {(x["assembly"], x["query"]) for x in L}
assemblies = sorted({r["assembly"] for r in rows})
queries = sorted({r["query"] for r in rows})
for asm in assemblies:
    for q in queries:
        xs = [x for x in L if x["assembly"] == asm and x["query"] == q]
        good = [x for x in xs if QUAL(x)]
        own = [x for x in good if x["owned"]]
        amb = [x for x in good if not x["owned"] and x["ambiguous"]]
        best = max(xs, key=lambda x: x["bits"]) if xs else None
        if own:      call = "present"
        elif amb:    call = "present_ambiguous"
        elif xs:     call = "degraded"
        else:        call = "absent"
        gene, tag, _ = q.split("|")
        calls[(asm, q)] = dict(
            assembly=asm, gene=gene, systematic_id=tag, call=call,
            copy_number=len(own), n_loci=len(xs),
            best_pident=round(best["pident"], 2) if best else 0.0,
            best_qcov=round(best["qcov"], 2) if best else 0.0,
            best_locus=f"{best['contig']}:{best['lo']}-{best['hi']}" if best else "")

os.makedirs("results", exist_ok=True)
with open("results/layer1_marker_calls.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(next(iter(calls.values())).keys()))
    w.writeheader()
    for k in sorted(calls, key=lambda k: (calls[k]["gene"], k[0])): w.writerow(calls[k])
print("wrote", len(calls), "calls")

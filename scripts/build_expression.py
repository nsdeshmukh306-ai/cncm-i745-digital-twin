"""Build gut-zone expression fold-change tables from GEO GSE18 (Gasch et al. 2000).

Output: data/expression_zones.csv with one row per (zone, systematic_id),
carrying the mean log2 ratio across the matched samples and its SD, so the
E-Flux ensemble can propagate measurement uncertainty.
"""
import gzip, glob, re, os
import numpy as np, pandas as pd

# zone -> (GEO platform, [sample accessions], provenance string)
ZONE_SAMPLES = {
    "duodenum": ("GPL51", ["GSM929", "GSM926", "GSM928", "GSM930"],
                 "heat shock to 37 C, 20 min (GSE18; Gasch et al. 2000)"),
    "ileum":    ("GPL64", ["GSM1112", "GSM1113", "GSM1114", "GSM1115",
                           "GSM1116", "GSM1117"],
                 "constant 0.32 mM H2O2, 10-60 min (GSE18)"),
    "colon":    ("GPL52", ["GSM972", "GSM1039", "GSM1040", "GSM938", "GSM939"],
                 "1 M sorbitol hyperosmotic shock, 5-90 min (GSE18)"),
}


def load_platform(gpl):
    """Return {spot_id: systematic ORF name} from a GEO .annot table."""
    path = f"data/geo/{gpl}.annot.gz"
    rows, header, started = [], None, False
    with gzip.open(path, "rt", errors="ignore") as fh:
        for line in fh:
            if line.startswith("!platform_table_begin"):
                started = True; continue
            if line.startswith("!platform_table_end"):
                break
            if started:
                p = line.rstrip("\n").split("\t")
                if header is None:
                    header = p
                else:
                    rows.append(p)
    df = pd.DataFrame(rows, columns=header)
    idc = header[0]
    # the ORF systematic name lives in whichever column matches Y??nnn[WC]
    orf_col, best = None, 0
    pat = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")
    for c in header[1:]:
        n = df[c].astype(str).str.strip().str.match(pat).sum()
        if n > best:
            best, orf_col = n, c
    return dict(zip(df[idc], df[orf_col].astype(str).str.strip())), best


def load_matrix(gpl):
    f = f"data/geo/GSE18-{gpl}_series_matrix.txt.gz"
    with gzip.open(f, "rt", errors="ignore") as fh:
        lines = fh.readlines()
    i = next(n for n, l in enumerate(lines) if l.startswith("!series_matrix_table_begin"))
    hdr = [x.strip('"') for x in lines[i + 1].rstrip("\n").split("\t")]
    data = [l.rstrip("\n").split("\t") for l in lines[i + 2:] if not l.startswith("!")]
    df = pd.DataFrame(data, columns=hdr).set_index("ID_REF")
    return df.apply(pd.to_numeric, errors="coerce")


if __name__ == "__main__":
    out, prov = [], []
    for zone, (gpl, samples, note) in ZONE_SAMPLES.items():
        spot2orf, n_orf = load_platform(gpl)
        mat = load_matrix(gpl)
        use = [s for s in samples if s in mat.columns]
        sub = mat[use]
        orf = pd.Series({k: v for k, v in spot2orf.items()})
        sub = sub.copy()
        sub["orf"] = [spot2orf.get(i, "") for i in sub.index]
        sub = sub[sub.orf.str.match(r"^Y[A-P][LR]\d{3}[WC]")]
        g = sub.groupby("orf")[use]
        mean = g.mean().mean(axis=1)
        sd = g.mean().std(axis=1).fillna(0.25).clip(lower=0.1)
        for orfid in mean.index:
            if np.isfinite(mean[orfid]):
                out.append(dict(zone=zone, systematic_id=orfid,
                                log2_ratio=round(float(mean[orfid]), 4),
                                sd_log2=round(float(sd[orfid]), 4),
                                n_samples=len(use)))
        prov.append(dict(zone=zone, platform=gpl, samples=";".join(use),
                         n_genes=int(np.isfinite(mean).sum()), source=note))
        print(f"{zone}: {gpl} {len(use)} samples, {int(np.isfinite(mean).sum())} genes "
              f"(platform ORF annotations: {n_orf})")
    pd.DataFrame(out).to_csv("data/expression_zones.csv", index=False)
    pd.DataFrame(prov).to_csv("data/expression_provenance.csv", index=False)
    print("wrote data/expression_zones.csv", len(out), "rows")

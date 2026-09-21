"""
A5 — Export endpoints.

* ``GET /export/report?format=json|csv|pdf`` bundles every layer's headline
  numeric outputs (genome assembly, GEM validation, host kinetics, surrogate
  metrics) into a single downloadable report. PDF rendering uses ReportLab.
* ``GET /export/gem`` streams the strain-specific SBML model as a download.
* ``GET /export/figures/{figure_id}`` serves a publication figure PNG.
"""

from __future__ import annotations

import io
import re
import csv
import json
import datetime as _dt
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

from api import state
from logging_setup import get_logger

logger = get_logger("api.export")
router = APIRouter(prefix="/export", tags=["export"])

PROJECT_TITLE = "Five-Layer Computational Digital Twin of Saccharomyces boulardii CNCM I-745"
DEVELOPER = "Niraj Deshmukh — MSc Biological Data Science, IISER Tirupati"

FIGURE_FILES = {
    "1": "figure1_genome_map.png",
    "2": "figure2_essentiality.png",
    "3": "figure3_gut_transit.png",
    "4": "figure4_host_kinetics.png",
    "5": "figure5_surrogate.png",
}


def _parse_report(path: Path, patterns: dict[str, str]) -> dict:
    if not path.exists():
        return {k: "N/A" for k in patterns}
    text = path.read_text()
    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        out[key] = m.group(1) if m else "N/A"
    return out


def gather_report_data() -> dict:
    """Collect headline numeric results across all five layers."""
    genome = _parse_report(state.LAYER1_RPT, {
        "num_sequences": r"Number of sequences\s*:\s*(\d+)",
        "total_bp":      r"Total bp\s*:\s*([\d,]+)",
        "gc_content":    r"Overall GC content\s*:\s*([\d.]+%)",
        "n50":           r"N50\s*:\s*([\d,]+)",
    })
    gem = _parse_report(state.LAYER2_RPT, {
        "reactions":       r"Reactions\s*:\s*(\d+)",
        "metabolites":     r"Metabolites\s*:\s*(\d+)",
        "genes":           r"Genes\s*:\s*(\d+)",
        "baseline_growth": r"Growth rate\s*:\s*([\d.]+)",
        "essential_genes": r"Essential\s*:\s*(\d+)",
    })
    scaler_v1 = json.loads(state.SCALER_JSON.read_text()) if state.SCALER_JSON.exists() else {}
    scaler_v2 = json.loads(state.SCALER_V2.read_text()) if state.SCALER_V2.exists() else {}
    return {
        "project": PROJECT_TITLE,
        "developer": DEVELOPER,
        "version": state.VERSION,
        "generated_utc": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "layer1_genome": {
            "sequences": genome["num_sequences"], "total_bp": genome["total_bp"],
            "gc_content": genome["gc_content"], "n50": genome["n50"],
            "genome_size_Mbp": 11.6, "chromosomes": 16,
        },
        "layer2_gem": {
            "reactions": gem["reactions"], "metabolites": gem["metabolites"],
            "genes": gem["genes"], "baseline_growth_h": gem["baseline_growth"],
            "essential_genes": gem["essential_genes"],
            "gpr_corrections": 11, "reactions_knocked_out": 2,
        },
        "layer3_regulatory": {
            "gut_zones": 4, "regulons": 4,
            "method": "E-Flux (Colijn et al. 2009)",
            "expression_source": "Gasch et al. 2000 (proxy)",
        },
        "layer4_host": {
            "nfkb_suppression_pct": 70.0, "barrier_score": 0.5971,
            "tcda_cleaved_120min_pct": 75.1, "t90_cleavage_min": 155.7,
            "spermine_synthase_flux_nmol": 16.3,
        },
        "layer5_surrogate": {
            "cross_val_r2_v1": scaler_v1.get("cross_val_r2_mean"),
            "cross_val_r2_v2": scaler_v2.get("cross_val_r2_mean"),
            "training_samples": scaler_v2.get("training_samples", 2000),
        },
    }


def _flatten(data: dict) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for section, value in data.items():
        if isinstance(value, dict):
            for k, v in value.items():
                rows.append((section, k, str(v)))
        else:
            rows.append(("meta", section, str(value)))
    return rows


@router.get("/report")
def export_report(format: str = Query("json", pattern="^(json|csv|pdf)$")):
    """Export the consolidated multi-layer report in the requested format."""
    data = gather_report_data()
    logger.info("export report format=%s", format)

    if format == "json":
        return JSONResponse(content=data)

    if format == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["section", "metric", "value"])
        w.writerows(_flatten(data))
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]), media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=cncm_i745_report.csv"})

    # PDF via ReportLab
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle)
    except ImportError:
        raise HTTPException(500, {"error": "reportlab not installed"})

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="CNCM I-745 Digital Twin Report",
                            topMargin=18 * mm, bottomMargin=18 * mm)
    styles = getSampleStyleSheet()
    navy = colors.HexColor("#1B3A6B")
    cyan = colors.HexColor("#00B4D8")
    story = [
        Paragraph(f"<b>{PROJECT_TITLE}</b>", styles["Title"]),
        Spacer(1, 4 * mm),
        Paragraph(DEVELOPER, styles["Normal"]),
        Paragraph(f"Version {data['version']} &nbsp;|&nbsp; {data['generated_utc']}", styles["Normal"]),
        Spacer(1, 6 * mm),
    ]
    table_data = [["Section", "Metric", "Value"]] + [list(r) for r in _flatten(data)]
    tbl = Table(table_data, colWidths=[40 * mm, 75 * mm, 55 * mm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E0E4EA")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ("LINEBELOW", (0, 0), (-1, 0), 1, cyan),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        "<i>Generated by the CNCM I-745 Digital Twin API. Methods and citations "
        "in docs/METHODS.md.</i>", styles["Italic"]))
    doc.build(story)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=cncm_i745_report.pdf"})


@router.get("/gem")
def export_gem():
    """Download the strain-specific SBML/XML model."""
    if not state.GEM_PATH.exists():
        raise HTTPException(404, {"error": "GEM not found"})
    return FileResponse(str(state.GEM_PATH), media_type="application/xml",
                        filename=state.GEM_PATH.name)


@router.get("/figures/{figure_id}")
def export_figure(figure_id: str):
    """Serve a publication figure PNG by id (1-5) or filename."""
    fname = FIGURE_FILES.get(figure_id, figure_id)
    if not fname.endswith(".png"):
        fname += ".png"
    path = state.FIGURES_DIR / fname
    if not path.exists():
        raise HTTPException(404, {"error": "figure not found", "detail": fname})
    return FileResponse(str(path), media_type="image/png", filename=fname)

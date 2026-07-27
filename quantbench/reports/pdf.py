"""
quantbench.reports.pdf
======================
Genere un PDF de SYNTHESE des etats financiers (4 ans) par titre — notre propre
document (aucun souci de droits, contrairement a l'hebergement des depots
officiels). Le profil renvoie aussi vers le depot SEC officiel.
"""

from __future__ import annotations

import os


def financial_summary_pdf(profile: dict, out_path: str) -> str | None:
    """Ecrit un PDF de synthese a `out_path`. Retourne le chemin ou None."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                        Paragraph, Spacer)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    except Exception:
        return None

    s = profile.get("statements")
    if not s or not s.get("years"):
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    styles = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=styles["Title"], fontSize=15, spaceAfter=2,
                       textColor=colors.HexColor("#10151c"))
    sub = ParagraphStyle("s", parent=styles["Normal"], fontSize=9,
                         textColor=colors.HexColor("#5b6472"))
    note = ParagraphStyle("n", parent=styles["Normal"], fontSize=7.5,
                          textColor=colors.HexColor("#8a94a3"), spaceBefore=8)

    val = profile.get("valuation", {})
    years = s["years"]
    lines = [("Chiffre d'affaires", "revenue"), ("Résultat opérationnel", "ebit"),
             ("Résultat net", "net_income"), ("Cash-flow d'exploitation", "cfo"),
             ("Actif total", "total_assets"), ("Capitaux propres", "equity"),
             ("Dette", "total_debt")]

    def cell(v):
        return "—" if v is None else f"{v:,.1f}".replace(",", " ")

    table_data = [["en Md USD"] + list(years)]
    for lab, k in lines:
        table_data.append([lab] + [cell(v) for v in (s.get(k) or [])])

    ncols = len(years) + 1
    t = Table(table_data, colWidths=[55 * mm] + [(115 * mm) / (ncols - 1)] * (ncols - 1))
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#b8892b")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#dde2e9")),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    story = [
        Paragraph(f"{profile.get('name', profile.get('ticker'))} "
                  f"({profile.get('ticker')})", h),
        Paragraph(f"{profile.get('sector', '')} · États financiers de synthèse — "
                  f"QuantBench · valorisation : {val.get('method', '—')}", sub),
        Spacer(1, 10),
        t,
        Paragraph("Source : SEC Financial Statement Data Sets (dépôts 10-K), montants en "
                  "milliards USD. Document de synthèse QuantBench — outil éducatif, pas un "
                  "conseil d'investissement. Pour les états officiels complets, consulter "
                  "SEC EDGAR.", note),
    ]
    SimpleDocTemplate(out_path, pagesize=A4,
                      topMargin=18 * mm, bottomMargin=15 * mm,
                      leftMargin=18 * mm, rightMargin=18 * mm).build(story)
    return out_path


__all__ = ["financial_summary_pdf"]

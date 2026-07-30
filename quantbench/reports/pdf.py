"""
quantbench.reports.pdf
======================
PDF de synthèse par titre — NOTRE rapport (valorisation + activité + états 4 ans
+ forensique), généré pour CHAQUE titre couvert (fiable, contrairement aux PDF
glossy des sociétés qui ne sont pas accessibles par une API universelle gratuite).
Le profil renvoie aussi vers le dépôt officiel SEC.
"""

from __future__ import annotations

import os

_GOLD = "#b8892b"
_INK = "#10151c"
_MUT = "#5b6472"


def _fmt(v):
    return "—" if v is None else f"{v:,.1f}".replace(",", " ")


def financial_summary_pdf(profile: dict, out_path: str) -> str | None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                        Paragraph, Spacer)
    except Exception:
        return None

    s = profile.get("statements")
    if not s or not s.get("years"):
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    val = profile.get("valuation", {}) or {}
    fx = (profile.get("forensics") or {}).get("scores", {}) or {}
    base = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=base["Title"], fontSize=16, leading=18,
                           textColor=colors.HexColor(_INK), spaceAfter=0)
    small = ParagraphStyle("s", parent=base["Normal"], fontSize=8.5,
                           textColor=colors.HexColor(_MUT))
    body = ParagraphStyle("b", parent=base["Normal"], fontSize=9, leading=12,
                          textColor=colors.HexColor(_INK))
    foot = ParagraphStyle("f", parent=base["Normal"], fontSize=7,
                          textColor=colors.HexColor("#8a94a3"), spaceBefore=10)

    story = []

    # --- Bandeau titre ---
    banner = Table([[Paragraph(f"<font color='white'><b>QuantBench</b></font>", small),
                     Paragraph("<font color='white'>Rapport de synthèse</font>",
                               ParagraphStyle("r", parent=small, alignment=2))]],
                   colWidths=[90 * mm, 80 * mm])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(_GOLD)),
                                ("TOPPADDING", (0, 0), (-1, -1), 4),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                                ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    story += [banner, Spacer(1, 8),
              Paragraph(f"{profile.get('name', profile.get('ticker'))} "
                        f"<font color='{_GOLD}'>({profile.get('ticker')})</font>", title),
              Paragraph(f"{profile.get('sector', '')}"
                        f"{' · ' + profile.get('industry') if profile.get('industry') else ''}", small),
              Spacer(1, 8)]

    # --- Valorisation ---
    up = val.get("upside")
    up_s = "—" if up is None else f"{up * 100:+.0f} %"
    vbox = Table([["Cours", "Valeur intrinsèque / action", "Upside", "Méthode"],
                  [_money(val.get("price")), _money(val.get("value_per_share")),
                   up_s, val.get("method", "—")]],
                 colWidths=[28 * mm, 48 * mm, 24 * mm, 70 * mm])
    vbox.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(_MUT)),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 1), colors.HexColor("#f4f6f8")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dde2e9")),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7)]))
    story += [vbox, Spacer(1, 10)]

    # --- Note de risque ---
    # Elle repond a une question DIFFERENTE de l'upside, et l'omettre du rapport
    # laisserait croire qu'une decote est une opportunite. Les dimensions sont triees
    # de la plus risquee a la moins risquee : le lecteur doit voir d'abord ce qui
    # degrade la note, non une liste alphabetique.
    risque = profile.get("risque") or {}
    if risque.get("grade"):
        fam = {"A": "#1a7f5a", "B": "#2f7db8", "C": "#a8791f",
               "D": "#c2612a", "F": "#a33b47"}[risque["grade"][0]]
        dispo = [d for d in risque.get("dimensions") or [] if d.get("rang") is not None]
        dispo.sort(key=lambda d: -d["rang"])
        pires = " · ".join(f"{d['nom']} {round(d['rang'] * 100)}" for d in dispo[:4])
        plafonds = ", ".join(risque.get("plafonds_appliques") or [])
        rbox = Table([[Paragraph(f"<font size=20 color='{fam}'><b>"
                                 f"{risque['grade']}</b></font>", body),
                       Paragraph("<b>Note de risque</b> — probabilité de perdre "
                                 "durablement sa mise, distincte de la décote.<br/>"
                                 f"<font size=7.5 color='{_MUT}'>Dimensions les plus "
                                 f"dégradées (0 = meilleur cas observé, 100 = pire) : "
                                 f"{pires or '—'}"
                                 + (f"<br/>Plafond appliqué : {plafonds}" if plafonds else "")
                                 + "</font>", body)]],
                     colWidths=[20 * mm, 150 * mm])
        rbox.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dde2e9")),
            ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor(fam)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 7)]))
        story += [rbox, Spacer(1, 10)]

    # --- Activité ---
    if profile.get("summary"):
        story += [Paragraph("<b>Activité</b>", body),
                  Paragraph(profile["summary"][:500], body), Spacer(1, 9)]

    # --- États financiers ---
    story += [Paragraph("<b>États financiers</b> (Md USD)", body), Spacer(1, 3)]
    lines = [("Chiffre d'affaires", "revenue"), ("Résultat opérationnel", "ebit"),
             ("Résultat net", "net_income"), ("Cash-flow d'exploitation", "cfo"),
             ("Actif total", "total_assets"), ("Capitaux propres", "equity"),
             ("Dette", "total_debt")]
    tdata = [["Md USD"] + list(s["years"])]
    for lab, k in lines:
        tdata.append([lab] + [_fmt(v) for v in (s.get(k) or [])])
    n = len(s["years"]) + 1
    t = Table(tdata, colWidths=[52 * mm] + [(118 * mm) / (n - 1)] * (n - 1))
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_GOLD)),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#dde2e9")),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [t, Spacer(1, 10)]

    # --- Forensique ---
    fparts = []
    if fx.get("piotroski_f") is not None:
        fparts.append(f"Piotroski {fx['piotroski_f']}/9")
    if fx.get("beneish_m") is not None:
        fparts.append(f"Beneish M {fx['beneish_m']}"
                      f"{' (à investiguer)' if fx.get('beneish_flag') else ''}")
    if fx.get("altman_z") is not None:
        fparts.append(f"Altman Z″ {fx['altman_z']}")
    if fparts:
        story += [Paragraph("<b>Forensique</b> — " + " · ".join(fparts), body)]
        for flag in (profile.get("forensics") or {}).get("flags", [])[:3]:
            story += [Paragraph("⚠ " + flag, small)]

    story += [Paragraph("Source : SEC Financial Statement Data Sets. Document de synthèse "
                        "QuantBench — outil éducatif, <b>pas un conseil d'investissement</b>. "
                        "Valorisation sur cas de base ; signaux forensiques statistiques à "
                        "investiguer. Rapport officiel complet : SEC EDGAR.", foot)]

    SimpleDocTemplate(out_path, pagesize=A4, topMargin=14 * mm, bottomMargin=12 * mm,
                      leftMargin=18 * mm, rightMargin=18 * mm).build(story)
    return out_path


def _money(v):
    return "—" if v is None else f"{v:,.2f} $".replace(",", " ")


__all__ = ["financial_summary_pdf"]

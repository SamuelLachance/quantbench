"""
quantbench.valuation.build_universal
====================================
Construit un DcfInputs a partir des fondamentaux UNIVERSELS (yfinance, deja
convertis en USD par data.universal). Permet de valoriser US + Canada avec le
meme moteur DCF FCFF que le connecteur SEC, mais sur la source universelle.
"""

from __future__ import annotations

import json as _json
import os as _os

import numpy as np

from . import DcfInputs
from ..data import market
from ..data.build import _estimate_growth, _clamp

_DEFAULT_ERP = 0.045          # prime de risque d'un marche mature (Etats-Unis)

# Prime de risque PAYS (methode Damodaran : ERP total = ERP mature + CRP du pays
# d'operation, deduite du spread de defaut souverain ajuste de la volatilite
# relative des actions). Sans elle, une societe bresilienne ou chinoise est
# actualisee au cout du capital americain -> valeur fortement surestimee.
_CRP = {
    # marches matures
    "US": 0.0, "CA": 0.0, "DE": 0.0, "CH": 0.0, "NL": 0.0, "SE": 0.0, "NO": 0.0,
    "DK": 0.0, "SG": 0.0, "AU": 0.0, "NZ": 0.0, "LU": 0.0, "FI": 0.0,
    "GB": 0.006, "FR": 0.008, "BE": 0.008, "IE": 0.010, "AT": 0.006, "JP": 0.010,
    "KR": 0.010, "TW": 0.010, "IL": 0.014, "ES": 0.020, "IT": 0.026, "PT": 0.024,
    # emergents
    "CN": 0.014, "CL": 0.014, "PL": 0.017, "MY": 0.020, "TH": 0.023, "MX": 0.026,
    "IN": 0.026, "ID": 0.026, "PH": 0.026, "PE": 0.026, "ZA": 0.043, "BR": 0.037,
    "CO": 0.037, "VN": 0.043, "GR": 0.043, "TR": 0.075, "EG": 0.086, "NG": 0.086,
    "AR": 0.115, "PK": 0.115, "RU": 0.115, "UA": 0.115,
}
_CRP_DEFAUT = 0.030           # pays non liste : prime emergente prudente

# Taux d'impot sur les societes par PAYS (taux marginal legal). Appliquer 21 %
# (taux federal americain) a une societe canadienne, allemande ou japonaise
# faussait mecaniquement tous ses flux apres impot.
_TAUX_IMPOT = {
    "US": 0.21, "CA": 0.265, "GB": 0.25, "DE": 0.30, "FR": 0.258, "IT": 0.24,
    "ES": 0.25, "NL": 0.258, "BE": 0.25, "CH": 0.15, "IE": 0.125, "SE": 0.206,
    "NO": 0.22, "DK": 0.22, "FI": 0.20, "AT": 0.23, "PT": 0.21, "GR": 0.22,
    "JP": 0.306, "CN": 0.25, "HK": 0.165, "TW": 0.20, "KR": 0.24, "SG": 0.17,
    "IN": 0.252, "AU": 0.30, "NZ": 0.28, "IL": 0.23, "ZA": 0.27, "BR": 0.34,
    "MX": 0.30, "CL": 0.27, "CO": 0.35, "AR": 0.35, "TR": 0.25, "RU": 0.20,
    "ID": 0.22, "TH": 0.20, "MY": 0.24, "PH": 0.25, "VN": 0.20, "PL": 0.19,
    "LU": 0.2494, "BM": 0.15, "KY": 0.15, "VG": 0.15, "PA": 0.25,
}
_IMPOT_DEFAUT = 0.25


def tax_rate(country) -> float:
    """Taux d'impot marginal du pays de la societe."""
    if not country:
        return _IMPOT_DEFAUT
    return _TAUX_IMPOT.get(str(country).strip().upper()[:2], _IMPOT_DEFAUT)


def country_erp(country, erp_mature=_DEFAULT_ERP):
    """ERP total = ERP mature + prime de risque pays (Damodaran)."""
    if not country:
        return erp_mature
    return erp_mature + _CRP.get(str(country).strip().upper()[:2], _CRP_DEFAUT)


# --------------------------------------------------------------------------- #
# Reperes par INDUSTRIE puis par SECTEUR (mesures sur l'univers reel)
# --------------------------------------------------------------------------- #
_STATS_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            "industry_stats.json")
try:
    with open(_STATS_PATH, encoding="utf-8") as _f:
        _STATS = _json.load(_f)
except Exception:
    _STATS = {"industries": {}, "secteurs": {}, "global": {}}


def repere(fund, cle, defaut=None):
    """Repere mesure, du plus fin au plus large : INDUSTRIE -> SECTEUR -> GLOBAL.
    Damodaran publie ses references par industrie ; a defaut on remonte d'un cran
    plutot que d'imposer une constante unique a toutes les societes."""
    for niveau, clef in (("industries", fund.get("industry")),
                         ("secteurs", fund.get("sector"))):
        g = _STATS.get(niveau, {}).get(clef or "")
        if g and g.get(cle) is not None:
            return g[cle]
    g = _STATS.get("global", {})
    return g[cle] if g.get(cle) is not None else defaut


def beta_ascendant(fund, tx):
    """BETA ASCENDANT (bottom-up), methode canonique de Damodaran.

    Il recommande explicitement de NE PAS utiliser le beta de regression d'un
    titre : bruite, instable, et carrement faux sur une cotation peu liquide (0,20
    pour la fonciere mexicaine Fibra UNO, 0,24 pour China Minsheng — ces titres
    bougent peu faute d'ECHANGES, pas faute de RISQUE).

    On part du beta d'ACTIVITE de l'industrie, mesure en desendettant les betas de
    ses membres, puis on le RE-ENDETTE au levier propre de la societe et au taux
    d'impot de SON pays :

        beta_levier = beta_desendette_industrie x (1 + (1 - t) x D/E)

    Le risque devient ainsi une propriete de l'activite exercee, corrigee de la
    structure de bilan et de la fiscalite locale — adaptatif sur les trois axes.
    Retourne (beta_levier, beta_desendette, source)."""
    bu = repere(fund, "beta_desendette")
    if bu is None:
        # aucun repere : on desendette le beta publie, faute de mieux
        b = fund.get("beta") or 1.0
        de0 = _safe_div(fund.get("total_debt"), fund.get("market_cap")) or 0.0
        return b, b / (1 + (1 - tx) * de0), "beta publie"
    de = _safe_div(fund.get("total_debt"), fund.get("market_cap")) or 0.0
    de = _clamp(de, 0.0, 5.0)
    return bu * (1 + (1 - tx) * de), bu, "industrie"


def build_dcf_from_fundamentals(fund: dict, *, margin_override: float | None = None,
                                erp: float | None = None, rf: float | None = None):
    """Retourne (DcfInputs, meta) depuis un dict de fondamentaux universal.get_fundamentals.
    margin_override : force la marge operationnelle (ex. marge normalisee pour un cyclique).
    erp : si None, ERP mature + prime de risque du pays de la societe (Damodaran)."""
    if erp is None:
        erp = country_erp(fund.get("country"))
    rev = fund.get("revenue")
    if not rev or rev <= 0:
        raise ValueError(f"CA indisponible pour {fund.get('ticker')}")

    rev_hist = [x for x in (fund.get("revenue_history") or []) if x]
    if len(rev_hist) >= 2:
        g_start, _ = _estimate_growth(rev_hist)
    else:
        g_start = 0.08

    op_margin = margin_override if margin_override is not None else fund.get("operating_margin")
    if op_margin is None:
        op_margin = _safe_div(fund.get("ebit"), rev) or 0.10
    op_margin = _clamp(op_margin, -0.20, 0.75)
    tx = tax_rate(fund.get("country"))          # taux d'impot du pays de la societe

    debt = fund.get("total_debt") or 0.0
    cash = fund.get("cash") or 0.0
    equity_book = fund.get("book_equity") or 0.0
    market_cap = fund.get("market_cap") or rev
    invested_raw = max(equity_book + debt - cash, 0.05 * rev)
    s2c = _clamp(rev / invested_raw, 0.3, 6.0)
    # COHERENCE : le ROIC doit porter sur le capital REELLEMENT utilise par le
    # modele (rev / s2c borne), pas sur le capital comptable brut. Sinon une
    # societe a capitaux propres negatifs (scission, rachats d'actions) affiche un
    # capital investi derisoire -> ROIC plafonne a 60% -> reinvestissement quasi
    # nul -> valeur surestimee (cas Embecta).
    invested = rev / s2c
    nopat = op_margin * rev * (1.0 - tx)
    cur_roic = _clamp(_safe_div(nopat, invested) or 0.12, 0.02, 0.40)

    rf = rf if rf is not None else market.risk_free_rate()
    lev_beta, unlev, _src = beta_ascendant(fund, tx)
    cost_equity = rf + lev_beta * erp
    term_roic = _clamp(cost_equity + 0.02, 0.07, max(cur_roic, 0.08))

    # Croissance perpetuelle : plafonnee par le taux sans risque (Damodaran : une
    # societe ne peut croitre indefiniment plus vite que l'economie). Une activite
    # en DECLIN structurel ne doit pas etre supposee re-accelerer a +2,8% : on borne
    # alors la croissance terminale par sa tendance (plancher -1%).
    # Cout de la dette : spread synthetique CROISSANT avec le levier (notation
    # synthetique de Damodaran). Un spread constant (rf + 120 bp) faisait decroitre
    # le WACC sans borne quand la dette augmentait -- pathologie Modigliani-Miller
    # sans couts de detresse : plus une societe s'endettait, plus elle "valait".
    lev = debt / (debt + market_cap) if (debt + market_cap) > 0 else 0.0
    kd = rf + 0.010 + 0.10 * _clamp(lev, 0.0, 1.0) ** 2

    term = min(rf, 0.028)
    if g_start < 0:
        term = min(term, max(g_start, -0.01))
    x = DcfInputs(
        revenue_base=rev,
        g1_begin=g_start, g1_end=0.80 * g_start + 0.20 * term,
        g2_begin=0.80 * g_start + 0.20 * term, g2_end=0.45 * g_start + 0.55 * term,
        g3_begin=0.45 * g_start + 0.55 * term, g3_end=term,
        len1=3, len2=4, len3=3,
        current_operating_margin=op_margin,
        # Une societe ne peut pas perdre de l'argent A PERPETUITE (elle serait
        # liquidee) : la marge TERMINALE ne descend pas sous zero. Les societes
        # durablement deficitaires sont routees ailleurs (jeune / mature en perte).
        terminal_operating_margin=op_margin if op_margin > 0 else 0.02,
        margin_converge_start=3,
        current_tax_rate=tx, marginal_tax_rate=tx, tax_converge_start=5,
        current_sales_to_capital=s2c, terminal_sales_to_capital=s2c, s2c_converge_start=3,
        risk_free_rate=rf, erp=erp,
        unlevered_beta=unlev, terminal_unlevered_beta=_clamp(unlev, 0.8, 1.2),
        beta_converge_start=5,
        current_pretax_kd=kd, terminal_pretax_kd=kd, kd_converge_start=5,
        equity_value=market_cap, debt_value=debt, cash_and_non_operating=cash,
        reinvestment_mode="roic", current_roic=cur_roic, terminal_roic=term_roic,
        roic_converge_start=5,
    )
    meta = {"g_start": g_start, "op_margin": op_margin, "s2c": s2c,
            "cur_roic": cur_roic, "term_roic": term_roic, "rf": rf,
            "beta": lev_beta, "erp": erp}
    return x, meta


def project(fund, years=20, margin_override=None):
    """Projection COMPLETE année par année (toutes les colonnes du DCF Damodaran)
    via le moteur value_dcf. Chaque enregistrement contient : CA, croissance, marge,
    EBIT, taux d'impôt, EBI (EBIT après impôt), réinvestissement, FCFF, ROIC, capital
    investi, WACC, coût des FP, facteur d'actualisation, valeur actualisée."""
    import numpy as np
    from .dcf import value_dcf
    x, _ = build_dcf_from_fundamentals(fund, margin_override=margin_override)
    l1 = max(1, years // 4)
    l3 = max(1, years // 4)
    l2 = max(1, years - l1 - l3)
    x.len1, x.len2, x.len3 = l1, l2, l3
    res = value_dcf(x)
    rev, g, m = res["revenues"], res["growth"], res["margins"]
    eat, reinv, fcff = res["ebit_after_tax"], res["reinvestment"], res["fcff"]
    wacc, coe, roic, ic = res["wacc"], res["cost_of_equity"], res["roic"], res["invested_capital"]
    disc = np.cumprod(1.0 + np.asarray(wacc, dtype=float))
    fin = lambda v: (v == v and v is not None)          # non-NaN

    out = []
    for i in range(x.n_years):
        ebit = float(rev[i] * m[i])
        eati = float(eat[i])
        tax = (1 - eati / ebit) if ebit else None
        out.append({
            "year": i + 1,
            "revenue": round(float(rev[i]), 2),
            "revenue_growth_pct": round(float(g[i]) * 100, 2),
            "operating_margin_pct": round(float(m[i]) * 100, 2),
            "ebit": round(ebit, 2),
            "tax_rate_pct": round(tax * 100, 2) if tax is not None else None,
            "ebit_after_tax": round(eati, 2),
            "reinvestment": round(float(reinv[i]), 2),
            "fcff": round(float(fcff[i]), 2),
            "roic_pct": round(float(roic[i]) * 100, 2) if fin(roic[i]) else None,
            "invested_capital": round(float(ic[i]), 2) if fin(ic[i]) else None,
            "wacc_pct": round(float(wacc[i]) * 100, 2),
            "cost_of_equity_pct": round(float(coe[i]) * 100, 2),
            "discount_factor": round(float(disc[i]), 4),
            "pv_fcff": round(float(fcff[i] / disc[i]), 2),
        })
    return out


def _safe_div(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


__all__ = ["build_dcf_from_fundamentals"]

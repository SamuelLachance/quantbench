"""
quantbench.valuation.build_universal
====================================
Construit un DcfInputs a partir des fondamentaux UNIVERSELS (yfinance, deja
convertis en USD par data.universal). Permet de valoriser US + Canada avec le
meme moteur DCF FCFF que le connecteur SEC, mais sur la source universelle.
"""

from __future__ import annotations

import numpy as np

from . import DcfInputs
from ..data import market
from ..data.build import _estimate_growth, _clamp

_DEFAULT_ERP = 0.045


def build_dcf_from_fundamentals(fund: dict, *, margin_override: float | None = None,
                                erp: float = _DEFAULT_ERP):
    """Retourne (DcfInputs, meta) depuis un dict de fondamentaux universal.get_fundamentals.
    margin_override : force la marge operationnelle (ex. marge normalisee pour un cyclique)."""
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

    debt = fund.get("total_debt") or 0.0
    cash = fund.get("cash") or 0.0
    equity_book = fund.get("book_equity") or 0.0
    market_cap = fund.get("market_cap") or rev
    invested = max(equity_book + debt - cash, 0.05 * rev)
    s2c = _clamp(rev / invested, 0.3, 6.0)

    nopat = op_margin * rev * 0.75
    cur_roic = _clamp(_safe_div(nopat, invested) or 0.12, 0.02, 0.60)

    rf = market.risk_free_rate()
    lev_beta = fund.get("beta") or 1.1
    de = _safe_div(debt, market_cap) or 0.0
    unlev = lev_beta / (1 + (1 - 0.25) * de) if de >= 0 else lev_beta
    cost_equity = rf + lev_beta * erp
    term_roic = _clamp(cost_equity + 0.02, 0.07, max(cur_roic, 0.08))

    term = min(rf, 0.028)
    x = DcfInputs(
        revenue_base=rev,
        g1_begin=g_start, g1_end=0.80 * g_start + 0.20 * term,
        g2_begin=0.80 * g_start + 0.20 * term, g2_end=0.45 * g_start + 0.55 * term,
        g3_begin=0.45 * g_start + 0.55 * term, g3_end=term,
        len1=3, len2=4, len3=3,
        current_operating_margin=op_margin, terminal_operating_margin=op_margin,
        margin_converge_start=3,
        current_tax_rate=0.21, marginal_tax_rate=0.25, tax_converge_start=5,
        current_sales_to_capital=s2c, terminal_sales_to_capital=s2c, s2c_converge_start=3,
        risk_free_rate=rf, erp=erp,
        unlevered_beta=unlev, terminal_unlevered_beta=_clamp(unlev, 0.8, 1.2),
        beta_converge_start=5,
        current_pretax_kd=rf + 0.012, terminal_pretax_kd=rf + 0.012, kd_converge_start=5,
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

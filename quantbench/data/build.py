"""
quantbench.data.build
=====================
Assemble les donnees SEC + marche en un `DcfInputs` calibre + un jeu de lois
pour le Monte Carlo. Toutes les hypotheses sont derivees des donnees quand c'est
possible ; sinon des defauts documentes sont utilises et l'incertitude est
poussee dans le Monte Carlo.

Choix methodologiques (corriges vs. script legacy) :
* ERP = prime de risque implicite fixe (~4.5 %, ordre de grandeur Damodaran),
  et NON le rendement passe du marche (biais retrospectif).
* Croissance : mediane historique du CA, convergeant vers le taux sans risque.
* Beta non-leve = beta leve / (1 + (1 - t) * D/E), D = dette, E = capitalisation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..valuation import DcfInputs
from . import edgar, market

_B = 1e9
_DEFAULT_ERP = 0.045


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _estimate_growth(revenues):
    """Croissance de depart calibree : melange du dernier YoY, du CAGR 3 ans et du
    CAGR complet, borne. Corrige le biais de la mediane historique qui sous-estime
    l'acceleration recente (ex. NVDA). Retourne (g_start, diagnostics)."""
    revs = [r for r in revenues if r > 0]
    n = len(revs)
    if n < 2:
        return 0.05, {}
    g_last = revs[-1] / revs[-2] - 1
    k3 = min(3, n - 1)
    g_cagr3 = (revs[-1] / revs[-1 - k3]) ** (1 / k3) - 1
    g_cagr_full = (revs[-1] / revs[0]) ** (1 / (n - 1)) - 1
    blend = 0.5 * g_last + 0.3 * g_cagr3 + 0.2 * g_cagr_full
    g_start = _clamp(blend, -0.05, 0.45)
    diag = {"g_last": round(g_last, 4), "g_cagr3": round(g_cagr3, 4),
            "g_cagr_full": round(g_cagr_full, 4), "g_start": round(g_start, 4)}
    return g_start, diag


def build_dcf_inputs(ticker: str, erp: float = _DEFAULT_ERP):
    """Retourne (base: DcfInputs, distributions: dict, meta: dict)."""
    ticker = ticker.upper()
    facts = edgar.get_facts(edgar.get_cik(ticker))

    rev_series = edgar.annual_series(facts, edgar.TAGS["revenue"])
    if len(rev_series) < 2:
        raise ValueError(f"Historique de CA insuffisant pour {ticker}.")
    revenues = [v for _, v in rev_series]
    revenue_base = revenues[-1] / _B

    op_income = edgar.latest(facts, edgar.TAGS["operating_income"])
    pretax = edgar.latest(facts, edgar.TAGS["pretax_income"])
    tax_exp = edgar.latest(facts, edgar.TAGS["tax_expense"])
    interest = edgar.latest(facts, edgar.TAGS["interest_expense"])
    cash = edgar.latest(facts, edgar.TAGS["cash"], "instant", 0.0) or 0.0
    ms_c = edgar.latest(facts, edgar.TAGS["marketable_current"], "instant", 0.0) or 0.0
    ms_n = edgar.latest(facts, edgar.TAGS["marketable_noncurrent"], "instant", 0.0) or 0.0
    equity_book = edgar.latest(facts, edgar.TAGS["equity"], "instant", 0.0) or 0.0
    debt = edgar.total_debt(facts)
    # Actions : dei (instant) ; fallback actions diluees (multi-classes : GOOGL, META...)
    sh_series = edgar.annual_series(facts, edgar.TAGS["shares"], "instant")
    if not sh_series:
        sh_series = edgar.annual_series(facts, edgar.TAGS["shares_diluted"], "duration")
    if not sh_series:
        raise ValueError(f"Nombre d'actions introuvable pour {ticker}.")
    shares, shares_date = sh_series[-1][1], sh_series[-1][0]
    # Corrige un eventuel split d'action survenu APRES la date de reference du 10-K.
    shares *= market.split_factor_since(ticker, shares_date)

    # --- Marche ---
    price = market.latest_price(ticker)
    rf = market.risk_free_rate()
    lev_beta = market.levered_beta(ticker)

    market_cap = price * shares / _B
    debt_b = debt / _B
    cash_b = (cash + ms_c + ms_n) / _B
    de_ratio = debt_b / market_cap if market_cap > 0 else 0.0

    # --- Marges, impots, reinvestissement ---
    op_margin = _clamp(op_income / revenues[-1], -0.5, 0.75) if op_income else 0.10
    eff_tax = _clamp(tax_exp / pretax, 0.0, 0.35) if (pretax and tax_exp) else 0.21
    # Capital investi = capitaux propres comptables + dette - cash/actifs non-op
    # (Damodaran : le cash n'est pas du capital operationnel). Evite de sur-estimer
    # l'intensite capitalistique des entreprises riches en tresorerie.
    invested = equity_book / _B + debt_b - cash_b
    s2c = _clamp(revenue_base / invested, 0.3, 6.0) if invested > 0 else 2.0

    # cout de la dette : interet / dette (borne), sinon rf + spread
    if interest and debt > 0:
        kd = _clamp(abs(interest) / debt, 0.01, 0.15)
    else:
        kd = rf + 0.012

    # --- Croissance calibree, en declin vers le terminal ---
    g_start, g_diag = _estimate_growth(revenues)
    term = min(rf, 0.028)
    g1_begin = g_start
    g1_end = 0.80 * g_start + 0.20 * term
    g2_begin = g1_end
    g2_end = 0.45 * g_start + 0.55 * term
    g3_begin = g2_end
    g3_end = term

    # --- Beta non-leve ---
    unlev = lev_beta / (1 + (1 - 0.25) * de_ratio) if de_ratio >= 0 else lev_beta

    # --- ROIC (reinvestissement Damodaran) ---
    nopat = (op_income or 0.0) * (1 - eff_tax)
    invested_dollars = invested * _B
    cur_roic = _clamp(nopat / invested_dollars, 0.02, 0.60) if invested_dollars > 0 else 0.12
    cost_equity_approx = rf + lev_beta * erp
    # ROIC terminal : cout du capital + faible rente durable, borne sous le ROIC courant
    term_roic = _clamp(cost_equity_approx + 0.02, 0.07, max(cur_roic, 0.08))

    base = DcfInputs(
        revenue_base=revenue_base,
        g1_begin=g1_begin, g1_end=g1_end,
        g2_begin=g2_begin, g2_end=g2_end,
        g3_begin=g3_begin, g3_end=g3_end,
        len1=3, len2=4, len3=3, conv1=1, conv2=1, conv3=1,
        current_operating_margin=op_margin, terminal_operating_margin=op_margin,
        margin_converge_start=3,
        current_tax_rate=eff_tax, marginal_tax_rate=0.25, tax_converge_start=5,
        current_sales_to_capital=s2c, terminal_sales_to_capital=s2c,
        s2c_converge_start=3,
        risk_free_rate=rf, erp=erp,
        unlevered_beta=unlev, terminal_unlevered_beta=_clamp(unlev, 0.8, 1.2),
        beta_converge_start=5,
        current_pretax_kd=kd, terminal_pretax_kd=kd, kd_converge_start=5,
        equity_value=market_cap, debt_value=debt_b, cash_and_non_operating=cash_b,
        additional_roic_in_perpetuity=0.0,
        reinvestment_mode="roic", current_roic=cur_roic, terminal_roic=term_roic,
        roic_converge_start=5,
    )

    # --- Lois pour le Monte Carlo (incertitude autour du cas de base) ---
    distributions = {
        "g1_begin":                  stats.norm(g1_begin, max(0.02, abs(g_start) * 0.35)),
        "g2_end":                    stats.norm(g2_end, 0.02),
        "terminal_operating_margin": stats.norm(op_margin, max(0.015, abs(op_margin) * 0.12)),
        "erp":                       stats.norm(erp, 0.005),
        "unlevered_beta":            stats.norm(unlev, 0.15),
        "current_roic":              stats.norm(cur_roic, max(0.03, cur_roic * 0.15)),
        "terminal_roic":             stats.norm(term_roic, 0.01),
        "g3_end":                    stats.norm(g3_end, 0.003),
    }
    correlations = [
        ("terminal_operating_margin", "current_roic", 0.3),
        ("g1_begin", "g2_end", 0.4),
    ]

    meta = {
        "ticker": ticker,
        "fiscal_year_end": rev_series[-1][0],
        "price": round(price, 2),
        "shares": shares / _B,
        "market_cap": round(market_cap, 1),
        "risk_free_rate": rf,
        "levered_beta": round(lev_beta, 3),
        "unlevered_beta": round(unlev, 3),
        "operating_margin": round(op_margin, 4),
        "effective_tax": round(eff_tax, 4),
        "sales_to_capital": round(s2c, 3),
        "current_roic": round(cur_roic, 4),
        "terminal_roic": round(term_roic, 4),
        "cost_of_debt": round(kd, 4),
        "hist_growth_median": round(g_start, 4),
        "growth_last_yoy": g_diag.get("g_last"),
        "growth_cagr3": g_diag.get("g_cagr3"),
        "debt": round(debt_b, 1),
        "cash_and_non_operating": round(cash_b, 1),
        "correlations": correlations,
    }
    return base, distributions, meta


__all__ = ["build_dcf_inputs"]

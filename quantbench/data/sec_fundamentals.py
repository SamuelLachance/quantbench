"""
quantbench.data.sec_fundamentals
================================
Couche de donnees US fondee sur la SEC (companyfacts API, gratuite, officielle,
~10 req/s -> tout le NASDAQ en ~10 min sans throttling). Produit :
  * un dict de fondamentaux compatible `universal.get_fundamentals` (pour route.py)
  * une structure de comptes compatible `forensics.get_financials` (metriques annuelles)
  * le SECTEUR derive du code SIC (submissions API) -> routage sans yfinance

Prix / beta : recuperes via Yahoo chart (leger, market.py). Tout est en USD
(emetteurs US -> comptes en USD).
"""

from __future__ import annotations

import functools

import requests

from . import edgar, market

_UA = edgar._UA
_B = 1e9


@functools.lru_cache(maxsize=8192)
def submission_meta(cik: str) -> dict:
    r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                     headers=_UA, timeout=30)
    r.raise_for_status()
    j = r.json()
    return {"sic": j.get("sic"), "sicDescription": j.get("sicDescription") or "",
            "name": j.get("name"), "exchanges": j.get("exchanges") or [],
            "tickers": j.get("tickers") or []}


@functools.lru_cache(maxsize=8192)
def annual_report_docs(cik: str) -> dict:
    """Cherche le PDF du rapport annuel glossy (formulaire ARS) et le 10-K sur
    SEC EDGAR. Retourne {ars_pdf, tenk} (URLs directes) — sources officielles
    gratuites. ars_pdf est None si la société ne dépose pas d'ARS (ex. AAPL)."""
    try:
        padded = str(cik).zfill(10)
        j = requests.get(f"https://data.sec.gov/submissions/CIK{padded}.json",
                         headers=_UA, timeout=30).json()
        rec = j["filings"]["recent"]
        forms, docs, accs = rec["form"], rec["primaryDocument"], rec["accessionNumber"]
    except Exception:
        return {"ars_pdf": None, "tenk": None}
    ci = int(cik)
    ars = tenk = None
    for i in range(len(forms)):
        a = accs[i].replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{ci}/{a}/{docs[i]}"
        if forms[i] == "ARS" and docs[i].lower().endswith(".pdf") and ars is None:
            ars = url
        elif forms[i] == "10-K" and docs[i] and tenk is None:
            tenk = url
        if ars and tenk:
            break
    return {"ars_pdf": ars, "tenk": tenk}


def sic_to_sector(sic) -> str:
    """Mappe un code SIC vers un secteur (chaine reconnue par route.classify)."""
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    if 6000 <= s <= 6799:
        return "Financial Services"
    if s in range(1300, 1400) or s in range(2900, 3000) or s == 1311:
        return "Energy"
    if s in range(1000, 1300) or s in range(1400, 1500) or s in range(3300, 3400):
        return "Basic Materials"
    return "Other"


# Tags US-GAAP pour la forensique (au-dela de edgar.TAGS)
_F_TAGS = {
    "revenue": edgar.TAGS["revenue"],
    "cogs": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "ebit": edgar.TAGS["operating_income"],
    "sga": ["SellingGeneralAndAdministrativeExpense",
            "GeneralAndAdministrativeExpense"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liab": ["LiabilitiesCurrent"],
    "net_ppe": ["PropertyPlantAndEquipmentNet"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "inventory": ["InventoryNet"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "equity": edgar.TAGS["equity"],
    "total_liab": ["Liabilities"],
    "shares": edgar.TAGS["shares"] + edgar.TAGS["shares_diluted"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "dep_amort": ["DepreciationDepletionAndAmortization",
                  "DepreciationAmortizationAndAccretionNet",
                  "DepreciationAndAmortization"],
}
_INSTANT = {"total_assets", "current_assets", "current_liab", "net_ppe",
            "receivables", "inventory", "long_term_debt", "retained_earnings",
            "equity", "total_liab", "shares"}


def get_financials(facts) -> dict | None:
    """Structure compatible forensics.get_financials, depuis les faits SEC.

    CRUCIAL : toutes les metriques sont alignees sur les MEMES annees (sinon les
    ratios t vs t-1 comparent des exercices differents). On construit une carte
    {annee: valeur} par metrique, puis des tableaux sur l'intersection des annees
    des metriques CENTRALES (plus recent d'abord)."""
    maps = {}
    for key, tags in _F_TAGS.items():
        kind = "instant" if key in _INSTANT else "duration"
        maps[key] = {d: v for d, v in edgar.annual_series(facts, tags, kind)}

    core = [set(maps[k]) for k in ("revenue", "net_income", "total_assets")
            if maps.get(k)]
    if len(core) < 3:
        return None
    years = sorted(set.intersection(*core), reverse=True)   # plus recent d'abord
    if len(years) < 2:
        return None

    F = {"years": years}
    for key in _F_TAGS:
        F[key] = [maps[key].get(y) for y in years]          # aligne; None si absent
    F["total_debt"] = F["long_term_debt"]
    ca, cl = F["current_assets"], F["current_liab"]
    F["working_capital"] = [
        (ca[i] - cl[i]) if (ca[i] is not None and cl[i] is not None) else None
        for i in range(len(years))]
    return F


def get_fundamentals(ticker: str) -> dict:
    """Dict compatible universal.get_fundamentals, source SEC + prix/beta Yahoo."""
    cik = edgar.get_cik(ticker)
    facts = edgar.get_facts(cik)
    meta = submission_meta(cik)
    sector = sic_to_sector(meta["sic"])

    def latest(key, kind):
        s = edgar.annual_series(facts, _F_TAGS.get(key, []), kind)
        return s[-1][1] if s else None

    revenue = latest("revenue", "duration")
    ebit = latest("ebit", "duration")
    ni = latest("net_income", "duration")
    equity = latest("equity", "instant")
    debt = edgar.total_debt(facts)
    cash = edgar.latest(facts, edgar.TAGS["cash"], "instant", 0.0) or 0.0
    shares = latest("shares", "instant") or edgar.latest(
        facts, edgar.TAGS["shares_diluted"], "duration")
    rev_hist = [v for _, v in edgar.annual_series(facts, _F_TAGS["revenue"])]  # oldest->newest

    price = market.latest_price(ticker)
    try:
        beta = market.levered_beta(ticker)
    except Exception:
        beta = 1.1
    market_cap = (price * shares / _B) if (price and shares) else None

    def b(x):
        return None if x is None else x / _B

    return {
        "ticker": ticker.upper(), "name": meta.get("name") or ticker.upper(),
        "sector": sector, "industry": meta.get("sicDescription"),
        "price": price, "market_cap": market_cap, "shares": shares, "beta": beta,
        "currency_ok": True, "price_currency": "USD", "financial_currency": "USD",
        "revenue": b(revenue), "revenue_history": [x / _B for x in rev_hist],
        "ebit": b(ebit), "net_income": b(ni),
        "total_debt": b(debt), "cash": b(cash), "book_equity": b(equity),
        "operating_margin": (ebit / revenue) if (ebit and revenue) else None,
        "roe": (ni / equity) if (ni and equity) else None,
        "sic": meta.get("sic"),
    }


def us_stock_data(ticker: str) -> dict:
    """Renvoie {fund, financials} pour le pipeline US-SEC."""
    cik = edgar.get_cik(ticker)
    facts = edgar.get_facts(cik)
    return {"fund": get_fundamentals(ticker), "financials": get_financials(facts)}


__all__ = ["get_fundamentals", "get_financials", "us_stock_data",
           "submission_meta", "sic_to_sector"]

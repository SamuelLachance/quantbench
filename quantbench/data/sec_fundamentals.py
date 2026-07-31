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
from ..series import serie_continue

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


_FORM_LABEL = {"10-K": "Rapport annuel (10-K)", "10-K/A": "Rapport annuel (10-K/A)",
               "10-Q": "Rapport trimestriel (10-Q)", "8-K": "Communiqué (8-K)",
               "DEF 14A": "Circulaire de procuration", "ARS": "Rapport annuel (glossy)",
               "20-F": "Rapport annuel (20-F)", "40-F": "Rapport annuel (40-F)",
               "6-K": "Rapport intermédiaire (6-K)"}
_DOC_FORMS = tuple(_FORM_LABEL)


@functools.lru_cache(maxsize=8192)
def annual_report_docs(cik: str) -> dict:
    """Via SEC EDGAR (un seul appel) : PDF du rapport annuel glossy (ARS), 10-K,
    et la liste des DERNIERS documents investisseurs (rapports, communiqués)."""
    try:
        padded = str(cik).zfill(10)
        j = requests.get(f"https://data.sec.gov/submissions/CIK{padded}.json",
                         headers=_UA, timeout=30).json()
        rec = j["filings"]["recent"]
        forms, docs, accs = rec["form"], rec["primaryDocument"], rec["accessionNumber"]
        dates = rec.get("filingDate", [""] * len(forms))
    except Exception:
        return {"ars_pdf": None, "tenk": None, "documents": []}
    ci = int(cik)
    ars = tenk = None
    documents = []
    for i in range(len(forms)):
        form, doc = forms[i], docs[i]
        if not doc:
            continue
        a = accs[i].replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{ci}/{a}/{doc}"
        if form == "ARS" and doc.lower().endswith(".pdf") and ars is None:
            ars = url
        if form == "10-K" and tenk is None:
            tenk = url
        if form in _DOC_FORMS and len(documents) < 8:
            documents.append({"form": form, "label": _FORM_LABEL.get(form, form),
                              "date": dates[i], "url": url,
                              "is_pdf": doc.lower().endswith(".pdf")})
    return {"ars_pdf": ars, "tenk": tenk, "documents": documents}


def sic_to_sector(sic) -> str:
    """Mappe un code SIC vers un secteur (chaine reconnue par route.classify).

    DEUX SECTEURS ETAIENT INATTEIGNABLES, et ce sont precisement les deux qui
    changent de METHODE de valorisation :

      - « Utilities » n'existait pas. Un service public reglemente (SIC 4911,
        4931) tombait en « Other », donc en DCF FCFF standard — alors qu'il doit
        etre valorise par capitalisation des benefices cote equite. Son activite
        est tres capitalistique et financee par dette a dessein : le DCF
        d'entreprise lui sort une equite negative sur une societe parfaitement
        solvable.
      - « Real Estate » n'existait pas non plus. Une fonciere (SIC 6798, 6500-6599)
        tombait dans la tranche 6000-6799 et ressortait « Financial Services »,
        donc valorisee en rendement excedentaire sur ses fonds propres comptables
        — alors qu'une fonciere se capitalise sur son FFO, les amortissements
        immobiliers etant purement comptables.

    Les deux perdaient au passage leur exemption d'Altman (`_NO_ALTMAN` dans
    route.py), qui existe justement parce que ce score n'a pas de sens sur un
    bilan de service public ou de fonciere.

    L'ORDRE DES TESTS COMPTE : l'immobilier doit etre reconnu AVANT la tranche
    financiere, qui l'englobe.
    """
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    # Services publics : electricite, gaz, eau, assainissement (4900-4999).
    if 4900 <= s <= 4999:
        return "Utilities"
    # Immobilier, AVANT la tranche financiere qui le contient : 6500-6599
    # (exploitants et agents) et 6798 (fonds de placement immobilier).
    if 6500 <= s <= 6599 or s == 6798:
        return "Real Estate"
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
    # SERIE CONTINUE : `annual_series` ne rend que les exercices DEPOSES. Un
    # exercice absent des donnees EDGAR disparaissait de la liste sans laisser de
    # trace, et deux elements voisins pouvaient couvrir deux ans.
    _annees, rev_hist = serie_continue(edgar.annual_series(facts, _F_TAGS["revenue"]))

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
        "revenue": b(revenue), "revenue_history": [None if x is None else x / _B for x in rev_hist],
        "ebit": b(ebit), "net_income": b(ni),
        "total_debt": b(debt), "cash": b(cash), "book_equity": b(equity),
        "operating_margin": (ebit / revenue) if (ebit is not None and revenue) else None,
        "roe": (ni / equity) if (ni is not None and equity) else None,
        "sic": meta.get("sic"),
    }


def us_stock_data(ticker: str) -> dict:
    """Renvoie {fund, financials} pour le pipeline US-SEC."""
    cik = edgar.get_cik(ticker)
    facts = edgar.get_facts(cik)
    return {"fund": get_fundamentals(ticker), "financials": get_financials(facts)}


__all__ = ["get_fundamentals", "get_financials", "us_stock_data",
           "submission_meta", "sic_to_sector"]

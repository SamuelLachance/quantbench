"""
quantbench.data.edgar
=====================
Fondamentaux via l'API XBRL de la SEC (data.sec.gov) : gratuit, officiel, sans
cle. On mappe un ticker -> CIK -> `companyfacts`, puis on extrait des series
ANNUELLES robustes (10-K, duree ~365 j pour les flux, dedup par date de fin en
gardant le depot le plus recent).

La SEC demande un User-Agent identifiant + limite a ~10 req/s.
"""

from __future__ import annotations

import functools
import os
from datetime import date

import requests

# La SEC demande un User-Agent identifiant avec un contact. Configurable via la
# variable d'environnement QUANTBENCH_SEC_UA (mettez votre email) ; defaut generique.
_UA = {"User-Agent": os.environ.get(
    "QUANTBENCH_SEC_UA", "QuantBench research tool (contact via GitHub issues)")}
_BASE = "https://data.sec.gov"
_TIMEOUT = 30


@functools.lru_cache(maxsize=1)
def _ticker_map() -> dict:
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=_UA, timeout=_TIMEOUT)
    r.raise_for_status()
    return {row["ticker"].upper(): str(row["cik_str"]).zfill(10)
            for row in r.json().values()}


def get_cik(ticker: str) -> str:
    m = _ticker_map()
    key = ticker.upper()
    if key not in m:
        raise KeyError(f"Ticker '{ticker}' introuvable dans l'index SEC "
                       f"(entreprise non americaine ou non cotee ?).")
    return m[key]


@functools.lru_cache(maxsize=64)
def get_facts(cik: str) -> dict:
    r = requests.get(f"{_BASE}/api/xbrl/companyfacts/CIK{cik}.json",
                     headers=_UA, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _units_for(facts: dict, tag: str):
    for ns in ("us-gaap", "dei", "ifrs-full"):
        node = facts["facts"].get(ns, {}).get(tag)
        if node:
            return node["units"]
    return None


def annual_series(facts: dict, tags, kind: str = "duration") -> list:
    """Serie annuelle [(date_fin, valeur)], la plus recente en dernier.

    tags : liste de tags candidats FUSIONNES par date de fin. Indispensable car
           les entreprises changent de tag au fil du temps (ex. NVDA passe de
           RevenueFromContractWithCustomer... a Revenues) : prendre le premier
           tag non vide donnerait des donnees perimees.
    kind : 'duration' (flux : CA, resultat...) filtre les durees ~365 j ;
           'instant' (bilan : dette, cash...) prend les points de fin d'exercice.
    En cas de meme date rapportee par plusieurs tags, on garde le depot le plus
    recent.
    """
    best = {}                                      # date_fin -> (date_depot, valeur)
    for tag in _as_list(tags):
        units = _units_for(facts, tag)
        if not units:
            continue
        rows = units[next(iter(units))]            # unite principale (USD, shares)
        for r in rows:
            if r.get("form") != "10-K":
                continue
            end = r.get("end")
            if not end:
                continue
            if kind == "duration":
                start = r.get("start")
                if not start:
                    continue
                span = (date.fromisoformat(end) - date.fromisoformat(start)).days
                if not (350 <= span <= 380):
                    continue
            filed = r.get("filed", "")
            if end not in best or filed > best[end][0]:
                best[end] = (filed, r["val"])
    return [(e, v) for e, (f, v) in sorted(best.items())]


def latest(facts: dict, tags, kind: str = "duration", default=None):
    s = annual_series(facts, tags, kind)
    return s[-1][1] if s else default


def _as_list(x):
    return [x] if isinstance(x, str) else list(x)


# --- Tags canoniques (avec fallbacks par ordre de preference) ---
TAGS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet"],
    "operating_income": ["OperatingIncomeLoss"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "interest_expense": ["InterestExpense", "InterestExpenseDebt",
                         "InterestIncomeExpenseNet"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "marketable_current": ["MarketableSecuritiesCurrent"],
    "marketable_noncurrent": ["MarketableSecuritiesNoncurrent"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "long_term_debt": ["LongTermDebt"],
    "lt_debt_noncurrent": ["LongTermDebtNoncurrent"],
    "lt_debt_current": ["LongTermDebtCurrent"],
    "short_term_borrowings": ["ShortTermBorrowings", "DebtCurrent"],
    "shares": ["EntityCommonStockSharesOutstanding"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding",
                       "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
                       "WeightedAverageNumberOfSharesOutstandingBasic"],
}


def total_debt(facts: dict) -> float:
    """Dette totale : prend LongTermDebt s'il existe, sinon somme les composantes."""
    ltd = latest(facts, TAGS["long_term_debt"], "instant")
    if ltd is not None:
        return float(ltd)
    parts = [latest(facts, TAGS[k], "instant", 0.0) or 0.0
             for k in ("lt_debt_noncurrent", "lt_debt_current", "short_term_borrowings")]
    return float(sum(parts))


__all__ = ["get_cik", "get_facts", "annual_series", "latest", "total_debt", "TAGS"]

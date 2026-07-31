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

# La SEC exige un User-Agent identifiant avec un contact, et REFUSE (403) ceux
# qu'elle juge insuffisants.
#
# Deux pieges ici, verifies en direct contre la SEC :
#   1. Un secret GitHub NON DEFINI devient la CHAINE VIDE, pas une variable
#      absente. `os.environ.get(cle, defaut)` ne rend alors JAMAIS le defaut : le
#      User-Agent part vide, et la SEC repond 403. D'ou le `or`.
#   2. La chaine de defaut d'origine — « QuantBench research tool (contact via
#      GitHub issues) » — est elle-meme refusee en 403, alors que la meme sans la
#      parenthese passe. Un repli qui echoue n'est pas un repli.
# En aval, `annual_report_docs` avale toute exception : sans ces deux corrections,
# les seize mille fiches perdaient leurs documents SEC sans qu'un seul job echoue.
_UA = {"User-Agent": (os.environ.get("QUANTBENCH_SEC_UA")
                      or "QuantBench research tool")}
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


def total_debt(facts: dict):
    """Dette totale, ou None quand AUCUN poste de dette n'est balise.

    L'ABSENCE N'EST PAS UN ZERO — c'est l'image en miroir du piege du zero que ce
    depot a deja rencontre cinq fois, et elle est plus dangereuse encore. La
    fonction rendait 0.0 quand aucun des trois tags de repli n'existait, valeur
    strictement indiscernable d'une societe reellement sans dette. `latest(...,
    0.0) or 0.0` ecrasait l'ignorance en certitude.

    Ce que cela coutait : la valeur d'entreprise devient egale a la valeur des
    fonds propres, et l'upside se trouve gonfle de la TOTALITE de l'endettement non
    balise, sans qu'aucun controle ne bronche — `validate.py` ne teste que la borne
    haute (dette superieure a l'actif), jamais l'absence. Les emetteurs touches
    sont ceux qui balisent leur dette sous des libelles hors des trois tags du
    repli : financieres, foncieres, petites capitalisations.

    La fonction savait pourtant deja faire la difference dix lignes plus haut, ou
    le poste global passe par `default=None` puis un test `is not None`.

    Renvoie None si rien n'est balise, 0.0 si au moins un poste existe et vaut zero.
    """
    ltd = latest(facts, TAGS["long_term_debt"], "instant")
    if ltd is not None:
        return float(ltd)
    parts = [latest(facts, TAGS[k], "instant", None)
             for k in ("lt_debt_noncurrent", "lt_debt_current", "short_term_borrowings")]
    connus = [p for p in parts if p is not None]
    if not connus:
        return None                      # rien de balise : on ne sait pas, on le dit
    return float(sum(connus))


__all__ = ["get_cik", "get_facts", "annual_series", "latest", "total_debt", "TAGS"]

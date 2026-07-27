"""
quantbench.data.fmp
===================
Couche de données UNIQUE via Financial Modeling Prep (API « stable », palier
Ultimate) : univers (US + Canada), fondamentaux (bulk), profils, historique de
prix, news. Fiable, non-Yahoo, couverture complète. Tout est ramené en USD.

Clé via la variable d'environnement FMP_API_KEY (jamais en dur dans le dépôt).
"""

from __future__ import annotations

import csv
import functools
import io
import os
import re

import requests

# Actions privilégiées / notes / débentures émises par banques & sociétés (quasi-
# obligataire, pas des actions ordinaires) : à exclure de l'univers. Le modèle
# excess-return les valorise comme des capitaux propres -> upside aberrant.
# Prudence : NE PAS attraper les ADR ("American Depositary Shares" = ordinaire) ni
# les sociétés dont le nom contient "Preferred" (ex. Preferred Bank).
_PREF_NAME = re.compile(
    r"\bPFD\b|\bPREF(ERRED)?\s+(STOCK|SHARES|SHS|SEC|SERIES|EQUITY)"
    r"|PERPETUAL\s+(RED\s+)?PREF|CUM\s+PERP|\bRATE\s+RESET\b|\bRST\s+PFD\b"
    r"|\b(SUB(ORDINATED)?|JR|JUNIOR|SR|SENIOR)\s+(NOTES?|DEBENT)"
    r"|\bDEBENTURES?\b|\bTRUST\s+PREF|\d+(\.\d+)?\s*%\s", re.I)
_PREF_TICKER = re.compile(r"-P[FR]?[A-Z]\.(TO|V)$", re.I)


def _is_preferred(symbol, name):
    return bool(_PREF_TICKER.search(symbol) or (name and _PREF_NAME.search(name)))


def _sane_beta(b):
    """Beta FMP fiable ? Les nano-caps illiquides donnent des betas aberrants
    (ex. −29) → coût du capital négatif → DCF qui explose. Hors [0.1, 3.5] =
    régression bidon → on retombe sur le beta de marché (1.1)."""
    b = _num(b)
    return b if (b is not None and 0.1 <= b <= 3.5) else 1.1

_BASE = "https://financialmodelingprep.com/stable"
_TIMEOUT = 120


def _key():
    k = os.environ.get("FMP_API_KEY")
    if not k:
        raise RuntimeError("FMP_API_KEY manquante (variable d'environnement).")
    return k


def _get(path, retries=3):
    import time
    url = f"{_BASE}/{path}" + ("&" if "?" in path else "?") + f"apikey={_key()}"
    for attempt in range(retries):
        r = requests.get(url, timeout=_TIMEOUT)
        if r.status_code == 429:                    # rate limit -> backoff
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def statements(symbol, limit=6):
    """3 états annuels par ticker (endpoints à débit élevé). Retourne
    {income:{year:row}, balance:{year:row}, cashflow:{year:row}}."""
    out = {"income": {}, "balance": {}, "cashflow": {}}
    for ep, kind in (("income-statement", "income"),
                     ("balance-sheet-statement", "balance"),
                     ("cash-flow-statement", "cashflow")):
        try:
            for row in _json(f"{ep}?symbol={symbol}&period=annual&limit={limit}"):
                y = row.get("fiscalYear") or str(row.get("date", ""))[:4]
                try:
                    out[kind][int(y)] = row
                except (TypeError, ValueError):
                    continue
        except Exception:
            continue
    return out


def profile(symbol):
    try:
        j = _json(f"profile?symbol={symbol}")
        return j[0] if j else {}
    except Exception:
        return {}


def _json(path):
    return _get(path).json()


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Change → USD (via FMP forex)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=32)
def fx_to_usd(currency: str):
    if not currency or currency.upper() == "USD":
        return 1.0
    c = currency.upper()
    for sym, inv in ((f"{c}USD", False), (f"USD{c}", True)):
        try:
            j = _json(f"quote?symbol={sym}")
            p = _num(j[0].get("price")) if j else None
            if p and p > 0:
                return (1.0 / p) if inv else p
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
# Univers (screener)
# --------------------------------------------------------------------------- #
def screener(exchanges=("NASDAQ",)):
    """Sociétés tangibles (hors ETF/fonds) des bourses données. Retourne
    {symbol: {sector, industry, beta, price, market_cap, currency, exchange, name}}."""
    out = {}
    for ex in exchanges:
        try:
            rows = _json(f"company-screener?exchange={ex}&isEtf=false&isFund=false"
                         f"&isActivelyTrading=true&limit=15000")
        except Exception:
            continue
        for r in rows:
            sym = r.get("symbol")
            if not sym:
                continue
            if _is_preferred(sym, r.get("companyName")):   # actions privilégiées / notes : exclues
                continue
            out[sym] = {"name": r.get("companyName"), "sector": r.get("sector"),
                        "industry": r.get("industry"), "beta": _num(r.get("beta")),
                        "price": _num(r.get("price")),
                        "market_cap": _num(r.get("marketCap")),
                        "currency": None, "exchange": ex}
    return out


# --------------------------------------------------------------------------- #
# Fondamentaux en bulk (toutes sociétés, 1 CSV par année/état)
# --------------------------------------------------------------------------- #
def _bulk_csv(endpoint, year, keep):
    r = _get(f"{endpoint}?year={year}&period=annual")
    reader = csv.DictReader(io.StringIO(r.text))
    rows = {}
    for row in reader:
        s = row.get("symbol")
        if s and (keep is None or s in keep):
            rows[s] = row
    return rows


def bulk_statements(symbols, years):
    """{symbol: {income:{y:row}, balance:{y:row}, cashflow:{y:row}}} pour l'univers."""
    keep = set(symbols)
    data = {}
    eps = [("income-statement-bulk", "income"),
           ("balance-sheet-statement-bulk", "balance"),
           ("cash-flow-statement-bulk", "cashflow")]
    for y in years:
        for endpoint, kind in eps:
            try:
                rows = _bulk_csv(endpoint, y, keep)
            except Exception:
                continue
            for s, row in rows.items():
                e = data.setdefault(s, {"income": {}, "balance": {}, "cashflow": {}})
                e[kind][y] = row
    return data


# --------------------------------------------------------------------------- #
# Profils (descriptions) en bulk
# --------------------------------------------------------------------------- #
def profile_descriptions(symbols, max_parts=6):
    keep = set(symbols)
    out = {}
    for part in range(max_parts):
        try:
            r = _get(f"profile-bulk?part={part}")
        except Exception:
            break
        reader = csv.DictReader(io.StringIO(r.text))
        any_row = False
        for row in reader:
            any_row = True
            s = row.get("symbol")
            if s in keep and s not in out:
                out[s] = {"description": (row.get("description") or "")[:650],
                          "industry": row.get("industry")}
        if not any_row:
            break
    return out


# --------------------------------------------------------------------------- #
# Historique de prix (signal court terme) + news (par ticker)
# --------------------------------------------------------------------------- #
def history_closes(symbol, days=400):
    try:
        j = _json(f"historical-price-eod/light?symbol={symbol}")
        closes = [_num(d.get("price")) for d in j][::-1]      # ancien -> récent
        return [c for c in closes if c][-days:]
    except Exception:
        return []


def news(symbol, limit=8):
    try:
        j = _json(f"news/stock?symbols={symbol}&limit={limit}")
        return [{"title": n.get("title"), "publisher": n.get("publisher"),
                 "url": n.get("url"), "pubDate": n.get("publishedDate")} for n in j]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Conversion vers nos structures (forensics F + fundamentals) — tout en USD
# --------------------------------------------------------------------------- #
def financials_from_fmp(entry):
    """Structure compatible forensics.get_financials (ratios -> devise neutre,
    on garde la devise d'origine, cohérente par société)."""
    inc, bal, cf = entry.get("income", {}), entry.get("balance", {}), entry.get("cashflow", {})
    yrs = sorted(set(inc) & set(bal), reverse=True)
    if len(yrs) < 2:
        return None

    def col(src, field):
        return [_num(src.get(y, {}).get(field)) for y in yrs]

    F = {"years": [str(y) for y in yrs],
         "revenue": col(inc, "revenue"), "cogs": col(inc, "costOfRevenue"),
         "gross_profit": col(inc, "grossProfit"), "ebit": col(inc, "operatingIncome"),
         "sga": col(inc, "sellingGeneralAndAdministrativeExpenses"),
         "net_income": col(inc, "netIncome"), "shares": col(inc, "weightedAverageShsOutDil"),
         "total_assets": col(bal, "totalAssets"), "current_assets": col(bal, "totalCurrentAssets"),
         "current_liab": col(bal, "totalCurrentLiabilities"),
         "net_ppe": col(bal, "propertyPlantEquipmentNet"),
         "receivables": col(bal, "netReceivables"), "inventory": col(bal, "inventory"),
         "total_debt": col(bal, "totalDebt"), "long_term_debt": col(bal, "longTermDebt"),
         "retained_earnings": col(bal, "retainedEarnings"),
         "equity": col(bal, "totalStockholdersEquity"), "total_liab": col(bal, "totalLiabilities"),
         "cfo": col(cf, "operatingCashFlow"), "dep_amort": col(cf, "depreciationAndAmortization")}
    ca, cl = F["current_assets"], F["current_liab"]
    F["working_capital"] = [(ca[i] - cl[i]) if (ca[i] is not None and cl[i] is not None)
                            else None for i in range(len(yrs))]
    return F


_EX_CUR = {"NASDAQ": "USD", "NYSE": "USD", "AMEX": "USD", "TSX": "CAD", "TSXV": "CAD"}


def fundamentals_from_fmp(symbol, sr, entry, desc):
    """Dict compatible universal.get_fundamentals — montants convertis en USD."""
    inc, bal = entry.get("income", {}), entry.get("balance", {})
    yrs = sorted(set(inc) & set(bal), reverse=True)
    if not yrs:
        return None
    y = yrs[0]
    rep_cur = inc.get(y, {}).get("reportedCurrency") or "USD"
    fxs = fx_to_usd(rep_cur) or 1.0                       # états -> USD
    price_cur = _EX_CUR.get((sr.get("exchange") or "").upper(), "USD")
    fxp = fx_to_usd(price_cur) or 1.0                     # cours/capi -> USD
    B = 1e9
    g = lambda src, f: _num(src.get(y, {}).get(f))
    rev, ebit, ni = g(inc, "revenue"), g(inc, "operatingIncome"), g(inc, "netIncome")
    eq, debt = g(bal, "totalStockholdersEquity"), g(bal, "totalDebt")
    cash = g(bal, "cashAndCashEquivalents")
    tot_assets = g(bal, "totalAssets")
    # Garde-fou données FMP corrompues (ex. RDZN cash=6.6e12 pour 55 M$ de CA) :
    # trésorerie et dette ne peuvent PAS dépasser l'actif total -> on plafonne.
    if tot_assets and tot_assets > 0:
        if cash is not None:
            cash = min(cash, tot_assets)
        if debt is not None:
            debt = min(max(debt, 0.0), tot_assets)
    shares = g(inc, "weightedAverageShsOutDil")
    rev_hist = [_num(inc.get(yy, {}).get("revenue")) for yy in sorted(inc)]
    rev_hist = [v * fxs / B for v in rev_hist if v is not None]
    price, mcap = _num(sr.get("price")), _num(sr.get("market_cap"))
    b = lambda v: None if v is None else v * fxs / B
    return {
        "ticker": symbol, "name": sr.get("name") or symbol, "sector": sr.get("sector"),
        "industry": (desc or {}).get("industry") or sr.get("industry"),
        "summary": (desc or {}).get("description"),
        "price": price * fxp if price else None,
        "market_cap": mcap * fxp / B if mcap else None,
        "shares": shares, "beta": _sane_beta(sr.get("beta")),
        "currency_ok": True, "price_currency": price_cur, "financial_currency": rep_cur,
        "revenue": b(rev), "revenue_history": rev_hist, "ebit": b(ebit), "net_income": b(ni),
        "total_debt": b(debt), "cash": b(cash), "book_equity": b(eq),
        "operating_margin": (ebit / rev) if (ebit and rev) else None,
        "roe": (ni / eq) if (ni and eq) else None,
        "cik": inc.get(y, {}).get("cik"), "sic": None,
    }


__all__ = ["screener", "bulk_statements", "profile_descriptions", "history_closes",
           "news", "fx_to_usd", "financials_from_fmp", "fundamentals_from_fmp"]


"""
quantbench.data.universal
=========================
Source de fondamentaux UNIVERSELLE et gratuite via yfinance : couvre les marches
US (NASDAQ) ET canadiens (.TO / .V), d'un meme pipeline. Fournit aussi le
`sector`/`industry`, indispensable pour ROUTER chaque titre vers la bonne
methode de valorisation (DCF FCFF, FCFE financieres, cyclique normalise...).

Note de robustesse : yfinance est une source non-officielle (scraping Yahoo),
sujette a du rate-limiting sur de gros volumes. A cadencer/cacher pour un batch
quotidien de milliers de tickers.
"""

from __future__ import annotations

import warnings

from . import market

warnings.filterwarnings("ignore")

_B = 1e9


def _safe_ratio(num, den):
    if num is None or den is None or den == 0:
        return None
    return num / den


def _first(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None and v == v:            # non-None et non-NaN
            return v
    return None


def _row(df, *labels):
    """Valeur annuelle la plus recente d'une ligne d'un etat financier yfinance."""
    if df is None:
        return None
    for lab in labels:
        if lab in df.index:
            s = df.loc[lab].dropna()
            if len(s):
                return float(s.iloc[0])
    return None


def _series(df, *labels):
    """Serie annuelle (ancienne -> recente) d'une ligne d'etat financier."""
    if df is None:
        return []
    for lab in labels:
        if lab in df.index:
            s = df.loc[lab].dropna()
            if len(s):
                return [float(x) for x in s.values][::-1]
    return []


def get_fundamentals(ticker: str) -> dict:
    """Retourne un dict de fondamentaux normalises (monnaie native, en milliards
    pour les montants). Champs a None si indisponibles."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    info = tk.info or {}
    try:
        inc = tk.income_stmt
    except Exception:
        inc = None
    try:
        bal = tk.balance_sheet
    except Exception:
        bal = None

    # --- Devises : cours vs etats financiers peuvent DIFFERER (piege Yahoo) ---
    price_cur = info.get("currency")
    fin_cur = info.get("financialCurrency") or price_cur
    fx_price = market.fx_to_usd(price_cur)       # 1 unite de la devise du cours -> USD
    fx_fin = market.fx_to_usd(fin_cur)           # 1 unite de la devise des comptes -> USD
    currency_ok = (fx_price is not None) and (fx_fin is not None)

    def usd_price(x):                            # montants en devise du COURS -> Md USD
        if x is None or fx_price is None:
            return None
        return x * fx_price / _B

    def usd_fin(x):                              # montants en devise des COMPTES -> Md USD
        if x is None or fx_fin is None:
            return None
        return x * fx_fin / _B

    price = _first(info, "currentPrice", "regularMarketPrice", "previousClose")
    ebit = _row(inc, "Operating Income", "EBIT", "Total Operating Income As Reported")
    book_equity = _row(bal, "Stockholders Equity", "Common Stock Equity",
                       "Total Equity Gross Minority Interest")
    pretax = _row(inc, "Pretax Income", "Income Before Tax")
    tax = _row(inc, "Tax Provision", "Income Tax Expense")

    # Tout converti en USD (deux groupes de devises distincts) :
    price_usd = (price * fx_price) if (price is not None and fx_price) else None
    market_cap = usd_price(_first(info, "marketCap"))
    revenue = usd_fin(_first(info, "totalRevenue") or _row(inc, "Total Revenue"))
    ebit_usd = usd_fin(ebit)
    ebitda = usd_fin(_first(info, "ebitda"))
    net_income = usd_fin(_first(info, "netIncomeToCommon") or _row(inc, "Net Income"))
    total_debt = usd_fin(_first(info, "totalDebt") or _row(bal, "Total Debt"))
    cash = usd_fin(_first(info, "totalCash") or _row(bal, "Cash And Cash Equivalents"))
    equity_usd = usd_fin(book_equity)
    ev = None if market_cap is None else market_cap + (total_debt or 0) - (cash or 0)

    return {
        "ticker": ticker.upper(),
        "name": _first(info, "shortName", "longName") or ticker.upper(),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        # devises (transparence) — tout le reste est deja converti en USD
        "price_currency": price_cur,
        "financial_currency": fin_cur,
        "fx_price_to_usd": fx_price,
        "fx_financial_to_usd": fx_fin,
        "currency_ok": currency_ok,
        # cours / capitalisation (USD)
        "price": price_usd,
        "market_cap": market_cap,
        "shares": _first(info, "sharesOutstanding", "impliedSharesOutstanding"),
        "beta": _first(info, "beta"),
        # compte de resultat (Md USD)
        "revenue": revenue,
        "revenue_history": [usd_fin(x) for x in _series(inc, "Total Revenue", "Operating Revenue")],
        "ebit": ebit_usd,
        "ebitda": ebitda,
        "net_income": net_income,
        "pretax_income": usd_fin(pretax),
        "tax": usd_fin(tax),
        # bilan (Md USD)
        "total_debt": total_debt,
        "cash": cash,
        "book_equity": equity_usd,
        "enterprise_value": ev,
        # ratios RECALCULES en USD (ne pas se fier aux ratios yfinance : melange de devises)
        "operating_margin": _safe_ratio(ebit_usd, revenue),
        "net_margin": _safe_ratio(net_income, revenue),
        "roe": _safe_ratio(net_income, equity_usd),
        "trailing_pe": _safe_ratio(market_cap, net_income),
        "price_to_book": _safe_ratio(market_cap, equity_usd),
        "ev_to_ebitda": _safe_ratio(ev, ebitda),
        "dividend_yield": _first(info, "dividendYield"),
        "payout_ratio": _first(info, "payoutRatio"),
    }


__all__ = ["get_fundamentals"]

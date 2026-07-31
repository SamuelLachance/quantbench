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


def _sinon(*valeurs):
    """Premiere valeur RENSEIGNEE, zero compris. Remplace `a or b`.

    CINQUIEME OCCURRENCE DU PIEGE DU ZERO. `_first` sait deja distinguer 0 de None,
    mais son resultat etait aussitot passe a `or`, qui, lui, ne le sait pas : en
    Python, `0 or x` vaut x. Un resultat net exactement NUL, une dette NULLE, un
    chiffre d'affaires NUL — soit le cas normal d'une biotechnologie avant son
    premier produit, ou d'une societe a l'equilibre parfait — declenchaient donc le
    repli sur les etats financiers, puis valaient None si ceux-ci manquaient.
    Un poste comptable a zero est une DONNEE, pas une absence de donnee.
    """
    for v in valeurs:
        if v is not None and v == v:            # non-None et non-NaN
            return v
    return None


def _first(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None and v == v:            # non-None et non-NaN
            return v
    return None


def exercice_commun(df, *lignes):
    """Colonne la plus recente ou TOUTES les lignes citees sont renseignees.

    Sans elle, `_row` rendait la valeur la plus recente DE CHAQUE LIGNE prise
    isolement. Deux lignes dont les trous ne tombent pas aux memes exercices
    rendaient donc des grandeurs d'ANNEES DIFFERENTES, que l'appelant mettait
    ensuite en rapport : une marge d'exploitation formee du resultat de 2025 et du
    chiffre d'affaires de 2024, un rendement des fonds propres formee du benefice
    d'une annee et du capital d'une autre. Le rapport n'a alors aucun sens, et rien
    ne le signale.

    Les colonnes yfinance sont des dates de cloture, PLUS RECENTE EN TETE.
    Rend `None` si aucune colonne ne porte toutes les lignes.
    """
    if df is None:
        return None
    presentes = [l for l in lignes if l in df.index]
    if not presentes:
        return None
    for col in df.columns:
        if all(df.loc[l, col] == df.loc[l, col] for l in presentes):   # aucun NaN
            return col
    return None


def _row(df, *labels, col=None):
    """Valeur d'une ligne d'un etat financier yfinance.

    Avec `col`, lit CET exercice et rend `None` si la ligne n'y est pas renseignee :
    on ne va pas chercher ailleurs une valeur qui manque ici, sans quoi le rapport
    forme en aval melangerait deux annees. Sans `col`, comportement d'origine — la
    valeur la plus recente de la ligne — reserve aux grandeurs qui ne servent a
    aucun rapport.
    """
    if df is None:
        return None
    for lab in labels:
        if lab in df.index:
            if col is not None:
                v = df.loc[lab, col]
                return float(v) if v == v else None
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
    # UN SEUL EXERCICE DE REFERENCE PAR ETAT, pour que les rapports formes en aval
    # — marge d'exploitation, taux d'impot, rendement des fonds propres — portent
    # sur la meme annee des deux cotes.
    ex_inc = exercice_commun(inc, "Total Revenue", "Operating Income")
    ex_bal = exercice_commun(bal, "Stockholders Equity")
    ebit = _row(inc, "Operating Income", "EBIT", "Total Operating Income As Reported",
                col=ex_inc)
    book_equity = _row(bal, "Stockholders Equity", "Common Stock Equity",
                       "Total Equity Gross Minority Interest", col=ex_bal)
    pretax = _row(inc, "Pretax Income", "Income Before Tax", col=ex_inc)
    tax = _row(inc, "Tax Provision", "Income Tax Expense", col=ex_inc)

    # Tout converti en USD (deux groupes de devises distincts) :
    price_usd = (price * fx_price) if (price is not None and fx_price) else None
    market_cap = usd_price(_first(info, "marketCap"))
    revenue = usd_fin(_sinon(_first(info, "totalRevenue"), _row(inc, "Total Revenue", col=ex_inc)))
    ebit_usd = usd_fin(ebit)
    ebitda = usd_fin(_first(info, "ebitda"))
    net_income = usd_fin(_sinon(_first(info, "netIncomeToCommon"),
                            _row(inc, "Net Income", col=ex_inc)))
    total_debt = usd_fin(_sinon(_first(info, "totalDebt"), _row(bal, "Total Debt", col=ex_bal)))
    # PERIMETRE CONSTANT, QUELLE QUE SOIT LA SOURCE QUI REPOND. `totalCash` de
    # Yahoo agrege la tresorerie ET les placements a court terme ; la ligne de
    # bilan « Cash And Cash Equivalents » ne porte que la premiere. Se replier de
    # l'une sur l'autre faisait donc changer la DEFINITION du poste selon la source
    # disponible, sans que rien ne le signale — et cette tresorerie entre
    # directement dans la valeur d'entreprise, donc dans tout multiple qui en
    # decoule. On somme explicitement les deux composantes du bilan pour retrouver
    # le perimetre de Yahoo, plutot que de choisir la plus etroite par accident.
    tresorerie_bilan = _row(bal, "Cash And Cash Equivalents", col=ex_bal)
    placements_court_terme = _row(bal, "Other Short Term Investments",
                                  "Short Term Investments", col=ex_bal)
    if tresorerie_bilan is not None and placements_court_terme is not None:
        tresorerie_bilan += placements_court_terme
    cash = usd_fin(_sinon(_first(info, "totalCash"), tresorerie_bilan))
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
        # RECOPIES BRUTS DE YAHOO, ET SANS CONSOMMATEUR. Aucune ligne du depot ne
        # les lit — ni le routage, ni la notation, ni une page. Ils restent parce
        # qu'ils ne coutent rien, mais leur unite n'est PAS etablie : Yahoo a livre
        # `dividendYield` tantot en fraction (0,023) tantot en pourcentage (2,3)
        # selon les versions de son API, et rien ici ne tranche. Quiconque les
        # branchera un jour doit donc verifier l'unite AVANT, et non decouvrir un
        # rendement de 230 % sur une fiche.
        "dividend_yield": _first(info, "dividendYield"),
        "payout_ratio": _first(info, "payoutRatio"),
    }


__all__ = ["get_fundamentals"]

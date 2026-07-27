"""
quantbench.forensics.scores
============================
Analyse forensique des etats financiers : detection de signaux d'INCOHERENCE /
manipulation de resultat et de POINTS POSITIFS discrets, avec suivi annee sur
annee. Fonde sur des modeles academiques etablis, calcules sur 4 ans de comptes
(yfinance). Tous les indicateurs sont des RATIOS etats/etats -> la devise
s'annule (aucune conversion FX necessaire).

Modeles :
  * Beneish M-Score (8 variables)  -> probabilite de manipulation de resultat
  * Altman Z''-Score               -> risque de detresse (variante tous secteurs,
                                       capitaux propres COMPTABLES -> pas de FX)
  * Piotroski F-Score (0-9)        -> qualite/amelioration des fondamentaux (les
                                       "points positifs" qui passent inapercus)
  * Accruals de Sloan (NI-CFO)/TA  -> qualite des resultats
  * Divergence Resultat net vs Cash-flow d'exploitation (annee sur annee)

CADRE ETHIQUE : ce sont des SIGNAUX STATISTIQUES A INVESTIGUER, jamais une preuve
de fraude. Les libelles restent neutres. Ne jamais affirmer "fraude" sur une
societe nommee.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

_INCOME = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "cogs": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "gross_profit": ["Gross Profit"],
    "ebit": ["Operating Income", "EBIT", "Total Operating Income As Reported"],
    "sga": ["Selling General And Administration"],
    "net_income": ["Net Income", "Net Income Common Stockholders",
                   "Net Income Continuous Operations"],
}
_BALANCE = {
    "total_assets": ["Total Assets"],
    "current_assets": ["Current Assets"],
    "current_liab": ["Current Liabilities"],
    "net_ppe": ["Net PPE"],
    "receivables": ["Receivables", "Accounts Receivable"],
    "inventory": ["Inventory"],
    "total_debt": ["Total Debt"],
    "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "retained_earnings": ["Retained Earnings"],
    "working_capital": ["Working Capital"],
    "equity": ["Stockholders Equity", "Common Stock Equity"],
    "total_liab": ["Total Liabilities Net Minority Interest"],
    "shares": ["Ordinary Shares Number", "Share Issued"],
}
_CASHFLOW = {
    "cfo": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "dep_amort": ["Depreciation And Amortization", "Depreciation Amortization Depletion"],
    "capex": ["Capital Expenditure"],
}


def get_financials(ticker: str):
    """Retourne un dict {metrique: [valeurs, plus recent -> plus ancien]} aligne
    sur les annees communes aux 3 etats. None si donnees insuffisantes."""
    import yfinance as yf
    tk = yf.Ticker(ticker)
    inc, bal, cf = tk.income_stmt, tk.balance_sheet, tk.cashflow
    if inc is None or bal is None or inc.empty or bal.empty:
        return None
    cols = [c for c in inc.columns if c in bal.columns]
    if cf is not None and not cf.empty:
        cols = [c for c in cols if c in cf.columns]
    cols = sorted(cols, reverse=True)              # plus recent d'abord
    if len(cols) < 2:
        return None

    def ser(df, labels):
        if df is None:
            return [None] * len(cols)
        for lab in labels:
            if lab in df.index:
                row = df.loc[lab]
                return [float(row[c]) if (c in row.index and row[c] == row[c]) else None
                        for c in cols]
        return [None] * len(cols)

    F = {"ticker": ticker.upper(), "years": [str(c)[:10] for c in cols]}
    for name, labs in _INCOME.items():
        F[name] = ser(inc, labs)
    for name, labs in _BALANCE.items():
        F[name] = ser(bal, labs)
    for name, labs in _CASHFLOW.items():
        F[name] = ser(cf, labs)
    return F


def _r(a, b):
    """Division sure : None si numerateur/denominateur manquant ou denom nul."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def beneish_m_score(F) -> float | None:
    """M-Score de Beneish (8 variables), annee t vs t-1. > -1.78 -> zone
    'manipulateur probable' (a investiguer)."""
    g = lambda k, i: F[k][i]
    try:
        rev0, rev1 = g("revenue", 0), g("revenue", 1)
        dsri = _r(_r(g("receivables", 0), rev0), _r(g("receivables", 1), rev1))
        gm1 = _r(rev1 - g("cogs", 1), rev1)
        gm0 = _r(rev0 - g("cogs", 0), rev0)
        gmi = _r(gm1, gm0)
        aqi0 = 1 - _r(g("current_assets", 0) + g("net_ppe", 0), g("total_assets", 0))
        aqi1 = 1 - _r(g("current_assets", 1) + g("net_ppe", 1), g("total_assets", 1))
        aqi = _r(aqi0, aqi1)
        sgi = _r(rev0, rev1)
        depi1 = _r(g("dep_amort", 1), g("dep_amort", 1) + g("net_ppe", 1))
        depi0 = _r(g("dep_amort", 0), g("dep_amort", 0) + g("net_ppe", 0))
        depi = _r(depi1, depi0)
        sgai = _r(_r(g("sga", 0), rev0), _r(g("sga", 1), rev1))
        ltd0 = g("long_term_debt", 0) or g("total_debt", 0)
        ltd1 = g("long_term_debt", 1) or g("total_debt", 1)
        lev0 = _r((ltd0 or 0) + g("current_liab", 0), g("total_assets", 0))
        lev1 = _r((ltd1 or 0) + g("current_liab", 1), g("total_assets", 1))
        lvgi = _r(lev0, lev1)
        tata = _r(g("net_income", 0) - g("cfo", 0), g("total_assets", 0))
        parts = [dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata]
        if any(p is None for p in parts):
            return None
        return (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
                + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    except (TypeError, ZeroDivisionError, IndexError):
        return None


def altman_z_score(F, i=0) -> float | None:
    """Altman Z''-Score (variante tous secteurs, capitaux propres comptables :
    aucune donnee de marche -> pas de probleme de devise). < 1.1 detresse,
    > 2.6 sain."""
    g = lambda k: F[k][i]
    try:
        ta = g("total_assets")
        x1 = _r(g("working_capital"), ta)
        x2 = _r(g("retained_earnings"), ta)
        x3 = _r(g("ebit"), ta)
        x4 = _r(g("equity"), g("total_liab"))
        if any(x is None for x in (x1, x2, x3, x4)):
            return None
        return 3.25 + 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    except (TypeError, ZeroDivisionError, IndexError):
        return None


def piotroski_f_score(F) -> dict | None:
    """F-Score de Piotroski (0-9) : force et AMELIORATION des fondamentaux.
    Retourne {score, max, tests} ; >=7 solide, <=2 faible."""
    g = lambda k, i: F[k][i]
    try:
        roa0 = _r(g("net_income", 0), g("total_assets", 0))
        roa1 = _r(g("net_income", 1), g("total_assets", 1))
        cfo0 = g("cfo", 0)
        tests = {}
        tests["ROA positif"] = (roa0 is not None and roa0 > 0)
        tests["Cash-flow d'exploitation positif"] = (cfo0 is not None and cfo0 > 0)
        tests["ROA en hausse"] = (roa0 is not None and roa1 is not None and roa0 > roa1)
        tests["Cash-flow > resultat net (accruals sains)"] = (
            cfo0 is not None and g("net_income", 0) is not None and cfo0 > g("net_income", 0))
        lev0 = _r(g("long_term_debt", 0) or g("total_debt", 0), g("total_assets", 0))
        lev1 = _r(g("long_term_debt", 1) or g("total_debt", 1), g("total_assets", 1))
        tests["Levier en baisse"] = (lev0 is not None and lev1 is not None and lev0 < lev1)
        cr0 = _r(g("current_assets", 0), g("current_liab", 0))
        cr1 = _r(g("current_assets", 1), g("current_liab", 1))
        tests["Liquidite courante en hausse"] = (cr0 is not None and cr1 is not None and cr0 > cr1)
        sh0, sh1 = g("shares", 0), g("shares", 1)
        tests["Pas de dilution"] = (sh0 is not None and sh1 is not None and sh0 <= sh1 * 1.001)
        gm0 = _r(g("gross_profit", 0), g("revenue", 0))
        gm1 = _r(g("gross_profit", 1), g("revenue", 1))
        tests["Marge brute en hausse"] = (gm0 is not None and gm1 is not None and gm0 > gm1)
        at0 = _r(g("revenue", 0), g("total_assets", 0))
        at1 = _r(g("revenue", 1), g("total_assets", 1))
        tests["Rotation de l'actif en hausse"] = (at0 is not None and at1 is not None and at0 > at1)
        score = sum(1 for v in tests.values() if v)
        return {"score": score, "max": 9, "tests": tests}
    except (TypeError, ZeroDivisionError, IndexError):
        return None


def accruals_ratio(F, i=0) -> float | None:
    """Accruals de Sloan = (Resultat net - Cash-flow d'exploitation) / Actif total.
    Eleve et positif -> resultats peu adosses au cash (qualite faible)."""
    try:
        return _r(F["net_income"][i] - F["cfo"][i], F["total_assets"][i])
    except (TypeError, IndexError):
        return None


def _growth(series, i0=0, i1=1):
    a, b = series[i0], series[i1]
    if a is None or b is None or b == 0:
        return None
    return a / b - 1.0


def analyze(ticker: str, financials=None) -> dict:
    """Analyse forensique complete d'un titre : scores + suivi annee/annee +
    points positifs et alertes (libelles NEUTRES, jamais 'fraude')."""
    F = financials or get_financials(ticker)
    if F is None:
        return {"ticker": ticker.upper(), "available": False}

    m = beneish_m_score(F)
    z = altman_z_score(F)
    piotroski = piotroski_f_score(F)
    accr = accruals_ratio(F)

    positives, flags = [], []

    # --- Points positifs (souvent discrets) ---
    if piotroski and piotroski["score"] >= 7:
        positives.append(f"Fondamentaux solides et en amelioration (F-Score {piotroski['score']}/9)")
    if F["cfo"][0] is not None and F["net_income"][0] is not None and F["cfo"][0] > F["net_income"][0] > 0:
        positives.append("Resultats bien adosses au cash (cash-flow > resultat net)")
    gm0 = _r(F["gross_profit"][0], F["revenue"][0])
    gm1 = _r(F["gross_profit"][1], F["revenue"][1])
    if gm0 is not None and gm1 is not None and gm0 > gm1:
        positives.append(f"Marge brute en expansion ({gm1*100:.1f}% -> {gm0*100:.1f}%)")
    if z is not None and z > 2.6:
        positives.append(f"Bilan sain, risque de detresse faible (Z-Score {z:.1f})")
    sh0, sh1 = F["shares"][0], F["shares"][1]
    if sh0 is not None and sh1 is not None and sh0 < sh1 * 0.999:
        positives.append(f"Reduction du nombre d'actions / rachats ({(sh0/sh1-1)*100:+.1f}%)")

    # --- Alertes (a investiguer, PAS des accusations) ---
    if m is not None and m > -1.78:
        flags.append(f"M-Score de Beneish en zone 'manipulateur probable' ({m:.2f} > -1,78) "
                     "- qualite du resultat a investiguer")
    if accr is not None and accr > 0.10:
        flags.append(f"Accruals eleves ({accr*100:.1f}% de l'actif) - resultats peu adosses au cash")
    if z is not None and z < 1.1:
        flags.append(f"Z-Score en zone de detresse ({z:.1f}) - solidite du bilan fragile")
    # Divergence resultat net vs cash-flow (type Stellantis : profits en hausse, cash qui decroche)
    g_ni = _growth(F["net_income"])
    g_cfo = _growth(F["cfo"])
    if g_ni is not None and g_cfo is not None and g_ni > 0.05 and g_cfo < -0.05:
        flags.append(f"Resultat net en hausse ({g_ni*100:+.0f}%) mais cash-flow d'exploitation "
                     f"en baisse ({g_cfo*100:+.0f}%) - divergence a surveiller")
    # Creances / stocks croissant plus vite que le CA
    g_rev = _growth(F["revenue"])
    g_rec = _growth(F["receivables"])
    g_inv = _growth(F["inventory"])
    if g_rev is not None and g_rec is not None and g_rec > g_rev + 0.10:
        flags.append(f"Creances croissant plus vite que le CA (+{g_rec*100:.0f}% vs +{g_rev*100:.0f}%) "
                     "- reconnaissance de revenus a verifier")
    if g_rev is not None and g_inv is not None and g_inv > g_rev + 0.15:
        flags.append(f"Stocks croissant plus vite que le CA (+{g_inv*100:.0f}% vs +{g_rev*100:.0f}%) "
                     "- possible sur-stockage / canal de distribution")

    return {
        "ticker": ticker.upper(),
        "available": True,
        "years": F["years"],
        "scores": {
            "beneish_m": None if m is None else round(m, 2),
            "beneish_flag": (m is not None and m > -1.78),
            "altman_z": None if z is None else round(z, 2),
            "piotroski_f": piotroski["score"] if piotroski else None,
            "piotroski_tests": piotroski["tests"] if piotroski else None,
            "accruals_ratio": None if accr is None else round(accr, 4),
        },
        "trends": {
            "net_income": F["net_income"], "cfo": F["cfo"], "revenue": F["revenue"],
        },
        "positives": positives,
        "flags": flags,
    }


__all__ = ["get_financials", "beneish_m_score", "altman_z_score",
           "piotroski_f_score", "accruals_ratio", "analyze"]

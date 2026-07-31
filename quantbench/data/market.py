"""
quantbench.data.market
======================
Donnees de marche gratuites : prix (Yahoo chart API v8, endpoint toujours vivant),
taux sans risque 10 ans (FRED DGS10, CSV sans cle), et beta calcule par regression
des rendements contre l'indice.
"""

from __future__ import annotations

import csv
import functools
import io as _io
from datetime import date, datetime, timezone

import numpy as np
import requests

_TIMEOUT = 20
_YUA = {"User-Agent": "Mozilla/5.0 (compatible; QuantBench/1.0)"}


@functools.lru_cache(maxsize=256)
def _chart(symbol: str, rng: str = "5y", interval: str = "1mo"):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?range={rng}&interval={interval}")
    r = requests.get(url, headers=_YUA, timeout=_TIMEOUT)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ind = res["indicators"]
    adj = None
    if "adjclose" in ind and ind["adjclose"]:
        adj = ind["adjclose"][0].get("adjclose")
    if not adj:
        adj = ind["quote"][0].get("close")
    return [v for v in adj if v is not None]


def latest_price(symbol: str) -> float:
    return float(_chart(symbol, "5d", "1d")[-1])


class TauxSansRisqueIndisponible(RuntimeError):
    """Aucune source n'a su fournir le rendement du 10 ans americain.

    Volontairement FATALE. Le taux sans risque entre dans le cout des fonds propres
    ET dans le cout de la dette de CHAQUE societe : lui substituer une constante en
    silence deplacerait toutes les valorisations du site d'un coup, sans que rien ne
    le signale. Mieux vaut ne rien publier que publier faux.
    """


# Bande de vraisemblance du rendement 10 ans, en fraction. Elle ne sert pas a juger
# du niveau des taux mais a detecter une ERREUR DE LECTURE : un decalage de colonne
# dans le CSV du Tresor rendrait le 1 mois ou le 30 ans, un facteur d'echelle
# manque rendrait 4,68 au lieu de 0,0468. Toute valeur hors bande est un defaut de
# parsage, jamais un fait de marche.
_BANDE_TAUX = (0.0, 0.25)


def _borne(v):
    return v if v is not None and _BANDE_TAUX[0] <= v <= _BANDE_TAUX[1] else None


def _taux_tresor() -> float | None:
    """Source PRIMAIRE : le Tresor americain publie lui-meme sa courbe des taux.

    C'est l'ORIGINE de la donnee — FRED n'en est que le republicateur. Le flux est
    plus rapide et n'a pas connu les indisponibilites de fredgraph.csv.
    Le CSV est trie PLUS RECENT EN TETE et couvre l'annee civile demandee ; le
    1er janvier, cette annee-la est vide, d'ou le repli sur la precedente.
    """
    for an in (date.today().year, date.today().year - 1):
        url = ("https://home.treasury.gov/resource-center/data-chart-center/"
               f"interest-rates/daily-treasury-rates.csv/{an}/all"
               f"?type=daily_treasury_yield_curve&field_tdr_date_value={an}"
               "&page&_format=csv")
        try:
            r = requests.get(url, timeout=_TIMEOUT, headers=_YUA)
            r.raise_for_status()
            lignes = list(csv.reader(_io.StringIO(r.text)))
            if len(lignes) < 2:
                continue
            col = lignes[0].index("10 Yr")
            for ligne in lignes[1:]:
                brut = (ligne[col] or "").strip()
                if brut:
                    t = _borne(float(brut) / 100.0)
                    if t is not None:
                        return t
        except Exception:
            continue
    return None


def _taux_fred() -> float | None:
    """Source de SECOURS. Conservee telle quelle : elle a servi jusqu'ici."""
    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
                         timeout=_TIMEOUT)
        r.raise_for_status()
        for line in reversed(r.text.strip().splitlines()[1:]):
            val = line.split(",")[-1].strip()
            if val not in (".", ""):
                return _borne(float(val) / 100.0)
    except Exception:
        return None
    return None


@functools.lru_cache(maxsize=1)
def risk_free_rate() -> float:
    """Rendement du Treasury 10 ans, en fraction decimale.

    DEUX SOURCES INDEPENDANTES. Il n'y en avait qu'une, sans repli : le jour ou
    fredgraph.csv ne repond pas — ce qui arrive — l'exception remontait jusqu'au
    moteur DCF et TOUTES les valorisations du build echouaient une a une. Pire,
    `lru_cache` ne memorise pas une exception : chaque societe repayait le delai
    d'attente complet, seize mille fois.

    L'echec est donc memorise ici, explicitement, pour qu'il coute une fois et non
    par titre.
    """
    if _ECHEC_MEMORISE:
        raise TauxSansRisqueIndisponible(_ECHEC_MEMORISE[0])
    # Chaque source attrape deja ses propres pannes et rend None. Le filet ci-dessous
    # ne protege pas d'un reseau indisponible mais d'un DEFAUT de la source elle-meme
    # — un format change, une colonne disparue — qui ne doit pas empecher d'essayer
    # la suivante ni, surtout, de MEMORISER l'echec.
    for source in (_taux_tresor, _taux_fred):
        try:
            t = source()
        except Exception:
            t = None
        if t is not None:
            return t
    _ECHEC_MEMORISE.append(
        "Ni le Tresor americain ni FRED n'ont fourni le rendement 10 ans. "
        "Aucune valorisation ne peut etre calculee sans taux sans risque.")
    raise TauxSansRisqueIndisponible(_ECHEC_MEMORISE[0])


# Porte l'echec entre les appels : une liste, parce que `lru_cache` ne memorise
# que les retours et jamais les exceptions.
_ECHEC_MEMORISE: list[str] = []


@functools.lru_cache(maxsize=128)
def fx_to_usd(currency: str):
    """Taux de conversion : 1 unite de `currency` = X USD. USD -> 1.0.

    Retourne None si la devise est inconnue (le code appelant DOIT alors signaler
    le titre plutot que de mal le valoriser). Gere les cotations en pence (GBp/GBX).
    """
    if not currency:
        return None
    c = currency.upper()
    if c == "USD":
        return 1.0
    if c in ("GBX", "GBP."):                     # pence britanniques -> livres
        gbp = fx_to_usd("GBP")
        return gbp / 100.0 if gbp else None
    for sym, invert in ((f"{c}USD=X", False), (f"USD{c}=X", True)):
        try:
            v = _chart(sym, "5d", "1d")
            if v and v[-1] and v[-1] > 0:
                return (1.0 / float(v[-1])) if invert else float(v[-1])
        except Exception:
            continue
    return None


@functools.lru_cache(maxsize=256)
def split_factor_since(symbol: str, since_iso: str) -> float:
    """Facteur cumule de fractionnement d'actions depuis `since_iso` (YYYY-MM-DD).

    Corrige le decalage entre le nombre d'actions du dernier 10-K (a une date de
    reference) et le cours actuel post-split : actions_courantes = actions_10K x
    facteur. Ex. un split 10:1 apres la date de reference -> facteur 10.
    """
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?range=5y&interval=1d&events=split")
    try:
        r = requests.get(url, headers=_YUA, timeout=_TIMEOUT)
        r.raise_for_status()
        events = r.json()["chart"]["result"][0].get("events", {}).get("splits", {})
    except Exception:
        return 1.0
    if not events:
        return 1.0
    since = date.fromisoformat(since_iso)
    factor = 1.0
    for s in events.values():
        d = datetime.fromtimestamp(int(s["date"]), tz=timezone.utc).date()
        if d > since:
            num = float(s.get("numerator", 1))
            den = float(s.get("denominator", 1))
            if den > 0:
                factor *= num / den
    return factor


def quote(symbol: str) -> dict:
    """Donnees de marche legeres via yfinance fast_info : prix, actions,
    capitalisation, beta — le tout CONVERTI en USD. Complement de la SEC (qui
    n'a ni prix ni capitalisation)."""
    import yfinance as yf
    out = {"price": None, "shares": None, "market_cap": None,
           "currency": None, "beta": None}
    try:
        fi = yf.Ticker(symbol).fast_info
        cur = (getattr(fi, "currency", None) or "USD").upper()
        fx = fx_to_usd(cur) or 1.0
        price = getattr(fi, "last_price", None)
        shares = getattr(fi, "shares", None)
        mcap = getattr(fi, "market_cap", None)
        out["currency"] = cur
        out["price"] = price * fx if price else None
        out["shares"] = shares
        out["market_cap"] = mcap * fx / 1e9 if mcap else None      # Md USD
    except Exception:
        pass
    if out["price"] is None:                       # repli sur le chart
        try:
            out["price"] = latest_price(symbol)
        except Exception:
            pass
    try:
        out["beta"] = levered_beta(symbol)
    except Exception:
        out["beta"] = None
    return out


def levered_beta(symbol: str, market: str = "%5EGSPC") -> float:
    """Beta leve : cov(rendements titre, marche) / var(marche), sur ~5 ans mensuels."""
    a = np.asarray(_chart(symbol), dtype=float)
    m = np.asarray(_chart(market), dtype=float)
    n = min(len(a), len(m))
    a, m = a[-n:], m[-n:]
    ra = np.diff(a) / a[:-1]
    rm = np.diff(m) / m[:-1]
    mask = np.isfinite(ra) & np.isfinite(rm)
    ra, rm = ra[mask], rm[mask]
    var = np.var(rm)
    if var == 0 or ra.size < 12:
        return 1.0
    return float(np.cov(ra, rm)[0, 1] / var)


__all__ = ["latest_price", "risk_free_rate", "levered_beta",
           "TauxSansRisqueIndisponible"]

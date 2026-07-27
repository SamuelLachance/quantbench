"""
quantbench.data.market
======================
Donnees de marche gratuites : prix (Yahoo chart API v8, endpoint toujours vivant),
taux sans risque 10 ans (FRED DGS10, CSV sans cle), et beta calcule par regression
des rendements contre l'indice.
"""

from __future__ import annotations

import functools
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


@functools.lru_cache(maxsize=1)
def risk_free_rate() -> float:
    """Rendement du Treasury 10 ans (DGS10), en fraction decimale."""
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
                     timeout=_TIMEOUT)
    r.raise_for_status()
    for line in reversed(r.text.strip().splitlines()[1:]):
        val = line.split(",")[-1].strip()
        if val not in (".", ""):
            return float(val) / 100.0
    raise RuntimeError("Aucune valeur DGS10 valide dans le flux FRED.")


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


__all__ = ["latest_price", "risk_free_rate", "levered_beta"]

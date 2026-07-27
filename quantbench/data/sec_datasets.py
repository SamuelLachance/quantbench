"""
quantbench.data.sec_datasets
============================
Backbone US complet et gratuit : SEC *Financial Statement Data Sets* (fichiers
trimestriels sub.txt/num.txt). Contrairement a l'API companyfacts (incomplete
pour certaines majors, ex. XOM), ces jeux sont CANONIQUES et complets.

Un 10-K porte ~3 ans de comparatifs (qtrs=4 pour les flux annuels, qtrs=0 pour
les postes de bilan instantanes). En unissant plusieurs trimestres on obtient
4-5 ans par societe. Les ZIP trimestriels sont IMMUABLES -> mis en cache.

Produit, par CIK, des structures compatibles forensics/route (memes _F_TAGS,
memes formes que sec_fundamentals).
"""

from __future__ import annotations

import io
import os
import zipfile
from datetime import date

import requests

from . import edgar
from .sec_fundamentals import _F_TAGS, _INSTANT, sic_to_sector

_BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/"
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data_cache", "sec")

# Ensemble des tags a extraire (aplati depuis _F_TAGS + cash)
_WANT = set()
for _labs in _F_TAGS.values():
    _WANT |= set(_labs)
_WANT |= set(edgar.TAGS["cash"])


def recent_quarters(n: int, as_of: date | None = None):
    """n derniers trimestres (plus recent d'abord), au format (annee, q)."""
    d = as_of or date.today()
    y, q = d.year, (d.month - 1) // 3 + 1
    out = []
    for _ in range(n):
        out.append((y, q))
        q -= 1
        if q == 0:
            q, y = 4, y - 1
    return out


def quarter_zip(year: int, q: int, cache_dir: str = _CACHE):
    """Chemin du ZIP trimestriel (telecharge+cache si absent). None si 404."""
    name = f"{year}q{q}.zip"
    path = os.path.join(cache_dir, name)
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        return path
    r = requests.get(_BASE + name, headers=edgar._UA, timeout=180)
    if r.status_code != 200:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(r.content)
    return path


def build_facts(quarters, ciks=None, cache_dir: str = _CACHE) -> dict:
    """Parse les trimestres et retourne {cik: entry}. entry = sic, name,
    dur{tag:{year:val}} (flux annuels), inst{tag:{year:val}} (bilan).
    `ciks` (iterable) filtre l'univers pour limiter la memoire."""
    keep = set(str(int(c)) for c in ciks) if ciks else None
    data: dict = {}

    for (y, q) in quarters:
        path = quarter_zip(y, q, cache_dir)
        if not path:
            continue
        with zipfile.ZipFile(path) as z:
            # sub.txt : adsh -> metadonnees des 10-K
            sub = {}
            with z.open("sub.txt") as fh:
                rd = io.TextIOWrapper(fh, encoding="latin-1")
                h = rd.readline().rstrip("\n").split("\t")
                ci = {c: i for i, c in enumerate(h)}
                for line in rd:
                    f = line.rstrip("\n").split("\t")
                    if f[ci["form"]] not in ("10-K", "10-K/A"):
                        continue
                    cik = f[ci["cik"]]
                    if keep and cik not in keep:
                        continue
                    sub[f[ci["adsh"]]] = (cik, f[ci["sic"]], f[ci["name"]],
                                          f[ci["filed"]])
            if not sub:
                continue
            # num.txt : faits numeriques (streaming)
            with z.open("num.txt") as fh:
                rd = io.TextIOWrapper(fh, encoding="latin-1")
                h = rd.readline().rstrip("\n").split("\t")
                nci = {c: i for i, c in enumerate(h)}
                iA, iT, iD, iQ, iU = (nci["adsh"], nci["tag"], nci["ddate"],
                                      nci["qtrs"], nci["uom"])
                iSeg, iCo, iV = nci["segments"], nci["coreg"], nci["value"]
                for line in rd:
                    f = line.rstrip("\n").split("\t")
                    meta = sub.get(f[iA])
                    if not meta:
                        continue
                    tag = f[iT]
                    if tag not in _WANT or f[iSeg] or f[iCo]:
                        continue
                    if f[iU] not in ("USD", "shares"):
                        continue
                    bucket = "inst" if f[iQ] == "0" else ("dur" if f[iQ] == "4" else None)
                    if bucket is None:
                        continue
                    try:
                        val = float(f[iV])
                    except ValueError:
                        continue
                    cik, sic, name, filed = meta
                    year = int(f[iD][:4])
                    e = data.setdefault(cik, {"cik": cik, "sic": sic, "name": name,
                                              "dur": {}, "inst": {}, "_f": {}})
                    k = (bucket, tag, year)
                    if k not in e["_f"] or filed > e["_f"][k]:
                        e["_f"][k] = filed
                        e[bucket].setdefault(tag, {})[year] = val
                    if filed > e.get("_latest", ""):        # metadonnees du depot le plus recent
                        e["_latest"], e["sic"], e["name"] = filed, sic, name
    return data


def _merge_years(entry, key):
    # Les actions peuvent etre instantanees (dei cover) OU duration (moyenne
    # ponderee diluee) -> on cherche dans les deux buckets.
    if key == "shares":
        buckets = ("inst", "dur")
    else:
        buckets = ("inst",) if key in _INSTANT else ("dur",)
    merged = {}
    for bucket in buckets:
        for lab in _F_TAGS.get(key, []):
            d = entry[bucket].get(lab)
            if d:
                for yr, v in d.items():
                    merged.setdefault(yr, v)          # 1er tag candidat present gagne
    return merged


def extract_financials(entry) -> dict | None:
    """Structure compatible forensics.get_financials (annees alignees)."""
    maps = {k: _merge_years(entry, k) for k in _F_TAGS}
    core = [set(maps[k]) for k in ("revenue", "net_income", "total_assets") if maps.get(k)]
    if len(core) < 3:
        return None
    years = sorted(set.intersection(*core), reverse=True)
    if len(years) < 2:
        return None
    F = {"years": [str(y) for y in years]}
    for k in _F_TAGS:
        F[k] = [maps[k].get(y) for y in years]
    F["total_debt"] = F["long_term_debt"]
    ca, cl = F["current_assets"], F["current_liab"]
    F["working_capital"] = [(ca[i] - cl[i]) if (ca[i] is not None and cl[i] is not None)
                            else None for i in range(len(years))]
    return F


def extract_fundamentals(entry, ticker, quote) -> dict:
    """Dict compatible universal.get_fundamentals : fondamentaux SEC + donnees de
    marche (`quote` = market.quote : prix/actions/capi/beta en USD)."""
    def latest(key):
        m = _merge_years(entry, key)
        return m[max(m)] if m else None

    revenue = latest("revenue")
    ebit = latest("ebit")
    ni = latest("net_income")
    equity = latest("equity")
    debt = latest("long_term_debt") or 0.0
    cash_m = {}
    for lab in edgar.TAGS["cash"]:
        d = entry["inst"].get(lab)
        if d:
            cash_m = d
            break
    cash = cash_m[max(cash_m)] if cash_m else 0.0
    rev_map = _merge_years(entry, "revenue")
    rev_hist = [rev_map[y] for y in sorted(rev_map)]     # ancien -> recent

    B = 1e9
    price = quote.get("price")
    shares = quote.get("shares") or latest("shares")     # marche, sinon SEC
    beta = quote.get("beta") or 1.1
    mcap = quote.get("market_cap")
    if mcap is None and price and shares:
        mcap = price * shares / B
    bb = lambda x: None if x is None else x / B
    return {
        "ticker": ticker.upper(), "name": entry.get("name") or ticker.upper(),
        "sector": sic_to_sector(entry.get("sic")), "industry": None,
        "price": price, "market_cap": mcap, "shares": shares, "beta": beta,
        "currency_ok": True, "price_currency": "USD", "financial_currency": "USD",
        "revenue": bb(revenue), "revenue_history": [x / B for x in rev_hist],
        "ebit": bb(ebit), "net_income": bb(ni),
        "total_debt": bb(debt), "cash": bb(cash), "book_equity": bb(equity),
        "operating_margin": (ebit / revenue) if (ebit and revenue) else None,
        "roe": (ni / equity) if (ni and equity) else None,
        "sic": entry.get("sic"),
    }


__all__ = ["recent_quarters", "quarter_zip", "build_facts",
           "extract_financials", "extract_fundamentals"]

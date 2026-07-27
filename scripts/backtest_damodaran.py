"""Backtest SANS biais du survivant sur le S&P 500 point-in-time.

Reconstitue les membres du S&P 500 à une date T (via l'historique des changements
FMP), y compris les sociétés depuis délistées/faillies. Pour chacune : modèle
Damodaran (MC 10k) + forensique (Altman/Beneish/Piotroski) reconstitués AS-OF T
(zéro look-ahead), puis rendement RÉEL jusqu'à aujourd'hui — ou jusqu'au délisting
(la chute est capturée). On teste si le modèle a identifié valeurs et faillites.

Usage : FMP_API_KEY=... python scripts/backtest_damodaran.py [--T 2013-07-01]
        [--rf 0.025] [--workers 16]
"""

import concurrent.futures as cf
import json
import os
import re
import sys
import warnings
from datetime import date, timedelta

import numpy as np

try:                                    # Windows: éviter le crash cp1252 sur → − ² …
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                # scripts/ -> build_site_fmp
sys.path.insert(0, os.path.dirname(_HERE))               # racine -> package quantbench
from quantbench.data import fmp                          # noqa: E402
from quantbench.forensics import analyze                 # noqa: E402
from quantbench.valuation.route import value_stock       # noqa: E402
import build_site_fmp as bs                              # noqa: E402

warnings.filterwarnings("ignore")
_TODAY = date(2026, 7, 24)

# Raison de retrait = faillite / mise sous tutelle (action anéantie) vs acquisition.
_FAIL_PAT = re.compile(
    r"receivership|bankrupt|chapter\s*11|liquidat|insolven|delist|wind[- ]?down",
    re.I)


def _events():
    return [e for e in fmp._json("historical-sp500-constituent") if e.get("date")]


def removal_reasons(evs):
    """{ticker retiré -> raison la plus récente} pour classer faillite vs rachat."""
    out = {}
    for e in sorted(evs, key=lambda e: e["date"]):        # ancien -> récent (récent écrase)
        rt = e.get("removedTicker")
        if rt:
            out[rt] = e.get("reason") or ""
    return out


def reconstruct_sp500(T, evs=None):
    """Ensemble des tickers du S&P 500 à la date T (annule les changements postérieurs)."""
    cur = {r["symbol"] for r in fmp._json("sp500-constituent") if r.get("symbol")}
    evs = evs if evs is not None else _events()
    evs = sorted(evs, key=lambda e: e["date"], reverse=True)   # plus récent d'abord
    Td = date.fromisoformat(T)
    s = set(cur)
    for e in evs:
        try:
            d = date.fromisoformat(e["date"])
        except Exception:
            continue
        if d <= Td:
            break
        if e.get("symbol"):
            s.discard(e["symbol"])                        # annule l'ajout
        if e.get("removedTicker"):
            s.add(e["removedTicker"])                     # remet le retiré
    return s


def history(symbol, frm="2012-01-01"):
    try:
        return fmp._json(f"historical-price-eod/light?symbol={symbol}&from={frm}&to=2026-07-27")
    except Exception:
        return []


def nearest(hist, target):
    if not hist:
        return None, None
    td = date.fromisoformat(target)
    best = min(hist, key=lambda r: abs((date.fromisoformat(r["date"]) - td).days))
    return fmp._num(best.get("price")), best["date"]


def cutoff_year(T):
    """Dernière année fiscale RÉELLEMENT publiée avant T (pas de look-ahead).
    10-K déposé ~60-90 j après clôture déc. : dispo l'année suivante vers mars."""
    y, m = int(T[:4]), int(T[5:7])
    return y - 1 if m >= 4 else y - 2


def trailing_beta(hist, spy, T, lookback_days=760):
    """Beta glissant réel à la date T (régression des rendements quotidiens ~2 ans
    du titre sur le SPY), au lieu d'un beta figé — évite l'artefact de WACC uniforme."""
    if not hist or not spy:
        return None
    td = date.fromisoformat(T)
    lo = td - timedelta(days=lookback_days)

    def px(h):
        out = {}
        for r in h:
            p = fmp._num(r.get("price"))
            if not p or p <= 0:
                continue
            try:
                d = date.fromisoformat(r["date"])
            except Exception:
                continue
            if lo <= d <= td:
                out[d] = p
        return out

    a, b = px(hist), px(spy)
    common = sorted(set(a) & set(b))
    if len(common) < 60:
        return None
    pa = np.array([a[d] for d in common])
    pb = np.array([b[d] for d in common])
    ra, rb = pa[1:] / pa[:-1] - 1.0, pb[1:] / pb[:-1] - 1.0
    var = float(np.var(rb))
    if var <= 0:
        return None
    beta = float(np.cov(ra, rb)[0, 1] / var)
    return min(max(beta, 0.3), 2.5)                    # borne raisonnable


def asof(entry, year):
    return {k: {y: r for y, r in entry[k].items() if y <= year}
            for k in ("income", "balance", "cashflow")}


def backtest_one(symbol, T, rf, reasons=None, spy=None):
    reasons = reasons or {}
    entry = fmp.statements(symbol, limit=16)
    eT = asof(entry, cutoff_year(T))                   # <= dernière année publiée avant T
    if len(set(eT["income"]) & set(eT["balance"])) < 2:
        return None
    hist = history(symbol)
    pT, dT = nearest(hist, T)
    if not pT or pT <= 0 or not hist:
        return None
    last_close, last_date = fmp._num(hist[0].get("price")), hist[0]["date"]
    if not last_close:
        return None
    ld = date.fromisoformat(last_date)
    years = max((ld - date.fromisoformat(T)).days / 365.25, 0.5)
    if years < 1.5:
        return None
    delisted = ld < _TODAY - timedelta(days=150)
    total = last_close / pT - 1.0
    # Faillite / mise sous tutelle : le dernier prix coté (souvent halté) sur-estime la
    # valeur récupérée — l'action a été anéantie. On force le rendement réel à ≈ −99 %.
    failure = delisted and bool(_FAIL_PAT.search(reasons.get(symbol, "")))
    if failure:
        total = -0.99
    ann = (1.0 + total) ** (1.0 / years) - 1.0

    yrs = sorted(set(eT["income"]) & set(eT["balance"]), reverse=True)
    shares = fmp._num(eT["income"][yrs[0]].get("weightedAverageShsOutDil"))
    if not shares or shares <= 0:
        return None
    beta = trailing_beta(hist, spy, T) or 1.1          # beta réel glissant (sinon défaut)
    srT = {"exchange": "NASDAQ", "price": pT, "market_cap": pT * shares,
           "beta": beta, "sector": None, "industry": None, "name": symbol}
    fundT = fmp.fundamentals_from_fmp(symbol, srT, eT, {})
    if not fundT or not fundT.get("revenue"):
        return None
    FT = fmp.financials_from_fmp(eT)
    forT = analyze(symbol, financials=FT) if FT else None
    valT = value_stock(symbol, fund=fundT, forensic=forT, F=FT)
    if not valT.get("ok"):
        return None
    mc = bs.run_mc(fundT, valT.get("category"), n=10000, rf=rf)
    if not mc or not fundT.get("market_cap"):
        return None
    sc = (forT or {}).get("scores", {})
    return {"ticker": symbol, "category": valT["category"],
            "upside": mc["median"] / fundT["market_cap"] - 1.0,
            "altman_z": sc.get("altman_z"), "piotroski": sc.get("piotroski_f"),
            "beneish_flag": sc.get("beneish_flag"),
            "ann": ann, "total": total, "years": round(years, 1),
            "delisted": delisted, "failure": failure}


def _p(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")


def _spear(x, y):
    return _p(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))


def report(res, spy_ann, T):
    ann = np.array([r["ann"] for r in res])
    up = np.array([r["upside"] for r in res])
    n = len(res)
    nd = sum(r["delisted"] for r in res)
    print(f"\n{'='*70}\nBACKTEST S&P 500 point-in-time (sans biais du survivant) — {T} → 2026")
    print(f"{n} sociétés (dont {nd} délistées/disparues depuis) | "
          f"rendement annualisé SPY : {spy_ann*100:+.1f}%/an\n")

    print("Corrélation signal(T) ↔ rendement annualisé futur (Spearman rang) :")
    print(f"  upside DCF/MC : {_spear(up, ann):+.3f}")
    z = [(r['altman_z'], r['ann']) for r in res if r['altman_z'] is not None]
    if z:
        print(f"  Altman Z''    : {_spear([a for a,_ in z],[b for _,b in z]):+.3f}  "
              f"(détresse -> Z bas -> devrait corréler + avec le rendement)")
    pio = [(r['piotroski'], r['ann']) for r in res if r['piotroski'] is not None]
    if pio:
        print(f"  Piotroski F   : {_spear([a for a,_ in pio],[b for _,b in pio]):+.3f}")

    order = sorted(res, key=lambda r: r["upside"])
    q = np.array_split(order, 5)
    print("\nQuintiles par upside (bas→haut) — rendement annualisé moyen :")
    for i, g in enumerate(q, 1):
        m = np.mean([r["ann"] for r in g])
        dl = sum(r["delisted"] for r in g)
        print(f"  Q{i} : {m*100:+6.1f}%/an  (n={len(g)}, délistées={dl})")
    ls = np.mean([r["ann"] for r in q[-1]]) - np.mean([r["ann"] for r in q[0]])
    print(f"  Spread Q5−Q1 : {ls*100:+.1f} pts/an")

    # --- Détection des FAILLITES / gros perdants ---
    fails = [r for r in res if r["failure"] or r["total"] < -0.6
             or (r["delisted"] and r["total"] < -0.4)]
    print(f"\n--- Casses (rendement total < −60%, ou délistée < −40%) : {len(fails)} sociétés ---")
    if fails:
        zf = [r["altman_z"] for r in fails if r["altman_z"] is not None]
        print(f"  Médiane upside du modèle : {np.median([r['upside'] for r in fails])*100:+.0f}% "
              f"(vs {np.median(up)*100:+.0f}% pour l'ensemble)")
        if zf:
            print(f"  Médiane Altman Z'' : {np.median(zf):.1f} | en détresse (Z<1.1) : "
                  f"{np.mean(np.array(zf)<1.1)*100:.0f}%")
        print(f"  Signalées Beneish : {np.mean([bool(r['beneish_flag']) for r in fails])*100:.0f}%")
        # détection : le modèle les a-t-il classées dans le pire quintile d'upside ?
        thr = np.percentile(up, 40)
        flagged = np.mean([r["upside"] <= thr for r in fails]) * 100
        print(f"  Classées dans le bas 40% d'upside (jugées chères) : {flagged:.0f}%")
        print("  Le modèle recommandait (upside jugé → rendement réel) :")
        for r in sorted(fails, key=lambda r: r["total"])[:8]:
            print(f"    {r['ticker']:6} upside {r['upside']*100:+5.0f}%  →  réel {r['total']*100:+.0f}%")

    winners = sorted(res, key=lambda r: -r["ann"])[:max(10, n // 10)]
    print(f"\n--- Top 10% gagnants : médiane upside {np.median([r['upside'] for r in winners])*100:+.0f}% ---")
    print("  Exemples :", ", ".join(f"{r['ticker']}({r['ann']*100:+.0f}%/an)" for r in winners[:6]))
    print(f"\nRendement annualisé médian global : {np.median(ann)*100:+.1f}%/an | "
          f"% battant le SPY : {np.mean(ann>spy_ann)*100:.0f}%")


def main(T="2013-07-01", rf=0.025, workers=16):
    evs = _events()
    reasons = removal_reasons(evs)
    universe = sorted(reconstruct_sp500(T, evs))
    print(f"S&P 500 reconstitué au {T} : {len(universe)} membres. rf={rf:.1%}. Backtest…")
    spy = history("SPY")
    pT, _ = nearest(spy, T)
    spy_ann = (fmp._num(spy[0]["price"]) / pT) ** (1 / ((_TODAY - date.fromisoformat(T)).days / 365.25)) - 1 if spy and pT else 0.0

    res = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(backtest_one, s, T, rf, reasons, spy): s for s in universe}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            try:
                r = fut.result()
                if r:
                    res.append(r)
            except Exception:
                pass
            if i % 100 == 0:
                print(f"  {i}/{len(universe)} (ok={len(res)})")
    try:
        out = os.path.join(_HERE, f"_backtest_{T}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"T": T, "rf": rf, "spy_ann": spy_ann, "res": res}, f)
        print(f"[résultats bruts -> {out}]")
    except Exception as e:
        print(f"[dump JSON échoué: {e}]")
    report(res, spy_ann, T)


if __name__ == "__main__":
    args = sys.argv[1:]
    kw = {"T": "2013-07-01", "rf": 0.025, "workers": 16}
    while args:
        a = args.pop(0)
        if a == "--T": kw["T"] = args.pop(0)
        elif a == "--rf": kw["rf"] = float(args.pop(0))
        elif a == "--workers": kw["workers"] = int(args.pop(0))
    main(**kw)

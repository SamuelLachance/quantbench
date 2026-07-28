"""Statistiques SECTORIELLES calculees sur l'univers reel.

Remplace les constantes arbitraires du moteur de valorisation par des reperes
mesures secteur par secteur — c'est la pratique de Damodaran, qui publie ses
moyennes par industrie plutot que d'imposer un chiffre unique a toute societe :

  * marge operationnelle mediane  -> marge cible d'une societe jeune/deficitaire
                                     (au lieu d'un 12 % arbitraire identique pour
                                     une biotech et un distributeur)
  * ventes/capital medianes       -> intensite capitalistique du secteur
  * intensite d'actifs corporels  -> taux de recuperation en liquidation (un
                                     immeuble se revend, pas un logiciel)
  * beta median                   -> repli quand le beta d'un titre est aberrant

Usage : FMP_API_KEY=... python scripts/build_sector_stats.py [--par-secteur 60]
"""

import concurrent.futures as cf
import json
import os
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from quantbench.data import fmp                       # noqa: E402

OUT = os.path.join(os.path.dirname(_HERE), "quantbench", "valuation", "sector_stats.json")


def mesure(sym, sr):
    """Mesures brutes d'une societe (aucune valorisation, juste des ratios)."""
    try:
        e = fmp.statements(sym, limit=4)
        F = fmp.financials_from_fmp(e)
        f = fmp.fundamentals_from_fmp(sym, sr, e, {})
        if not f or not f.get("revenue") or f["revenue"] <= 0:
            return None
        rev = f["revenue"]
        out = {"sector": f.get("sector")}
        if f.get("ebit") is not None:
            out["marge"] = f["ebit"] / rev
        inv = (f.get("book_equity") or 0) + (f.get("total_debt") or 0) - (f.get("cash") or 0)
        if inv > 0:
            out["s2c"] = rev / inv
        ta = f.get("total_assets")
        ppe = (F or {}).get("net_ppe", [None])[0]
        if ta and ta > 0 and ppe is not None:
            out["corporel"] = max(0.0, min(1.0, (ppe / 1e9) / ta))
        if f.get("beta"):
            out["beta"] = f["beta"]
        return out
    except Exception:
        return None


def med(vals, lo, hi, defaut):
    vals = [v for v in vals if v is not None and lo <= v <= hi]
    return round(float(st.median(vals)), 4) if len(vals) >= 5 else defaut


def main(par_secteur=60):
    uni = fmp.screener(["NASDAQ", "NYSE", "TSX", "TSXV"])
    par = {}
    for s, r in uni.items():
        sec = r.get("sector")
        if sec:
            par.setdefault(sec, []).append((s, r.get("market_cap") or 0))
    # les plus grosses de chaque secteur : donnees les plus fiables
    ech = []
    for sec, lst in par.items():
        lst.sort(key=lambda x: -x[1])
        ech += [s for s, _ in lst[:par_secteur]]
    print(f"{len(par)} secteurs, {len(ech)} societes echantillonnees…")

    res = []
    with cf.ThreadPoolExecutor(max_workers=14) as ex:
        futs = {ex.submit(mesure, s, uni[s]): s for s in ech}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            try:
                r = fut.result()
                if r:
                    res.append(r)
            except Exception:
                pass
            if i % 200 == 0:
                print(f"  {i}/{len(ech)}")

    stats = {}
    for sec in sorted(par):
        g = [r for r in res if r.get("sector") == sec]
        if len(g) < 5:
            continue
        corporel = med([r.get("corporel") for r in g], 0.0, 1.0, 0.30)
        stats[sec] = {
            "n": len(g),
            "marge": med([r.get("marge") for r in g], -0.5, 0.75, 0.10),
            "s2c": med([r.get("s2c") for r in g], 0.1, 10.0, 1.5),
            "corporel": corporel,
            "beta": med([r.get("beta") for r in g], 0.1, 3.5, 1.1),
            # Taux de recuperation en liquidation : un actif CORPOREL se revend
            # (immeuble, centrale, gisement), un actif incorporel beaucoup moins.
            "recuperation": round(min(0.80, max(0.30, 0.30 + 0.50 * corporel)), 3),
        }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"\n{'secteur':26} {'n':>4} {'marge':>7} {'s2c':>6} {'corp.':>6} {'recup':>6} {'beta':>5}")
    for sec, v in sorted(stats.items()):
        print(f"  {sec[:24]:24} {v['n']:>4} {v['marge']*100:>6.1f}% {v['s2c']:>6.2f} "
              f"{v['corporel']:>6.2f} {v['recuperation']:>6.2f} {v['beta']:>5.2f}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    args = sys.argv[1:]
    n = 60
    while args:
        a = args.pop(0)
        if a == "--par-secteur":
            n = int(args.pop(0))
    main(n)

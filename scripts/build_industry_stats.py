"""Statistiques par INDUSTRIE et par SECTEUR, mesurees sur l'univers reel.

Reproduit la demarche de Damodaran, qui publie ses reperes par industrie plutot
que d'imposer un chiffre unique : chaque parametre du modele est ainsi propre a
l'activite de la societe, et non herite d'une constante globale.

Parametre central : le BETA DESENDETTE PAR INDUSTRIE. Damodaran recommande
explicitement de ne PAS utiliser le beta de regression d'un titre — bruite, et
carrement faux sur une cotation peu liquide (0,20 pour la fonciere mexicaine Fibra
UNO, 0,24 pour China Minsheng : ces titres bougent peu faute d'ECHANGES, pas faute
de RISQUE). On prend le beta d'ACTIVITE de l'industrie, desendette de la structure
financiere de chacun :
        beta_desendette = beta_levier / (1 + (1 - t) x D/E)
puis on le RE-ENDETTE au levier propre de la societe valorisee et au taux d'impot
de SON pays. Le risque devient ainsi une propriete de l'activite exercee, corrigee
de la structure de bilan — exactement la construction de Damodaran.

Sont egalement mesurees par industrie : marge operationnelle, ventes/capital,
ROIC, ratio d'endettement et intensite d'actifs corporels (qui fixe le taux de
recuperation en liquidation).

Usage : FMP_API_KEY=... python scripts/build_industry_stats.py [--par-groupe 40]
"""

import concurrent.futures as cf
import json
import os
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from quantbench.data import fmp                                      # noqa: E402
from quantbench.valuation.build_universal import tax_rate            # noqa: E402

OUT = os.path.join(os.path.dirname(_HERE), "quantbench", "valuation",
                   "industry_stats.json")
MIN_OBS = 5                     # au-dessous, l'industrie retombe sur son secteur


def mesure(sym, sr):
    """Mesures brutes d'une societe : aucun jugement, uniquement des ratios."""
    try:
        e = fmp.statements(sym, limit=4)
        F = fmp.financials_from_fmp(e)
        f = fmp.fundamentals_from_fmp(sym, sr, e, {})
        if not f or not f.get("revenue") or f["revenue"] <= 0:
            return None
        rev, mcap = f["revenue"], f.get("market_cap")
        debt, cash = f.get("total_debt") or 0.0, f.get("cash") or 0.0
        be = f.get("book_equity") or 0.0
        out = {"secteur": f.get("sector"), "industrie": f.get("industry")}
        if f.get("ebit") is not None:
            out["marge"] = f["ebit"] / rev
        inv = be + debt - cash
        if inv > 0:
            out["s2c"] = rev / inv
            if f.get("ebit") is not None:
                out["roic"] = f["ebit"] * (1 - tax_rate(f.get("country"))) / inv
        if mcap and mcap > 0:
            de = debt / mcap
            out["dette_sur_capitalisation"] = de
            b = f.get("beta")
            # Beta DESENDETTE : on retire l'effet du levier pour isoler le risque
            # d'ACTIVITE, seul repere comparable entre societes d'une industrie.
            if b and 0.1 <= b <= 3.5 and de >= 0:
                out["beta_desendette"] = b / (1 + (1 - tax_rate(f.get("country"))) * de)
        # Part d'actifs REALISABLES : ce qui se revend en cas de liquidation, soit
        # l'actif total DIMINUE du goodwill et des incorporels. Mesurer les seules
        # immobilisations corporelles nettes etait faux pour l'immobilier, dont la
        # plupart des foncieres inscrivent leurs immeubles en "investment property"
        # et non en immobilisations : le secteur ressortait a 0,6 % d'actifs
        # corporels alors que ses actifs sont, precisement, les plus realisables.
        # Ratio de deux champs du MEME bilan, donc independant de la devise.
        bal = e.get("balance", {})
        if bal:
            b0 = bal[max(bal)]
            ta_b = fmp._num(b0.get("totalAssets"))
            incorp = (fmp._num(b0.get("goodwillAndIntangibleAssets"))
                      or ((fmp._num(b0.get("goodwill")) or 0.0)
                          + (fmp._num(b0.get("intangibleAssets")) or 0.0)))
            if ta_b and ta_b > 0 and incorp is not None:
                out["corporel"] = max(0.0, min(1.0, 1.0 - incorp / ta_b))
        return out
    except Exception:
        return None


def med(vals, lo, hi):
    vals = [v for v in vals if v is not None and lo <= v <= hi]
    return round(float(st.median(vals)), 4) if len(vals) >= MIN_OBS else None


def agrege(groupe):
    corporel = med([r.get("corporel") for r in groupe], 0.0, 1.0)
    return {
        "n": len(groupe),
        "beta_desendette": med([r.get("beta_desendette") for r in groupe], 0.05, 3.0),
        "marge": med([r.get("marge") for r in groupe], -0.5, 0.75),
        "s2c": med([r.get("s2c") for r in groupe], 0.1, 10.0),
        "roic": med([r.get("roic") for r in groupe], -0.2, 0.60),
        "dette_sur_capitalisation": med([r.get("dette_sur_capitalisation")
                                         for r in groupe], 0.0, 5.0),
        "corporel": corporel,
        # Recuperation en liquidation : un actif CORPOREL se revend (centrale,
        # immeuble, gisement), un actif incorporel beaucoup moins.
        "recuperation": (round(min(0.80, max(0.30, 0.30 + 0.50 * corporel)), 3)
                         if corporel is not None else None),
    }


def main(par_groupe=40):
    uni = fmp.screener(["NASDAQ", "NYSE", "TSX", "TSXV"])
    par_ind = {}
    for s, r in uni.items():
        ind = r.get("industry")
        if ind:
            par_ind.setdefault(ind, []).append((s, r.get("market_cap") or 0))
    ech = []
    for ind, lst in par_ind.items():
        lst.sort(key=lambda x: -x[1])
        ech += [s for s, _ in lst[:par_groupe]]     # les plus grosses : donnees fiables
    print(f"{len(par_ind)} industries, {len(ech)} societes echantillonnees…")

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
            if i % 400 == 0:
                print(f"  {i}/{len(ech)}")

    industries, secteurs = {}, {}
    for ind in sorted({r.get("industrie") for r in res if r.get("industrie")}):
        g = [r for r in res if r.get("industrie") == ind]
        if len(g) >= MIN_OBS:
            industries[ind] = agrege(g)
    for sec in sorted({r.get("secteur") for r in res if r.get("secteur")}):
        g = [r for r in res if r.get("secteur") == sec]
        if len(g) >= MIN_OBS:
            secteurs[sec] = agrege(g)
    # Repere GLOBAL : dernier recours quand ni l'industrie ni le secteur ne sont
    # renseignes, pour qu'aucune societe ne reste sans parametre.
    glob = agrege(res)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"industries": industries, "secteurs": secteurs, "global": glob},
                  f, ensure_ascii=False, indent=1)

    print(f"\n{len(industries)} industries, {len(secteurs)} secteurs mesures.")
    print(f"\n{'industrie':34} {'n':>4} {'beta_des':>9} {'marge':>7} {'s2c':>6} {'D/cap':>7}")
    for ind, v in sorted(industries.items(),
                         key=lambda kv: -(kv[1]["beta_desendette"] or 0))[:12]:
        print(f"  {ind[:32]:32} {v['n']:>4} {str(v['beta_desendette']):>9} "
              f"{str(v['marge']):>7} {str(v['s2c']):>6} {str(v['dette_sur_capitalisation']):>7}")
    print("  …")
    for ind, v in sorted(industries.items(),
                         key=lambda kv: (kv[1]["beta_desendette"] or 9))[:6]:
        print(f"  {ind[:32]:32} {v['n']:>4} {str(v['beta_desendette']):>9} "
              f"{str(v['marge']):>7} {str(v['s2c']):>6} {str(v['dette_sur_capitalisation']):>7}")
    print(f"\nglobal : beta_desendette={glob['beta_desendette']} marge={glob['marge']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    args = sys.argv[1:]
    n = 40
    while args:
        a = args.pop(0)
        if a == "--par-groupe":
            n = int(args.pop(0))
    main(n)

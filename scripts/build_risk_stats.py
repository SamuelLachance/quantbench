"""Calibrage de la note de risque — MESURE tout ce que le score utilise.

Aucun nombre de la notation n'est pose a la main. Ce script produit
`quantbench/risk/risk_calibration.json`, que le build quotidien se contente de LIRE :

  quantiles        grille de centiles de chaque dimension, au niveau qui lui convient
                   (univers, secteur ou industrie)
  liquidite        regression du volume echange sur la taille, PAR PLACE DE COTATION
  bornes_grades    les douze bornes qui separent les treize grades
  plafonds         le niveau de chaque plafond, mesure et non choisi
  poids            ponderation des dimensions

POPULATION DE REFERENCE. Les quantiles ne sont pas mesures sur l'univers entier mais
sur les societes dont les comptes sont RENSEIGNES ET COHERENTS : au moins six
exercices, aucune reparation appliquee, identite du bilan respectee. Calibrer sur
l'univers entier — majoritairement de gre a gre, dont une moitie ne couvre pas ses
interets — ferait glisser toute l'echelle vers le bas et rendrait A+ atteignable en
etant simplement moins mauvais que des coquilles.

Usage : FMP_API_KEY=... python scripts/build_risk_stats.py [--limite 4000]
"""

import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")
_ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_ICI))

from quantbench.data import fmp                                       # noqa: E402
from quantbench.data.repair import reparer                            # noqa: E402
from quantbench.data.validate import valider                          # noqa: E402
from quantbench.risk.dimensions import DIMENSIONS, mesurer            # noqa: E402
from quantbench.risk.score import GRADES, noter                       # noqa: E402

SORTIE = os.path.join(os.path.dirname(_ICI), "quantbench", "risk",
                      "risk_calibration.json")
CENTILES = 100
# Effectif minimal pour qu'une grille sectorielle ou industrielle soit publiee. Ce
# n'est pas un seuil d'opinion : en dessous, l'intervalle de confiance des centiles
# est plus large que l'ecart entre deux grades, la grille ne separerait donc rien.
N_MIN_GRILLE = 40
# Exercices exiges pour entrer dans la population de reference.
EXERCICES_MIN = 6


def collecter(sym, uni):
    try:
        e = fmp.statements(sym, limit=10)
        F = fmp.financials_from_fmp(e)
        f = fmp.fundamentals_from_fmp(sym, uni[sym], e, {})
        if not f or not f.get("price"):
            return None
        f["exchange"] = uni[sym].get("exchange")
        # Volume echange : un appel de plus par titre, mais c'est la seule mesure
        # directe de LIQUIDITE, et la regression du volume sur la taille ne peut se
        # calibrer sans lui.
        try:
            f["volume_dollars_median"] = fmp.volume_dollars_median(
                fmp.history_ohlcv(sym, days=90))
        except Exception:                              # noqa: BLE001
            f["volume_dollars_median"] = None
        motifs = valider(f, F, e)
        reparations = reparer(sym, f, F, e, motifs) if motifs else []
        if reparations:
            motifs = valider(f, F, e)
        exercices = len(set(e.get("income", {})) & set(e.get("balance", {})))
        return {
            "sym": sym, "fund": f, "F": F, "motifs": motifs,
            "reparations": reparations, "exercices": exercices,
            "reference": (exercices >= EXERCICES_MIN and not reparations
                          and not any("identite du bilan" in m for m in motifs)),
        }
    except Exception:                                  # noqa: BLE001
        return None


def grille(valeurs):
    """Les 99 centiles internes d'une serie — la table qu'on gele."""
    v = sorted(x for x in valeurs
               if x is not None and isinstance(x, float) and math.isfinite(x))
    if len(v) < N_MIN_GRILLE:
        return None
    out = []
    for i in range(1, CENTILES):
        pos = i / CENTILES * (len(v) - 1)
        bas = int(math.floor(pos))
        haut = min(bas + 1, len(v) - 1)
        out.append(round(v[bas] + (v[haut] - v[bas]) * (pos - bas), 6))
    return out


def regression(points):
    """Moindres carres de log(volume) sur log(capitalisation)."""
    if len(points) < N_MIN_GRILLE:
        return None
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    b = (n * sxy - sx * sy) / denom
    return {"a": round((sy - b * sx) / n, 6), "b": round(b, 6), "n": n}


def main(limite):
    uni = fmp.screener(["NASDAQ", "NYSE", "TSX", "TSXV", "OTC"])
    syms = sorted(uni)
    random.seed(20260730)
    if limite and limite < len(syms):
        syms = random.sample(syms, limite)
    print(f"univers {len(uni)} | echantillon de calibrage {len(syms)}")

    lignes = []
    with cf.ThreadPoolExecutor(max_workers=14) as ex:
        for i, r in enumerate(ex.map(lambda s: collecter(s, uni), syms), 1):
            if r:
                lignes.append(r)
            if i % 500 == 0:
                print(f"  {i}/{len(syms)}  retenues {len(lignes)}")

    reference = [l for l in lignes if l["reference"]]
    print(f"\nexploitables {len(lignes)} | population de REFERENCE {len(reference)} "
          f"({len(reference) / max(len(lignes), 1) * 100:.0f} %)")

    # --- Passe 0 : regression de liquidite, PAR PLACE DE COTATION ---------------
    # Elle doit preceder la mesure des quantiles : la dimension de liquidite n'est pas
    # le volume brut mais son RESIDU sur la taille, et le calibrage doit mesurer
    # EXACTEMENT la grandeur que la notation classera ensuite. Mesurer les centiles du
    # volume brut puis y chercher le rang d'un residu revient a confronter des
    # -log10(volume) de l'ordre de -6 a -9 a des residus de l'ordre de l'unite : tout
    # residu depasse alors tout centile, et l'univers ENTIER — Apple comprise, la
    # valeur la plus echangee au monde — ressort au 96e centile d'illiquidite.
    # --- Regression de liquidite, par place de cotation -------------------------
    par_place = {}
    for l in reference:
        f = l["fund"]
        vol, cap = f.get("volume_dollars_median"), f.get("market_cap")
        if vol and vol > 0 and cap and cap > 0:
            par_place.setdefault(f.get("exchange") or "?", []).append(
                (math.log10(cap * 1e9), math.log10(vol)))
    liquidite = {k: r for k, v in par_place.items() if (r := regression(v))}
    tous = [p for v in par_place.values() for p in v]
    if (r := regression(tous)):
        liquidite["global"] = r

    # --- Passe 1 : grilles de centiles, mesurees sur la reference ---------------
    # Une table par REGIME en plus des tables globale, sectorielle et
    # industrielle : plusieurs dimensions changent de NATURE selon le regime — la
    # solvabilite d'une banque est un coussin de fonds propres, celle d'un
    # industriel une couverture d'interets — et les ranger ensemble revient a
    # classer des metres contre des kilogrammes.
    par_dim = {cle: {"global": [], "secteurs": {}, "industries": {}, "regimes": {}}
               for cle, _n, _f, _niv in DIMENSIONS}
    for l in reference:
        s = mesurer(l["fund"], l["F"], l["motifs"], l["reparations"],
                    {"liquidite": liquidite})
        reg = s.pop("__regime__", None)
        sec = l["fund"].get("sector") or "?"
        ind = l["fund"].get("industry") or "?"
        for cle, (signal, _lib) in s.items():
            if signal is None or signal == math.inf:
                continue
            par_dim[cle]["global"].append(signal)
            par_dim[cle]["secteurs"].setdefault(sec, []).append(signal)
            par_dim[cle]["industries"].setdefault(ind, []).append(signal)
            if reg:
                par_dim[cle]["regimes"].setdefault(reg, []).append(signal)

    quantiles = {}
    for cle, paquets in par_dim.items():
        q = {"global": grille(paquets["global"])}
        q["secteurs"] = {k: g for k, v in paquets["secteurs"].items()
                         if (g := grille(v))}
        q["industries"] = {k: g for k, v in paquets["industries"].items()
                           if (g := grille(v))}
        q["regimes"] = {k: g for k, v in paquets["regimes"].items()
                        if (g := grille(v))}
        quantiles[cle] = q
        print(f"  {cle} : global n={len(paquets['global'])}, "
              f"{len(q['secteurs'])} secteurs, {len(q['industries'])} industries, "
              f"{len(q['regimes'])} regimes")

    cal = {
        "version": "1",
        "n_reference": len(reference),
        "n_echantillon": len(lignes),
        "quantiles": quantiles,
        "liquidite": liquidite,
        # POIDS UNIFORMES, ET C'EST UN CHOIX ASSUME. Aucune dimension n'entre avec un
        # poids de conviction : tant que la variable de resultat n'est pas construite
        # — effondrement durable du cours a douze mois — nous n'avons AUCUNE mesure
        # qui autoriserait a declarer une dimension plus predictive qu'une autre.
        # L'uniformite est la seule ponderation qui n'affirme rien.
        "poids": {cle: 1.0 for cle, _n, _f, _niv in DIMENSIONS},
        "origine_poids": "uniforme — non encore estimee sur une variable de resultat",
        # Idem : le melange avec le maillon faible reste a zero tant que son gain
        # n'est pas demontre.
        "lambda_maillon_faible": 0.0,
        # SEUIL DE DESACCORD SUR LA BASE ACTIONNAIRE — un ORDRE DE GRANDEUR.
        # J'ai d'abord tente de le MESURER, comme tout le reste : 99e centile de la
        # population de reference. Il est ressorti a un facteur 177, c'est-a-dire
        # qu'il ne detectait rien. La raison est instructive et vaut d'etre ecrite :
        # la queue de cette distribution est peuplee de rapports ADR LEGITIMES —
        # un certificat vaut couramment 10, 20, 100 ou 200 actions ordinaires — qui
        # ne sont pas des erreurs mais des changements d'unite. Un percentile ne peut
        # donc pas separer l'erreur de la structure : il mesure les deux ensemble.
        # Le facteur dix n'est pas ici une opinion sur la VALEUR — c'est le constat
        # que deux nombres ne designent pas la meme chose. Assume comme tel.
        "seuil_base_actionnaire": 1.0,
    }

    # --- Passe 2 : bornes de grades et niveaux de plafond ----------------------
    # Les scores sont recalcules avec les grilles qu'on vient de mesurer.
    # Un plafond n'est jamais choisi : son niveau est le score MEDIAN des societes qui
    # portent la modalite. Il exprime "on ne peut pas faire mieux que ce que fait
    # typiquement une societe dans cette situation", ce qui est une mesure et non un
    # choix. Le score est celui d'AVANT plafonnement, sans quoi la mesure se
    # mordrait la queue.
    from quantbench.risk.score import _modalites
    par_modalite = {}
    for l in lignes:
        r = noter(l["fund"], l["F"], l["motifs"], l["reparations"], cal=cal)
        sig = mesurer(l["fund"], l["F"], l["motifs"], l["reparations"], cal)
        for m in _modalites(l["fund"], l["F"], l["motifs"], sig):
            par_modalite.setdefault(m, []).append(r["score"])
    plafonds = {}
    for m, v in par_modalite.items():
        v.sort()
        plafonds[m] = round(v[len(v) // 2], 4)
        print(f"  plafond {m:28} n={len(v):5} niveau={plafonds[m]}")
    cal["plafonds"] = plafonds

    # BORNES DE GRADES : treize intervalles d'AMPLITUDE EGALE en log-odds, entre deux
    # bornes mesurees sur la population de reference.
    # Un decoupage par quantiles serait tentant mais il est faux ici : les plafonds
    # posent le score exactement sur leur niveau, creant des MASSES. Deux quantiles
    # consecutifs tombaient alors sur la meme valeur et un grade entier restait vide —
    # D+ n'a recu aucun titre au premier calibrage. L'amplitude egale y est insensible.
    # Le log-odds plutot que le score brut : il etire les extremites, ou se joue la
    # difference entre "fragile" et "condamne", et resserre le milieu ou elle est
    # sans consequence.
    scores_ref = sorted(noter(l["fund"], l["F"], l["motifs"], l["reparations"],
                              cal=cal)["score"] for l in reference)
    n = len(scores_ref)

    def _logit(s):
        s = min(max(s, 0.005), 0.995)
        return math.log(s / (1.0 - s))

    bas, haut = _logit(scores_ref[int(0.02 * (n - 1))]), _logit(scores_ref[int(0.98 * (n - 1))])
    pas = (haut - bas) / len(GRADES)
    bornes = [round(1.0 / (1.0 + math.exp(-(bas + i * pas))), 4)
              for i in range(1, len(GRADES))]
    cal["bornes_grades"] = bornes
    print(f"\nbornes de grades ({len(bornes)}) : {bornes}")

    with open(SORTIE, "w", encoding="utf-8") as f:
        json.dump(cal, f, ensure_ascii=False, indent=1)
    print(f"-> {SORTIE}")

    # --- Controle : distribution des grades sur l'echantillon ------------------
    from collections import Counter
    c = Counter(noter(l["fund"], l["F"], l["motifs"], l["reparations"],
                      cal=cal)["grade"] for l in lignes)
    print("\ndistribution des grades sur l'echantillon complet :")
    for g in GRADES:
        k = c.get(g, 0)
        print(f"  {g:3} {k:5} {'#' * int(60 * k / max(len(lignes), 1))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limite", type=int, default=4000)
    main(p.parse_args().limite)

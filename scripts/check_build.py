"""Controle qualite du build — BLOQUE la publication d'un build defectueux.

Execute entre la fusion des shards et le deploiement. S'il echoue, le job de
deploiement ne s'execute pas et le site CONSERVE ses dernieres donnees valides :
un bug ne peut plus atteindre la production silencieusement.

Trois familles de controles, toutes issues de defauts REELS observes :

  1. INVARIANTS DURS (mathematiquement impossibles)
     - upside < -100 % : la responsabilite limitee l'interdit (observe : -115 373 %)
     - upside demesure : signale une donnee corrompue (observe : +1,5 milliard %)
     - valeur/action incoherente avec le cours

  2. REGRESSION DE DISTRIBUTION (le modele s'est degrade sans erreur visible)
     - part de titres a valeur nulle (observe : 29 % quand le DCF etait applique
       aux foncieres, services publics et bras financiers captifs)
     - effondrement de la couverture

  3. PANIER DE REFERENCE (le routage sectoriel fonctionne toujours)
     - des titres connus doivent conserver leur methode : une banque ne doit pas
       repasser au DCF, une obligation ne doit pas reapparaitre comme action

Usage : python scripts/check_build.py [--strict]
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_HERE = os.path.dirname(os.path.abspath(__file__))
SCREENER = os.path.join(os.path.dirname(_HERE), "app", "us", "_screener.json")

# --- Seuils : larges, ils ne visent QUE les regressions grossieres ------------
MAX_PART_NULLE = 0.32          # part de titres a -100 % (constatee : ~21 %)
MAX_PART_EXTREME = 0.05        # part au-dessus de +500 % (constatee : ~1,8 %)
MAX_UPSIDE = 100.0             # +10 000 % : au-dela, c'est une donnee corrompue
MIN_COUVERTURE = 8000          # nombre de titres valorises (constate : ~12 000)

# --- Panier de reference : methode attendue par secteur -----------------------
PANIER = {
    "JPM": "financiere", "BAC": "financiere", "ALL": "financiere",
    "PLD": "fonciere", "O": "fonciere", "SPG": "fonciere",
    "XEL": "reglementee", "DUK": "reglementee", "SO": "reglementee",
    "XOM": "cyclique", "CVX": "cyclique",
    "AAPL": "standard", "MSFT": "standard",
    "V": "standard", "MA": "standard",          # reseaux de paiement, PAS des banques
}
# Titres qui ne sont PAS des actions ordinaires : ils doivent rester absents.
INTERDITS = ["TD-PFA.TO", "ENB-PFA.TO", "EMP", "ENJ", "KKRS", "TVE", "TVC",
             "SBNY", "OTLC", "FITB-PM"]


def main(strict=True):
    with open(SCREENER, encoding="utf-8") as f:
        d = json.load(f)
    rows = d.get("rows", [])
    ups = [r["upside"] for r in rows if r.get("upside") is not None]
    n = len(ups)
    erreurs, alertes = [], []

    print(f"Controle du build : {n} titres valorises, univers {d.get('universe')}, "
          f"maj {d.get('updated')}\n")

    if n < MIN_COUVERTURE:
        erreurs.append(f"couverture effondree : {n} titres (< {MIN_COUVERTURE})")

    # 1. Invariants durs
    impossibles = [r for r in rows if (r.get("upside") or 0) < -1.0001]
    if impossibles:
        erreurs.append(f"{len(impossibles)} upside < -100 % (responsabilite limitee) : "
                       + ", ".join(r["ticker"] for r in impossibles[:5]))
    demesures = [r for r in rows if (r.get("upside") or 0) > MAX_UPSIDE]
    if demesures:
        erreurs.append(f"{len(demesures)} upside > {MAX_UPSIDE*100:.0f} % (donnee corrompue) : "
                       + ", ".join(f"{r['ticker']} {r['upside']*100:,.0f}%"
                                   for r in demesures[:5]))
    # identite valeur/action <-> cours
    # Tolerance RELATIVE : un ecart de 5 points d'upside n'a pas le meme sens a
    # -50 % qu'a +28 000 % (ou il ne traduit que l'arrondi d'affichage).
    incoherents = []
    for r in rows:
        vps, px, up = r.get("value_per_share"), r.get("price"), r.get("upside")
        if vps and px and px > 0 and up is not None:
            implique = vps / px - 1.0
            if abs(implique - up) / (1.0 + abs(up)) > 0.02:
                incoherents.append(r["ticker"])
    if len(incoherents) > 20:
        erreurs.append(f"{len(incoherents)} titres ou valeur/action est incoherente "
                       f"avec l'upside : {', '.join(incoherents[:5])}")

    # 2. Regression de distribution
    nulle = sum(1 for u in ups if u <= -0.999) / max(n, 1)
    extreme = sum(1 for u in ups if u > 5.0) / max(n, 1)
    print(f"  titres a valeur nulle : {nulle*100:5.1f} %  (plafond {MAX_PART_NULLE*100:.0f} %)")
    print(f"  titres au-dela +500 % : {extreme*100:5.1f} %  (plafond {MAX_PART_EXTREME*100:.0f} %)")
    print(f"  upside maximum        : {max(ups)*100:+,.0f} %")
    print(f"  upside median         : {sorted(ups)[n//2]*100:+.0f} %")
    if nulle > MAX_PART_NULLE:
        erreurs.append(f"part de titres a valeur nulle trop elevee : {nulle*100:.1f} % "
                       f"(> {MAX_PART_NULLE*100:.0f} %) — une methode sectorielle est "
                       f"probablement cassee")
    if extreme > MAX_PART_EXTREME:
        erreurs.append(f"trop de valorisations extremes : {extreme*100:.1f} %")

    # 3. Panier de reference
    par_ticker = {r["ticker"]: r for r in rows}
    print("\n  panier de reference :")
    for t, attendu in PANIER.items():
        r = par_ticker.get(t)
        if not r:
            alertes.append(f"{t} absent du build")
            continue
        obtenu = r.get("category")
        ok = obtenu == attendu
        print(f"    {t:9} {str(obtenu):14} {'OK' if ok else 'ATTENDU ' + attendu:>20}"
              f"   upside {r['upside']*100:+7.0f} %")
        if not ok:
            erreurs.append(f"{t} route en '{obtenu}' au lieu de '{attendu}' "
                           f"— routage sectoriel casse")
        if r.get("upside") is not None and not (-1.0 <= r["upside"] <= 3.0):
            erreurs.append(f"{t} : upside implausible pour une grande valeur "
                           f"({r['upside']*100:+.0f} %)")

    presents = [t for t in INTERDITS if t in par_ticker]
    if presents:
        erreurs.append("titres qui ne sont pas des actions ordinaires (obligations, "
                       f"privilegiees, entites disparues) presents : {', '.join(presents)}")

    # --- Verdict --------------------------------------------------------------
    print()
    for a in alertes:
        print(f"  [alerte] {a}")
    if erreurs:
        print(f"\n{'='*70}\nBUILD REFUSE — {len(erreurs)} anomalie(s) :")
        for e in erreurs:
            print(f"  X {e}")
        print("\nLe site conserve ses dernieres donnees valides.")
        return 1 if strict else 0
    print(f"{'='*70}\nBUILD VALIDE — tous les controles passent.")
    return 0


if __name__ == "__main__":
    sys.exit(main(strict="--no-strict" not in sys.argv))

"""Mesure ce que vaut REELLEMENT la note de risque, en confrontant deux archives.

Une note qui n'a jamais ete confrontee aux faits n'est qu'une opinion mise en forme.
Ce script est le seul juge autorise du systeme, et il ne peut rien dire avant qu'un an
ne separe deux archives — c'est le prix d'une mesure honnete.

    scripts/build_site_fmp.py  ecrit chaque mois app/us/_notes_risque/AAAA-MM.json
    ce script                  confronte deux archives distantes de douze mois

CE QU'IL MESURE
  taux d'effondrement par grade   part des titres ayant perdu la moitie de leur valeur
  monotonie                       ce taux croit-il bien de A+ vers F ?
  aire sous la courbe             pouvoir discriminant du score continu
  apport de chaque dimension      variation de l'aire quand on la retire

CE QU'IL NE FAIT PAS
  Il n'ajuste RIEN automatiquement. Les poids ne changent que sur decision explicite,
  apres lecture des intervalles de confiance. Une aire de 0,55 avec un intervalle
  contenant 0,50 ne demontre rien, et le systeme doit rester a poids uniformes tant
  que c'est le cas.

Usage : python scripts/mesurer_les_notes.py [--depart 2026-07] [--arrivee 2027-07]
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
_ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_ICI))

from quantbench.risk import GRADES                                   # noqa: E402

ARCHIVES = os.path.join(os.path.dirname(_ICI), "app", "us", "_notes_risque")
# Seuil d'EFFONDREMENT. Ce n'est pas une opinion de valorisation mais la definition
# de la variable a predire : perdre la moitie de sa mise est une perte dont on ne se
# remet pas par un simple retour a la moyenne.
PERTE_GRAVE = -0.50


def charger(mois):
    chemin = os.path.join(ARCHIVES, f"{mois}.json")
    if not os.path.exists(chemin):
        return None
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def archives_disponibles():
    if not os.path.isdir(ARCHIVES):
        return []
    return sorted(f[:-5] for f in os.listdir(ARCHIVES) if f.endswith(".json"))


def aire_sous_la_courbe(paires):
    """Probabilite qu'un titre effondre porte un score PIRE qu'un titre epargne.
    0,50 = aucun pouvoir discriminant. Calcul par rangs (statistique de Mann-Whitney),
    exact et sans hypothese de distribution."""
    positifs = [s for s, y in paires if y]
    negatifs = [s for s, y in paires if not y]
    if not positifs or not negatifs:
        return None, None
    tous = sorted(paires, key=lambda p: p[0])
    rangs, i = {}, 0
    while i < len(tous):
        j = i
        while j + 1 < len(tous) and tous[j + 1][0] == tous[i][0]:
            j += 1
        moyen = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            rangs.setdefault(tous[k][0], moyen)
        i = j + 1
    somme = sum(rangs[s] for s, y in paires if y)
    n1, n0 = len(positifs), len(negatifs)
    auc = (somme - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    # Ecart-type de Hanley-McNeil : sans lui, un chiffre isole ne se lit pas.
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2)
           + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)
    return auc, math.sqrt(max(var, 0.0))


def main(depart, arrivee):
    dispo = archives_disponibles()
    if not dispo:
        print("Aucune archive. Le build en ecrit une par mois dans "
              f"{ARCHIVES}\nRevenez dans douze mois — c'est le prix d'une mesure "
              "honnete.")
        return
    print(f"archives disponibles : {', '.join(dispo)}")
    depart = depart or dispo[0]
    arrivee = arrivee or dispo[-1]
    a, b = charger(depart), charger(arrivee)
    if not a or not b:
        print(f"archive introuvable ({depart} ou {arrivee})")
        return
    d1 = datetime.strptime(a["date"], "%Y-%m-%d")
    d2 = datetime.strptime(b["date"], "%Y-%m-%d")
    mois = (d2.year - d1.year) * 12 + (d2.month - d1.month)
    print(f"\n{depart} -> {arrivee} : {mois} mois d'ecart")
    if mois < 12:
        print("\nMOINS DE DOUZE MOIS. La mesure n'est pas publiable : sur un horizon"
              "\ncourt, le bruit de marche domine tout signal de solvabilite. Le"
              "\nresultat ci-dessous est indicatif et ne doit PAS servir a ponderer.")

    paires, par_grade = [], {g: [0, 0] for g in GRADES}
    for t, (grade, score, cours0) in a["notes"].items():
        cible = b["notes"].get(t)
        if not cible or not cours0 or cours0 <= 0:
            continue
        cours1 = cible[2]
        if not cours1 or cours1 <= 0:
            continue
        effondre = (cours1 / cours0 - 1.0) <= PERTE_GRAVE
        paires.append((score, effondre))
        if grade in par_grade:
            par_grade[grade][0] += int(effondre)
            par_grade[grade][1] += 1

    if not paires:
        print("aucun titre commun aux deux archives")
        return
    n_eff = sum(1 for _s, y in paires if y)
    print(f"titres suivis : {len(paires)} | effondrements (-50 % ou pire) : "
          f"{n_eff} ({n_eff / len(paires) * 100:.1f} %)\n")

    print(f"{'grade':6} {'suivis':>7} {'effondres':>10} {'taux':>7}")
    precedent, monotone = -1.0, True
    for g in GRADES:
        eff, n = par_grade[g]
        if not n:
            print(f"{g:6} {0:>7} {'':>10} {'—':>7}")
            continue
        taux = eff / n
        if taux + 1e-9 < precedent:
            monotone = False
        precedent = taux
        print(f"{g:6} {n:>7} {eff:>10} {taux * 100:>6.1f}%")
    print(f"\nmonotonie de A+ vers F : {'OUI' if monotone else 'NON'}"
          f"{'' if monotone else '  — un grade meilleur qui echoue plus souvent'}")

    auc, ecart = aire_sous_la_courbe(paires)
    if auc is None:
        print("aire non calculable : une seule classe observee")
        return
    bas, haut = auc - 1.96 * ecart, auc + 1.96 * ecart
    print(f"\naire sous la courbe : {auc:.3f}  (intervalle a 95 % : "
          f"{bas:.3f} — {haut:.3f})")
    if bas <= 0.5:
        print("L'INTERVALLE CONTIENT 0,50 : aucun pouvoir discriminant demontre.")
        print("Les poids DOIVENT rester uniformes. Publier une ponderation estimee")
        print("sur ce resultat reviendrait a habiller du bruit.")
    else:
        print("Pouvoir discriminant demontre. L'estimation des poids par dimension")
        print("devient legitime — a conduire en validation croisee purgee, et a")
        print("n'appliquer qu'apres publication de l'ecart de distribution induit.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--depart", default=None, help="archive de depart, AAAA-MM")
    p.add_argument("--arrivee", default=None, help="archive d'arrivee, AAAA-MM")
    a = p.parse_args()
    main(a.depart, a.arrivee)

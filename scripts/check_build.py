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
from collections import Counter
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
SCREENER = os.path.join(os.path.dirname(_HERE), "app", "us", "_screener.json")

from quantbench.risk import GRADES                                    # noqa: E402

# Couverture minimale de la note de risque. Ce n'est pas un seuil de qualite mais un
# detecteur de PANNE : la note est calculable pour TOUT titre valorise — meme sans
# donnees, la dimension de confiance en tient lieu — donc une couverture effondree ne
# peut signifier qu'une chose, le fichier de calibrage est absent ou illisible.
MIN_PART_NOTEE = 0.95          # POSE, avec une marge large : detecteur de panne
# Un grade concentrant la moitie de l'univers signale un calibrage degenere : bornes
# ecrasees, table de quantiles vide, ou signal devenu constant. C'est exactement ce
# qui s'est produit quand la dimension de liquidite classait TOUT l'univers au 96e
# centile, Apple comprise, parce que le calibrage mesurait une grandeur differente de
# celle que la notation classait.
MAX_PART_UN_GRADE = 0.50       # POSE : une moitie de l'univers dans un grade

# --- Seuils : larges, ils ne visent QUE les regressions grossieres ------------
MAX_PART_NULLE = 0.32          # POSE au-dessus du constate (~24 %)
MAX_PART_EXTREME = 0.05        # POSE au-dessus du constate (~1,7 %)
# SEUIL D'INSPECTION, ET NON DE CORRUPTION. Le commentaire precedent affirmait
# "au-dela, c'est une donnee corrompue" : c'est FAUX, et la mesure le refute. La
# pente log-log du nombre de titres au-dela d'un upside x passe continument de -0,73
# a -3,49 : la queue est une loi de puissance, sans aucune discontinuite ou placer une
# frontiere entre "extreme" et "corrompu". Le point de coupe est donc CONVENTIONNEL et
# assume comme tel — il declenche une inspection, il ne diagnostique rien.
MAX_UPSIDE = 100.0             # +10 000 %, point de coupe conventionnel assume
MIN_COUVERTURE = 8000          # nombre de titres valorises (constate : ~15 500)

# NOMBRE DE DEMESUREES TOLEREES — mesure, et c'etait tout le probleme.
#
# La valeur precedente etait 10, POSEE. Or la distribution AU REPOS de ce compteur —
# le meme modele rejoue sur 250 seances de cours reels, sans aucune modification du
# code — s'etend de 8 a 19, de mediane 15 et d'ecart-type 3,5. Le seuil se trouvait
# donc SOUS LA MEDIANE de sa propre distribution au repos : le garde-fou refusait un
# build identique a lui-meme environ neuf fois sur dix, et onze des quarante derniers
# deploiements ont echoue, tous a cette etape. Un garde-fou qui bloque le cas normal
# ne protege plus de rien, il empeche seulement la publication.
#
# Le niveau retenu est le quantile 99,5 % de cette distribution au repos, soit
# mediane + 2,58 ecarts-types, arrondi au-dessus du maximum observe (19).
# A REMESURER a chaque revision du modele : le compteur au repos se deplace avec lui.
MAX_NB_DEMESURES = 25          # mesure : au repos 8-19, mediane 15, ecart-type 3,5

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
    # Une poignee de nano-caps aberrantes ne doit pas bloquer un build par ailleurs
    # sain : on BLOQUE si le defaut est nombreux ou s'il touche une societe d'une
    # taille significative (la, c'est le modele qui derape, pas une donnee isolee).
    demesures = [r for r in rows if (r.get("upside") or 0) > MAX_UPSIDE]
    grosses = [r for r in demesures if (r.get("market_cap") or 0) >= 1.0]
    if demesures:
        msg = (f"{len(demesures)} upside > {MAX_UPSIDE*100:.0f} % : "
               + ", ".join(f"{r['ticker']} {r['upside']*100:,.0f}%" for r in demesures[:5]))
        (erreurs if (len(demesures) > MAX_NB_DEMESURES or grosses) else alertes).append(msg)
    # DEUX MESURES JOURNALISEES SANS BLOQUER. Une regression du MODELE deplace la
    # capitalisation concernee, pas seulement le compte : une poignee de nano-caps
    # corrompues pese 1e-8 de l'univers, une route cassee en pese des pour cent.
    # Elles ne deviendront bloquantes qu'apres observation de leur propre distribution
    # au repos a travers des revisions REELLES du modele — pas des rejeux de cours,
    # auxquels elles sont par construction insensibles.
    cap_univers = sum(r.get("market_cap") or 0.0 for r in rows)
    cap_demesuree = sum(r.get("market_cap") or 0.0 for r in demesures)
    print(f"  capitalisation demesuree : {cap_demesuree:,.3f} Md$ sur "
          f"{cap_univers:,.0f} Md$ ({cap_demesuree / max(cap_univers, 1e-9):.2e} de l'univers)")
    if demesures:
        plus_grosse = max(demesures, key=lambda r: r.get("market_cap") or 0.0)
        print(f"  plus grosse demesuree    : {plus_grosse['ticker']} "
              f"{plus_grosse.get('market_cap') or 0:,.3f} Md$")
        # Une route CASSEE concentre ses degats sur une seule methode ; des donnees
        # individuellement corrompues s'eparpillent. C'est le vrai discriminant.
        par_route = Counter(r.get("category") for r in demesures)
        route, n_route = par_route.most_common(1)[0]
        print(f"  concentration par route  : {n_route}/{len(demesures)} en '{route}'")
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

    # 4. NOTE DE RISQUE
    # Une note absente ne se voit pas : la fiche affiche simplement un tiret, et rien
    # n'alerte. Les controles ci-dessous portent donc sur la COUVERTURE et sur la
    # FORME de la distribution, jamais sur la note d'un titre en particulier.
    notes = [r.get("note_risque") for r in rows]
    notees = [g for g in notes if g]
    print(f"\n  note de risque : {len(notees)}/{len(rows)} titres notes "
          f"({len(notees) / max(len(rows), 1) * 100:.1f} %)")
    if rows and len(notees) / len(rows) < MIN_PART_NOTEE:
        erreurs.append(f"seulement {len(notees) / len(rows) * 100:.0f} % des titres "
                       f"portent une note de risque — le calibrage est probablement "
                       f"absent ou illisible")
    if notees:
        c = Counter(notees)
        print("    " + "  ".join(f"{g}:{c.get(g, 0)}" for g in GRADES))
        # UN SEUL GRADE CONCENTRANT LA MOITIE DE L'UNIVERS signale un calibrage
        # degenere — bornes ecrasees, table de quantiles vide, ou signal constant.
        pire = max(c.values()) / len(notees)
        if pire > MAX_PART_UN_GRADE:
            domine = max(c, key=c.get)
            erreurs.append(f"le grade {domine} concentre {pire * 100:.0f} % des titres "
                           f"— calibrage degenere")
        # Les grandes valeurs du panier ne peuvent pas ressortir en risque extreme :
        # si Apple ou Microsoft tombent en D ou F, une dimension est cassee.
        for t in ("AAPL", "MSFT", "V", "JPM", "XOM", "KO"):
            g = (par_ticker.get(t) or {}).get("note_risque")
            if g and g[0] in ("D", "F"):
                erreurs.append(f"{t} note {g} — une dimension de risque est cassee")

    # Aucune valeur non finie ne doit atteindre le JSON : `Infinity` et `NaN` ne sont
    # pas du JSON valide et rendent le fichier ENTIER illisible dans un navigateur,
    # alors que Python les relit sans broncher.
    # Le test porte sur la SYNTAXE et non sur le texte : chercher la sous-chaine
    # "Infinity" bloquerait le deploiement des qu'un emetteur s'appelle "Infinity
    # Stone Ventures Corp" — il y en a un dans l'univers. `parse_constant` n'est
    # appele que sur les litteraux non finis, jamais sur le contenu d'une chaine.
    def _non_fini(litteral):
        raise ValueError(litteral)

    for chemin in (SCREENER,):
        try:
            with open(chemin, encoding="utf-8") as f:
                json.load(f, parse_constant=_non_fini)
        except ValueError as ex:
            erreurs.append(f"valeur non finie ({ex}) dans {os.path.basename(chemin)} "
                           f"— le navigateur echouera a lire le fichier entier")

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

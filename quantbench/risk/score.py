"""Agregation des dimensions en une note de A+ a F.

TROIS GELS, qui font la difference entre une NOTE et un CLASSEMENT :

  1. les tables de quantiles de chaque dimension sont mesurees une fois sur une
     population de reference datee, puis LUES par le build ;
  2. les bornes de grade sont mesurees en meme temps, sur la meme population ;
  3. les niveaux de plafond aussi.

Sans ces gels, tous les rangs seraient calcules sur la coupe du jour, leurs marges
seraient uniformes PAR CONSTRUCTION, et la distribution des notes serait invariante :
dans une crise ou toutes les couvertures d'interets s'effondrent ensemble, personne ne
serait degrade. Avec eux, la distribution des grades a le DROIT de se deplacer, ce qui
est le comportement attendu d'une notation et ce qui la rend surveillable.

Benefice technique decisif : aucune statistique de l'univers du jour n'est requise,
donc les cinq shards du build restent independants et aucune passe d'agregation
supplementaire n'est necessaire.
"""

from __future__ import annotations

import json
import math
import os

from .dimensions import DIMENSIONS, mesurer

# Treize grades. Le NOMBRE est une contrainte d'affichage assumee comme telle ; tout
# le reste — les bornes qui les separent — est mesure.
GRADES = ("A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F")

_ICI = os.path.dirname(os.path.abspath(__file__))
_FICHIER = os.path.join(_ICI, "risk_calibration.json")


def _charger():
    try:
        with open(_FICHIER, encoding="utf-8") as f:
            return json.load(f)
    except Exception:                                  # noqa: BLE001
        return None


_CAL = _charger()


def rang(cle, signal, cal, secteur=None, industrie=None, niveau="univers",
         regime=None):
    """Rang percentile du signal dans la table GELEE — la part de la population de
    reference qui fait STRICTEMENT mieux.

    Le percentile est UNILATERAL, et ce n'est pas un detail : avec un percentile a
    rang moyen, une societe irreprochable sur une dimension a mode propre — zero
    dilution, zero motif de validation, ecart de resultat nul — recevrait 0,50 alors
    qu'elle merite 0. Une societe parfaite ressortait ainsi a la moyenne."""
    if signal is None:
        return None
    # Les MODALITES bornent l'echelle sans passer par la table : -inf designe le
    # meilleur cas possible de la dimension (aucune dette, autofinancee, aucune
    # dilution), +inf le pire (aucun chiffre d'affaires). Les faire passer par les
    # centiles les melangerait aux MESURES : une societe sans charge d'interets se
    # retrouvait classee plus risquee qu'une societe couvrant les siennes trois fois.
    if signal == math.inf:
        return 1.0
    if signal == -math.inf:
        return 0.0
    tables = (cal or {}).get("quantiles", {}).get(cle) or {}
    grille = None
    # TABLE PROPRE AU REGIME, quand le signal CHANGE DE NATURE avec lui.
    #
    # La solvabilite d'un industriel est une couverture d'interets, celle d'une
    # banque un coussin de fonds propres tangibles : deux grandeurs sans commune
    # mesure, rangees dans la meme table de centiles. Consequence mesuree : toute
    # l'etendue plausible d'un bilan bancaire — de 20 % a 1 % de fonds propres —
    # se comprimait sur 0,101 d'echelle, ENTIEREMENT au-dessus de la mediane de
    # risque. La dimension ne separait plus rien la ou la solvabilite est
    # precisement la question, et penalisait toutes les banques sans distinction.
    #
    # La table du regime prime donc sur celle du secteur et sur la globale. Son
    # absence fait retomber sur le comportement d'avant, si bien qu'un calibrage
    # ancien reste lisible.
    if regime:
        grille = (tables.get("regimes") or {}).get(regime)
    if grille:
        pass
    elif niveau == "secteur" and secteur:
        grille = tables.get("secteurs", {}).get(secteur)
    elif niveau == "industrie" and industrie:
        grille = (tables.get("industries", {}).get(industrie)
                  or tables.get("secteurs", {}).get(secteur or ""))
    if not grille:
        grille = tables.get("global")
    if not grille:
        return None
    # `grille` est la liste des centiles du signal, du plus faible au plus fort.
    n = len(grille)
    bas = 0
    for i, borne in enumerate(grille):
        if signal > borne:
            bas = i + 1
        else:
            break
    return min(1.0, max(0.0, bas / n))


def _modalites(fund, F, motifs, mesures, seuil_base=None, reg=None):
    """Faits VERIFIABLES qui plafonnent la note. Ce ne sont pas des jugements : chacun
    est une observation qu'aucune qualite par ailleurs ne compense."""
    from ..valuation.route import valeur_de_liquidation
    from .dimensions import consommation_selon_le_regime
    out = []
    be = fund.get("book_equity")
    # MEME mesure que la dimension d'autonomie : corrigee du capex de CROISSANCE.
    # Les plafonds lisaient le flux libre BRUT, si bien qu'un service public regule —
    # qui finance ses investissements par la dette, par construction, avec un
    # rendement garanti par le regulateur — etait declare a court d'autonomie. Duke
    # Energy et Southern Company ressortaient a D+, E.ON aussi.
    conso = consommation_selon_le_regime(fund, reg)
    # "NE CONSOMME PAS" N'EST PAS "GENERE". Les deux plafonds ci-dessous etaient
    # gardes par `conso > 0`, et la consommation vaut zero dans DEUX cas tres
    # differents : la societe encaisse, ou son tableau de flux est ABSENT. 1812
    # Brewing porte 13,7 M$ de passif pour 5,0 M$ d'actif et des fonds propres de
    # -8,7 M$, sans aucun tableau de flux publie : elle echappait aux deux plafonds
    # et ressortait notee B.
    # On exige donc une PREUVE de generation de tresorerie pour epargner une societe,
    # au lieu de se contenter de l'absence de preuve du contraire.
    cfo = fund.get("cfo")
    capex = fund.get("capex")
    genere = (cfo is not None and conso <= 0)
    # PASSIF NON COUVERT PAR L'ACTIF REALISABLE.
    # La condition portait sur `passif > actif` — trop stricte : la decote de
    # realisation peut rendre l'actif insuffisant alors meme que sa valeur COMPTABLE
    # depasse le passif. Une societe pre-revenu ressortait ainsi notee B avec une
    # equite valorisee a zero.
    # Mais la retirer sans rien mettre a la place serait pire : la valeur de
    # liquidation d'APPLE est nulle — 0,65 x 359 Md$ d'actif ne couvre pas 285 Md$ de
    # passif — et toute grande societe serait plafonnee. Une liquidation ne se pose
    # que pour une societe qui NE GENERE PAS de tresorerie ; celle qui encaisse n'est
    # pas liquidee, quelle que soit la decote theorique sur son actif.
    if valeur_de_liquidation(fund) <= 0 and not genere:
        out.append("passif_non_couvert")
    if be is not None and be <= 0 and not genere:
        out.append("fonds_propres_absorbes")
    if conso > 0:
        cash = max(fund.get("cash") or 0.0, 0.0)
        if cash / conso < 1.0:
            out.append("moins_d_un_an_d_autonomie")
    for m in (motifs or []):
        if "identite du bilan" in m or "taux de change" in m:
            out.append("bilan_non_verifiable")
            break
    if not ((F or {}).get("years")):
        out.append("aucun_exercice_exploitable")
    # DEUX SOURCES QUI NE PARLENT PAS DE LA MEME SOCIETE.
    # Le nombre d'actions deduit du marche et celui depose aux comptes divergent
    # parfois d'un ORDRE DE GRANDEUR : All American Gold affiche 96,6 millions de
    # titres implicites pour 1,877 MILLIARD deposes, CTR Investments 140 millions
    # pour 3,7 milliards. La capitalisation ne porte alors pas sur la meme entite que
    # les comptes, et tout rapport entre les deux est vide de sens.
    # On ne REECRIT rien — aucune source ne prouve que l'autre a tort, et les
    # corrections tentees allaient neuf fois sur douze dans le sens qui gonfle la
    # valeur. Mais un desaccord de cette ampleur PLAFONNE la note : ce n'est plus une
    # societe risquee, c'est une societe dont on ne sait pas de quoi on parle.
    # Le seuil est MESURE (percentile eleve de la distribution du desaccord, ecrit
    # dans le calibrage) et non pose : le repli ne sert qu'en son absence.
    publiees, deduites = fund.get("actions_publiees"), fund.get("shares")
    if publiees and deduites and publiees > 0 and deduites > 0:
        if abs(math.log10(deduites / publiees)) > (seuil_base or 1.0):
            out.append("base_actionnaire_incoherente")
    return out


def _debiaiser_le_maillon_faible(maximum: float, k: int) -> float:
    """Transformation integrale de probabilite du MAXIMUM de k rangs uniformes.

    P(max <= m) = m**k. Sous l'hypothese nulle — des rangs independants et
    uniformes — cette quantite suit donc une loi uniforme, ce qui est exactement
    la propriete recherchee : une societe aux comptes riches ne doit pas etre
    penalisee d'avoir plus de dimensions mesurables.

    Expose a part pour etre TESTABLE sur sa propriete, et non sur son texte.
    """
    return float(maximum) ** int(k)


def noter(fund, F=None, motifs=None, reparations=None, cal=None):
    """Note de risque complete. Retourne toujours un resultat : une note doit exister
    pour TOUS les titres, y compris ceux dont les donnees sont pauvres — l'incertitude
    sur la donnee est elle-meme un risque, et c'est la dimension D7 qui la porte."""
    cal = cal if cal is not None else _CAL
    signaux = mesurer(fund, F, motifs, reparations, cal)
    reg = signaux.pop("__regime__")

    poids = (cal or {}).get("poids") or {}
    secteur, industrie = fund.get("sector"), fund.get("industry")
    detail, total, somme_poids, rangs = [], 0.0, 0.0, []
    for cle, nom, _f, niveau in DIMENSIONS:
        signal, libelle = signaux.get(cle, (None, None))
        r = rang(cle, signal, cal, secteur, industrie, niveau, regime=reg)
        w = float(poids.get(cle, 1.0))
        # LES INFINIS NE DOIVENT JAMAIS SORTIR DU MODULE. `json.dumps` de Python les
        # ecrit `Infinity` et `-Infinity`, qui ne sont PAS du JSON valide : le
        # navigateur echoue alors a lire la fiche ENTIERE, pas seulement la note, et
        # le message affiche est "profil indisponible" — un defaut invisible cote
        # serveur, ou Python relit sans broncher ce qu'il vient d'ecrire.
        # Ici l'infini est une MODALITE, deja traduite en rang : le signal chiffre
        # n'a pas a la porter.
        detail.append({"cle": cle, "nom": nom, "libelle": libelle,
                       "signal": (None if signal is None or not math.isfinite(signal)
                                  else round(signal, 4)),
                       "rang": None if r is None else round(r, 3),
                       "poids": w, "niveau": niveau})
        if r is not None and w > 0:
            total += w * r
            somme_poids += w
            rangs.append(r)

    if not rangs:
        # Aucune dimension mesurable : la note EXISTE quand meme, et elle dit
        # exactement cela. C'est la branche de repli, plafonnee.
        return _finaliser(1.0, ["aucune_dimension_mesurable"], detail, reg, cal,
                          repli=True)

    moyenne = total / somme_poids
    # MAILLON FAIBLE, DEBIAISE. Une societe ne fait pas defaut "en moyenne", elle fait
    # defaut par son point le plus faible. Mais le maximum de k rangs uniformes vaut
    # k/(k+1) en esperance — 0,75 a trois dimensions, 0,90 a huit : une societe aux
    # comptes riches serait mecaniquement moins bien notee qu'une societe opaque, a
    # risque egal. On corrige par la transformation integrale de probabilite.
    #
    # LA TRANSFORMATION EST m**k, ET NON 1-(1-m)**k. Cette derniere est la fonction
    # de repartition du MINIMUM de k uniformes, appliquee au MAXIMUM : au lieu de
    # debiaiser, elle sature. Mesure sur 400 000 tirages, la mediane obtenue valait
    # 0,991 a trois dimensions et 1,000 a partir de cinq, la ou une transformation
    # debiaisee doit rendre une loi UNIFORME, donc une mediane de 0,5. Une societe
    # aux comptes riches se voyait donc attribuer un maillon faible pratiquement
    # maximal quels que soient ses rangs — l'exact contraire du but annonce.
    #
    # La bonne transformation est celle du maximum : P(max <= m) = m**k, uniforme
    # sous l'hypothese nulle. Verifie : mediane 0,500 pour k de 2 a 8.
    #
    # Sans consequence sur les notes PUBLIEES a ce jour, `lambda_maillon_faible`
    # valant zero dans le calibrage : le terme etait calcule puis jete. C'etait une
    # mine, pas un incendie — et elle aurait explose le jour ou quelqu'un aurait
    # donne du poids a ce terme.
    k = len(rangs)
    maximum = max(rangs)
    debiaise = _debiaiser_le_maillon_faible(maximum, k)
    lam = float((cal or {}).get("lambda_maillon_faible", 0.0))
    score = (1.0 - lam) * moyenne + lam * debiaise
    return _finaliser(score, _modalites(fund, F, motifs, signaux,
                                       (cal or {}).get("seuil_base_actionnaire"), reg),
                      detail, reg, cal)


def _finaliser(score, modalites, detail, reg, cal, repli=False):
    bornes = (cal or {}).get("bornes_grades")
    plafonds = (cal or {}).get("plafonds") or {}
    applique = []
    for m in modalites:
        niveau = plafonds.get(m)
        if niveau is not None and niveau > score:
            score = niveau
            applique.append(m)
        elif niveau is not None:
            applique.append(m)
    if repli:
        score = max(score, plafonds.get("aucune_dimension_mesurable", score))

    if bornes:
        i = sum(1 for b in bornes if score > b)
        grade = GRADES[min(i, len(GRADES) - 1)]
    else:
        grade = None                                  # calibrage absent
    return {
        "grade": grade,
        "score": round(float(score), 4),
        "regime": reg,
        "dimensions": detail,
        "plafonds_appliques": applique,
        "calibrage": (cal or {}).get("version"),
    }


__all__ = ["GRADES", "noter", "rang"]

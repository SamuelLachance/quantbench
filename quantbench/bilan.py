"""Reconnaitre une activite DE BILAN a sa structure, et nulle part ailleurs.

Une banque, un assureur, un preteur : leur dette est la MATIERE PREMIERE et non un
financement. La notion de valeur d'entreprise n'a chez eux aucun sens, et les
valoriser par actualisation des flux libres produit des absurdites — Apollo,
etiquetee « Asset Management » mais detenant l'assureur Athene, ressortait a
+532 %. Le libelle sectoriel du fournisseur ne suffit pas : FDCTech porte
« Financial Services » et vend des logiciels.

La signature est donc structurelle : des fonds propres MINCES rapportes a l'actif
(banques 8 a 10 %, assureurs 5 a 15 %) et un actif LOURD rapporte au chiffre
d'affaires.

POURQUOI CE MODULE EXISTE. Cette regle etait ecrite TROIS FOIS — dans le routage
de valorisation, dans la notation du risque, et dans la construction des entrees
du modele — et les trois copies avaient diverge. Table de verite mesuree avant
unification :

    cas                        routage   notation   entrees du modele
    banque ordinaire            oui        oui           oui
    fonds propres NULS          non        OUI           OUI
    fonds propres NEGATIFS      OUI        OUI           OUI
    fonds propres absents       non        non           OUI

Une meme societe changeait donc de nature selon le module qui la regardait.

LE DEFAUT DE FOND, commun aux trois : des fonds propres NEGATIFS satisfont
« fonds propres inferieurs a 15 % de l'actif ». Une societe dont les pertes ont
absorbe le capital etait donc reconnue comme une banque et valorisee par le
rendement excedentaire. Mesure sur 385 societes tirees au hasard dans l'univers :
8 cas, dont Hyliion Holdings — 1,31 Md$ d'actif pour MOINS 4,06 Md$ de fonds
propres et 4 M$ de chiffre d'affaires — et Edenor. Des fonds propres negatifs sont
la signature d'une DETRESSE, que le routage traite ailleurs et autrement ; ils ne
sont pas la signature d'une activite de bilan, qui suppose des fonds propres
minces mais bien REELS.

La condition est donc `0 < fonds propres / actif < 15 %`, borne basse comprise.
"""
from __future__ import annotations

__all__ = ["est_une_activite_de_bilan", "PART_FONDS_PROPRES", "ACTIF_SUR_CA"]

# Part maximale des fonds propres dans l'actif au-dela de laquelle la structure
# n'est plus celle d'un bilan. CONSTANTE POSEE, ancree sur des faits externes :
# les banques publient 8 a 10 % de fonds propres sur actif, les assureurs 5 a 15 %.
# Elle n'est pas mesuree sur notre univers, et personne ne devrait le lire ainsi.
PART_FONDS_PROPRES = 0.15

# Actif rapporte au chiffre d'affaires, au-dela duquel le bilan est LOURD. Meme
# statut : posee. Un industriel ordinaire tourne autour de 1 a 2 ; au-dela de 4,
# l'actif ne sert plus a produire un chiffre d'affaires mais EST l'activite.
ACTIF_SUR_CA = 4.0


def est_une_activite_de_bilan(total_actif, fonds_propres, chiffre_affaires) -> bool:
    """La structure du bilan est-elle celle d'une banque, d'un assureur, d'un preteur ?

    Les trois grandeurs sont attendues dans la MEME unite ; seuls des rapports en
    sont tires, donc l'unite elle-meme est indifferente — mais les melanger ne
    l'est pas.

    Rend faux des qu'une grandeur manque : on ne devine pas la nature d'une societe
    a partir de ce qu'on ignore d'elle. C'est le comportement du routage, et non
    celui des deux autres copies, qui traitaient une donnee absente comme la preuve
    d'une activite de bilan.
    """
    if not total_actif or total_actif <= 0:
        return False
    if fonds_propres is None or chiffre_affaires is None:
        return False
    if chiffre_affaires <= 0:
        return False
    # Borne BASSE incluse, et c'est tout l'objet de ce module : des fonds propres
    # nuls ou negatifs ne sont pas des fonds propres minces, ce sont des fonds
    # propres absents ou detruits.
    if fonds_propres <= 0:
        return False
    return ((fonds_propres / total_actif) < PART_FONDS_PROPRES
            and (total_actif / chiffre_affaires) >= ACTIF_SUR_CA)

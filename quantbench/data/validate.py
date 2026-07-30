"""Validation des donnees d'entree — une valorisation ne vaut que ce que valent
ses entrees.

Principe : ne valoriser une societe QUE si chacune des donnees qui alimentent le
modele a ete VERIFIEE par un controle objectif. Une donnee invalide ne produit pas
une valorisation approximative, elle produit une valorisation FAUSSE — donc sans
valeur. Mieux vaut couvrir moins de titres et pouvoir repondre de chacun.

Chaque controle est une verite comptable ou une identite de marche, jamais un
jugement sur la societe :

  devise          le taux de change doit etre OBTENU, jamais suppose a 1 (des
                  montants indonesiens pris pour des dollars : erreur de 18 000x)
  capitalisation  coherente avec le chiffre d'affaires et les fonds propres — sinon
                  elle porte sur une ligne ADR et non sur la societe
  bilan           actif = passif + fonds propres (identite fondamentale)
  tresorerie      ne peut exceder l'actif total
  chiffre d'affaires  ne peut valoir plusieurs fois l'actif total (entite de
                  financement publiant les comptes consolides du groupe)
  fraicheur       des comptes vieux de plus de deux ans ne decrivent plus la societe

Les motifs de rejet sont RETOURNES, pas avales : le build les agrege pour rendre
visible ce qui n'est pas couvert et pourquoi.
"""

from __future__ import annotations

from datetime import datetime, timezone

# QUATRE SEUILS ONT ETE SUPPRIMES ICI, ET C'EST UNE CORRECTION DE PRINCIPE.
#
#   MIN_CAP_SUR_CA = 0.03            "on ne cote pas a 3 % de ses ventes"
#   MIN_CAP_SUR_FONDS_PROPRES = 0.05 "ni a 5 % de ses fonds propres"
#   MIN_CAP_SUR_EBIT = 2.0           "une societe rentable vaut plus de 2x son EBIT"
#   MAX_CA_SUR_ACTIF = 5.0           "aucune activite ne tourne a 5x son actif"
#
# Ces quatre regles ne controlaient pas une DONNEE mais un RESULTAT. Elles
# confrontaient une grandeur d'ACTIONNAIRE — la capitalisation — a des grandeurs
# d'ENTREPRISE — chiffre d'affaires, resultat operationnel, actif net — sans jamais
# ajouter la dette : elles mesuraient donc le LEVIER, et rejetaient les societes
# endettees. Charter Communications, 19,57 Md$ de capitalisation pour 12,73 Md$ de
# resultat operationnel, etait ecartee au motif que le rapport valait 1,54 — c'est-a-
# dire uniquement parce qu'elle porte 95,8 Md$ de dette. Sa valeur d'entreprise
# rapportee au resultat operationnel vaut 9,0.
# Sur les titres de cette famille dont la capitalisation a ete CONFIRMEE par une
# source independante, 76 % repassent le seuil des qu'on raisonne en valeur
# d'entreprise. Et les depassements du rapport chiffre d'affaires sur actif sont
# reels, confirmes au dollar pres par les depots : World Kinect 6,3 fois
# (distributeur de carburant), The Real Brokerage 15,5 (commissions brutes),
# Bullish 61,9 (volume negocie).
# 601 titres etaient ecartes par ces seuils, dont 456 sans aucun autre motif.
#
# Ce qu'ils attrapaient LEGITIMEMENT — des lignes obligataires prises pour des
# actions — releve du TYPE DE TITRE et est desormais traite a la source, dans
# `fmp._is_preferred`, sur le libelle de l'emission.
#
# Ne subsistent que des IDENTITES et des impossibilites physiques.
MAX_FONDS_PROPRES_SUR_CAP = 200.0   # erreur d'unites, pas un jugement de valeur
TOLERANCE_BILAN = 0.05              # 5 % d'ecart admis sur l'identite comptable
ANCIENNETE_MAX = 2                  # exercices


def valider(fund: dict, F: dict | None, entry: dict | None = None) -> list[str]:
    """Retourne la liste des motifs d'invalidite. Vide = donnees exploitables."""
    motifs = []
    if not fund:
        return ["fondamentaux absents"]

    # --- Devise -------------------------------------------------------------
    if fund.get("fx_indisponible"):
        motifs.append(f"taux de change indisponible ({fund.get('financial_currency')})")

    # --- Marche -------------------------------------------------------------
    prix, cap = fund.get("price"), fund.get("market_cap")
    if not prix or prix <= 0:
        motifs.append("cours indisponible")
    if not cap or cap <= 0:
        motifs.append("capitalisation indisponible")
    actions = fund.get("shares")
    if not actions or actions <= 0:
        motifs.append("nombre d'actions indeterminable")

    rev = fund.get("revenue")
    be = fund.get("book_equity")
    ta = fund.get("total_assets")

    # --- Erreur d'UNITES sur les fonds propres ------------------------------
    # Seul controle conserve de cette famille, et il ne juge aucune valorisation :
    # des fonds propres deux cents fois superieurs a la capitalisation ne traduisent
    # pas une decote mais un montant hors d'echelle — Oncotelic declarait 262 000
    # milliards de dollars de capitaux propres pour 16 M$ de capitalisation.
    if cap and cap > 0 and be and be > MAX_FONDS_PROPRES_SUR_CAP * cap:
        motifs.append("fonds propres hors d'echelle face a la capitalisation")

    # --- Identites comptables ----------------------------------------------
    if ta and ta > 0:
        if (fund.get("cash") or 0) > ta * 1.001:
            motifs.append("tresorerie superieure a l'actif total")
        if (fund.get("total_debt") or 0) > ta * 1.001:
            motifs.append("dette superieure a l'actif total")
        tl = fund.get("total_liab")          # deja converti en USD, comme ta et be
        # Le bilan equilibre sur les fonds propres TOTAUX, minoritaires inclus.
        # `book_equity` ne retient que la part attribuable aux actionnaires — c'est
        # la bonne grandeur pour VALORISER, mais pas pour verifier une identite
        # comptable : confronter l'identite a la part attribuable rejetait toute
        # societe detenant des filiales non integralement possedees.
        fp = fund.get("total_equity")
        if fp is None:
            fp = be
        # `tl` est compare a None et non teste en verite : un passif NUL est falsy,
        # si bien que le bilan des societes sans dettes n'etait jamais controle. Vingt-
        # cinq lignes de l'univers y echappaient, dont quinze a fonds propres nuls ou
        # negatifs — exactement la population que les routes de liquidation traitent.
        if tl is not None and fp is not None:
            ecart = abs(ta - (tl + fp)) / ta
            if ecart > TOLERANCE_BILAN:
                motifs.append(f"identite du bilan violee (ecart {ecart:.0%} entre "
                              f"l'actif et passif + fonds propres)")

    # --- Flux de tresorerie -------------------------------------------------
    # Manhattan Bridge Capital publiait 117,9 M$ d'amortissements et 4,93 MILLIARDS
    # de flux d'exploitation pour 8,7 M$ de chiffre d'affaires : le FFO capitalise
    # sur ces chiffres donnait 2,6 Md$ de valeur pour une societe de 48 M$.
    da, cfo = fund.get("dep_amort"), fund.get("cfo")
    if ta and ta > 0:
        if da is not None and abs(da) > ta:
            motifs.append("amortissements superieurs a l'actif total")
        if cfo is not None and abs(cfo) > 5 * ta:
            motifs.append("flux d'exploitation hors d'echelle face a l'actif")
    if rev and rev > 0 and cfo is not None and abs(cfo) > 20 * rev:
        motifs.append(f"flux d'exploitation de {abs(cfo)/rev:.0f}x le chiffre d'affaires")

    # --- Fraicheur ----------------------------------------------------------
    if entry:
        annees = set(entry.get("income", {})) & set(entry.get("balance", {}))
        if not annees:
            motifs.append("aucun exercice complet (compte de resultat + bilan)")
        else:
            dernier = max(annees)
            courante = datetime.now(timezone.utc).year
            if dernier < courante - ANCIENNETE_MAX:
                motifs.append(f"comptes perimes (dernier exercice {dernier})")

    return motifs


__all__ = ["valider"]

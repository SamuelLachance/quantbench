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
    # UN ACTIF TOTAL NUL AVEC UN PASSIF OU DES FONDS PROPRES NON NULS EST IMPOSSIBLE.
    # Le controle d'identite ci-dessous est garde par `if ta and ta > 0`, si bien
    # qu'un actif a ZERO le desactivait entierement — meme angle mort que celui du
    # passif nul. Entergy New Orleans publie 0 d'actif total pour 91 M$ de passif et
    # 16,9 Md$ de fonds propres, en realite ceux du groupe Entergy entier : la liasse
    # ne decrit pas l'entite cotee. Elle ressortait a +2 566 %.
    tl_, fp_ = fund.get("total_liab"), fund.get("total_equity")
    if (not ta) and ((tl_ and abs(tl_) > 1e-9) or (fp_ and abs(fp_) > 1e-9)):
        motifs.append("actif total absent alors que le passif et les fonds propres "
                      "sont renseignes — la liasse ne decrit pas l'entite cotee")

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
    # L'ANCIENNETE N'EST PLUS UN MOTIF DE REJET. Des comptes vieux de trois ans ne
    # sont pas des comptes FAUX : ce sont les derniers comptes connus, et Damodaran
    # valorise sur les derniers comptes disponibles quels qu'ils soient, en signalant
    # leur date. Rejeter revenait a declarer mortes des societes bien vivantes — 13 %
    # des lignes ainsi ecartees deposaient encore a la SEC, dont Brookfield
    # Infrastructure Bermuda pour 13,3 Md$.
    # Le seuil separait de surcroit les clotures de juin des clotures de decembre et
    # non les vivantes des mortes : trente-six lignes tombaient exactement a la
    # bascule, toutes closes le 30 juin. Il oscillait avec le calendrier — 34 lignes
    # perdues en juillet, 428 en janvier.
    # La date est desormais MESUREE et PUBLIEE (`fraicheur_des_comptes`), et elle
    # declenche la reconstitution sur douze mois glissants. Une societe morte se
    # constate sur un fait date, pas sur l'age d'une donnee du fournisseur.
    if entry and not (set(entry.get("income", {})) & set(entry.get("balance", {}))):
        motifs.append("aucun exercice complet (compte de resultat + bilan)")

    return motifs


def fraicheur_des_comptes(entry, reference=None):
    """(date de cloture du dernier exercice, age en MOIS). (None, None) si inconnu.

    L'age se compte en MOIS depuis la CLOTURE, non en millesimes d'exercice. Le
    millesime est decale chez le fournisseur pour 3 a 4 % des lignes — un exercice
    clos le 30 juin 2024 est etiquete 2023 — et la comparaison d'annees civiles cree
    une falaise au 1er janvier qui n'a aucun sens economique."""
    if not entry:
        return None, None
    annees = set(entry.get("income", {})) & set(entry.get("balance", {}))
    if not annees:
        return None, None
    ligne = entry["income"][max(annees)]
    brut = str(ligne.get("date") or "")[:10]
    try:
        clos = datetime.strptime(brut, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None, None
    ref = reference or datetime.now(timezone.utc)
    mois = (ref.year - clos.year) * 12 + (ref.month - clos.month) - (ref.day < clos.day)
    return brut, mois


__all__ = ["valider", "fraicheur_des_comptes"]

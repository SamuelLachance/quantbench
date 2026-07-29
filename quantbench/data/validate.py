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

# Seuils : bornes de PLAUSIBILITE larges, pas des opinions de valorisation.
MIN_CAP_SUR_CA = 0.03          # une societe ne se traite pas a 3 % de ses ventes
MIN_CAP_SUR_FONDS_PROPRES = 0.05
MAX_CA_SUR_ACTIF = 5.0         # les activites les plus legeres tournent a 2-3x
MAX_FONDS_PROPRES_SUR_CAP = 200.0
TOLERANCE_BILAN = 0.05         # 5 % d'ecart admis sur l'identite comptable
ANCIENNETE_MAX = 2             # exercices


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

    # --- Coherence capitalisation <-> comptes -------------------------------
    if cap and cap > 0:
        if rev and rev > 0 and cap < MIN_CAP_SUR_CA * rev:
            motifs.append(f"capitalisation incoherente avec le chiffre d'affaires "
                          f"({cap/rev:.1%} du CA) — porte probablement sur une ligne ADR")
        if be and be > 0:
            if cap < MIN_CAP_SUR_FONDS_PROPRES * be:
                motifs.append(f"capitalisation incoherente avec les fonds propres "
                              f"({cap/be:.1%} des capitaux propres)")
            elif be > MAX_FONDS_PROPRES_SUR_CAP * cap:
                motifs.append("fonds propres hors d'echelle face a la capitalisation")

    # --- Identites comptables ----------------------------------------------
    if ta and ta > 0:
        if (fund.get("cash") or 0) > ta * 1.001:
            motifs.append("tresorerie superieure a l'actif total")
        if (fund.get("total_debt") or 0) > ta * 1.001:
            motifs.append("dette superieure a l'actif total")
        if rev and rev > MAX_CA_SUR_ACTIF * ta:
            motifs.append(f"chiffre d'affaires de {rev/ta:.0f}x l'actif total — "
                          f"entite de financement publiant les comptes du groupe")
        tl = (F or {}).get("total_liab", [None])[0]
        if tl and be is not None:
            ecart = abs(ta - (tl / 1e9 + be)) / ta
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

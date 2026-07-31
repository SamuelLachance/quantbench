"""
quantbench.valuation.route
==========================
Routage d'un titre vers la methode de valorisation adaptee a sa nature (methode
Damodaran), puis valorisation. Chaque resultat porte sa methode, sa categorie et
un niveau de confiance.

Routage (par secteur + signaux) :
  financiere        -> valorisation des capitaux propres (excess-return / residual income)
  cyclique          -> DCF sur benefices NORMALISES (marge moyenne du cycle)
  jeune/deficitaire -> DCF top-down sur revenus x probabilite de survie
  detresse          -> DCF going-concern pondere par proba de defaut + liquidation
  standard          -> DCF FCFF classique
  (repli)           -> multiple relatif quand l'intrinseque echoue
"""

from __future__ import annotations

import json as _json
import os as _os

import numpy as np

from ..data.universal import get_fundamentals
from ..data import market
from ..forensics import analyze as forensic_analyze, get_financials
from ..forensics.scores import default_probability
from .build_universal import (build_dcf_from_fundamentals, country_erp,
                              pays_exploitation)
from .dcf import value_dcf
from ..bilan import bilan_lourd_et_endette, est_une_activite_de_bilan


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _coe(fund):
    """Cout des fonds propres = rf + beta x ERP, ERP incluant la prime de risque
    PAYS (Damodaran) — sinon une banque chinoise ou bresilienne serait actualisee
    au cout du capital americain.

    PLANCHER DE BETA : une cotation OTC/ADR peu liquide produit un beta de
    regression artificiellement bas (0,20 pour la fonciere mexicaine Fibra UNO,
    0,24 pour la banque China Minsheng) — le titre bouge peu faute d'echanges, pas
    faute de risque. Un tel beta effondre le cout des fonds propres et fait
    exploser tout multiple de capitalisation. On le plancherise a 60 % de la
    mediane MESUREE du secteur, et on impose au cout des fonds propres de rester
    au-dessus du taux sans risque augmente de 3 points : aucune action n'est moins
    risquee qu'une obligation d'Etat."""
    rf = market.risk_free_rate()
    from .build_universal import beta_ascendant, tax_rate
    pays = pays_exploitation(fund)
    from .build_universal import prime_taille
    beta, _unlev, _src = beta_ascendant(fund, tax_rate(pays))
    erp = country_erp(pays)
    ke = rf + beta * erp + prime_taille(fund.get("market_cap"))
    return max(ke, rf + 0.03), rf


# Le Z-score d'Altman est calibré sur des industriels : Altman lui-même et
# Damodaran l'excluent pour les financières ; foncières et services publics ont
# structurellement un Z bas (actifs lourds, dette élevée) sans être en détresse.
_NO_ALTMAN = ("financial", "real estate", "utilities")
# Route "detresse" reservee au risque de defaut REEL (Z''-EMS < 3.20 ~ notation
# CCC+ ou moins). Entre 3.20 et 5.85 la societe est speculative mais en activite :
# DCF normal, le risque passe par le cout du capital.
Z_DETRESSE_ROUTE = 3.20


def sect(fund, cle, defaut):
    """Repere MESURE, du plus fin au plus large : INDUSTRIE -> SECTEUR -> GLOBAL.
    Source UNIQUE, partagee avec le moteur DCF. Deux fichiers de statistiques
    coexistaient et le routage lisait le plus ancien : le taux de recuperation de la
    sante y valait 0,354 au lieu de 0,624, et celui de l'immobilier 0,303 au lieu de
    0,795. Un repere doit avoir une seule definition."""
    from .build_universal import repere
    v = repere(fund, cle)
    return defaut if v is None else v


def _financiere_de_bilan(fund) -> bool:
    """Une societe n'est une FINANCIERE au sens de la valorisation que si son
    BILAN est son outil de production (la dette y est une matiere premiere) :
    banques, assurances, courtiers, preteurs. Les reseaux de paiement (Visa,
    Mastercard), gerants d'actifs (BlackRock), bourses et fournisseurs de donnees
    (S&P Global, CME) sont des metiers de COMMISSIONS : leurs capitaux propres ne
    portent aucune information de valeur -> DCF classique. Les valoriser en
    multiple de valeur comptable donnait Visa a -73 % et Mastercard a -92 %."""
    ind = (fund.get("industry") or "").lower()
    ta, rev = fund.get("total_assets"), fund.get("revenue")
    be = fund.get("book_equity")
    # Le LEVIER prime sur l'etiquette : des fonds propres inferieurs a 15 % de
    # l'actif avec un bilan lourd est la signature d'une activite de bilan
    # (banques 8-10 %, assureurs 5-15 %). Apollo, etiquetee "Asset Management",
    # detient l'assureur Athene : 460 Md$ d'actif pour 4,8 % de fonds propres —
    # elle etait valorisee en DCF et ressortait a +532 %.
    # SOURCE UNIQUE. Cette regle etait ecrite ici, dans la notation du risque et
    # dans la construction des entrees du modele, et les trois copies avaient
    # diverge — notamment sur des fonds propres nuls, negatifs ou absents.
    if est_une_activite_de_bilan(ta, be, rev):
        return True
    # UN COURTIER N'EST PAS UN ASSUREUR. Le mot « insurance » attrapait aussi les
    # COURTIERS, qui encaissent une commission sans porter le moindre risque a leur
    # bilan : Arthur J. Gallagher (33 % de fonds propres sur actif), Brown & Brown
    # (42 %), Aon, Marsh & McLennan, Willis Towers Watson. Un assureur du S&P 500
    # porte entre 16 et 39 % de fonds propres et immobilise ses reserves ; un
    # courtier n'immobilise rien. Les valoriser en multiple de valeur comptable est
    # la meme faute que celle qui donnait Visa a -73 %.
    # Le test precede les mots-cles d'inclusion, sans quoi « insurance » l'emporte.
    if "broker" in ind and "insurance" in ind:
        return False
    if any(k in ind for k in ("bank", "insurance", "mortgage", "thrift")):
        return True
    if any(k in ind for k in ("asset management", "stock exchange", "financial data",
                              "shell", "conglomerate")):
        return False
    # DERNIER RECOURS : LE POIDS DU BILAN, MAIS AVEC SA CONDITION DE FONDS PROPRES.
    # Ce repli testait le seul rapport actif sur chiffre d'affaires, sans rien dire
    # des fonds propres — il suffisait donc d'un bilan lourd pour etre declare
    # financiere de bilan. Global Payments, processeur de paiements comme Visa,
    # porte 43 % de fonds propres sur actif et un actif de 6,9 fois son chiffre
    # d'affaires, ce dernier venant d'ECARTS D'ACQUISITION et non de prets : elle
    # etait valorisee en rendement excedentaire sur ses fonds propres comptables.
    # Un bilan lourd finance par des FONDS PROPRES n'est pas une activite de bilan ;
    # c'est le LEVIER qui la definit, et `est_une_activite_de_bilan` le sait deja.
    if ta and rev and rev > 0:
        return bilan_lourd_et_endette(ta, be, rev)
    return True


def _trop_leger_pour_une_fonciere(fund) -> bool:
    """Un CHIFFRE D'AFFAIRES de plusieurs fois l'actif exclut la route immobiliere.

    Une fonciere est, par construction, l'activite la plus capitalistique qui soit :
    son chiffre d'affaires est le loyer de son parc et ne peut en depasser une
    fraction. Une societe du secteur "immobilier" qui encaisse dix a quinze fois son
    actif ne detient pas d'immeubles — elle vend un SERVICE immobilier : courtage,
    administration de biens, promotion pour compte de tiers. Capitaliser son flux
    comme une rente perpetuelle de loyers revient a valoriser un courtier comme un
    immeuble ; The Real Brokerage (15,5 fois son actif) et AGNT (10,8) ressortaient
    ainsi a +144 % et +188 %.

    Le seuil n'est pas une opinion de valorisation mais une impossibilite physique :
    aucun parc immobilier ne tourne trois fois par an. Ces societes rejoignent le DCF
    d'entreprise, qui convient a une activite de service."""
    rev, ta = fund.get("revenue"), fund.get("total_assets")
    return bool(rev and ta and ta > 0 and (rev / ta) > 3.0)


def _est_holding(fund) -> bool:
    """Societe de PORTEFEUILLE (holding d'investissement) : son "chiffre
    d'affaires" est le revenu de ses participations, pas une activite. Un DCF y
    est denue de sens — AB Industrivarden ressortait a +540 % avec une croissance
    et une marge toutes deux COLLEES A LEURS PLAFONDS (45 % et 75 %), signe que le
    modele etait nourri de donnees non operationnelles.
    Discriminant mesure : un holding degage un revenu tres faible au regard de ses
    capitaux propres (Industrivarden 0,19) la ou une vraie societe de commissions
    exploite ses fonds propres (BlackRock 0,43, S&P Global 0,49, ICE 0,44)."""
    ind = (fund.get("industry") or "").lower()
    if not any(k in ind for k in ("asset management", "conglomerate", "shell",
                                  "holding", "closed-end")):
        return False
    rev, be = fund.get("revenue"), fund.get("book_equity")
    return bool(rev is not None and be and be > 0 and (rev / be) < 0.30)


def _roe_normalise(fund, F):
    """ROE median sur l'historique : une annee deprimee ou dopee par un element
    exceptionnel ne doit pas fixer la valeur (Capital One : 2,2 % une annee)."""
    roes = []
    if F:
        # FONDS PROPRES D'OUVERTURE, jamais de cloture. Le rendement d'un exercice
        # se rapporte au capital dont la societe DISPOSAIT pour le produire ; les
        # fonds propres de cloture contiennent deja le benefice qu'on leur rapporte,
        # plus toute augmentation de capital de l'annee. Le rendement mesure valait
        # donc ROE / (1 + croissance des fonds propres) — sous-estime a chaque
        # exercice, dans le meme sens, si bien que la mediane n'y remedie pas.
        # C'est en outre ce que suppose la forme fermee de `value_financial` :
        # l'exces de l'annee t y porte sur BE_{t-1}. Les deux bouts du modele
        # parlaient de deux capitaux differents.
        # Les series arrivent PLUS RECENT EN TETE (`sorted(..., reverse=True)`) :
        # l'ouverture de l'exercice i est donc la cloture de l'exercice i+1. Le plus
        # ancien exercice n'a pas d'ouverture connue et sort de la mesure.
        eq = F.get("equity") or []
        for i, ni in enumerate(F.get("net_income") or []):
            ouverture = eq[i + 1] if i + 1 < len(eq) else None
            if ni is not None and ouverture and ouverture > 0:
                roes.append(ni / ouverture)
    r = float(np.median(roes)) if len(roes) >= 3 else fund.get("roe")
    return None if r is None else max(-1.0, min(r, 0.40))


def _hist_margins(F):
    """Marges opérationnelles historiques (EBIT/CA)."""
    if not F:
        return []
    return [e / r for e, r in zip(F.get("ebit", []), F.get("revenue", []))
            if e is not None and r]


def conversion_en_tresorerie(F):
    """Part du benefice comptable cumule qui s'est reellement transformee en
    TRESORERIE, mesuree sur tout l'historique disponible.

    Un benefice qui n'est jamais encaisse n'a pas de valeur. FDCTech affiche sur dix
    ans 3 M$ de resultat d'exploitation cumule pour 31 M$ de tresorerie CONSOMMEE :
    son chiffre d'affaires a ete multiplie par 76 sans qu'un dollar ne rentre. Le
    modele capitalisait ce benefice comptable a l'infini.

    On mesure sur le CUMUL et non annee par annee : un decalage de besoin en fonds
    de roulement se resorbe d'un exercice a l'autre, une decennie de non-conversion
    non. Retourne None quand la mesure n'a pas de sens (moins de quatre exercices,
    ou benefice cumule negatif — il n'y a alors rien a convertir)."""
    if not F:
        return None
    # LES DEUX SERIES SE FILTRENT ENSEMBLE, jamais separement. Chacune etait
    # purgee de ses trous DANS SON COIN, puis les deux etaient tronquees a la
    # longueur commune : un exercice sans EBIT decalait toute la serie des flux
    # d'un cran, et le rapport cumulait alors des EBIT et des flux d'ANNEES
    # DIFFERENTES. Mesure sur l'univers : 1,3 % des societes presentent ce
    # decalage. Les series arrivent alignees par construction — `financials_from_fmp`
    # les bâtit toutes sur la meme liste d'exercices — c'est donc le filtrage qui
    # cassait cet alignement.
    paires = [(e, c) for e, c in zip(F.get("ebit") or [], F.get("cfo") or [])
              if e is not None and c is not None]
    if len(paires) < 4:
        return None
    somme_ebit = sum(e for e, _ in paires)
    if somme_ebit <= 0:
        return None
    return sum(c for _, c in paires) / somme_ebit


def classify(fund: dict, forensic: dict | None, F: dict | None = None) -> str:
    sec = (fund.get("sector") or "").lower()
    ebit, ni = fund.get("ebit"), fund.get("net_income")
    be = fund.get("book_equity")
    z = (forensic or {}).get("scores", {}).get("altman_z")
    rev = fund.get("revenue")
    if rev is None or rev <= 0:
        # Pre-revenu (biotech clinique, mineur d'exploration), holding, SPAC :
        # pas de flux a actualiser -> valeur d'actif net (methode Damodaran).
        return "actif_net"

    # --- UNE CONTINUITE D'EXPLOITATION SE CONSTATE, ELLE NE SE SUPPOSE PAS ------
    # Un DCF actualise les flux FUTURS d'une societe en activite, a partir d'un
    # exercice de reference cense la decrire. Qingdao Footwear etait valorisee sur
    # son exercice 2010 — quinze ans et demi plus tard — a +1 282 295 %, et Alabama
    # Aircraft sur 2008. Ces societes n'ont rien publie depuis : rien n'atteste
    # qu'elles exploitent encore quoi que ce soit.
    # Le critere ne porte pas sur l'age comme jugement de valeur mais sur un FAIT DE
    # PUBLICATION : une societe en activite depose des comptes chaque annee. Passe
    # deux exercices annuels manques — delai de depot compris — la continuite n'est
    # plus attestee par rien, et seul subsiste un droit sur les actifs derniers
    # constates, que l'erosion temporelle ramene a leur valeur reelle.
    # Ce n'est pas un rejet : la societe reste valorisee et publiee. C'est le choix
    # d'une METHODE applicable, au meme titre que le secteur ou le levier.
    mois = fund.get("age_des_comptes_mois")
    if mois is not None and mois > 30:
        return "actif_net"

    deficitaire = (ebit is not None and ebit < 0) or (ni is not None and ni < 0)
    # Le Z-score d'Altman est calibre sur des industriels : Altman lui-meme et
    # Damodaran l'excluent pour les financieres ; foncieres et services publics ont
    # structurellement un Z bas sans etre en detresse.
    z_ok = z is not None and not any(s in sec for s in _NO_ALTMAN)

    # LA DETRESSE SE MESURE EN TRESORERIE, PAS EN RESULTAT COMPTABLE.
    # Une depreciation d'actifs ne fait sortir aucun euro de la societe. Branicks a
    # publie en 2024 un EBIT de -288,7 M EUR et une perte nette de -365,5 M EUR pour
    # un flux d'exploitation de +54,8 M EUR et un FFO de +52,2 M EUR : la perte etait
    # integralement due a la reevaluation de son parc (-6,9 %), et son propre rapport
    # annuel precise que le resultat "reflete des ajustements de valorisation, non une
    # faiblesse operationnelle". Juger la detresse sur le resultat envoyait donc a la
    # LIQUIDATION une fonciere qui encaisse ses loyers.
    # Le defaut est structurel, non anecdotique : sous IFRS l'immobilier est inscrit a
    # la JUSTE VALEUR, si bien que toute correction de marche traverse le compte de
    # resultat de TOUTES les foncieres europeennes, canadiennes et asiatiques. Il en
    # va de meme de toute depreciation de goodwill, dans n'importe quel secteur.
    encaisse = (fund.get("cfo") or 0) > 0

    # --- LA DETRESSE PRIME SUR LE SECTEUR -----------------------------------
    # Une societe dont les pertes ont absorbe les fonds propres, ou dont le Z-score
    # signale un risque de defaut avere, ne se valorise pas comme une consoeur en
    # bonne sante — quel que soit son secteur. L'ordre inverse laissait SunPower,
    # en faillite avec des fonds propres NEGATIFS, etre valorisee comme un cyclique
    # ordinaire a +2 116 % : la branche "energie" repondait avant tout controle de
    # solvabilite. Le meme angle mort touchait l'immobilier, les services publics et
    # les financieres.
    if deficitaire and not encaisse and be is not None and be <= 0:
        return "detresse"                      # fonds propres absorbes par les pertes
    if deficitaire and not encaisse and z_ok and z < Z_DETRESSE_ROUTE:
        return "detresse"

    # --- Routage sectoriel : societes en continuite d'exploitation -----------
    if "real estate" in sec and not _trop_leger_pour_une_fonciere(fund):
        return "fonciere"                      # REIT : FFO/NAV, jamais le FCFF
    if "utilities" in sec:
        return "reglementee"                   # service public : cote equite
    if _est_holding(fund):
        return "holding"                       # portefeuille : actif net reevalue
    if "financial" in sec:
        return "financiere" if _financiere_de_bilan(fund) else "standard"
    if any(s in sec for s in ("energy", "materials")):
        return "cyclique"

    if deficitaire:
        # "JEUNE" VEUT DIRE COURTE HISTOIRE, PAS "JAMAIS RENTABLE".
        # Le critere ne comptait que les exercices benificiaires : une societe
        # publiant depuis dix ans sans en avoir aucun etait donc traitee en jeune
        # pousse, et valorisee sur la marge MEDIANE DE SON SECTEUR — qu'elle n'a
        # precisement jamais approchee. Sur 186 societes ainsi routees, 66 disposaient
        # de dix exercices et treize depassaient le milliard de chiffre d'affaires :
        # DiDi Global (32,6 Md$), Carvana (20,3), NIO (12,6), Roku (4,7).
        # Une societe qui publie depuis longtemps a un HISTORIQUE : c'est lui qui doit
        # servir de reference, non la mediane de ses consoeurs. La route des societes
        # jeunes ne se justifie que faute d'historique — c'est la definition meme de
        # l'approche descendante chez Damodaran.
        ms = _hist_margins(F)
        if len(ms) >= 6 or sum(1 for m in ms if m > 0) >= 2:
            return "mature_deficitaire"
        return "jeune/deficitaire"
    if z_ok and z < Z_DETRESSE_ROUTE and not encaisse:
        return "detresse"
    return "standard"


def marge_de_cycle(F):
    """MARGE NORMALISEE SUR LE CYCLE = somme des EBIT / somme des chiffres d'affaires.

    Une marge est un RAPPORT DE DEUX SOMMES, non la moyenne de rapports annuels.
    Nous prenions la moyenne (ou la mediane) des marges de chaque exercice, ce qui
    accorde le meme poids a une annee de 8,6 M$ de CA qu'a une annee de 107,6 M$.
    Pop Culture a triple son chiffre d'affaires en deux ans en perdant de l'argent :
    ses marges anciennes, gagnees sur un dixieme du volume actuel, dominaient la
    moyenne et le modele appliquait 18 % de marge a un CA qui n'en a jamais degage.
    Le rapport des sommes est naturellement pondere par le volume, donc insensible
    a un changement d'echelle.

    Toutes les annees comptent, y compris deficitaires : normaliser SUR LE CYCLE est
    precisement l'objet de la manoeuvre chez Damodaran. Ne moyenner que les
    exercices benificiaires — ce que faisait la route des societes matures en perte
    — revient a definir la rentabilite normale comme "ce que la societe gagne quand
    elle gagne", et garantit une reponse optimiste.

    Retourne None si la mesure n'a pas de sens (moins de trois exercices)."""
    if not F:
        return None
    eb, rv = F.get("ebit") or [], F.get("revenue") or []
    # PRESENCE, ET NON VERITE. Le filtre testait `if r`, ce qui ecarte un chiffre
    # d'affaires exactement NUL — donc les exercices ou une societe n'a rien vendu
    # et beaucoup perdu. Elle disparaissait du numerateur ET du denominateur, alors
    # qu'elle doit entrer au numerateur (la perte est reelle) et n'ajouter rien au
    # denominateur (la vente est nulle). Sur un cas mesure, la marge de cycle
    # passait de -192 % a +5,3 % : le modele appliquait a l'infini une marge
    # POSITIVE a une societe qui a brule de l'argent trois ans sans un dollar de
    # recette. C'est exactement ce que la docstring ci-dessus condamne — definir la
    # rentabilite normale comme « ce que la societe gagne quand elle gagne ».
    # Mesure sur l'univers : 14 % des societes ont au moins un exercice a chiffre
    # d'affaires nul, et la correction rend leur marge de cycle plus severe sans
    # exception de signe.
    paires = [(e, r) for e, r in zip(eb, rv) if e is not None and r is not None]
    if len(paires) < 3:
        return None
    ca = sum(r for _, r in paires)
    return (sum(e for e, _ in paires) / ca) if ca > 0 else None


def marge_normalisee(fund, F):
    """Marge NORMALISEE appliquee quand la marge du dernier exercice s'ecarte
    fortement de la marge de cycle. Un produit exceptionnel (accord de licence,
    cession) gonfle la marge de l'annee et, extrapole a l'infini, multiplie la
    valeur. Damodaran normalise systematiquement dans ce cas."""
    cyc = marge_de_cycle(F)
    cur = fund.get("operating_margin")
    if cyc is None or cur is None or cyc <= 0:
        return None
    if cur > 1.5 * cyc or cur < 0.5 * cyc:
        return cyc
    return None


def _dcf_value(fund, margin_override=None, method="DCF FCFF"):
    x, _ = build_dcf_from_fundamentals(fund, margin_override=margin_override)
    res = value_dcf(x)
    return {"equity_value": res["equity_value"], "method": method, "confidence": "moyenne"}


def value_financial(fund, F=None):
    """Financieres DE BILAN — modele de rendement excedentaire (Damodaran).
    V = BE x [1 + (ROE - ke) x A], ou A actualise l'exces de rentabilite sur une
    periode d'avantage concurrentiel de 10 ans, au terme de laquelle ROE = ke (les
    rentes sont competees). L'ecart de rentabilite est CONSTANT sur ces dix ans puis
    NUL — la forme de Damodaran — et non decroissant : le libelle publie annoncait
    une "erosion" que le calcul ne fait pas. Cela remplace un multiplicateur PERPETUEL qui explosait
    des que ke - g devenait minuscule (assureurs a faible beta : Allstate ressortait
    a +121 %) et qu'il fallait brider par des bornes arbitraires figeant le P/B de
    TOUTE financiere dans [0,40 ; 5,00]."""
    be = fund.get("book_equity")
    roe = _roe_normalise(fund, F)
    if not be or be <= 0 or roe is None:
        return None
    # Un resultat net SUPERIEUR au chiffre d'affaires signale un element non
    # recurrent (sortie de faillite) : le ROE qui en decoule n'est pas reproductible.
    ni, rev = fund.get("net_income"), fund.get("revenue")
    if ni is not None and rev and rev > 0 and ni > rev:
        return {"equity_value": be, "confidence": "faible",
                "method": "Valeur comptable (resultat net non recurrent)"}
    ke, rf = _coe(fund)
    g = min(rf, 0.03)
    ke = max(ke, g + 0.02)
    n = 10                                   # periode d'avantage concurrentiel
    q = (1.0 + g) / (1.0 + ke)
    A = (n / (1.0 + ke)) if abs(1.0 - q) < 1e-9 else         (1.0 / (1.0 + ke)) * (1.0 - q ** n) / (1.0 - q)
    val = be * (1.0 + (roe - ke) * A)
    return {"equity_value": max(val, 0.0), "confidence": "moyenne",
            "method": "Rendement excedentaire, avantage concurrentiel 10 ans (financiere de bilan)",
            "roe_normalise": round(roe, 4), "pb_implicite": round(1.0 + (roe - ke) * A, 2)}


def value_regulated(fund, F=None):
    """Services publics REGULES (electricite, gaz, eau) — methode Damodaran.
    Le DCF d'entreprise echoue ici : ces societes sont extremement capitalistiques
    et endettees, si bien que "valeur d'entreprise moins dette" devient negatif
    alors qu'elles sont parfaitement solvables. Leur rentabilite est en outre
    FIXEE par le regulateur (ROE autorise stable) et leur distribution elevee : on
    les valorise COTE EQUITE, par capitalisation des benefices avec une croissance
    FONDAMENTALE coherente g = ROE x taux de retention (identite de Damodaran),
    donc un taux de distribution = 1 - g/ROE.

    Benefices NORMALISES sur la mediane historique : un service public regule ne
    perd pas durablement d'argent, sa tarification etant fixee pour couvrir ses
    couts. Un exercice deficitaire (couverture energetique, sinistre, provision)
    ne doit ni fixer sa valeur ni faire basculer vers un DCF inapplicable —
    Centrica, en perte une annee, ressortait a +1 043 %."""
    be = fund.get("book_equity")
    if not be or be <= 0:
        return None
    # Resultat net normalise. Il se calcule en MARGE — un rapport — et non sur les
    # montants de l'historique : celui-ci est publie en DEVISE LOCALE tandis que la
    # capitalisation a laquelle on compare le resultat est en dollars. Diviser la
    # mediane historique par un milliard revenait a traiter des pesos ou des bahts
    # comme des dollars, surestimant la valeur d'exactement le taux de change :
    # x1 400 pour l'argentine Edenor (+6 864 %), x33 pour la thailandaise EGCO
    # (+2 207 %). Un rapport est, lui, sans unite ; applique au chiffre d'affaires
    # deja converti, il redonne un montant en dollars.
    # Rapport des sommes, non moyenne de rapports : ces societes traversent des
    # exercices deficitaires (couverture energetique, gel tarifaire) sur des volumes
    # tres inegaux.
    # LES DEUX SERIES SE FILTRENT PAR PAIRES. Chacune etait purgee de ses trous
    # dans son coin puis tronquee a la longueur commune : un exercice sans resultat
    # net decalait toute la serie des chiffres d'affaires d'un cran, et la marge
    # normalisee rapportait alors des resultats et des ventes d'ANNEES DIFFERENTES.
    # Le filtre `if x` sur le chiffre d'affaires ecartait de surcroit un exercice a
    # recettes NULLES, qui est une donnee et non une absence — meme piege que dans
    # `marge_de_cycle`.
    ni = None
    paires = [(a, b) for a, b in zip((F or {}).get("net_income") or [],
                                     (F or {}).get("revenue") or [])
              if a is not None and b is not None]
    rev_usd = fund.get("revenue")
    if len(paires) >= 3 and rev_usd:
        total = sum(b for _, b in paires)
        if total > 0:
            ni = (sum(a for a, _ in paires) / total) * rev_usd
    if ni is None:
        ni = fund.get("net_income")
    roe = _roe_normalise(fund, F)
    if not ni or ni <= 0 or not roe or roe <= 0:
        return None
    ke, rf = _coe(fund)
    g = min(rf, 0.028, roe * 0.9)               # g ne peut exceder ce que le ROE finance
    ke = max(ke, g + 0.03)                      # ecart minimal : un multiple fini
    payout = max(0.0, 1.0 - g / roe)
    val = ni * payout * (1.0 + g) / (ke - g)
    if val <= 0:
        return None
    return {"equity_value": val, "confidence": "moyenne",
            "method": "Benefices capitalises cote equite (service public regule)",
            "payout": round(payout, 3), "g": round(g, 4),
            "resultat_normalise": round(ni, 3)}


def value_holding(fund, F=None):
    """Societe de portefeuille — ACTIF NET REEVALUE (somme des parties, Damodaran).

    Une holding ne s'actualise pas : elle detient des participations. Sa valeur est
    l'actif net, et l'ecart avec la capitalisation est la DECOTE DE HOLDING, qui est
    l'information utile pour l'investisseur.

    Fiabilite de l'actif net — le point cle. Les capitaux propres ne valent comme
    actif net que si les participations sont inscrites a la JUSTE VALEUR (IFRS 9,
    cas des holdings cotees europeennes). En cout historique, ils la sous-estiment.
    On ne peut pas le supposer : on le VERIFIE sur les donnees.
      1. Identite comptable : actif total - passif total doit redonner les capitaux
         propres. Sinon le bilan est inexploitable.
      2. Base d'evaluation : sous juste valeur, les capitaux propres suivent les
         marches et varient fortement d'un exercice a l'autre ; en cout historique,
         ils progressent regulierement. On mesure cette volatilite pour qualifier la
         confiance au lieu de l'affirmer.
      3. Plausibilite : une holding se traite typiquement entre 0,5 et 1,2 fois son
         actif net. Au-dela, les capitaux propres ne sont pas un actif net credible
         et on le signale plutot que d'afficher une fausse precision."""
    be = fund.get("book_equity")
    if not be or be <= 0:
        return None

    diag, confiance = {}, "moyenne"

    # 1. Identite comptable actif - passif = capitaux propres
    #
    # LES CAPITAUX PROPRES TOTAUX, minoritaires COMPRIS. L'identite comptable ne
    # connait pas la repartition entre actionnaires : actif moins passif rend
    # l'integralite des fonds propres. La comparer a la seule part ATTRIBUABLE
    # declarait « bilan non verifiable » toute holding dont le bilan boucle
    # pourtant parfaitement mais qui detient des filiales non integralement
    # possedees — c'est-a-dire le cas ordinaire d'une holding, et l'ecart mesure
    # y vaut exactement la part des minoritaires.
    ta, tl = fund.get("total_assets"), fund.get("total_liab")   # en USD tous deux
    fp_totaux = fund.get("total_equity")
    if fp_totaux is None:
        fp_totaux = be
    if ta and tl:
        ecart = abs((ta - tl) - fp_totaux) / max(ta, 1e-9)
        diag["identite_bilan_ok"] = bool(ecart < 0.05)
        if ecart >= 0.05:
            confiance = "faible"

    # 2. Base d'evaluation deduite de la volatilite des capitaux propres
    # LA SERIE NE SE PURGE PAS AVANT D'EN PRENDRE DES RAPPORTS D'ANNEES VOISINES.
    # Le filtre `if e` retirait les exercices manquants ET les fonds propres NULS,
    # puis les rapports etaient pris sur des indices DEVENUS adjacents : un trou au
    # milieu faisait comparer 2024 a 2022 comme s'il s'agissait d'une variation
    # annuelle, gonflant la volatilite mesuree. Or c'est elle qui decide si les
    # participations sont inscrites a la juste valeur — une holding au cout
    # historique, dont l'actif net est un PLANCHER, ressortait alors en confiance
    # moyenne au lieu de faible.
    # On apparie donc les exercices VOISINS, en exigeant les deux.
    eqs = (F or {}).get("equity") or []
    if len(eqs) >= 3:
        var = [abs(eqs[i] / eqs[i + 1] - 1.0) for i in range(len(eqs) - 1)
               if eqs[i] is not None and eqs[i + 1] is not None and eqs[i + 1] > 0]
        if len(var) >= 2:
            v = float(np.median(var))
            diag["variation_annuelle_capitaux_propres"] = round(v, 4)
            # Sous juste valeur les capitaux propres suivent les marches (>8 %/an) ;
            # une progression tres reguliere trahit un cout historique, qui
            # SOUS-ESTIME l'actif net -> valeur plancher, confiance reduite.
            diag["base_juste_valeur"] = bool(v >= 0.08)
            if v < 0.08:
                confiance = "faible"

    # 3. Plausibilite : decote/prime de holding hors norme -> actif net douteux
    mcap = fund.get("market_cap")
    if mcap and mcap > 0:
        ratio = mcap / be
        diag["cours_sur_actif_net"] = round(ratio, 3)
        diag["decote_nav_pct"] = round((ratio - 1.0) * 100, 1)
        if not (0.35 <= ratio <= 1.6):
            confiance = "faible"

    return {"equity_value": be, "confidence": confiance,
            "method": "Actif net reevalue (societe de portefeuille)",
            "actif_net": round(be, 2), **diag}


def valeur_de_liquidation(fund):
    """Valeur revenant a l'ACTIONNAIRE si l'actif etait realise.

        liquidation = max( tresorerie + taux x (actif - tresorerie) - PASSIF , 0 )

    Le taux de recuperation porte sur l'ACTIF ; le passif se retranche ENSUITE, a sa
    valeur nominale. Nous l'appliquions aux FONDS PROPRES, ce qui revient a supposer
    que les dettes subissent la meme decote AU BENEFICE DE L'ACTIONNAIRE — or elles
    sont nominales et prioritaires : dans une realisation d'actifs, le creancier est
    servi en entier avant que l'actionnaire ne recoive un centime.

    L'erreur vaut exactement (1 - taux) x passif. Elle est NULLE pour une societe sans
    dette — d'ou son invisibilite sur les cas simples — et croit avec le levier, donc
    elle est maximale precisement la ou la question de la liquidation se pose.

    Wesizwe Platinum, qui construit la mine de Bakubung sans avoir jamais produit,
    porte 1,83 Md$ de passif pour 2,15 Md$ d'actif. Sa valeur de liquidation
    ressortait a 0,25 Md$ pour une capitalisation de 16 M$, soit +1 462 %. Le calcul
    correct donne : 2,15 x 0,80 = 1,72 Md$ d'actif realisable, moins 1,83 Md$ de
    passif — l'equite ne vaut RIEN, ce que confirme le rapport annuel 2025, qui porte
    une reserve sur la continuite d'exploitation.

    La tresorerie est realisable a 100 %, le reste subit la decote du secteur selon
    son intensite d'actifs corporels : une centrale ou un immeuble se revend, un
    portefeuille de brevets beaucoup moins."""
    taux = sect(fund, "recuperation", 0.5)
    ta, tl = fund.get("total_assets"), fund.get("total_liab")
    cash = max(fund.get("cash") or 0.0, 0.0)
    if ta and ta > 0 and tl is not None:
        realisable = min(cash, ta) + taux * max(ta - cash, 0.0)
        # ACTIF MOINS PASSIF donne les fonds propres TOTAUX. L'actionnaire n'a droit
        # qu'a la part ATTRIBUABLE : ce qui revient aux minoritaires d'une filiale ne
        # lui reviendra jamais. Sans ce retrait, une societe aux fonds propres
        # attribuables NEGATIFS mais aux minoritaires importants ressortait avec une
        # valeur de liquidation positive — Qingdao Footwear a +1 282 295 %, Etao
        # International a +5 291 567 %, alors que l'actionnaire ne detient rien.
        #
        # LE PARTAGE EST PROPORTIONNEL, PAS NOMINAL. Les minoritaires etaient
        # retranches a leur valeur COMPTABLE, comme un creancier : tout le rabais
        # de realisation retombait alors sur la maison mere. Or ce sont des
        # ACTIONNAIRES — ils subissent la meme decote d'actif, et n'ont pas plus de
        # droit que nous sur ce qui reste apres le passif. Le net realisable se
        # partage donc au prorata des fonds propres.
        #
        # Et le partage n'a lieu QUE SI les deux grandeurs sont mesurees : le
        # `or 0.0` faisait passer des fonds propres attribuables INTROUVABLES pour
        # zero, les minoritaires absorbaient alors la totalite, et la valeur de
        # liquidation tombait mecaniquement a zero pour cause d'ignorance.
        net = realisable - tl
        te, be_att = fund.get("total_equity"), fund.get("book_equity")
        if te is not None and be_att is not None and te > 0 and be_att < te:
            net *= max(be_att, 0.0) / te
        return max(net, 0.0)
    # Bilan incomplet : on retombe sur les fonds propres, faute de pouvoir separer
    # actif et passif. Cette voie SURESTIME les societes endettees — c'est le defaut
    # que l'on vient de corriger — et ne doit servir qu'en dernier recours.
    be = fund.get("book_equity")
    if be is None or be <= 0:
        return 0.0
    liquide = min(be, cash)
    return liquide + taux * max(be - liquide, 0.0)


def erosion_par_les_pertes(fund, valeur):
    """Une valeur d'ACTIF ne tient que si la societe ne consume pas ses fonds propres.

    Les routes fondees sur l'actif retournaient les capitaux propres comme s'ils
    etaient intacts. Or une societe qui PERD de l'argent les absorbe : la fonciere
    allemande Branicks perd 322 M EUR par an sur 863 M EUR d'actif realisable, soit
    un peu plus de deux ans d'autonomie.

    On ne retranche pas les pertes de facon mecanique — cela aneantirait une biotech
    disposant de quatre ans de tresorerie, dont la valeur EST cette tresorerie. On
    pondere par une PROBABILITE DE SURVIE deduite du temps avant epuisement : au-dela
    de cinq ans, la societe a le temps de se redresser et garde toute sa valeur ; en
    deca, son actif est proportionnellement menace. Meme logique que la probabilite
    de survie des societes jeunes, appliquee ici a l'actif.

    La consommation se mesure en TRESORERIE et non en resultat comptable : une
    depreciation d'actifs reduit le resultat sans faire sortir un euro, elle ne
    ronge donc aucune autonomie. Brancher l'erosion sur le resultat net faisait
    perdre deux tiers de sa valeur a une societe qui ENCAISSAIT."""
    # La consommation est celle du FLUX LIBRE : exploitation MOINS investissements.
    # Une societe qui construit son outil de production consomme sa tresorerie par le
    # capex bien plus que par l'exploitation — Wesizwe Platinum brule 562 M ZAR
    # d'exploitation pour 1 238 M ZAR investis dans la mine de Bakubung. Ne regarder
    # que le flux d'exploitation lui pretait sept ans d'autonomie la ou son rapport
    # annuel porte une reserve sur la continuite d'exploitation.
    if valeur is None or valeur <= 0:
        return valeur, None
    p_survie = probabilite_de_survie(fund, valeur)
    if p_survie >= 0.999:
        return valeur, None
    return valeur * p_survie, round(p_survie, 2)


def consommation_est_mesurable(fund) -> bool:
    """Dispose-t-on de quoi mesurer la consommation de tresorerie ?

    `consommation_de_tresorerie` rend 0,0 dans DEUX situations que rien ne
    distinguait : une societe qui ne consomme pas — c'est une bonne nouvelle — et
    une societe dont nous n'avons ni tableau de flux ni resultat — nous n'en savons
    rien. Le zero d'IGNORANCE etait alors lu comme un zero de MESURE : la dimension
    d'autonomie decernait a une coquille vide le meilleur rang de l'univers, et la
    probabilite de survie lui accordait 1,0.

    Cette fonction ne juge pas, elle dit seulement si la question a une reponse.
    """
    return any(fund.get(k) is not None for k in ("cfo", "capex", "net_income"))


def consommation_de_tresorerie(fund):
    """Tresorerie consommee par an — FLUX LIBRE : exploitation MOINS investissements.

    Une societe qui construit son outil de production consomme sa tresorerie par le
    capex bien plus que par l'exploitation : Wesizwe Platinum brule 562 M ZAR
    d'exploitation pour 1 238 M ZAR investis dans la mine de Bakubung. Ne regarder que
    le flux d'exploitation lui pretait sept ans d'autonomie la ou son rapport annuel
    2025 porte une reserve sur la continuite d'exploitation.

    Retourne un nombre POSITIF quand la societe consomme, 0 quand elle degage."""
    cfo, capex, ni = fund.get("cfo"), fund.get("capex"), fund.get("net_income")
    # UN TABLEAU DE FLUX ENTIEREMENT NUL EST UN TABLEAU ABSENT, pas une societe a
    # l'equilibre parfait. Iridium World Communications publie pour 1998 une perte de
    # 107,6 M$ et des flux tous a zero : nous en concluions qu'elle ne consommait
    # rien, et sa valeur d'actif traversait intacte vingt-huit ans de faillite.
    # Le test ne porte pas sur l'age mais sur la COHERENCE INTERNE de la liasse : un
    # resultat non nul avec des flux exactement nuls ne peut pas etre vrai.
    if cfo in (None, 0.0) and (capex in (None, 0.0)) and ni:
        return max(-ni, 0.0)
    flux = None
    if cfo is not None:
        flux = cfo - abs(capex) if capex is not None else cfo
    elif ni is not None:
        flux = ni                              # a defaut, meilleure approximation
    return max(-(flux or 0.0), 0.0)


# HORIZON DE REDRESSEMENT, en annees. CONSTANTE POSEE, et declaree telle.
#
# Elle ne sort d'aucune mesure : nous n'avons pas d'historique de societes en
# consommation de tresorerie suivi jusqu'a leur issue, et la construire exigerait
# precisement le banc d'essai multi-periodes que l'archive mensuelle prepare.
# La forme de la rampe, elle, est justifiee — la probabilite suit le temps avant
# epuisement — mais la VALEUR du diviseur est une convention.
#
# Ce qu'elle coute, pour que le lecteur en juge : elle multiplie lineairement
# l'equite de toute societe deficitaire. A 2,5 ans d'autonomie, la moitie de la
# valeur disparait du seul fait de ce choix ; a 3 ans, 40 % ; a 5 ans, rien. La
# porter a 3 rendrait le modele plus indulgent, a 10 plus severe, dans les memes
# proportions. Un test de sensibilite verrouille cette dependance pour qu'elle
# reste visible.
_HORIZON_DE_REDRESSEMENT = 5.0


def probabilite_de_survie(fund, valeur_en_jeu=None):
    """Probabilite qu'une societe deficitaire tienne assez longtemps pour realiser la
    valeur qu'on lui prete. DEFINITION UNIQUE, partagee par toutes les routes.

    Elle se deduit du TEMPS AVANT EPUISEMENT : au-dela de cinq ans la societe a le
    temps de se redresser et garde toute sa valeur ; en deca, elle est
    proportionnellement menacee.

    Deux defauts corriges ici, tous deux presents dans la route des societes jeunes :
      - la consommation etait mesuree sur le RESULTAT NET. Sur 126 societes routees
        "jeune/deficitaire", 37 — soit 29 % — affichent un resultat et un flux de
        signes OPPOSES : GitLab perd 56 M$ comptables en encaissant 233 M$, NIO perd
        2,16 Md$ en encaissant 431 M$. Une perte comptable ne consomme aucune
        tresorerie ;
      - la probabilite etait PLANCHERISEE A 0,30. Une societe sans tresorerie et en
        pleine consommation se voyait donc creditee d'une chance sur trois de
        survivre, chiffre pose a la main et sans fondement. Elle vaut desormais zero
        quand l'autonomie est nulle, ce qui est la seule reponse defendable."""
    if not consommation_est_mesurable(fund):
        # Ni flux ni resultat : nous ne savons pas si elle consomme. Lui accorder
        # une survie certaine reviendrait a recompenser l'absence de comptes.
        return 0.0
    conso = consommation_de_tresorerie(fund)
    if conso <= 0:
        return 1.0                              # la societe ne consomme pas
    reference = valeur_en_jeu
    if reference is None:
        reference = max(fund.get("cash") or 0.0, 0.0)
    if reference <= 0:
        return 0.0
    autonomie = reference / conso
    # LE TEMPS ECOULE DEPUIS L'OBSERVATION SE RETRANCHE DE L'AUTONOMIE.
    # Un bilan est une PHOTOGRAPHIE datee. Une societe qui consommait sa tresorerie
    # a la date de la photo a continue de la consommer depuis : lui preter aujourd'hui
    # l'autonomie qu'elle avait alors revient a arreter le temps.
    # Iridium World Communications, en faillite depuis 1999, publiait en 1998
    # 119,7 M$ de fonds propres pour 107,6 M$ de perte annuelle — soit onze mois
    # d'autonomie. Nous la valorisions vingt-huit ans plus tard sur ces memes fonds
    # propres, a +5 360 %. GSV (comptes de 2008), CannaBusiness (2013) et Maudore
    # Minerals (2014) relevaient du meme aveuglement.
    # Ce n'est pas un seuil d'anciennete deguise : le temps ecoule n'ecarte rien par
    # lui-meme. Une societe qui ENCAISSE traverse trente ans sans perdre un centime
    # de valeur ici, et une societe qui consomme est penalisee des le premier mois.
    mois = fund.get("age_des_comptes_mois")
    if mois and mois > 0:
        autonomie -= mois / 12.0
    return max(0.0, min(1.0, autonomie / _HORIZON_DE_REDRESSEMENT))


def value_reit(fund):
    """Foncières (REIT) — méthode Damodaran : le FCFF est inapplicable car les
    amortissements immobiliers, purement comptables, écrasent l'EBIT et rendent la
    valeur d'entreprise inférieure à la dette. On capitalise le FFO (résultat net +
    amortissements), mesure de flux propre à l'immobilier. Valorisation CÔTÉ ÉQUITÉ
    (le FFO est après intérêts) : aucune dette n'est soustraite."""
    ni, da = fund.get("net_income"), fund.get("dep_amort")
    cfo = fund.get("cfo")
    # Le FFO n'a de sens que si l'exploitation ENCAISSE. C'est le flux de tresorerie,
    # et non l'EBIT, qui en juge : un promoteur en defaut affiche un resultat net
    # positif issu d'un abandon de creances alors qu'il ne rentre pas un euro, et
    # inversement une fonciere IFRS affiche un EBIT tres negatif tout en encaissant
    # ses loyers. Subordonner cette route a un EBIT positif ecartait Branicks, dont
    # l'EBIT 2024 est de -288,7 M EUR pour un FFO PUBLIE de +52,2 M EUR.
    if cfo is None or cfo <= 0:
        return None
    # FFO = resultat net + amortissements est une formule US GAAP, ou l'immobilier
    # est AMORTI. Sous IFRS il est inscrit a la JUSTE VALEUR : il n'est pas amorti
    # (amortissements quasi nuls) et le resultat net porte les REEVALUATIONS, non
    # monetaires et dans les deux sens. La formule surestime alors la generation de
    # tresorerie quand le marche monte (Fibra UNO : 1,41 Md$ de FFO annonce pour
    # 0,26 Md$ de flux reel) et la rend absurdement negative quand il baisse.
    # Le FLUX D'EXPLOITATION est, lui, immunise : le tableau de flux commence par
    # neutraliser toute variation de juste valeur. C'est donc lui l'ancrage, la
    # formule comptable ne servant plus qu'a le BORNER lorsqu'elle reste positive
    # (elle protege alors d'un flux gonfle par le besoin en fonds de roulement).
    # Verification sur comptes publies : Branicks 2024 annonce un FFO de 52,2 M EUR
    # pour 54,8 M EUR de flux d'exploitation — 5 % d'ecart.
    # La formule comptable ne vaut que si l'immeuble est REELLEMENT AMORTI. Sous
    # IFRS il ne l'est pas : RioCan passe 1 M$ d'amortissements pour 309 M$ de flux
    # d'exploitation, CAP REIT 5 M$ pour 405 M$. "Resultat net + amortissements" y
    # degenere en simple resultat net — lequel, sous juste valeur, ne mesure aucune
    # generation de tresorerie. La retenir bornait le FFO de RioCan a 50 M$ au lieu
    # de 309 M$ et sortait la premiere fonciere du Canada a -85 %.
    # Le test porte sur une grandeur MESURABLE — la materialite de l'amortissement
    # rapportee au flux — et non sur le referentiel comptable declare, que nos
    # donnees ne portent pas. Les foncieres americaines amortissent effectivement
    # (Prologis 2,6 Md$ pour 5,0 Md$ de flux, Realty Income 2,5 pour 4,0) et
    # gardent donc la borne comptable.
    gaap = (ni + da) if (ni is not None and da is not None) else None
    amortissement_materiel = da is not None and da >= 0.20 * cfo
    ffo = (min(gaap, cfo) if (gaap is not None and gaap > 0 and amortissement_materiel)
           else cfo)
    ke, rf = _coe(fund)
    g = min(rf, 0.028)
    ke = max(ke, g + 0.02)                      # écart minimal pour un multiple fini
    # LA CROISSANCE DOIT ETRE FINANCEE. Le FFO est un flux AVANT investissements :
    # le capitaliser avec une croissance perpetuelle revient a faire grandir la
    # fonciere pour toujours sans jamais entretenir ni agrandir ses immeubles.
    # A 8 % de cout des fonds propres et 2,8 % de croissance, l'ancienne formule
    # payait 19,8 fois le FFO ; le haut de la fourchette du marche.
    #
    # On applique l'identite que le moteur DCF s'impose deja en perpetuite : aucune
    # rente excessive, donc un rendement du capital reinvesti egal au cout des
    # fonds propres. La part a retenir vaut alors g / ke, et l'algebre se simplifie
    # remarquablement — le multiple devient (1 + g) / ke, et la croissance
    # n'apporte plus de valeur par elle-meme, exactement comme la valeur terminale
    # du DCF vaut le benefice divise par le WACC quand ROIC = WACC.
    #
    # C'est aussi la regle deja retenue pour les services publics regules, ou la
    # croissance suit g = rendement des fonds propres x taux de retention. Trois
    # methodes du site partagent desormais la meme discipline.
    retenu = g / ke                              # part du FFO reinvestie
    return {"equity_value": ffo * (1 - retenu) * (1 + g) / (ke - g),
            "method": "FFO capitalisé (foncière — Damodaran REIT)",
            "confidence": "moyenne", "ffo": round(ffo, 3),
            "multiple_ffo": round((1 - retenu) * (1 + g) / (ke - g), 2),
            "part_reinvestie": round(retenu, 4)}


def probabilite_de_realisation(fund, F, marge_visee):
    """Probabilite que la societe ATTEIGNE la marge sur laquelle on la valorise.

    Damodaran : quand une valorisation repose sur un REDRESSEMENT — marge normalisee
    d'une societe en perte, marge moyenne de cycle d'un cyclique deprime — la valeur
    doit etre ponderee par la probabilite d'y parvenir, l'alternative etant la
    liquidation. Nous n'appliquions cette ponderation qu'aux societes etiquetees
    "detresse" : partout ailleurs le redressement etait traite comme CERTAIN.

    La probabilite n'est pas supposee mais MESUREE : c'est la part des exercices ou
    la societe a effectivement atteint cette marge. Une societe qui l'a tenue chaque
    annee conserve toute sa valeur ; une societe qui ne l'a atteinte que deux fois
    sur six n'en garde qu'un tiers. C'est la distinction entre difficulte passagere
    et declin structurel, etablie sur les faits plutot que postulee."""
    ms = _hist_margins(F)
    if marge_visee is None or len(ms) < 3:
        return 1.0
    courante = fund.get("operating_margin")
    if courante is not None and courante >= marge_visee:
        return 1.0                       # aucun redressement suppose
    # TOLERANCE ADDITIVE, ET NON MULTIPLICATIVE. La bande de 10 % s'ecrivait
    # `marge_visee * 0.9`, ce qui ABAISSE le seuil pour une cible positive — le
    # comportement voulu — mais le RELEVE pour une cible negative : viser -10 % de
    # marge exigeait alors d'avoir fait -9 %, soit MIEUX que la cible. Un exercice
    # ayant reellement atteint l'objectif etait compte comme un echec, et la
    # probabilite de redressement sous-estimee — precisement pour les societes
    # deficitaires, ou la marge visee est le plus souvent negative.
    # `- 0,1 x |cible|` relache la contrainte dans les deux cas.
    seuil = marge_visee - 0.1 * abs(marge_visee)
    atteints = sum(1 for m in ms if m >= seuil)
    p = atteints / len(ms)
    # AUCUN PLANCHER. Cette ligne planchait la probabilite a 0,15 : une societe qui
    # n'a JAMAIS atteint la marge visee sur son historique se voyait tout de meme
    # crediter 15 % de la valeur de son redressement, via `_pondere_par_realisation`.
    # C'est le jumeau exact du plancher a 0,30 supprime de `probabilite_de_survie`
    # cent lignes plus haut, et pour la meme raison : la docstring de cette fonction
    # promet une probabilite MESUREE — « la part des exercices ou la societe a
    # effectivement atteint cette marge » — et un plancher est precisement ce qui
    # empeche la mesure de dire zero quand la reponse est zero.
    # Ce n'est pas non plus un lissage bayesien defendable : celui-ci vaudrait
    # 1/(n+2) et decroitrait avec le nombre d'exercices, la ou 0,15 ne bougeait pas.
    return float(min(1.0, max(0.0, p)))


def _pondere_par_realisation(r, fund, F, marge_visee):
    """Applique la probabilite de realisation : la valeur du redressement d'un cote,
    la valeur de liquidation de l'autre."""
    if not r or r.get("equity_value") is None:
        return r
    # Une equite NEGATIVE ne signale pas un redressement improbable mais l'ECHEC de
    # l'approche entreprise — typiquement une dette de financement captive (GM
    # Financial, Ford Credit) soustraite comme si elle etait operationnelle. La
    # ponderation la ramenait a zero puis la melangeait a une valeur de liquidation,
    # produisant un resultat positif qui masquait l'echec et empechait la bascule
    # vers le modele cote equite : General Motors ressortait a -78 % au lieu de -4 %.
    # On laisse donc passer le signal intact.
    if r["equity_value"] <= 0:
        return r
    p = probabilite_de_realisation(fund, F, marge_visee)
    if p >= 0.999:
        return r
    liq = valeur_de_liquidation(fund)
    r["equity_value"] = max(r["equity_value"], 0.0) * p + liq * (1.0 - p)
    r["probabilite_realisation"] = round(p, 2)
    r["confidence"] = "faible" if p < 0.5 else r.get("confidence", "moyenne")
    return r


def value_mature_loss(fund, F):
    """Société MATURE en perte temporaire : Damodaran valorise sur bénéfices
    NORMALISÉS plutôt que d'extrapoler une perte conjoncturelle à l'infini.

    La marge normale est celle du CYCLE ENTIER. Ne moyenner que les exercices
    benificiaires selectionnait la moitie favorable de l'histoire et rendait toute
    societe deficitaire mecaniquement sous-evaluee."""
    norm = marge_de_cycle(F)
    if norm is None:
        return None
    r = _dcf_value(fund, margin_override=norm,
                   method="DCF sur bénéfices normalisés (perte temporaire)")
    r["norm_margin"] = round(norm, 4)
    return _pondere_par_realisation(r, fund, F, norm)


def value_cyclical(fund, F):
    """Cyclique : Damodaran valorise sur la marge MOYENNE DE CYCLE, un exercice
    isole ne disant rien de la rentabilite normale d'une mine ou d'un raffineur.
    Le rapport des sommes pondere chaque exercice par son volume — indispensable
    ici, les cycles de matieres premieres faisant varier le chiffre d'affaires du
    simple au triple : CITIC Resources, negociant petrolier, degage 2 % de marge
    sur son cycle la ou la mediane de son secteur declare en affiche 15 %."""
    navg = marge_de_cycle(F)
    if navg is not None:
        r = _dcf_value(fund, margin_override=navg,
                       method="DCF sur bénéfices normalisés (cyclique)")
        r["norm_margin"] = round(navg, 4)
        return _pondere_par_realisation(r, fund, F, navg)
    return _dcf_value(fund, method="DCF FCFF")


def value_young(fund):
    # Marge cible = marge MEDIANE DU SECTEUR : une biotech et un distributeur
    # n'ont aucune raison de converger vers la meme rentabilite.
    om = fund.get("operating_margin")
    target = om if (om is not None and om > 0.05) else sect(fund, "marge", 0.10)
    base = _dcf_value(fund, margin_override=target,
                      method="DCF top-down sur revenus (jeune) × survie")
    # Probabilite de survie : DEFINITION UNIQUE, partagee avec les routes fondees sur
    # l'actif. Elle rapporte la TRESORERIE DISPONIBLE a la consommation annuelle de
    # FLUX LIBRE. La formule locale mesurait la consommation sur le RESULTAT NET —
    # 29 % de cette cohorte affiche un resultat et un flux de signes opposes — et
    # plancherisait la probabilite a 0,30, creditant d'une chance sur trois une
    # societe sans tresorerie aucune.
    surv = probabilite_de_survie(fund)
    # Recuperation en liquidation selon l'intensite d'ACTIFS CORPORELS du
    # secteur : une centrale ou un gisement se revend, un logiciel beaucoup moins.
    liq = valeur_de_liquidation(fund)
    return {"equity_value": max(base["equity_value"], 0) * surv + liq * (1 - surv),
            "method": base["method"], "confidence": "faible", "survival": round(surv, 2)}


def value_distressed(fund, forensic):
    z = (forensic or {}).get("scores", {}).get("altman_z")
    # UN Z INCALCULABLE NE REND PAS LA SITUATION MEILLEURE. Le repli generique de
    # 0,5 est INFERIEUR a toutes les valeurs que la table attribue dans la zone de
    # detresse (0,55 a 0,90) : une societe deja routee ici — fonds propres absorbes
    # par les pertes, ou Z sous le seuil, et sans generation de tresorerie — se
    # voyait donc mieux traitee parce qu'on ne savait pas la mesurer.
    # On retient la probabilite du SEUIL qui l'a fait entrer en detresse : au moins
    # aussi grave que la frontiere, sans inventer pire.
    pdef = default_probability(z, si_inconnu=default_probability(Z_DETRESSE_ROUTE))
    gc = _dcf_value(fund, method="DCF going-concern")
    # Recuperation en liquidation selon l'intensite d'ACTIFS CORPORELS du
    # secteur : une centrale ou un gisement se revend, un logiciel beaucoup moins.
    liq = valeur_de_liquidation(fund)
    return {"equity_value": max(gc["equity_value"], 0) * (1 - pdef) + liq * pdef,
            "method": f"DCF pondéré défaut (p={pdef:.0%}) + liquidation",
            "confidence": "faible", "p_default": round(pdef, 2)}


def value_assetbased(fund):
    """Sociétés pré-revenu / holdings / SPAC : pas de flux à actualiser. Plancher
    = valeur d'actif net comptable (capitaux propres), à défaut la trésorerie nette.
    Conservateur : le pipeline (biotech) ou les gisements (mines) ne sont pas capitalisés."""
    be = fund.get("book_equity")
    cash = fund.get("cash") or 0.0
    debt = fund.get("total_debt") or 0.0
    # LE PORTILLON TESTE LA PRESENCE DES DONNEES, PAS LEUR SIGNE.
    # Il exigeait un actif net POSITIF et renonçait sinon — si bien qu'une societe
    # dont le passif excede l'actif, cas ou la reponse "zero" est justement la bonne,
    # ressortait "valorisation impossible" et repartait dans la cascade de repli, qui
    # lui inventait une valeur. Qingdao Footwear, 10,2 M$ d'actif pour 16,8 M$ de
    # passif, ressortait ainsi a +1 282 295 %.
    if fund.get("total_assets") is None and be is None:
        return None
    nav = be if (be is not None) else (cash - debt)
    # VALEUR REALISABLE, pas valeur comptable. Cette route s'applique justement aux
    # societes incapables de degager des flux : leur actif ne vaut que ce qu'on en
    # tirerait, et le creancier est servi AVANT l'actionnaire. Le calcul est celui de
    # `valeur_de_liquidation` : decote sur l'ACTIF, puis retrait du PASSIF au nominal.
    realisable = valeur_de_liquidation(fund)
    if realisable <= 0:
        # L'actif realisable ne couvre pas les dettes : l'actionnaire ne recoit rien.
        # C'est une reponse, pas un echec de methode — la responsabilite limitee
        # plancherise a zero et la route ne doit surtout pas se rabattre ailleurs.
        return {"equity_value": 0.0,
                "method": "Valeur d'actif net réalisable (pré-revenu / holding)",
                "confidence": "faible", "actif_net_comptable": round(nav, 3),
                "taux_recuperation": round(sect(fund, "recuperation", 0.5), 2),
                "passif_non_couvert": True}
    realisable, consomme = erosion_par_les_pertes(fund, realisable)
    return {"equity_value": realisable,
            **({"probabilite_survie": consomme} if consomme else {}),
            "method": "Valeur d'actif net réalisable (pré-revenu / holding)",
            "confidence": "faible",
            "actif_net_comptable": round(nav, 3),
            "taux_recuperation": round(sect(fund, "recuperation", 0.5), 2)}


def value_stock(ticker: str, fund=None, forensic=None, F=None) -> dict:
    """Valorise un titre via la methode routee. Retourne un resultat unifie."""
    fund = fund or get_fundamentals(ticker)
    if not fund.get("currency_ok", True):
        return {"ticker": ticker.upper(), "ok": False, "reason": "devise introuvable"}
    if F is None:
        F = get_financials(ticker)
    if forensic is None:
        forensic = forensic_analyze(ticker, financials=F,
                                    secteur=(fund or {}).get("sector")) if F else None
    # Mesure portee sur `fund` pour que le moteur DCF y accede sans dependre du
    # module de routage. Neutre par construction quand elle n'est pas mesurable.
    if fund.get("conversion_tresorerie") is None:
        fund["conversion_tresorerie"] = conversion_en_tresorerie(F)
    cat = classify(fund, forensic, F)

    try:
        if cat == "actif_net":
            r = value_assetbased(fund)
        elif cat == "holding":
            r = value_holding(fund, F)
        elif cat == "fonciere":
            r = value_reit(fund)
        elif cat == "reglementee":
            r = value_regulated(fund, F)
        elif cat == "financiere":
            r = value_financial(fund, F)
        elif cat == "cyclique":
            r = value_cyclical(fund, F)
        elif cat == "mature_deficitaire":
            r = value_mature_loss(fund, F)
        elif cat == "jeune/deficitaire":
            r = value_young(fund)
        elif cat == "detresse":
            r = value_distressed(fund, forensic)
        else:
            mn = marge_normalisee(fund, F)
            r = _dcf_value(fund, margin_override=mn,
                           method="DCF FCFF sur marge normalisee" if mn is not None
                           else "DCF FCFF (standard)")
            # Des lors qu'on valorise sur une marge que la societe N'ATTEINT PAS,
            # on suppose un redressement — et un redressement se pondere par sa
            # probabilite. La ponderation n'etait appliquee qu'aux cycliques et aux
            # societes matures en perte ; la route standard, qui normalise pourtant
            # dans les memes termes, tenait le retour a la marge de cycle pour
            # ACQUIS. Yiren Digital, dont le resultat a chute de 96,5 % en 2025
            # apres l'interdiction du pret entre particuliers en Chine, etait ainsi
            # valorisee sur les 16,7 % de marge de la decennie ecoulee comme s'ils
            # devaient revenir a coup sur.
            if mn is not None:
                r = _pondere_par_realisation(r, fund, F, mn)
    except Exception:                              # noqa: BLE001
        r = None

    # L'approche ENTREPRISE (valeur d'entreprise − dette) est invalide quand la
    # dette n'est pas opérationnelle mais de FINANCEMENT (bras financier captif :
    # GM Financial, Ford Credit) : elle produit une équité négative pour une
    # société solvable. Damodaran : basculer sur un modèle CÔTÉ ÉQUITÉ.
    # Secteurs ou le DCF d'entreprise est INAPPLICABLE par construction : le repli
    # ne doit surtout pas y ramener un DCF, sinon on annule la correction meme —
    # Centrica (service public en perte) tombait dans le repli DCF et ressortait a
    # +1 043 %. Pour eux, l'actif net est le seul repli legitime.
    if cat in ("fonciere", "reglementee", "financiere", "holding") and (
            not r or r.get("equity_value") is None):
        r = value_assetbased(fund)
        if r:
            r["method"] = "Valeur d'actif net (methode sectorielle inapplicable)"
            r["confidence"] = "faible"
        return _finalise(ticker, fund, r, cat)

    # Ce repli ne vaut que pour l'approche ENTREPRISE, ou une equite negative signale
    # que la dette soustraite n'etait pas operationnelle. Les methodes deja COTE
    # EQUITE ne soustraient aucune dette : leur zero est une REPONSE, pas un echec.
    # Wesizwe Platinum, dont l'actif realisable ne couvre pas le pret China
    # Development Bank, ressortait ainsi ressuscitee a 0,5 Md$ apres avoir ete
    # correctement valorisee a zero.
    _COTE_EQUITE = ("actif_net", "holding", "fonciere", "reglementee", "financiere")
    if (r and r.get("equity_value") is not None and r["equity_value"] <= 0
            and cat not in _COTE_EQUITE):
        be = fund.get("book_equity")
        if be and be > 0:
            alt = value_financial(fund, F)            # residual income (borné)
            if alt and alt.get("equity_value", 0) > 0:
                alt["method"] = ("Residual income côté équité "
                                 "(dette de financement — approche entreprise inapplicable)")
                alt["confidence"] = "faible"
                r = alt

    # Repli en cascade si la méthode routée échoue (ex. financière à capitaux
    # propres négatifs mais rentable : StepStone) — on ne renonce qu'en dernier recours.
    if not r or r.get("equity_value") is None:
        rev = fund.get("revenue")
        if rev and rev > 0:                            # 1) DCF standard si CA dispo
            try:
                r = _dcf_value(fund, method="DCF FCFF (repli)")
            except Exception:
                r = None
    if not r or r.get("equity_value") is None:         # 2) valeur d'actif net
        r = value_assetbased(fund)
    if not r or r.get("equity_value") is None:         # 3) valeur comptable brute
        be = fund.get("book_equity")
        if be and be > 0:
            r = {"equity_value": be, "method": "Valeur comptable (repli)",
                 "confidence": "très faible"}
        else:
            return {"ticker": ticker.upper(), "ok": False,
                    "reason": "valorisation impossible", "category": cat}

    return _finalise(ticker, fund, r, cat)


def _finalise(ticker, fund, r, cat):
    """Met en forme le resultat : plancher de responsabilite limitee, valeur par
    action coherente avec le cours, upside."""
    if not r or r.get("equity_value") is None:
        return {"ticker": ticker.upper(), "ok": False,
                "reason": "valorisation impossible", "category": cat}
    shares, mcap = fund.get("shares"), fund.get("market_cap")
    eq = max(float(r["equity_value"]), 0.0)         # Md USD — responsabilite limitee : equite >= 0
    vps = eq * 1e9 / shares if shares else None
    upside = (eq / mcap - 1.0) if (mcap and mcap > 0) else None
    return {
        "ticker": ticker.upper(), "ok": True, "category": cat,
        "method": r["method"], "confidence": r.get("confidence"),
        "equity_value": round(eq, 2),
        # 2 decimales ecrasaient la precision des titres sous 1 $ (penny stocks) :
        # la valeur par action ne se reconciliait plus avec l'upside affiche.
        "value_per_share": (round(vps, 2) if abs(vps) >= 1.0 else round(vps, 6)) if vps else None,
        "price": fund.get("price"), "market_cap": mcap,
        "upside": round(upside, 4) if upside is not None else None,
        "extra": {k: v for k, v in r.items()
                  if k not in ("equity_value", "method", "confidence")},
    }


__all__ = ["classify", "value_stock"]

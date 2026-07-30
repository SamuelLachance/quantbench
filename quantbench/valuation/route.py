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
    if ta and ta > 0 and be and rev and rev > 0:
        if (be / ta) < 0.15 and (ta / rev) >= 4.0:
            return True
    if any(k in ind for k in ("bank", "insurance", "mortgage", "thrift")):
        return True
    if any(k in ind for k in ("asset management", "stock exchange", "financial data",
                              "shell", "conglomerate")):
        return False
    if ta and rev and rev > 0:
        return (ta / rev) >= 4.0        # poids du bilan : banques ~20x, Visa ~2,7x
    return True


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
        for ni, eq in zip(F.get("net_income", []), F.get("equity", [])):
            if ni is not None and eq and eq > 0:
                roes.append(ni / eq)
    r = float(np.median(roes)) if len(roes) >= 3 else fund.get("roe")
    return None if r is None else max(-1.0, min(r, 0.40))


def _hist_margins(F):
    """Marges opérationnelles historiques (EBIT/CA)."""
    if not F:
        return []
    return [e / r for e, r in zip(F.get("ebit", []), F.get("revenue", []))
            if e is not None and r]


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

    deficitaire = (ebit is not None and ebit < 0) or (ni is not None and ni < 0)
    # Le Z-score d'Altman est calibre sur des industriels : Altman lui-meme et
    # Damodaran l'excluent pour les financieres ; foncieres et services publics ont
    # structurellement un Z bas sans etre en detresse.
    z_ok = z is not None and not any(s in sec for s in _NO_ALTMAN)

    # --- LA DETRESSE PRIME SUR LE SECTEUR -----------------------------------
    # Une societe dont les pertes ont absorbe les fonds propres, ou dont le Z-score
    # signale un risque de defaut avere, ne se valorise pas comme une consoeur en
    # bonne sante — quel que soit son secteur. L'ordre inverse laissait SunPower,
    # en faillite avec des fonds propres NEGATIFS, etre valorisee comme un cyclique
    # ordinaire a +2 116 % : la branche "energie" repondait avant tout controle de
    # solvabilite. Le meme angle mort touchait l'immobilier, les services publics et
    # les financieres.
    if deficitaire and be is not None and be <= 0:
        return "detresse"                      # fonds propres absorbes par les pertes
    if deficitaire and z_ok and z < Z_DETRESSE_ROUTE:
        return "detresse"

    # --- Routage sectoriel : societes en continuite d'exploitation -----------
    if "real estate" in sec:
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
        # Societe MATURE en perte temporaire (deja rentable par le passe) :
        # Damodaran normalise les benefices — ce n'est pas une societe jeune.
        if sum(1 for m in _hist_margins(F) if m > 0) >= 2:
            return "mature_deficitaire"
        return "jeune/deficitaire"
    if z_ok and z < Z_DETRESSE_ROUTE:
        return "detresse"
    return "standard"


def marge_normalisee(fund, F):
    """Marge NORMALISEE : mediane des marges historiques quand la marge du dernier
    exercice s'en ecarte fortement. Un produit exceptionnel (accord de licence,
    cession) gonfle la marge de l'annee et, extrapole a l'infini, multiplie la
    valeur. Damodaran normalise systematiquement dans ce cas."""
    ms = _hist_margins(F)
    cur = fund.get("operating_margin")
    if len(ms) < 3 or cur is None:
        return None
    med = float(np.median(ms))
    if med <= 0:
        return None
    if cur > 1.5 * med or cur < 0.5 * med:
        return med
    return None


def _dcf_value(fund, margin_override=None, method="DCF FCFF"):
    x, _ = build_dcf_from_fundamentals(fund, margin_override=margin_override)
    res = value_dcf(x)
    return {"equity_value": res["equity_value"], "method": method, "confidence": "moyenne"}


def value_financial(fund, F=None):
    """Financieres DE BILAN — modele de rendement excedentaire (Damodaran).
    V = BE x [1 + (ROE - ke) x A], ou A actualise l'exces de rentabilite sur une
    periode d'avantage concurrentiel de 10 ans, au terme de laquelle ROE = ke (les
    rentes sont competees). Cela remplace un multiplicateur PERPETUEL qui explosait
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
            "method": "Rendement excedentaire a erosion 10 ans (financiere de bilan)",
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
    # Resultat net normalise : mediane de l'historique quand il est disponible.
    nis = [x for x in (F or {}).get("net_income", []) if x is not None]
    ni = float(np.median(nis)) / 1e9 if len(nis) >= 3 else fund.get("net_income")
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
    ta, tl = fund.get("total_assets"), fund.get("total_liab")   # en USD tous deux
    if ta and tl:
        ecart = abs((ta - tl) - be) / max(ta, 1e-9)
        diag["identite_bilan_ok"] = bool(ecart < 0.05)
        if ecart >= 0.05:
            confiance = "faible"

    # 2. Base d'evaluation deduite de la volatilite des capitaux propres
    eqs = [e for e in (F or {}).get("equity", []) if e]
    if len(eqs) >= 3:
        var = [abs(eqs[i] / eqs[i + 1] - 1.0) for i in range(len(eqs) - 1)
               if eqs[i + 1]]
        if var:
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
    de survie des societes jeunes, appliquee ici a l'actif."""
    perte = fund.get("net_income")
    if perte is None or perte >= 0 or valeur is None or valeur <= 0:
        return valeur, None
    annees = valeur / abs(perte)
    p_survie = min(1.0, annees / 5.0)
    if p_survie >= 0.999:
        return valeur, None
    return valeur * p_survie, round(p_survie, 2)


def value_reit(fund):
    """Foncières (REIT) — méthode Damodaran : le FCFF est inapplicable car les
    amortissements immobiliers, purement comptables, écrasent l'EBIT et rendent la
    valeur d'entreprise inférieure à la dette. On capitalise le FFO (résultat net +
    amortissements), mesure de flux propre à l'immobilier. Valorisation CÔTÉ ÉQUITÉ
    (le FFO est après intérêts) : aucune dette n'est soustraite."""
    ni, da, ebit = fund.get("net_income"), fund.get("dep_amort"), fund.get("ebit")
    if ni is None or da is None:
        return None
    # Le FFO n'a de sens que si l'EXPLOITATION est benificiaire. Un promoteur en
    # defaut affiche un resultat net positif issu d'un abandon de creances (gain
    # exceptionnel) alors que son EBIT est tres negatif : capitaliser ce FFO
    # reviendrait a valoriser une faillite comme une rente perenne.
    if ebit is not None and ebit <= 0:
        return None
    ffo = ni + da
    # FFO = resultat net + amortissements est une formule US GAAP, ou l'immobilier
    # est AMORTI. Sous IFRS il est inscrit a la JUSTE VALEUR : il n'est pas amorti
    # (amortissements quasi nuls) et le resultat net contient des PLUS-VALUES DE
    # REEVALUATION, non monetaires. Le FFO ainsi calcule surestime alors largement
    # la generation de tresorerie recurrente — Fibra UNO affichait 1,41 Md$ de FFO
    # pour 0,26 Md$ de flux d'exploitation reel. On borne donc le FFO par le flux
    # d'exploitation, qui est immunise contre les reevaluations comptables.
    cfo = fund.get("cfo")
    if cfo is not None and cfo > 0:
        ffo = min(ffo, cfo)
    if ffo <= 0:
        return None
    ke, rf = _coe(fund)
    g = min(rf, 0.028)
    ke = max(ke, g + 0.02)                      # écart minimal pour un multiple fini
    return {"equity_value": ffo * (1 + g) / (ke - g),
            "method": "FFO capitalisé (foncière — Damodaran REIT)",
            "confidence": "moyenne", "ffo": round(ffo, 3)}


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
    atteints = sum(1 for m in ms if m >= marge_visee * 0.9)
    p = atteints / len(ms)
    return float(min(1.0, max(0.15, p)))


def _pondere_par_realisation(r, fund, F, marge_visee):
    """Applique la probabilite de realisation : la valeur du redressement d'un cote,
    la valeur de liquidation de l'autre."""
    if not r or r.get("equity_value") is None:
        return r
    p = probabilite_de_realisation(fund, F, marge_visee)
    if p >= 0.999:
        return r
    liq = sect(fund, "recuperation", 0.5) * max(fund.get("book_equity") or 0.0, 0.0)
    r["equity_value"] = max(r["equity_value"], 0.0) * p + liq * (1.0 - p)
    r["probabilite_realisation"] = round(p, 2)
    r["confidence"] = "faible" if p < 0.5 else r.get("confidence", "moyenne")
    return r


def value_mature_loss(fund, F):
    """Société MATURE en perte temporaire : Damodaran valorise sur bénéfices
    NORMALISÉS (moyenne des marges positives passées) plutôt que d'extrapoler une
    perte conjoncturelle à l'infini."""
    ms = [m for m in _hist_margins(F) if m > 0]
    if not ms:
        return None
    norm = float(np.mean(ms))
    r = _dcf_value(fund, margin_override=norm,
                   method="DCF sur bénéfices normalisés (perte temporaire)")
    r["norm_margin"] = round(norm, 4)
    return _pondere_par_realisation(r, fund, F, norm)


def value_cyclical(fund, F):
    if F:
        margins = [e / r for e, r in zip(F["ebit"], F["revenue"])
                   if e is not None and r]
        if margins:
            navg = float(np.mean(margins))
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
    cash, ni = fund.get("cash") or 0.0, fund.get("net_income") or 0.0
    burn = -ni if ni < 0 else 0.0
    surv = _clip(0.3 + 0.15 * (cash / burn), 0.3, 0.9) if burn > 0 else 0.85
    # Recuperation en liquidation selon l'intensite d'ACTIFS CORPORELS du
    # secteur : une centrale ou un gisement se revend, un logiciel beaucoup moins.
    liq = sect(fund, "recuperation", 0.5) * max(fund.get("book_equity") or 0.0, 0.0)
    return {"equity_value": max(base["equity_value"], 0) * surv + liq * (1 - surv),
            "method": base["method"], "confidence": "faible", "survival": round(surv, 2)}


def value_distressed(fund, forensic):
    z = (forensic or {}).get("scores", {}).get("altman_z")
    pdef = default_probability(z)      # table de notation Altman/Damodaran
    gc = _dcf_value(fund, method="DCF going-concern")
    # Recuperation en liquidation selon l'intensite d'ACTIFS CORPORELS du
    # secteur : une centrale ou un gisement se revend, un logiciel beaucoup moins.
    liq = sect(fund, "recuperation", 0.5) * max(fund.get("book_equity") or 0.0, 0.0)
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
    nav = be if (be is not None and be > 0) else (cash - debt)
    if nav is None or nav <= 0:
        return None
    # VALEUR REALISABLE, pas valeur comptable. Cette route s'applique justement aux
    # societes incapables de degager des flux : leur actif ne vaut que ce qu'on en
    # tirerait. La tresorerie est realisable a 100 %, le reste subit la decote du
    # secteur selon son intensite d'actifs CORPOREL (une centrale ou un immeuble se
    # revend, un portefeuille de brevets beaucoup moins). Les autres routes de
    # liquidation appliquaient deja cette decote — celle-ci retournait la valeur
    # comptable INTEGRALE, une incoherence interne du modele.
    liquide = min(nav, max(cash, 0.0))
    reste = max(nav - liquide, 0.0)
    realisable = liquide + sect(fund, "recuperation", 0.5) * reste
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
        forensic = forensic_analyze(ticker, financials=F) if F else None
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

    if r and r.get("equity_value") is not None and r["equity_value"] <= 0:
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

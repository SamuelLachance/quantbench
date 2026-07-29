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
from .build_universal import build_dcf_from_fundamentals, country_erp
from .dcf import value_dcf


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _coe(fund):
    """Cout des fonds propres = rf + beta x ERP, ERP incluant la prime de risque
    PAYS (Damodaran) — sinon une banque chinoise ou bresilienne serait actualisee
    au cout du capital americain."""
    rf = market.risk_free_rate()
    beta = fund.get("beta") or 1.1
    erp = country_erp(fund.get("country"))
    return rf + beta * erp, rf


# Le Z-score d'Altman est calibré sur des industriels : Altman lui-même et
# Damodaran l'excluent pour les financières ; foncières et services publics ont
# structurellement un Z bas (actifs lourds, dette élevée) sans être en détresse.
_NO_ALTMAN = ("financial", "real estate", "utilities")
# Route "detresse" reservee au risque de defaut REEL (Z''-EMS < 3.20 ~ notation
# CCC+ ou moins). Entre 3.20 et 5.85 la societe est speculative mais en activite :
# DCF normal, le risque passe par le cout du capital.
Z_DETRESSE_ROUTE = 3.20


_SECT_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "sector_stats.json")
try:
    with open(_SECT_PATH, encoding="utf-8") as _f:
        SECTEURS = _json.load(_f)
except Exception:
    SECTEURS = {}


def sect(fund, cle, defaut):
    """Repere MESURE du secteur (medianes calculees sur l'univers reel par
    scripts/build_sector_stats.py) plutot qu'une constante identique pour tous."""
    s = SECTEURS.get(fund.get("sector") or "")
    v = s.get(cle) if s else None
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
    if any(k in ind for k in ("bank", "insurance", "mortgage", "thrift")):
        return True
    if any(k in ind for k in ("asset management", "stock exchange", "financial data",
                              "shell", "conglomerate")):
        return False
    ta, rev = fund.get("total_assets"), fund.get("revenue")
    if ta and rev and rev > 0:
        return (ta / rev) >= 4.0        # poids du bilan : banques ~20x, Visa ~2,7x
    return True


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
    z = (forensic or {}).get("scores", {}).get("altman_z")
    rev = fund.get("revenue")
    if rev is None or rev <= 0:
        # Pré-revenu (biotech clinique, mineur d'exploration), holding, SPAC :
        # pas de flux à actualiser -> valeur d'actif net (méthode Damodaran).
        return "actif_net"
    if "real estate" in sec:
        return "fonciere"                      # REIT : FFO/NAV, jamais le FCFF
    if "utilities" in sec:
        return "reglementee"                   # service public : cote equite
    if "financial" in sec:
        return "financiere" if _financiere_de_bilan(fund) else "standard"
    if any(s in sec for s in ("energy", "materials")):
        return "cyclique"
    neg = (ebit is not None and ebit < 0) or (ni is not None and ni < 0)
    z_ok = z is not None and not any(s in sec for s in _NO_ALTMAN)
    if neg and z_ok and z < Z_DETRESSE_ROUTE:
        return "detresse"
    if neg:
        # Société MATURE en perte temporaire (déjà rentable par le passé) :
        # Damodaran normalise les bénéfices — ce n'est pas une société jeune.
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


def value_regulated(fund):
    """Services publics REGULES (electricite, gaz, eau) — methode Damodaran.
    Le DCF d'entreprise echoue ici : ces societes sont extremement capitalistiques
    et endettees, si bien que "valeur d'entreprise moins dette" devient negatif
    alors qu'elles sont parfaitement solvables et rentables. Leur rentabilite est
    en outre FIXEE par le regulateur (ROE autorise stable) et leur distribution
    elevee : on les valorise donc COTE EQUITE, par capitalisation des benefices
    avec une croissance FONDAMENTALE coherente g = ROE x taux de retention
    (identite de Damodaran), donc un taux de distribution = 1 - g/ROE."""
    ni, be, roe = fund.get("net_income"), fund.get("book_equity"), fund.get("roe")
    if not ni or ni <= 0 or not be or be <= 0 or not roe or roe <= 0:
        return None
    ke, rf = _coe(fund)
    g = min(rf, 0.028, roe * 0.9)               # g ne peut exceder ce que le ROE finance
    ke = max(ke, g + 0.02)
    payout = max(0.0, 1.0 - g / roe)
    val = ni * payout * (1.0 + g) / (ke - g)
    if val <= 0:
        return None
    return {"equity_value": val, "confidence": "moyenne",
            "method": "Benefices capitalises cote equite (service public regule)",
            "payout": round(payout, 3), "g": round(g, 4)}


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
    if ffo <= 0:
        return None
    ke, rf = _coe(fund)
    g = min(rf, 0.028)
    ke = max(ke, g + 0.02)                      # écart minimal pour un multiple fini
    return {"equity_value": ffo * (1 + g) / (ke - g),
            "method": "FFO capitalisé (foncière — Damodaran REIT)",
            "confidence": "moyenne", "ffo": round(ffo, 3)}


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
    return r


def value_cyclical(fund, F):
    if F:
        margins = [e / r for e, r in zip(F["ebit"], F["revenue"])
                   if e is not None and r]
        if margins:
            navg = float(np.mean(margins))
            r = _dcf_value(fund, margin_override=navg,
                           method="DCF sur bénéfices normalisés (cyclique)")
            r["norm_margin"] = round(navg, 4)
            return r
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
    return {"equity_value": nav,
            "method": "Valeur d'actif net (pré-revenu / holding)",
            "confidence": "faible"}


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
        elif cat == "fonciere":
            r = value_reit(fund)
        elif cat == "reglementee":
            r = value_regulated(fund)
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

    shares, mcap = fund.get("shares"), fund.get("market_cap")
    eq = max(float(r["equity_value"]), 0.0)         # Md USD — responsabilité limitée : équité ≥ 0
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

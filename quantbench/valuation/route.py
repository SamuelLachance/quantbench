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
    if "financial" in sec:
        return "financiere"
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


def _dcf_value(fund, margin_override=None, method="DCF FCFF"):
    x, _ = build_dcf_from_fundamentals(fund, margin_override=margin_override)
    res = value_dcf(x)
    return {"equity_value": res["equity_value"], "method": method, "confidence": "moyenne"}


def value_financial(fund):
    be, roe = fund.get("book_equity"), fund.get("roe")
    if not be or be <= 0 or roe is None:
        return None
    # Un resultat net SUPERIEUR au chiffre d'affaires est impossible en
    # exploitation : il signale un element non recurrent (gain de restructuration,
    # sortie de faillite, cession). Le ROE qui en decoule (WeightWatchers : 332 %)
    # n'est pas reproductible -> on retombe sur la valeur comptable.
    ni, rev = fund.get("net_income"), fund.get("revenue")
    if ni is not None and rev and rev > 0 and ni > rev:
        return {"equity_value": be, "confidence": "faible",
                "method": "Valeur comptable (resultat net non recurrent)"}
    roe = max(-1.0, min(roe, 0.40))          # ROE soutenable plafonne
    ke, rf = _coe(fund)
    g = min(rf, 0.03)
    ke = max(ke, g + 0.01)
    # Excess-return / residual income, avec multiplicateur BORNE (le spread ROE-Ke
    # ne persiste pas a l'infini : atténuation implicite, évite l'explosion en taux bas).
    # Une societe qui DETRUIT massivement ses fonds propres (ROE tres negatif) n'a pas
    # droit au meme plancher : sa valeur comptable est fictive (promoteur immobilier en
    # defaut, actifs a reevaluer). On laisse alors le multiplicateur descendre et on
    # supprime le plancher de 20 % des capitaux propres.
    detruit = roe is not None and roe < -0.20
    mult = max(-0.95 if detruit else -0.6, min((roe - ke) / (ke - g), 4.0))
    val = be * (1 + mult)
    return {"equity_value": max(val, 0.0 if detruit else 0.2 * be),
            "method": "Excess-return (capitaux propres — Damodaran financières)",
            "confidence": "moyenne"}


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
    om = fund.get("operating_margin")
    target = om if (om is not None and om > 0.05) else 0.12
    base = _dcf_value(fund, margin_override=target,
                      method="DCF top-down sur revenus (jeune) × survie")
    cash, ni = fund.get("cash") or 0.0, fund.get("net_income") or 0.0
    burn = -ni if ni < 0 else 0.0
    surv = _clip(0.3 + 0.15 * (cash / burn), 0.3, 0.9) if burn > 0 else 0.85
    liq = 0.5 * (fund.get("book_equity") or 0.0)
    return {"equity_value": max(base["equity_value"], 0) * surv + liq * (1 - surv),
            "method": base["method"], "confidence": "faible", "survival": round(surv, 2)}


def value_distressed(fund, forensic):
    z = (forensic or {}).get("scores", {}).get("altman_z")
    pdef = default_probability(z)      # table de notation Altman/Damodaran
    gc = _dcf_value(fund, method="DCF going-concern")
    liq = 0.5 * (fund.get("book_equity") or 0.0)
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
        elif cat == "financiere":
            r = value_financial(fund)
        elif cat == "cyclique":
            r = value_cyclical(fund, F)
        elif cat == "mature_deficitaire":
            r = value_mature_loss(fund, F)
        elif cat == "jeune/deficitaire":
            r = value_young(fund)
        elif cat == "detresse":
            r = value_distressed(fund, forensic)
        else:
            r = _dcf_value(fund, method="DCF FCFF (standard)")
    except Exception:                              # noqa: BLE001
        r = None

    # L'approche ENTREPRISE (valeur d'entreprise − dette) est invalide quand la
    # dette n'est pas opérationnelle mais de FINANCEMENT (bras financier captif :
    # GM Financial, Ford Credit) : elle produit une équité négative pour une
    # société solvable. Damodaran : basculer sur un modèle CÔTÉ ÉQUITÉ.
    if r and r.get("equity_value") is not None and r["equity_value"] <= 0:
        be = fund.get("book_equity")
        if be and be > 0:
            alt = value_financial(fund)            # residual income (borné)
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
        "value_per_share": round(vps, 2) if vps else None,
        "price": fund.get("price"), "market_cap": mcap,
        "upside": round(upside, 4) if upside is not None else None,
        "extra": {k: v for k, v in r.items()
                  if k not in ("equity_value", "method", "confidence")},
    }


__all__ = ["classify", "value_stock"]

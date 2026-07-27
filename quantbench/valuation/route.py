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
from .build_universal import build_dcf_from_fundamentals
from .dcf import value_dcf


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _coe(fund):
    rf = market.risk_free_rate()
    beta = fund.get("beta") or 1.1
    return rf + beta * 0.045, rf


def classify(fund: dict, forensic: dict | None) -> str:
    sec = (fund.get("sector") or "").lower()
    ebit, ni = fund.get("ebit"), fund.get("net_income")
    z = (forensic or {}).get("scores", {}).get("altman_z")
    if "financial" in sec:
        return "financiere"
    if any(s in sec for s in ("energy", "materials")):
        return "cyclique"
    neg = (ebit is not None and ebit < 0) or (ni is not None and ni < 0)
    if neg and z is not None and z < 1.1:
        return "detresse"
    if neg:
        return "jeune/deficitaire"
    if z is not None and z < 1.1:
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
    ke, rf = _coe(fund)
    g = min(rf, 0.03)
    ke = max(ke, g + 0.01)
    val = be + be * (roe - ke) / (ke - g)      # residual income / excess return
    return {"equity_value": max(val, 0.2 * be),
            "method": "Excess-return (capitaux propres — Damodaran financières)",
            "confidence": "moyenne"}


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
    pdef = 0.5 if z is None else _clip(1.0 - (z - 0.5) / 2.0, 0.05, 0.9)
    gc = _dcf_value(fund, method="DCF going-concern")
    liq = 0.5 * (fund.get("book_equity") or 0.0)
    return {"equity_value": max(gc["equity_value"], 0) * (1 - pdef) + liq * pdef,
            "method": f"DCF pondéré défaut (p={pdef:.0%}) + liquidation",
            "confidence": "faible", "p_default": round(pdef, 2)}


def value_stock(ticker: str, fund=None, forensic=None, F=None) -> dict:
    """Valorise un titre via la methode routee. Retourne un resultat unifie."""
    fund = fund or get_fundamentals(ticker)
    if not fund.get("currency_ok", True):
        return {"ticker": ticker.upper(), "ok": False, "reason": "devise introuvable"}
    if F is None:
        F = get_financials(ticker)
    if forensic is None:
        forensic = forensic_analyze(ticker, financials=F) if F else None
    cat = classify(fund, forensic)

    try:
        if cat == "financiere":
            r = value_financial(fund)
        elif cat == "cyclique":
            r = value_cyclical(fund, F)
        elif cat == "jeune/deficitaire":
            r = value_young(fund)
        elif cat == "detresse":
            r = value_distressed(fund, forensic)
        else:
            r = _dcf_value(fund, method="DCF FCFF (standard)")
    except Exception:                              # noqa: BLE001
        r = None

    if not r or r.get("equity_value") is None:
        be = fund.get("book_equity")
        if be and be > 0:
            r = {"equity_value": be * 2.0, "method": "Repli relatif (P/B ~2×)",
                 "confidence": "très faible"}
        else:
            return {"ticker": ticker.upper(), "ok": False,
                    "reason": "valorisation impossible", "category": cat}

    shares, mcap = fund.get("shares"), fund.get("market_cap")
    eq = r["equity_value"]                          # Md USD
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

"""
quantbench.service
==================
Logique metier reutilisable : construit le payload complet de valorisation d'un
ticker (cas de base + Monte Carlo + histogramme + trajectoires), consommee a la
fois par les scripts CLI et par l'API web.
"""

from __future__ import annotations

import numpy as np

from .valuation import value_dcf, monte_carlo_dcf
from .data import build_dcf_inputs

COMPANY_NAMES = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "NVDA": "NVIDIA Corp.",
    "AMZN": "Amazon.com", "GOOGL": "Alphabet Inc.", "META": "Meta Platforms",
    "NFLX": "Netflix", "ADBE": "Adobe", "AMD": "AMD",
}


def single_ticker_payload(ticker: str, n: int = 20000) -> dict:
    """Payload complet pour la page de valorisation d'un titre."""
    ticker = ticker.upper()
    base, dists, meta = build_dcf_inputs(ticker)
    point = value_dcf(base)
    mc = monte_carlo_dcf(base, dists, n=n, correlations=meta["correlations"],
                         shares_outstanding=meta["shares"],
                         current_market_cap=meta["market_cap"], seed=42)

    vals = mc["equity_values"]
    counts, edges = np.histogram(vals, bins=40)
    hist = [{"x": round(float((edges[i] + edges[i + 1]) / 2), 2), "y": int(counts[i])}
            for i in range(len(counts))]
    years = list(range(1, base.n_years + 1))

    return {
        "ticker": ticker, "name": COMPANY_NAMES.get(ticker, ticker), "unit": "Md$",
        "current_market_cap": meta["market_cap"], "shares_outstanding": meta["shares"],
        "current_price": meta["price"], "fiscal_year_end": meta["fiscal_year_end"],
        "point_estimate": round(float(point["equity_value"]), 1),
        "terminal_value_share": round(float(point["pv_terminal_value"])
                                      / float(point["value_of_operating_assets"]), 3),
        "meta": {k: meta[k] for k in ("operating_margin", "effective_tax",
                 "sales_to_capital", "current_roic", "terminal_roic",
                 "cost_of_debt", "risk_free_rate", "levered_beta", "unlevered_beta",
                 "hist_growth_median", "growth_last_yoy", "growth_cagr3",
                 "debt", "cash_and_non_operating")},
        "mc": {"n_valid": mc["n_valid"], "n_total": mc["n_total"],
               "median": round(mc["median"], 1), "mean": round(mc["mean"], 1),
               "std": round(mc["std"], 1),
               "percentiles": {str(k): round(v, 1) for k, v in mc["percentiles"].items()},
               "prob_undervalued": round(mc["prob_undervalued"], 4),
               "median_upside": round(mc["median_upside"], 4),
               "value_per_share": {k: round(v, 2)
                                   for k, v in mc["value_per_share"].items()}},
        "histogram": hist,
        "series": {"years": years,
                   "revenues": [round(float(v), 1) for v in point["revenues"]],
                   "fcff": [round(float(v), 2) for v in point["fcff"]],
                   "wacc": [round(float(v) * 100, 2) for v in point["wacc"]],
                   "margins": [round(float(v) * 100, 1) for v in point["margins"]],
                   "roic": [round(float(v) * 100, 1) for v in point["roic"]],
                   "growth": [round(float(v) * 100, 1) for v in point["growth"]]},
    }


__all__ = ["single_ticker_payload", "COMPANY_NAMES"]

"""Valorise un vrai ticker (donnees SEC EDGAR + FRED + Yahoo) et exporte le
payload du site.

Usage :  python scripts/value_ticker.py MSFT
"""

import json
import sys
from pathlib import Path

import numpy as np

from quantbench.valuation import value_dcf, monte_carlo_dcf
from quantbench.data import build_dcf_inputs

COMPANY_NAMES = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.",
                 "NVDA": "NVIDIA Corp.", "GOOGL": "Alphabet Inc."}


def run(ticker: str, n: int = 20000, write: bool = True) -> dict:
    base, dists, meta = build_dcf_inputs(ticker)
    point = value_dcf(base)
    mc = monte_carlo_dcf(base, dists, n=n, correlations=meta["correlations"],
                         shares_outstanding=meta["shares"],
                         current_market_cap=meta["market_cap"], seed=42)

    print(f"\n=== {ticker} (exercice {meta['fiscal_year_end']}) ===")
    print(f"  cours {meta['price']} $ | cap {meta['market_cap']} Md$ | "
          f"{meta['shares']:.3f} Md actions")
    print(f"  CA base {base.revenue_base:.1f} Md$ | marge op {meta['operating_margin']:.1%} | "
          f"croissance hist. {meta['hist_growth_median']:+.1%}")
    print(f"  rf {meta['risk_free_rate']:.2%} | beta levier {meta['levered_beta']} -> "
          f"non-levier {meta['unlevered_beta']} | S/C {meta['sales_to_capital']}")
    print(f"  [point] equite {point['equity_value']:,.0f} Md$")
    print(f"  [MC {mc['n_valid']}/{mc['n_total']}] mediane {mc['median']:,.0f} Md$ | "
          f"P(sous-val) {mc['prob_undervalued']:.1%} | upside {mc['median_upside']:+.1%}")
    print(f"  valeur/action {mc['value_per_share']['median']:.2f} $ "
          f"[{mc['value_per_share']['p10']:.2f} ; {mc['value_per_share']['p90']:.2f}] "
          f"vs cours {meta['price']} $")

    vals = mc["equity_values"]
    counts, edges = np.histogram(vals, bins=40)
    hist = [{"x": round(float((edges[i] + edges[i + 1]) / 2), 2), "y": int(counts[i])}
            for i in range(len(counts))]
    years = list(range(1, base.n_years + 1))
    payload = {
        "ticker": ticker, "name": COMPANY_NAMES.get(ticker, ticker), "unit": "Md$",
        "current_market_cap": meta["market_cap"], "shares_outstanding": meta["shares"],
        "current_price": meta["price"], "fiscal_year_end": meta["fiscal_year_end"],
        "point_estimate": round(float(point["equity_value"]), 1),
        "terminal_value_share": round(float(point["pv_terminal_value"])
                                      / float(point["value_of_operating_assets"]), 3),
        "meta": {k: meta[k] for k in ("operating_margin", "effective_tax",
                 "sales_to_capital", "current_roic", "terminal_roic",
                 "cost_of_debt", "risk_free_rate",
                 "levered_beta", "unlevered_beta", "hist_growth_median",
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
    if write:
        out = Path(__file__).resolve().parent.parent / "app" / "data.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  -> ecrit {out}")
    return payload


if __name__ == "__main__":
    tickers = sys.argv[1:] or ["MSFT"]
    for i, t in enumerate(tickers):
        run(t, write=(i == 0))   # le premier ticker alimente le site

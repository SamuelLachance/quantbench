"""Genere une valorisation Monte Carlo de demonstration et l'exporte en JSON.

Profil realiste d'une grande capitalisation tech (chiffres en milliards USD).
Les hypotheses incertaines sont tirees de lois -> distribution de valeur.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

from quantbench.valuation import DcfInputs, value_dcf, monte_carlo_dcf

# --- Cas de base : profil grande-cap tech (unite = milliards USD) ---
base = DcfInputs(
    revenue_base=100.0,
    g1_begin=0.14, g1_end=0.12, g2_begin=0.10, g2_end=0.06,
    g3_begin=0.05, g3_end=0.028,               # croissance terminale = 2.8 %
    len1=3, len2=4, len3=3, conv1=1, conv2=1, conv3=1,
    current_operating_margin=0.28, terminal_operating_margin=0.32,
    margin_converge_start=3,
    current_tax_rate=0.15, marginal_tax_rate=0.25, tax_converge_start=5,
    current_sales_to_capital=1.8, terminal_sales_to_capital=2.2,
    s2c_converge_start=3,
    risk_free_rate=0.042, erp=0.048,
    unlevered_beta=1.10, terminal_unlevered_beta=1.00, beta_converge_start=5,
    current_pretax_kd=0.05, terminal_pretax_kd=0.045, kd_converge_start=5,
    equity_value=430.0, debt_value=120.0, cash_and_non_operating=90.0,
    additional_roic_in_perpetuity=0.02,
)

point = value_dcf(base)
print(f"[Point estimate] Equity value = {point['equity_value']:,.0f} Md$")

# --- Hypotheses incertaines (lois marginales) ---
distributions = {
    "g1_begin":                 stats.norm(loc=0.14, scale=0.03),
    "g2_end":                   stats.norm(loc=0.06, scale=0.02),
    "terminal_operating_margin": stats.norm(loc=0.32, scale=0.03),
    "erp":                      stats.norm(loc=0.048, scale=0.006),
    "unlevered_beta":           stats.norm(loc=1.10, scale=0.15),
    "terminal_sales_to_capital": stats.norm(loc=2.2, scale=0.3),
    "g3_end":                   stats.norm(loc=0.028, scale=0.004),
}
# Correlations plausibles entre hypotheses
correlations = [
    ("terminal_operating_margin", "terminal_sales_to_capital", 0.35),
    ("g1_begin", "g2_end", 0.40),
]

CURRENT_MARKET_CAP = 430.0
SHARES = 15.0  # milliards d'actions

mc = monte_carlo_dcf(base, distributions, n=20000, correlations=correlations,
                     shares_outstanding=SHARES,
                     current_market_cap=CURRENT_MARKET_CAP, seed=42)

print(f"[Monte Carlo] {mc['n_valid']}/{mc['n_total']} scenarios valides")
print(f"  mediane equity   = {mc['median']:,.0f} Md$")
print(f"  P(sous-valorise) = {mc['prob_undervalued']:.1%}")
print(f"  upside median    = {mc['median_upside']:+.1%}")
print(f"  valeur/action    = {mc['value_per_share']['median']:.2f} $ "
      f"[{mc['value_per_share']['p10']:.2f} ; {mc['value_per_share']['p90']:.2f}]")

# --- Histogramme (bins) pour le site ---
vals = mc["equity_values"]
counts, edges = np.histogram(vals, bins=40)
hist = [{"x": float((edges[i] + edges[i + 1]) / 2), "y": int(counts[i])}
        for i in range(len(counts))]

# --- Trajectoires annuelles du cas de base (pour le site) ---
years = list(range(1, base.n_years + 1))
series = {
    "years": years,
    "revenues": [round(float(v), 1) for v in point["revenues"]],
    "fcff": [round(float(v), 2) for v in point["fcff"]],
    "wacc": [round(float(v) * 100, 2) for v in point["wacc"]],
    "margins": [round(float(v) * 100, 1) for v in point["margins"]],
    "roic": [round(float(v) * 100, 1) for v in point["roic"]],
    "growth": [round(float(v) * 100, 1) for v in point["growth"]],
}

payload = {
    "ticker": "DEMO",
    "name": "DEMO Corp (grande-cap tech)",
    "unit": "Md$",
    "current_market_cap": CURRENT_MARKET_CAP,
    "shares_outstanding": SHARES,
    "current_price": round(CURRENT_MARKET_CAP / SHARES, 2),
    "point_estimate": round(float(point["equity_value"]), 1),
    "terminal_value_share": round(float(point["pv_terminal_value"])
                                  / float(point["value_of_operating_assets"]), 3),
    "mc": {
        "n_valid": mc["n_valid"], "n_total": mc["n_total"],
        "median": round(mc["median"], 1),
        "mean": round(mc["mean"], 1),
        "std": round(mc["std"], 1),
        "percentiles": {str(k): round(v, 1) for k, v in mc["percentiles"].items()},
        "prob_undervalued": round(mc["prob_undervalued"], 4),
        "median_upside": round(mc["median_upside"], 4),
        "value_per_share": {k: round(v, 2)
                            for k, v in mc["value_per_share"].items()},
    },
    "histogram": hist,
    "series": series,
}

out = Path(__file__).resolve().parent.parent / "app" / "data.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"\nExporte -> {out}")

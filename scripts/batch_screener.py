"""Valorise un univers de tickers et exporte un tableau de screening.

Usage :
    python scripts/batch_screener.py            # univers NASDAQ par defaut
    python scripts/batch_screener.py AAPL MSFT  # liste explicite

Ecrit app/screener.json (trie par upside decroissant).
Note : univers restreint aux grandes capitalisations non-financieres, pour
lesquelles un DCF FCFF a du sens. Les financieres/REIT/entreprises deficitaires
demandent d'autres modeles (signalees en erreur, pas valorisees a tort).
"""

import json
import sys
import time
from pathlib import Path

from quantbench.valuation import value_dcf, monte_carlo_dcf
from quantbench.data import build_dcf_inputs

# Univers par defaut : principaux composants NASDAQ-100 adaptes au DCF.
NASDAQ = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "COST",
    "NFLX", "AMD", "PEP", "ADBE", "CSCO", "QCOM", "TXN", "AMAT", "INTU",
    "AMGN", "HON", "BKNG", "ISRG", "VRTX", "ADI", "REGN", "LRCX", "MU",
    "PANW", "KLAC", "SNPS", "CDNS", "MRVL", "ORLY", "CRWD", "FTNT", "ADP",
    "NXPI", "MNST", "CTAS", "WDAY", "ABNB", "ROP", "ODFL", "PYPL", "MELI",
]

NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
         "GOOGL": "Alphabet", "META": "Meta", "AVGO": "Broadcom", "TSLA": "Tesla",
         "COST": "Costco", "NFLX": "Netflix", "AMD": "AMD", "PEP": "PepsiCo",
         "ADBE": "Adobe", "CSCO": "Cisco", "QCOM": "Qualcomm", "TXN": "Texas Instr.",
         "AMAT": "Applied Mat.", "INTU": "Intuit", "AMGN": "Amgen", "HON": "Honeywell",
         "BKNG": "Booking", "ISRG": "Intuitive Surg.", "VRTX": "Vertex", "ADI": "Analog Dev.",
         "REGN": "Regeneron", "LRCX": "Lam Research", "MU": "Micron", "PANW": "Palo Alto",
         "KLAC": "KLA", "SNPS": "Synopsys", "CDNS": "Cadence", "MRVL": "Marvell",
         "ORLY": "O'Reilly", "CRWD": "CrowdStrike", "FTNT": "Fortinet", "ADP": "ADP",
         "NXPI": "NXP", "MNST": "Monster", "CTAS": "Cintas", "WDAY": "Workday",
         "ABNB": "Airbnb", "ROP": "Roper", "ODFL": "Old Dominion", "PYPL": "PayPal",
         "MELI": "MercadoLibre"}


def value_one(ticker: str, n: int = 4000) -> dict:
    base, dists, meta = build_dcf_inputs(ticker)
    point = value_dcf(base)
    mc = monte_carlo_dcf(base, dists, n=n, correlations=meta["correlations"],
                         shares_outstanding=meta["shares"],
                         current_market_cap=meta["market_cap"], seed=42)
    return {
        "ticker": ticker, "name": NAMES.get(ticker, ticker),
        "price": meta["price"], "market_cap": meta["market_cap"],
        "median_equity": round(mc["median"], 1),
        "value_per_share": round(mc["value_per_share"]["median"], 2),
        "vps_p10": round(mc["value_per_share"]["p10"], 2),
        "vps_p90": round(mc["value_per_share"]["p90"], 2),
        "upside": round(mc["median_upside"], 4),
        "prob_undervalued": round(mc["prob_undervalued"], 4),
        "op_margin": meta["operating_margin"], "growth": meta["hist_growth_median"],
        "roic": meta["current_roic"], "fy": meta["fiscal_year_end"],
    }


def main(tickers):
    rows, errors = [], []
    for i, t in enumerate(tickers, 1):
        try:
            rows.append(value_one(t))
            r = rows[-1]
            print(f"[{i}/{len(tickers)}] {t:6s} upside {r['upside']:+7.1%}  "
                  f"P(sv) {r['prob_undervalued']:5.1%}")
        except Exception as e:
            errors.append({"ticker": t, "error": str(e)[:140]})
            print(f"[{i}/{len(tickers)}] {t:6s} ECHEC : {str(e)[:80]}")
        time.sleep(0.25)                      # throttle SEC (~10 req/s max)

    # Isole les resultats implausibles (souvent : split d'action -> decalage
    # prix post-split / actions pre-split, ou tag de donnee aberrant).
    suspects = [r for r in rows if not (-0.98 <= r["upside"] <= 2.0)]
    for r in suspects:
        r["note"] = "upside implausible — probable split d'action ou donnee aberrante"
    clean = [r for r in rows if -0.98 <= r["upside"] <= 2.0]
    clean.sort(key=lambda r: -r["upside"])
    payload = {"universe_size": len(tickers), "n_ok": len(clean),
               "n_suspect": len(suspects), "n_fail": len(errors),
               "rows": clean, "suspects": suspects, "errors": errors}
    out = Path(__file__).resolve().parent.parent / "app" / "screener.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n-> {out}  ({len(clean)} classes, {len(suspects)} suspects, "
          f"{len(errors)} echecs)")
    return payload


if __name__ == "__main__":
    main(sys.argv[1:] or NASDAQ)

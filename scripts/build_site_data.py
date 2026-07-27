"""Build unifie des donnees du site statique (pour GitHub Pages + cron).

Genere, dans app/ :
  - screener.json          : tableau de l'univers (classes + suspects)
  - data.json              : ticker vedette (page d'accueil)
  - tickers/<T>.json       : payload complet par ticker (recherche statique)
  - tickers/_index.json    : liste des tickers disponibles

Usage :  python scripts/build_site_data.py [--n 4000] [--featured MSFT] [TICKERS...]
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_screener import NASDAQ, NAMES                     # noqa: E402
from quantbench.service import single_ticker_payload         # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app"
TICK = APP / "tickers"


def _row(p):
    mc, meta = p["mc"], p["meta"]
    return {"ticker": p["ticker"], "name": NAMES.get(p["ticker"], p["ticker"]),
            "price": p["current_price"], "market_cap": p["current_market_cap"],
            "median_equity": mc["median"],
            "value_per_share": mc["value_per_share"]["median"],
            "vps_p10": mc["value_per_share"]["p10"],
            "vps_p90": mc["value_per_share"]["p90"],
            "upside": mc["median_upside"], "prob_undervalued": mc["prob_undervalued"],
            "op_margin": meta["operating_margin"], "growth": meta["hist_growth_median"],
            "roic": meta["current_roic"], "fy": p["fiscal_year_end"]}


def main(universe, n=4000, featured="MSFT"):
    TICK.mkdir(parents=True, exist_ok=True)
    rows, suspects, errors, ok = [], [], [], []
    for i, t in enumerate(universe, 1):
        try:
            p = single_ticker_payload(t, n=n)
            (TICK / f"{t}.json").write_text(json.dumps(p, ensure_ascii=False),
                                            encoding="utf-8")
            ok.append(t)
            r = _row(p)
            if -0.98 <= r["upside"] <= 2.0:
                rows.append(r)
            else:
                r["note"] = "upside implausible — split ou donnee aberrante"
                suspects.append(r)
            print(f"[{i}/{len(universe)}] {t:6s} upside {r['upside']:+7.1%}")
        except Exception as e:                                # noqa: BLE001
            errors.append({"ticker": t, "error": str(e)[:140]})
            print(f"[{i}/{len(universe)}] {t:6s} ECHEC : {str(e)[:70]}")
        time.sleep(0.2)

    rows.sort(key=lambda r: -r["upside"])
    (APP / "screener.json").write_text(json.dumps(
        {"universe_size": len(universe), "n_ok": len(rows), "n_suspect": len(suspects),
         "n_fail": len(errors), "rows": rows, "suspects": suspects, "errors": errors},
        indent=2), encoding="utf-8")

    feat = featured if featured in ok else (ok[0] if ok else None)
    if feat:
        p = json.loads((TICK / f"{feat}.json").read_text(encoding="utf-8"))
        (APP / "data.json").write_text(json.dumps(p, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    (TICK / "_index.json").write_text(json.dumps(sorted(ok)), encoding="utf-8")
    print(f"\nOK: {len(rows)} classes, {len(suspects)} suspects, {len(errors)} echecs "
          f"| vedette={feat}")


if __name__ == "__main__":
    args = sys.argv[1:]
    n, featured, tickers = 4000, "MSFT", []
    while args:
        a = args.pop(0)
        if a == "--n":
            n = int(args.pop(0))
        elif a == "--featured":
            featured = args.pop(0)
        else:
            tickers.append(a.upper())
    main(tickers or NASDAQ, n=n, featured=featured)

"""Genere les donnees du site US (backbone SEC Data Sets + marche yfinance).

Pour chaque titre : valorisation routee (Damodaran) + forensique + news badgees
+ etats financiers 4 ans. Ecrit :
  app/us/<TICKER>.json   profil complet
  app/us/_screener.json  tableau (dashboard + screener personnalisable)

Usage : python scripts/build_us_site.py [--quarters 8] [--no-news] [TICKERS...]
"""

import json
import sys
import time
from pathlib import Path

from quantbench.data import edgar, sec_datasets as ds, market
from quantbench.forensics import analyze
from quantbench.valuation.route import value_stock
from quantbench.news import fetch_and_classify

# Univers V1 : grandes capitalisations US diversifiees (extensible).
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "ORCL", "NFLX",
    "AMD", "ADBE", "CRM", "CSCO", "INTC", "QCOM", "TXN", "AMAT", "INTU", "IBM",
    "NOW", "UBER", "PYPL", "MU", "ADI", "LRCX", "KLAC", "PANW", "SNPS", "CDNS",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "SPGI",
    "V", "MA", "BRK-B", "JNJ", "UNH", "LLY", "PFE", "MRK", "ABBV", "TMO",
    "ABT", "DHR", "BMY", "AMGN", "GILD", "CVS", "MDT", "ISRG", "VRTX", "REGN",
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY", "KMI", "WMB",
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "LOW", "HD",
    "TGT", "MDLZ", "CL", "MO", "PM", "GIS", "KHC", "MNST", "KDP", "STZ",
    "BA", "CAT", "GE", "HON", "UPS", "RTX", "LMT", "DE", "UNP", "MMM",
    "T", "VZ", "TMUS", "CMCSA", "DIS", "F", "GM", "DAL", "ADP", "NEE",
    "DUK", "SO", "LIN", "SHW", "FCX", "NEM", "NUE", "DOW", "PLD", "AMT",
]

APP = Path(__file__).resolve().parent.parent / "app"
US = APP / "us"


def resolve_ciks(tickers):
    out = {}
    for t in tickers:
        try:
            out[t] = edgar.get_cik(t)
        except Exception:                          # noqa: BLE001
            pass
    return out


def _statements(F):
    """Extrait un tableau d'etats financiers (Md USD) pour la page profil."""
    if not F:
        return None
    B = 1e9
    def col(k):
        return [round(v / B, 2) if v is not None else None for v in F.get(k, [])]
    return {"years": F["years"], "revenue": col("revenue"), "ebit": col("ebit"),
            "net_income": col("net_income"), "cfo": col("cfo"),
            "total_assets": col("total_assets"), "equity": col("equity"),
            "total_debt": col("total_debt")}


def build_one(ticker, entry, with_news=True):
    F = ds.extract_financials(entry)
    q = market.quote(ticker)
    fund = ds.extract_fundamentals(entry, ticker, q)
    forensic = analyze(ticker, financials=F) if F else None
    val = value_stock(ticker, fund=fund, forensic=forensic, F=F)
    if not val.get("ok"):
        return None, None
    news = fetch_and_classify(ticker, limit=8) if with_news else []
    profile = {
        "ticker": ticker, "name": fund.get("name"), "sector": fund.get("sector"),
        "industry": fund.get("industry"), "sic": fund.get("sic"),
        "valuation": val,
        "fundamentals": {k: fund.get(k) for k in (
            "price", "market_cap", "shares", "beta", "revenue", "ebit",
            "net_income", "total_debt", "cash", "book_equity",
            "operating_margin", "roe")},
        "forensics": forensic,
        "statements": _statements(F),
        "news": news,
        "filing_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={entry['cik']}&type=10-K&dateb=&owner=include&count=10",
    }
    row = {
        "ticker": ticker, "name": fund.get("name"), "sector": fund.get("sector"),
        "category": val.get("category"), "method": val.get("method"),
        "price": val.get("price"), "market_cap": val.get("market_cap"),
        "value_per_share": val.get("value_per_share"), "upside": val.get("upside"),
        "confidence": val.get("confidence"),
        "op_margin": fund.get("operating_margin"), "roe": fund.get("roe"),
        "piotroski": (forensic or {}).get("scores", {}).get("piotroski_f"),
        "beneish_flag": (forensic or {}).get("scores", {}).get("beneish_flag"),
        "n_flags": len((forensic or {}).get("flags", [])),
    }
    return profile, row


def main(tickers, quarters=8, with_news=True):
    US.mkdir(parents=True, exist_ok=True)
    ciks = resolve_ciks(tickers)
    print(f"CIKs résolus : {len(ciks)}/{len(tickers)}")
    facts = ds.build_facts(ds.recent_quarters(quarters), ciks=ciks.values())
    print(f"Facts SEC : {len(facts)} sociétés")
    rows, ok, fail = [], 0, 0
    for i, t in enumerate(tickers, 1):
        cik = ciks.get(t)
        entry = facts.get(str(int(cik))) if cik else None
        if not entry:
            fail += 1
            continue
        try:
            profile, row = build_one(t, entry, with_news=with_news)
            if profile:
                (US / f"{t}.json").write_text(json.dumps(profile, ensure_ascii=False),
                                              encoding="utf-8")
                rows.append(row)
                ok += 1
                if i % 10 == 0 or i <= 6:
                    up = "n/a" if row["upside"] is None else f"{row['upside']*100:+.0f}%"
                    print(f"[{i}/{len(tickers)}] {t:6s} {row['category']:12s} up={up}")
            else:
                fail += 1
        except Exception as e:                     # noqa: BLE001
            fail += 1
            print(f"[{i}/{len(tickers)}] {t:6s} ECHEC : {str(e)[:60]}")
        time.sleep(0.1)

    # tri par upside ; suspects isoles
    clean = [r for r in rows if r["upside"] is not None and -0.98 <= r["upside"] <= 3.0]
    suspects = [r for r in rows if r not in clean]
    clean.sort(key=lambda r: -(r["upside"] or -9))
    (US / "_screener.json").write_text(json.dumps(
        {"n_ok": len(clean), "n_suspect": len(suspects), "n_fail": fail,
         "generated_tickers": len(tickers), "rows": clean, "suspects": suspects},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {US}  ({len(clean)} classés, {len(suspects)} suspects, {fail} échecs)")


if __name__ == "__main__":
    args = sys.argv[1:]
    quarters, with_news, tickers = 8, True, []
    while args:
        a = args.pop(0)
        if a == "--quarters":
            quarters = int(args.pop(0))
        elif a == "--no-news":
            with_news = False
        else:
            tickers.append(a.upper())
    main(tickers or UNIVERSE, quarters=quarters, with_news=with_news)

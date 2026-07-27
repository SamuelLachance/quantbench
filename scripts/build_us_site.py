"""Génère le site US (tout le NASDAQ) — rebuild quotidien.

Pipeline : univers NASDAQ (nasdaqtrader) -> CIKs SEC -> fondamentaux SEC Data Sets
(bulk, cache) + données de marché yfinance (fast_info, threadé) -> valorisation
routée Damodaran + forensique + news badgées + PDF de synthèse.

Sorties (déployées comme artefact Pages, NON commitées -> pas de gonflement) :
  app/us/<T>.json, app/us/pdf/<T>.pdf, app/us/_screener.json

Usage : python scripts/build_us_site.py [--quarters 8] [--workers 24]
        [--limit N] [--no-news] [--no-pdf] [TICKERS...]
"""

import concurrent.futures as cf
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import requests

from quantbench.data import edgar, sec_datasets as ds
from quantbench.data.sec_datasets import sic_to_sector
from quantbench.forensics import analyze
from quantbench.valuation.route import value_stock
from quantbench.valuation.build_universal import project
from quantbench.news import fetch_and_classify
from quantbench.reports import financial_summary_pdf

warnings.filterwarnings("ignore")

APP = Path(__file__).resolve().parent.parent / "app"
US = APP / "us"
PDF = US / "pdf"

# Bêta par défaut par secteur (evite un appel reseau par titre a grande echelle).
SECTOR_BETA = {"Financial Services": 1.05, "Energy": 1.1, "Basic Materials": 1.15,
               "Other": 1.1, "Unknown": 1.1}


def nasdaq_universe():
    """Symboles NASDAQ (actions ordinaires) via Nasdaq Trader."""
    r = requests.get("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    lines = r.text.splitlines()
    hdr = lines[0].split("|")
    idx = {c: i for i, c in enumerate(hdr)}
    out = []
    for ln in lines[1:]:
        f = ln.split("|")
        if len(f) < len(hdr):
            continue
        sym = f[idx["Symbol"]]
        if f[idx.get("ETF", -1)] == "Y" or f[idx.get("Test Issue", -1)] == "Y":
            continue
        if not sym.isalpha():                      # exclut warrants/units/preferred
            continue
        out.append(sym)
    return out


def market_quote(ticker):
    """Marché + activité via yfinance .info (un appel : prix/actions/capi/beta/
    résumé/industrie), repli fast_info si .info échoue."""
    import yfinance as yf
    from quantbench.data.market import fx_to_usd
    tk = yf.Ticker(ticker)
    out = {"price": None, "shares": None, "market_cap": None, "beta": None,
           "summary": None, "industry": None}
    try:
        info = tk.info or {}
        cur = (info.get("currency") or "USD").upper()
        fx = fx_to_usd(cur) or 1.0
        p = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        out["price"] = p * fx if p else None
        out["shares"] = info.get("sharesOutstanding")
        mc = info.get("marketCap")
        out["market_cap"] = mc * fx / 1e9 if mc else None
        out["beta"] = info.get("beta")
        s = info.get("longBusinessSummary")
        out["summary"] = s[:650] if s else None
        out["industry"] = info.get("industry")
    except Exception:
        pass
    if out["price"] is None:
        try:
            fi = tk.fast_info
            cur = (getattr(fi, "currency", None) or "USD").upper()
            fx = fx_to_usd(cur) or 1.0
            p = getattr(fi, "last_price", None)
            out["price"] = p * fx if p else None
            out["shares"] = getattr(fi, "shares", None)
            mc = getattr(fi, "market_cap", None)
            out["market_cap"] = mc * fx / 1e9 if mc else None
        except Exception:
            pass
    return out


def _shortterm(price, beta, days=63):
    """Cône de projection court terme (bandes de probabilité log-normales) —
    illustratif (méthode quant type Simons : incertitude, pas prédiction)."""
    import math
    if not price:
        return None
    sigma = 0.22 + 0.12 * max(0.0, (beta or 1.0) - 0.8)      # vol annualisée proxy
    pts = []
    for step in range(0, days + 1, 7):
        tau = step / 252.0
        drift = -0.5 * sigma * sigma * tau
        band = 1.2816 * sigma * math.sqrt(tau)
        pts.append({"d": step,
                    "p50": round(price * math.exp(drift), 2),
                    "p10": round(price * math.exp(drift - band), 2),
                    "p90": round(price * math.exp(drift + band), 2)})
    return {"days": days, "sigma": round(sigma, 3), "points": pts}


def _results_summary(F):
    if not F:
        return None
    B = 1e9
    rev, ni = F.get("revenue"), F.get("net_income")
    r0 = rev[0] if rev and rev[0] is not None else None
    r1 = rev[1] if rev and len(rev) > 1 and rev[1] is not None else None
    n0 = ni[0] if ni and ni[0] is not None else None
    return {"fiscal_year": F["years"][0],
            "revenue": round(r0 / B, 1) if r0 else None,
            "rev_growth": round(r0 / r1 - 1, 4) if (r0 and r1) else None,
            "net_income": round(n0 / B, 1) if n0 else None,
            "net_margin": round(n0 / r0, 4) if (n0 and r0) else None}


def _statements(F):
    if not F:
        return None
    B = 1e9
    col = lambda k: [round(v / B, 2) if v is not None else None for v in F.get(k, [])]
    return {"years": F["years"], "revenue": col("revenue"), "ebit": col("ebit"),
            "net_income": col("net_income"), "cfo": col("cfo"),
            "total_assets": col("total_assets"), "equity": col("equity"),
            "total_debt": col("total_debt")}


def build_one(ticker, entry, with_news=True, with_pdf=True):
    F = ds.extract_financials(entry)
    q = market_quote(ticker)
    if not q.get("beta"):
        q["beta"] = SECTOR_BETA.get(sic_to_sector(entry.get("sic")), 1.1)
    fund = ds.extract_fundamentals(entry, ticker, q)
    if not fund.get("price"):
        return None, None
    forensic = analyze(ticker, financials=F) if F else None
    val = value_stock(ticker, fund=fund, forensic=forensic, F=F)
    if not val.get("ok"):
        return None, None
    news = fetch_and_classify(ticker, limit=8) if with_news else []
    try:
        proj = project(fund, years=20)
    except Exception:
        proj = None
    # lien direct vers le dernier 10-K (dossier de dépôt SEC)
    adsh = (entry.get("adsh") or "").replace("-", "")
    report_url = (f"https://www.sec.gov/Archives/edgar/data/{int(entry['cik'])}/{adsh}/"
                  if adsh else None)
    profile = {
        "ticker": ticker, "name": fund.get("name"), "sector": fund.get("sector"),
        "industry": q.get("industry") or fund.get("industry"), "sic": fund.get("sic"),
        "summary": q.get("summary"),
        "valuation": val,
        "fundamentals": {k: fund.get(k) for k in (
            "price", "market_cap", "shares", "beta", "revenue", "ebit",
            "net_income", "total_debt", "cash", "book_equity", "operating_margin", "roe")},
        "forensics": forensic, "statements": _statements(F), "news": news,
        "projection": proj, "results_summary": _results_summary(F),
        "shortterm": _shortterm(fund.get("price"), fund.get("beta")),
        "filing_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={entry['cik']}&type=10-K",
        "report_url": report_url,
        "pdf_url": None,
    }
    if with_pdf:
        p = financial_summary_pdf(profile, str(PDF / f"{ticker}.pdf"))
        if p:
            profile["pdf_url"] = f"pdf/{ticker}.pdf"
    (US / f"{ticker}.json").write_text(json.dumps(profile, ensure_ascii=False),
                                       encoding="utf-8")
    f = forensic or {}
    row = {"ticker": ticker, "name": fund.get("name"), "sector": fund.get("sector"),
           "category": val.get("category"), "method": val.get("method"),
           "price": val.get("price"), "market_cap": val.get("market_cap"),
           "value_per_share": val.get("value_per_share"), "upside": val.get("upside"),
           "confidence": val.get("confidence"), "op_margin": fund.get("operating_margin"),
           "roe": fund.get("roe"),
           "piotroski": f.get("scores", {}).get("piotroski_f"),
           "beneish_flag": f.get("scores", {}).get("beneish_flag"),
           "n_flags": len(f.get("flags", []))}
    return profile, row


def main(tickers, quarters=8, workers=24, with_news=True, with_pdf=True, limit=None):
    US.mkdir(parents=True, exist_ok=True)
    PDF.mkdir(parents=True, exist_ok=True)
    if limit:
        tickers = tickers[:limit]
    print(f"Univers : {len(tickers)} symboles")

    ciks = {}
    for t in tickers:
        try:
            ciks[t] = edgar.get_cik(t)
        except Exception:
            pass
    print(f"CIKs SEC résolus : {len(ciks)}")

    t0 = time.time()
    facts = ds.build_facts(ds.recent_quarters(quarters), ciks=ciks.values())
    print(f"Facts SEC : {len(facts)} sociétés ({time.time()-t0:.0f}s)")

    work = [(t, facts.get(str(int(ciks[t])))) for t in tickers if t in ciks]
    work = [(t, e) for t, e in work if e]
    print(f"À valoriser : {len(work)}")

    rows, done, fail = [], 0, 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, t, e, with_news, with_pdf): t for t, e in work}
        for fut in cf.as_completed(futs):
            try:
                _, row = fut.result()
                if row:
                    rows.append(row)
                    done += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
            if (done + fail) % 100 == 0:
                print(f"  {done+fail}/{len(work)} (ok={done})")

    # Bande d'upside crédible (au-delà = quasi toujours du bruit de données micro-cap).
    clean = [r for r in rows if r["upside"] is not None and -0.95 <= r["upside"] <= 1.5]
    suspects = [r for r in rows if r not in clean]
    clean.sort(key=lambda r: -(r["upside"] or -9))
    (US / "_screener.json").write_text(json.dumps(
        {"n_ok": len(clean), "n_suspect": len(suspects), "n_fail": fail,
         "universe": len(tickers),
         "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
         "rows": clean, "suspects": suspects},
        ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {done} valorisés, {len(suspects)} suspects, {fail} échecs "
          f"| total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    kw = {"quarters": 8, "workers": 24, "with_news": True, "with_pdf": True, "limit": None}
    tickers = []
    while args:
        a = args.pop(0)
        if a == "--quarters": kw["quarters"] = int(args.pop(0))
        elif a == "--workers": kw["workers"] = int(args.pop(0))
        elif a == "--limit": kw["limit"] = int(args.pop(0))
        elif a == "--no-news": kw["with_news"] = False
        elif a == "--no-pdf": kw["with_pdf"] = False
        else: tickers.append(a.upper())
    main(tickers or nasdaq_universe(), **kw)

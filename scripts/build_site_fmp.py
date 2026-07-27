"""Génère le site (US + Canada) 100 % via FMP (Ultimate) — rebuild quotidien.

Univers screener (NASDAQ + TSX + TSXV, hors ETF/fonds) -> fondamentaux bulk FMP
+ profils + historique + news -> valorisation routée Damodaran + forensique +
signal court terme + PDF. Tout converti en USD.

Sorties : app/us/<T>.json, app/us/pdf/<T>.pdf, app/us/_screener.json, _shortterm.json

Usage : QUANTBENCH_SEC_UA=... FMP_API_KEY=... python scripts/build_site_fmp.py
        [--exchanges NASDAQ,TSX,TSXV] [--years 6] [--workers 20] [--limit N]
"""

import concurrent.futures as cf
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

from quantbench.data import fmp
from quantbench.data.sec_fundamentals import annual_report_docs
from quantbench.forensics import analyze
from quantbench.valuation.route import value_stock
from quantbench.valuation.build_universal import project
from quantbench.shortterm.predict import predict as st_predict
from quantbench.reports import financial_summary_pdf

warnings.filterwarnings("ignore")

APP = Path(__file__).resolve().parent.parent / "app"
US = APP / "us"
PDF = US / "pdf"

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:
    _ET = timezone.utc


def _now_et():
    try:
        return datetime.now(_ET).strftime("%Y-%m-%d %H:%M ET")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _statements(F):
    if not F:
        return None
    col = lambda k: [round(v / 1e9, 2) if v is not None else None for v in F.get(k, [])]
    return {"years": F["years"], "revenue": col("revenue"), "ebit": col("ebit"),
            "net_income": col("net_income"), "cfo": col("cfo"),
            "total_assets": col("total_assets"), "equity": col("equity"),
            "total_debt": col("total_debt")}


def _results_summary(F):
    if not F:
        return None
    B = 1e9
    rev, ni = F.get("revenue"), F.get("net_income")
    r0 = rev[0] if rev and rev[0] is not None else None
    r1 = rev[1] if rev and len(rev) > 1 and rev[1] is not None else None
    n0 = ni[0] if ni and ni[0] is not None else None
    return {"fiscal_year": F["years"][0], "revenue": round(r0 / B, 1) if r0 else None,
            "rev_growth": round(r0 / r1 - 1, 4) if (r0 and r1) else None,
            "net_income": round(n0 / B, 1) if n0 else None,
            "net_margin": round(n0 / r0, 4) if (n0 and r0) else None}


def build_one(symbol, sr, with_news=True, with_pdf=True):
    entry = fmp.statements(symbol)                 # 3 appels par-ticker (débit élevé)
    if len(set(entry["income"]) & set(entry["balance"])) < 2:
        return None, None
    prof = fmp.profile(symbol)
    desc = {"description": prof.get("description"), "industry": prof.get("industry")}
    F = fmp.financials_from_fmp(entry)
    fund = fmp.fundamentals_from_fmp(symbol, sr, entry, desc)
    if not fund or not fund.get("price") or not fund.get("revenue"):
        return None, None
    forensic = analyze(symbol, financials=F) if F else None
    val = value_stock(symbol, fund=fund, forensic=forensic, F=F)
    if not val.get("ok"):
        return None, None
    signal = st_predict(fmp.history_closes(symbol))
    news = fmp.news(symbol, limit=8) if with_news else []
    try:
        proj = project(fund, years=20)
    except Exception:
        proj = None
    cik = fund.get("cik")
    ard = annual_report_docs(cik) if cik else {"ars_pdf": None, "tenk": None}
    filing = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K"
              if cik else None)
    profile = {
        "ticker": symbol, "name": fund.get("name"), "sector": fund.get("sector"),
        "industry": fund.get("industry"), "summary": fund.get("summary"),
        "exchange": sr.get("exchange"), "valuation": val,
        "fundamentals": {k: fund.get(k) for k in (
            "price", "market_cap", "shares", "beta", "revenue", "ebit", "net_income",
            "total_debt", "cash", "book_equity", "operating_margin", "roe")},
        "forensics": forensic, "statements": _statements(F), "news": news,
        "projection": proj, "results_summary": _results_summary(F), "shortterm": signal,
        "report_url": ard.get("tenk"), "ars_pdf_url": ard.get("ars_pdf"),
        "filing_url": filing, "pdf_url": None,
    }
    if with_pdf:
        if financial_summary_pdf(profile, str(PDF / f"{symbol}.pdf")):
            profile["pdf_url"] = f"pdf/{symbol}.pdf"
    (US / f"{symbol}.json").write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
    f = forensic or {}
    row = {"ticker": symbol, "name": fund.get("name"), "sector": fund.get("sector"),
           "exchange": sr.get("exchange"), "category": val.get("category"),
           "method": val.get("method"), "price": val.get("price"),
           "market_cap": val.get("market_cap"), "value_per_share": val.get("value_per_share"),
           "upside": val.get("upside"), "confidence": val.get("confidence"),
           "op_margin": fund.get("operating_margin"), "roe": fund.get("roe"),
           "piotroski": f.get("scores", {}).get("piotroski_f"),
           "beneish_flag": f.get("scores", {}).get("beneish_flag"),
           "n_flags": len(f.get("flags", [])),
           "p_up": (signal or {}).get("p_up"), "p_down": (signal or {}).get("p_down"),
           "bias": (signal or {}).get("bias")}
    return profile, row


def main(exchanges, years=6, workers=20, with_news=True, with_pdf=True, limit=None):
    US.mkdir(parents=True, exist_ok=True)
    PDF.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    uni = fmp.screener(exchanges)
    syms = sorted(uni, key=lambda s: -(uni[s].get("market_cap") or 0))   # plus grosses d'abord
    if limit:
        syms = syms[:limit]
    print(f"Univers {exchanges} : {len(syms)} sociétés")

    work = syms
    print(f"Valorisation par-ticker (FMP), {workers} threads…")
    rows, done, fail = [], 0, 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, s, uni[s], with_news, with_pdf): s for s in work}
        for fut in cf.as_completed(futs):
            try:
                _, row = fut.result()
                if row:
                    rows.append(row); done += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
            if (done + fail) % 200 == 0:
                print(f"  {done+fail}/{len(work)} (ok={done})")

    clean = [r for r in rows if r["upside"] is not None and -0.95 <= r["upside"] <= 1.5]
    suspects = [r for r in rows if r not in clean]
    clean.sort(key=lambda r: -(r["upside"] or -9))
    (US / "_screener.json").write_text(json.dumps(
        {"n_ok": len(clean), "n_suspect": len(suspects), "n_fail": fail,
         "universe": len(syms), "updated": _now_et(), "rows": clean, "suspects": suspects},
        ensure_ascii=False), encoding="utf-8")
    st = [r for r in rows if r.get("p_up") is not None]
    st.sort(key=lambda r: -(r["p_up"] or 0))
    (US / "_shortterm.json").write_text(json.dumps(
        {"n": len(st), "updated": _now_et(), "rows": st}, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {done} valorisés, {len(suspects)} suspects, {fail} échecs "
          f"| court terme {len(st)} | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    kw = {"years": 6, "workers": 20, "with_news": True, "with_pdf": True, "limit": None}
    exch = ["NASDAQ", "TSX", "TSXV"]
    while args:
        a = args.pop(0)
        if a == "--exchanges": exch = args.pop(0).split(",")
        elif a == "--years": kw["years"] = int(args.pop(0))
        elif a == "--workers": kw["workers"] = int(args.pop(0))
        elif a == "--limit": kw["limit"] = int(args.pop(0))
        elif a == "--no-news": kw["with_news"] = False
        elif a == "--no-pdf": kw["with_pdf"] = False
    main(exch, **kw)

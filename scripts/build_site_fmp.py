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

import numpy as np
from scipy import stats

from quantbench.data import fmp
from quantbench.data.sec_fundamentals import annual_report_docs
from quantbench.data.market import risk_free_rate
from quantbench.forensics import analyze
from quantbench.valuation import monte_carlo_dcf
from quantbench.valuation.route import value_stock
from quantbench.valuation.build_universal import project, build_dcf_from_fundamentals
from quantbench.shortterm.predict import predict as st_predict
from quantbench.reports import financial_summary_pdf


def _mc_stats(eq, mcap, shares):
    eq = np.asarray([v for v in eq if v == v and np.isfinite(v)], dtype=float)
    if eq.size < 50:
        return None
    perc = {str(p): round(float(np.percentile(eq, p)), 1)
            for p in (5, 10, 25, 50, 75, 90, 95)}
    counts, edges = np.histogram(eq, bins=30)
    hist = [{"x": round(float((edges[i] + edges[i + 1]) / 2), 1), "y": int(counts[i])}
            for i in range(len(counts))]
    vps = lambda q: round(float(np.percentile(eq, q)) * 1e9 / shares, 2) if shares else None
    return {"median": round(float(np.median(eq)), 1),
            "mean": round(float(eq.mean()), 1), "std": round(float(eq.std()), 1),
            "percentiles": perc, "histogram": hist,
            "vps": {"p10": vps(10), "p50": vps(50), "p90": vps(90)},
            "prob_undervalued": round(float((eq > mcap).mean()), 4) if mcap else None,
            "n": int(eq.size)}


def _route_margin(fund, category, F):
    """Marge opérationnelle normalisée selon la catégorie (miroir de route.py) :
    cyclique = marge moyenne du cycle ; jeune/déficitaire = marge cible (pas la
    marge courante négative, sinon valeur nulle à l'infini). Sinon None (marge brute)."""
    if category == "cyclique" and F:
        ms = [e / r for e, r in zip(F.get("ebit", []), F.get("revenue", []))
              if e is not None and r]
        if ms:
            return float(np.mean(ms))
    if category == "jeune/deficitaire":
        om = fund.get("operating_margin")
        return om if (om is not None and om > 0.05) else 0.12
    return None


def run_mc(fund, category, F=None, forensic=None, n=10000, rf=None):
    """Monte Carlo de valorisation COHÉRENT avec le routage Damodaran : marge
    normalisée (cyclique) ou cible (jeune/déficitaire), pondération survie/défaut,
    et équité plancher à 0 (responsabilité limitée : une action ne vaut jamais < 0).
    Excess-return simulé pour les financières. `rf` imposé pour le backtest."""
    shares, mcap = fund.get("shares"), fund.get("market_cap")
    if category == "actif_net":
        return None                                # valeur d'actif net = point, pas de MC
    try:
        if category == "financiere":
            be, roe = fund.get("book_equity"), fund.get("roe")
            beta = fund.get("beta") or 1.1
            if not be or be <= 0 or roe is None:
                return None
            rf = rf if rf is not None else risk_free_rate()
            g = min(rf, 0.03)
            rng = np.random.default_rng(42)
            roes = rng.normal(roe, max(0.02, abs(roe) * 0.2), n)
            betas = rng.normal(beta, 0.15, n)
            ke = np.maximum(rf + betas * 0.045, g + 0.01)
            mult = np.clip((roes - ke) / (ke - g), -0.6, 4.0)
            eq = np.maximum(be * (1 + mult), 0.2 * be)
        else:
            margin = _route_margin(fund, category, F)
            base, _ = build_dcf_from_fundamentals(fund, margin_override=margin, rf=rf)
            dists = {
                "g1_begin": stats.norm(base.g1_begin, max(0.02, abs(base.g1_begin) * 0.35)),
                "terminal_operating_margin": stats.norm(
                    base.terminal_operating_margin, max(0.015, abs(base.terminal_operating_margin) * 0.15)),
                "erp": stats.norm(base.erp, 0.005),
                "unlevered_beta": stats.norm(base.unlevered_beta, 0.15),
                "current_roic": stats.norm(base.current_roic, max(0.03, abs(base.current_roic) * 0.2)),
                "terminal_roic": stats.norm(base.terminal_roic, 0.01),
                "g3_end": stats.norm(base.g3_end, 0.003),
            }
            eq = monte_carlo_dcf(base, dists, n=n, current_market_cap=mcap,
                                 seed=42)["equity_values"]
            eq = np.maximum(eq, 0.0)                    # responsabilité limitée : équité ≥ 0
            # Pondération de la catégorie (miroir de route.value_young/value_distressed)
            if category == "jeune/deficitaire":
                ni = fund.get("net_income") or 0.0
                burn = -ni if ni < 0 else 0.0
                cash = fund.get("cash") or 0.0
                surv = min(max(0.3 + 0.15 * (cash / burn), 0.3), 0.9) if burn > 0 else 0.85
                liq = 0.5 * (fund.get("book_equity") or 0.0)
                eq = eq * surv + liq * (1.0 - surv)
            elif category == "detresse":
                z = (forensic or {}).get("scores", {}).get("altman_z")
                pdef = 0.5 if z is None else min(max(1.0 - (z - 0.5) / 2.0, 0.05), 0.9)
                liq = 0.5 * (fund.get("book_equity") or 0.0)
                eq = eq * (1.0 - pdef) + liq * pdef
    except Exception:
        return None
    return _mc_stats(eq, mcap, shares)

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
    if len(set(entry["income"]) & set(entry["balance"])) < 1:   # ≥1 an suffit (actif net)
        return None, None
    prof = fmp.profile(symbol)
    desc = {"description": prof.get("description"), "industry": prof.get("industry")}
    F = fmp.financials_from_fmp(entry)
    fund = fmp.fundamentals_from_fmp(symbol, sr, entry, desc)
    # Plus d'exigence de CA : les sociétés pré-revenu (biotech, mines, holdings) sont
    # valorisées sur l'actif net. Il faut juste un prix (pour l'upside).
    if not fund or not fund.get("price"):
        return None, None
    # Intégrité des données : rejeter les fondamentaux corrompus (nombre d'actions
    # implausible, cap quasi nulle) qui produisent des valorisations aberrantes.
    sh, mc0 = fund.get("shares"), fund.get("market_cap")
    if not sh or sh < 100_000 or not mc0 or mc0 < 0.002:   # <100k actions ou <2 M$
        return None, None
    forensic = analyze(symbol, financials=F) if F else None
    val = value_stock(symbol, fund=fund, forensic=forensic, F=F)
    if not val.get("ok"):
        return None, None
    # Monte Carlo : l'upside affiché est basé sur la MÉDIANE de la simulation
    mc = run_mc(fund, val.get("category"), F=F, forensic=forensic)
    if mc and fund.get("market_cap"):
        if mc["vps"].get("p50") is not None:
            val["value_per_share"] = mc["vps"]["p50"]
        val["upside"] = round(mc["median"] / fund["market_cap"] - 1.0, 4)
        val["upside_basis"] = "monte_carlo"
    # Garde-fou final : un upside démesuré (>500%) révèle une donnée résiduelle non
    # fiable (aucune société n'est crédiblement sous-évaluée de +5× via un DCF conservateur ;
    # les vraies sous-évaluations plafonnent ~150-200%).
    if val.get("upside") is not None and val["upside"] > 5.0:
        return None, None
    signal = st_predict(fmp.history_closes(symbol))
    news = fmp.news(symbol, limit=8) if with_news else []
    try:
        proj = project(fund, years=20)
    except Exception:
        proj = None
    cik = fund.get("cik")
    ard = annual_report_docs(cik) if cik else {"ars_pdf": None, "tenk": None, "documents": []}
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
        "montecarlo": mc, "documents": ard.get("documents", []),
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

    # AUCUN filtre arbitraire : on montre tous les titres valorisés. Seule exclusion
    # = upside non-fini (NaN/inf), qui n'est pas un nombre valide (pas une opinion du modèle).
    clean = [r for r in rows if r["upside"] is not None and np.isfinite(r["upside"])]
    invalid = [r for r in rows if r["upside"] is None or not np.isfinite(r["upside"])]
    clean.sort(key=lambda r: -(r["upside"] if r["upside"] is not None else -9))
    (US / "_screener.json").write_text(json.dumps(
        {"n_ok": len(clean), "n_suspect": 0, "n_invalid": len(invalid), "n_fail": fail,
         "universe": len(syms), "updated": _now_et(), "rows": clean, "suspects": []},
        ensure_ascii=False), encoding="utf-8")
    st = [r for r in rows if r.get("p_up") is not None]
    st.sort(key=lambda r: -(r["p_up"] or 0))
    (US / "_shortterm.json").write_text(json.dumps(
        {"n": len(st), "updated": _now_et(), "rows": st}, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {done} valorisés ({len(clean)} affichés, {len(invalid)} upside non-fini), "
          f"{fail} sans données | court terme {len(st)} | total {time.time()-t0:.0f}s")


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

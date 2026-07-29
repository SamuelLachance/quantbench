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
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

from quantbench.data import fmp
from quantbench.data.validate import valider
from quantbench.data.repair import reparer
from quantbench.data.sec_fundamentals import annual_report_docs
from quantbench.data.market import risk_free_rate
from quantbench.forensics import analyze
from quantbench.valuation import monte_carlo_dcf
from quantbench.valuation.route import value_stock
from quantbench.valuation.build_universal import (project, build_dcf_from_fundamentals,
                                                  country_erp, pays_exploitation)
from quantbench.shortterm.predict import predict as st_predict
from quantbench.reports import financial_summary_pdf


def _arrondi(v):
    """Arrondi a precision constante en CHIFFRES SIGNIFICATIFS : 2 decimales
    au-dela de 1 $, 6 en dessous (penny stocks)."""
    return round(v, 2) if abs(v) >= 1.0 else round(v, 6)


def _mc_stats(eq, mcap, shares):
    eq = np.asarray([v for v in eq if v == v and np.isfinite(v)], dtype=float)
    if eq.size < 50:
        return None
    # Precision : arrondir a 0,1 Md$ (soit 100 M$ !) rendait la valeur par action
    # incoherente avec l'upside pour toute societe de taille modeste — l'utilisateur
    # lisait un cours cible et un pourcentage qui ne se reconciliaient pas.
    perc = {str(p): round(float(np.percentile(eq, p)), 6)
            for p in (5, 10, 25, 50, 75, 90, 95)}
    counts, edges = np.histogram(eq, bins=30)
    hist = [{"x": round(float((edges[i] + edges[i + 1]) / 2), 1), "y": int(counts[i])}
            for i in range(len(counts))]
    # Arrondi ADAPTATIF : 2 decimales ecrasaient toute precision sur les titres a
    # moins de 1 $ — une valeur de 0,004 $ devenait 0,00 et l'upside implicite
    # affiche passait a -100 %, incoherent avec le pourcentage calcule.
    vps = lambda q: (_arrondi(float(np.percentile(eq, q)) * 1e9 / shares)
                     if shares else None)
    return {"median": round(float(np.median(eq)), 6),
            "mean": round(float(eq.mean()), 6), "std": round(float(eq.std()), 6),
            "percentiles": perc, "histogram": hist,
            "vps": {"p10": vps(10), "p50": vps(50), "p90": vps(90)},
            "prob_undervalued": round(float((eq > mcap).mean()), 4) if mcap else None,
            "n": int(eq.size)}


def _route_margin(fund, category, F):
    """Marge opérationnelle normalisée selon la catégorie (miroir de route.py) :
    cyclique = marge moyenne du cycle ; mature en perte = moyenne des marges
    POSITIVES passées ; jeune/déficitaire = marge cible (pas la marge courante
    négative, sinon valeur nulle à l'infini). Sinon None (marge brute)."""
    ms = [e / r for e, r in zip((F or {}).get("ebit", []), (F or {}).get("revenue", []))
          if e is not None and r]
    if category == "cyclique" and ms:
        return float(np.mean(ms))
    if category == "mature_deficitaire":
        pos = [m for m in ms if m > 0]
        if pos:
            return float(np.mean(pos))
    if category == "jeune/deficitaire":
        om = fund.get("operating_margin")
        return om if (om is not None and om > 0.05) else 0.12
    return None


_CAT_DCF = ("standard", "cyclique", "mature_deficitaire", "jeune/deficitaire", "detresse")
_METH_NON_DCF = ("Residual income", "Valeur comptable", "Valeur d'actif net",
                 "FFO capitalisé", "Benefices capitalises", "Rendement excedentaire",
                 "Actif net reevalue")

_POURQUOI = {
    "financiere": ("Banque, assurance ou courtier : la dette est leur MATIERE PREMIERE, "
                   "pas un financement — la notion de valeur d'entreprise n'a donc aucun "
                   "sens. On valorise directement les capitaux propres par le rendement "
                   "excedentaire (ROE face au cout des fonds propres), erode sur 10 ans "
                   "car les rentes se competent."),
    "fonciere": ("Fonciere (REIT) : les amortissements immobiliers, purement comptables, "
                 "ecrasent le resultat operationnel et rendraient la valeur d'entreprise "
                 "inferieure a la dette. Damodaran capitalise le FFO (resultat net + "
                 "amortissements), cote equite — aucune dette n'est soustraite."),
    "reglementee": ("Service public regule : rentabilite FIXEE par le regulateur, activite "
                    "tres capitalistique et endettee. Le DCF d'entreprise sortirait une "
                    "equite negative sur une societe parfaitement solvable. Benefices "
                    "capitalises cote equite, croissance = ROE x taux de retention."),
    "cyclique": ("Secteur cyclique (energie, matieres premieres) : la marge de l'exercice "
                 "est trompeuse selon la position dans le cycle. DCF sur benefices "
                 "NORMALISES — marge moyenne du cycle."),
    "mature_deficitaire": ("Societe mature en perte TEMPORAIRE : extrapoler la perte a "
                           "l'infini donnerait une valeur nulle. On normalise sur la "
                           "moyenne des marges positives passees."),
    "jeune/deficitaire": ("Societe jeune ou deficitaire : DCF descendant sur le chiffre "
                          "d'affaires avec une marge cible egale a la MEDIANE DU SECTEUR, "
                          "pondere par une probabilite de survie deduite de la tresorerie "
                          "et du rythme de consommation de cash."),
    "detresse": ("Risque de defaut avere (Z-score d'Altman) : DCF d'exploitation pondere "
                 "par la probabilite de defaut issue de la table de notation, complete "
                 "par la valeur de liquidation."),
    "actif_net": ("Aucun chiffre d'affaires (biotech clinique, minier d'exploration, "
                  "holding, SPAC) : il n'y a aucun flux a actualiser. Valeur d'actif net "
                  "comptable — le portefeuille de brevets ou les gisements ne sont pas "
                  "capitalises, methode volontairement prudente."),
    "standard": ("DCF FCFF classique : flux de tresorerie disponibles pour l'entreprise, "
                 "actualises au cout moyen pondere du capital, croissance decroissant vers "
                 "celle de l'economie."),
}


def _methodologie(fund, val, F):
    """Methode retenue, sa justification sectorielle, et les hypotheses REELLEMENT
    utilisees pour ce titre (toutes deduites de ses donnees et de son secteur)."""
    from quantbench.valuation.build_universal import (country_erp, tax_rate,
                                                      pays_exploitation)
    cat = val.get("category")
    pays = pays_exploitation(fund)
    h = {"beta": fund.get("beta"),
         "prime_risque_actions_pct": round(country_erp(pays) * 100, 2),
         "taux_impot_pct": round(tax_rate(pays) * 100, 1),
         "pays_exploitation": pays, "pays_declare": fund.get("country"),
         "secteur": fund.get("sector"), "industrie": fund.get("industry")}
    try:
        h["taux_sans_risque_pct"] = round(risk_free_rate() * 100, 2)
    except Exception:
        pass
    if cat in _CAT_DCF and not str(val.get("method") or "").startswith(_METH_NON_DCF):
        try:
            _, meta = build_dcf_from_fundamentals(
                fund, margin_override=_route_margin(fund, cat, F))
            h.update({"croissance_initiale_pct": round(meta["g_start"] * 100, 2),
                      "marge_operationnelle_pct": round(meta["op_margin"] * 100, 2),
                      "roic_courant_pct": round(meta["cur_roic"] * 100, 2),
                      "ventes_sur_capital": round(meta["s2c"], 2)})
        except Exception:
            pass
    for k in ("roe_normalise", "pb_implicite", "payout", "ffo", "norm_margin",
              "p_default", "survival"):
        if (val.get("extra") or {}).get(k) is not None:
            h[k] = val["extra"][k]
    return {"categorie": cat, "methode": val.get("method"),
            "pourquoi": _POURQUOI.get(cat, ""), "hypotheses": h,
            "dcf_applicable": proj_ok(cat, val)}


def proj_ok(cat, val):
    return bool(cat in _CAT_DCF
                and not str(val.get("method") or "").startswith(_METH_NON_DCF))


def run_mc(fund, category, F=None, forensic=None, method=None, n=10000, rf=None):
    """Monte Carlo de valorisation COHÉRENT avec le routage Damodaran : marge
    normalisée (cyclique) ou cible (jeune/déficitaire), pondération survie/défaut,
    et équité plancher à 0 (responsabilité limitée : une action ne vaut jamais < 0).
    Excess-return simulé pour les financières. `rf` imposé pour le backtest."""
    shares, mcap = fund.get("shares"), fund.get("market_cap")
    if category in ("actif_net", "fonciere", "reglementee", "financiere", "holding"):
        return None      # actif net, FFO, benefices regules, rendement excedentaire :
                         # valeurs POINT — la simulation reproduisait un autre modele
    # La simulation reproduit le DCF FCFF. Si la valorisation RETENUE vient d'une
    # autre methode (bascule cote equite pour dette de financement, valeur
    # comptable), simuler le DCF puis ECRASER l'upside avec sa mediane annulerait
    # justement la correction : General Motors repassait de -70 % a -100 %.
    if method and str(method).startswith(_METH_NON_DCF):
        return None
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
            erp_c = country_erp(pays_exploitation(fund))  # prime du pays d'exploitation
            ke = np.maximum(rf + betas * erp_c, g + 0.01)
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
                liq = 0.5 * max(fund.get("book_equity") or 0.0, 0.0)
                eq = np.maximum(eq * surv + liq * (1.0 - surv), 0.0)
            elif category == "detresse":
                from quantbench.forensics.scores import default_probability
                z = (forensic or {}).get("scores", {}).get("altman_z")
                pdef = default_probability(z)
                liq = 0.5 * max(fund.get("book_equity") or 0.0, 0.0)
                eq = np.maximum(eq * (1.0 - pdef) + liq * pdef, 0.0)
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
    # VALIDATION DES ENTREES : une donnee non verifiee ne donne pas une valorisation
    # approximative mais une valorisation FAUSSE. On ne valorise que ce dont on
    # peut repondre, et on RETOURNE le motif quand ce n'est pas le cas.
    motifs = valider(fund, F, entry)
    reparations = []
    if motifs:
        # On tente de CORRIGER avant de renoncer : capitalisation de la cotation
        # d'origine, identite du bilan, douze mois glissants, devise pivot.
        reparations = reparer(symbol, fund, F, entry, motifs)
        if reparations:
            motifs = valider(fund, F, entry)          # revalidation apres reparation
    sh, mc0 = fund.get("shares"), fund.get("market_cap")
    if not sh or sh < 100_000:
        motifs.append("nombre d'actions implausible")
    if not mc0 or mc0 < 0.002:
        motifs.append("capitalisation inferieure a 2 M$")
    if motifs:
        return None, {"__rejet__": motifs, "ticker": symbol,
                      "reparations_tentees": reparations}
    # PLAUSIBILITE : des capitaux propres 200 fois superieurs a la capitalisation
    # ne traduisent pas une opportunite mais une donnee non credible — erreur
    # d'unites (Oncotelic : 262 000 milliards $ de fonds propres pour 16 M$ de
    # capitalisation), serie obligataire prise pour une action (obligations de la
    # Tennessee Valley Authority), ou capitalisation perimee.
    be0 = fund.get("book_equity")
    # COHERENCE COMPTES : aucune societe ne realise un chiffre d'affaires plusieurs
    # fois superieur a son actif total (les plus legeres en capital tournent a 2-3x).
    # Un tel ecart signale une entite de FINANCEMENT qui publie le CA consolide du
    # groupe sans en porter le bilan : "KKR Group Finance Co. IX LLC" declarait
    # 19,5 Md$ de CA pour 1 Md$ d'actif et 14 M$ de fonds propres — c'est une
    # obligation cotee du groupe KKR, pas une action.

    # Un SERVICE PUBLIC REGULE se traite entre 1 et 2,5 fois ses fonds propres :
    # sa rentabilite est FIXEE par le regulateur. Un titre "de service public" cote
    # au quart de ses capitaux propres n'est donc pas une action ordinaire mais une
    # OBLIGATION COTEE de filiale -- Entergy Mississippi "1M BD 66" et Entergy New
    # Orleans cotent ~20 $ avec un coupon FIXE de 1,22 $, soit 6,1 % de rendement.
    if (fund.get("sector") or "").lower().startswith("utilit") and be0 and mc0 and be0 > 3 * mc0:
        return None, None
    forensic = analyze(symbol, financials=F) if F else None
    val = value_stock(symbol, fund=fund, forensic=forensic, F=F)
    if not val.get("ok"):
        return None, None
    # Monte Carlo : l'upside affiché est basé sur la MÉDIANE de la simulation
    mc = run_mc(fund, val.get("category"), F=F, forensic=forensic,
                method=val.get("method"))
    if mc and fund.get("market_cap"):
        if mc["vps"].get("p50") is not None:
            val["value_per_share"] = mc["vps"]["p50"]
        # Responsabilite limitee : l'upside ne peut pas descendre sous -100 %.
        val["upside"] = round(max(mc["median"] / fund["market_cap"] - 1.0, -1.0), 4)
        val["upside_basis"] = "monte_carlo"
    signal = st_predict(fmp.history_closes(symbol))
    news = fmp.news(symbol, limit=8) if with_news else []
    # La projection 20 ans n'a de sens que si la valorisation retenue EST un DCF.
    # Afficher un echeancier de flux actualises sous une banque valorisee par
    # rendement excedentaire, ou sous une fonciere valorisee au FFO, etait
    # trompeur. Elle utilise en outre EXACTEMENT la meme marge que la valorisation.
    proj = None
    if (val.get("category") in _CAT_DCF
            and not str(val.get("method") or "").startswith(_METH_NON_DCF)):
        try:
            proj = project(fund, years=20,
                           margin_override=_route_margin(fund, val.get("category"), F))
        except Exception:
            proj = None
    methodo = _methodologie(fund, val, F)
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
        "projection": proj, "methodologie": methodo,
        "reparations_donnees": reparations,
        "results_summary": _results_summary(F), "shortterm": signal,
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
           "bias": (signal or {}).get("bias"),
           # Signal court terme : le SCORE est continu et classable ; p_up n'existe
           # que si la calibration hors echantillon a demontre un pouvoir predictif.
           "st_score": (signal or {}).get("score"),
           "reversal": (signal or {}).get("reversal"),
           "momentum": (signal or {}).get("momentum"),
           "vol_annual": (signal or {}).get("vol_annual"),
           "trend": (signal or {}).get("trend"),
           "st_calibrated": (signal or {}).get("calibrated")}
    return profile, row


def _write_aggregates(all_rows, universe_size, fail):
    """Écrit _screener.json + _shortterm.json depuis l'ensemble des lignes."""
    clean = [r for r in all_rows if r["upside"] is not None and np.isfinite(r["upside"])]
    invalid = [r for r in all_rows if r["upside"] is None or not np.isfinite(r["upside"])]
    clean.sort(key=lambda r: -(r["upside"] if r["upside"] is not None else -9))
    (US / "_screener.json").write_text(json.dumps(
        {"n_ok": len(clean), "n_suspect": 0, "n_invalid": len(invalid), "n_fail": fail,
         "universe": universe_size, "updated": _now_et(), "rows": clean, "suspects": []},
        ensure_ascii=False), encoding="utf-8")
    # Classement sur le SCORE (continu, injectif) et non sur une probabilite bornee
    # qui creait des ex aequo massifs. On publie aussi le resultat de la calibration
    # hors echantillon : si elle ne demontre aucun edge, aucune probabilite n'est
    # affichee — le classement reste indicatif, la mesure est publiee telle quelle.
    st = [r for r in all_rows if r.get("st_score") is not None]
    st.sort(key=lambda r: -(r["st_score"] or 0))
    try:
        calib = json.loads((Path(__file__).resolve().parent.parent / "quantbench" /
                            "shortterm" / "shortterm_calibration.json")
                           .read_text(encoding="utf-8"))
    except Exception:
        calib = {}
    (US / "_shortterm.json").write_text(json.dumps(
        {"n": len(st), "updated": _now_et(),
         "calibrated": bool(calib.get("calibrated")),
         "calibration": calib.get("metrics"), "fiabilite": calib.get("fiabilite"),
         "rows": st}, ensure_ascii=False), encoding="utf-8")
    return len(clean), len(invalid), len(st)


def combine_shards(n_shards):
    """Job de combinaison : fusionne les _shard_*.json (produits par les jobs parallèles)
    en _screener.json + _shortterm.json. Les fiches par-titre + PDF sont déjà en place
    (téléchargées depuis les artefacts de chaque shard)."""
    all_rows, universe, fail = [], 0, 0
    for i in range(n_shards):
        p = US / f"_shard_{i}.json"
        if not p.exists():
            print(f"  ATTENTION : shard {i} manquant")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        all_rows.extend(d.get("rows", []))
        universe += d.get("universe", 0)
        fail += d.get("fail", 0)
    nok, ninv, nst = _write_aggregates(all_rows, universe, fail)
    print(f"-> COMBINE {n_shards} shards : {nok} affichés, {ninv} non-fini, "
          f"{fail} sans données | court terme {nst}")


def main(exchanges, years=6, workers=20, with_news=True, with_pdf=True, limit=None,
         shard=None, combine=None):
    US.mkdir(parents=True, exist_ok=True)
    PDF.mkdir(parents=True, exist_ok=True)
    if combine:                                          # job de fusion (pas de calcul)
        combine_shards(combine)
        return
    t0 = time.time()
    uni = fmp.screener(exchanges)
    syms = sorted(uni, key=lambda s: -(uni[s].get("market_cap") or 0))   # plus grosses d'abord
    if shard is not None:                                # partition stable (crc32) de l'univers
        i, n = shard
        syms = [s for s in syms if zlib.crc32(s.encode()) % n == i]
        print(f"SHARD {i}/{n}")
    if limit:
        syms = syms[:limit]
    print(f"Univers {exchanges} : {len(syms)} sociétés")

    print(f"Valorisation par-ticker (FMP), {workers} threads…")
    rows, done, fail = [], 0, 0
    rejets, repares = {}, {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, s, uni[s], with_news, with_pdf): s for s in syms}
        for fut in cf.as_completed(futs):
            try:
                _, row = fut.result()
                if row and "__rejet__" in row:
                    for m in row["__rejet__"]:
                        rejets[m.split(" (")[0].split(" —")[0]] =                             rejets.get(m.split(" (")[0].split(" —")[0], 0) + 1
                    for rp in row.get("reparations_tentees") or []:
                        repares[rp.split(" (")[0]] = repares.get(rp.split(" (")[0], 0) + 1
                    fail += 1
                elif row:
                    rows.append(row); done += 1
                    for rp in row.get("reparations") or []:
                        repares[rp] = repares.get(rp, 0) + 1
                else:
                    fail += 1
            except Exception:
                fail += 1
            if (done + fail) % 200 == 0:
                print(f"  {done+fail}/{len(syms)} (ok={done})")

    if rejets:
        print("\n  motifs de non-couverture (donnees non verifiables) :")
        for m, c in sorted(rejets.items(), key=lambda kv: -kv[1])[:12]:
            print(f"    {c:>6}  {m}")
    if repares:
        print("  reparations automatiques appliquees :")
        for m, c in sorted(repares.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {c:>6}  {m}")

    if shard is not None:                                # écrit les lignes de ce shard
        i, n = shard
        (US / f"_shard_{i}.json").write_text(json.dumps(
            {"rows": rows, "universe": len(syms), "fail": fail}, ensure_ascii=False),
            encoding="utf-8")
        print(f"\n-> SHARD {i}/{n} : {done} valorisés, {fail} sans données, "
              f"{len(rows)} lignes | {time.time()-t0:.0f}s")
    else:                                                # build mono-job : agrégats directs
        nok, ninv, nst = _write_aggregates(rows, len(syms), fail)
        print(f"\n-> {done} valorisés ({nok} affichés, {ninv} upside non-fini), "
              f"{fail} sans données | court terme {nst} | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    kw = {"years": 6, "workers": 20, "with_news": True, "with_pdf": True,
          "limit": None, "shard": None, "combine": None}
    exch = ["NASDAQ", "NYSE", "TSX", "TSXV", "OTC"]
    while args:
        a = args.pop(0)
        if a == "--exchanges": exch = args.pop(0).split(",")
        elif a == "--years": kw["years"] = int(args.pop(0))
        elif a == "--workers": kw["workers"] = int(args.pop(0))
        elif a == "--limit": kw["limit"] = int(args.pop(0))
        elif a == "--shard":
            p = args.pop(0).split("/"); kw["shard"] = (int(p[0]), int(p[1]))
        elif a == "--combine": kw["combine"] = int(args.pop(0))
        elif a == "--no-news": kw["with_news"] = False
        elif a == "--no-pdf": kw["with_pdf"] = False
    main(exch, **kw)

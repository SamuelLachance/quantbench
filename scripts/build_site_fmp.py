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
from quantbench.data.validate import valider, fraicheur_des_comptes
from quantbench.data.repair import reparer
from quantbench.data.sec_fundamentals import annual_report_docs
from quantbench.data.market import risk_free_rate
from quantbench.forensics import analyze
from quantbench.risk import noter as noter_risque
from quantbench.valuation import monte_carlo_dcf
from quantbench.valuation.route import value_stock
from quantbench.valuation.build_universal import (project, build_dcf_from_fundamentals,
                                                  country_erp, pays_exploitation)
from quantbench.shortterm.predict import predict as st_predict
from quantbench.reports import financial_summary_pdf


# CHRONOMETRE PAR ETAPE. Le build met une a trois heures et personne n'a jamais
# mesure ou passe ce temps : toute optimisation serait une supposition.
#
# Une seule horloge, parce qu'une seule fonctionne. `time.thread_time`,
# `time.process_time` et `os.times` renvoient TOUS ZERO sur ce Python/Windows, y
# compris pour une boucle qui brule une demi-seconde de CPU pur : impossible de
# separer calcul et attente par la mesure directe. Reste `perf_counter`, qui mesure
# le temps de MUR.
#
# Consequence a ne pas oublier en lisant le tableau : sous N threads, une etape de
# calcul pur accumule aussi tout le temps ou elle attend le GIL. Mesure a 12 threads,
# le Monte Carlo semblait couter 14,3 s par titre ; a 1 thread il en coute 1,7, et
# 0,35 s en isolation totale. C'est pourquoi le tableau annonce le facteur de
# parallelisme : au-dessus de 1, l'attribution par etape n'est qu'indicative, et
# seule une passe `--workers 1` donne le cout reel.
_CHRONO = {}
_CHRONO_VERROU = __import__("threading").Lock()


class _etape:
    """Accumule le temps de mur d'une etape sous son nom."""

    def __init__(self, nom):
        self.nom = nom

    def __enter__(self):
        self.mur = time.perf_counter()
        return self

    def __exit__(self, *_exc):
        d = time.perf_counter() - self.mur
        with _CHRONO_VERROU:
            n, mur = _CHRONO.get(self.nom, (0, 0.0))
            _CHRONO[self.nom] = (n + 1, mur + d)
        return False


def _rapport_chrono(duree_totale, workers=1):
    if not _CHRONO:
        return
    lignes = sorted(_CHRONO.items(), key=lambda kv: -kv[1][1])
    cumul = sum(m for _n, m in _CHRONO.values())
    facteur = cumul / duree_totale if duree_totale > 0 else 0.0
    fiable = "cout reel" if facteur <= 1.2 else \
             f"INDICATIF : x{facteur:.1f} de recouvrement, relancer a --workers 1"
    print(f"\n  temps par etape ({duree_totale/60:.1f} min de mur, "
          f"{workers} thread(s)) — {fiable}")
    for nom, (n, mur) in lignes:
        print(f"    {nom:26} {mur/60:7.1f} min {mur/cumul*100:5.1f} %  "
              f"{n:6} appels  {mur/max(n,1)*1000:7.0f} ms/appel")


# PLAFOND MENSUEL DE TRANSFERT chez le fournisseur de donnees, en octets.
# POSE, parce qu'il est contractuel et non mesure : c'est le forfait souscrit.
PLAFOND_MENSUEL = 150e9


def _rapport_bande_passante(n_titres, n_shards=1):
    """Ce que le build a telecharge, et ce que cela donne sur un mois.

    Le forfait autorise 150 Go par mois et le build en consommait assez pour
    l'epuiser avant la fin du mois, sans que personne ne sache quel appel
    telechargeait quoi. La mesure a designe un seul coupable : l'historique de
    cours, 83 % du total, telecharge sur cinq ans pour n'en garder que 400
    seances. Le bornage cote serveur a ramene le build de 2,44 Go a 1,03 Go.

    Ce rapport existe pour que cela ne se reperde pas en silence : une
    regression de bande passante ne casse rien, ne ralentit rien, et ne se voit
    que sur la facture — donc trop tard.
    """
    d = fmp.octets_consommes()
    if not d:
        return
    total = sum(o for _k, o in d.values())
    print(f"\n  bande passante ({total/1e6:.0f} Mo pour {n_titres} titres, "
          f"{total/max(n_titres,1)/1e3:.0f} ko/titre) :")
    for fam, (k, o) in sorted(d.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"    {fam:42} {k:6} appels {o/1e6:7.2f} Mo {o/total*100:5.1f} %")
    # Projection : ce shard represente 1/n_shards du build quotidien.
    mois = total * n_shards * 30
    part = mois / PLAFOND_MENSUEL * 100
    alerte = "" if part < 60 else ("   <-- SURVEILLER" if part < 90
                                   else "   <-- PLAFOND MENSUEL EN DANGER")
    print(f"    projection 30 jours : {mois/1e9:.0f} Go sur {PLAFOND_MENSUEL/1e9:.0f} Go "
          f"({part:.0f} %){alerte}")


# CALIBRATION DU SIGNAL COURT TERME, chargee une fois. Chaque fiche affiche
# « aucun pouvoir predictif (AUC ...) » : ce chiffre etait ECRIT EN DUR dans le
# HTML, alors que la mesure vit ici. Deux chiffres a deux endroits divergent
# toujours — il suffit d'une recalibration pour que la page mente.
def _calibration_court_terme():
    try:
        c = json.loads((Path(__file__).resolve().parent.parent / "quantbench" /
                        "shortterm" / "shortterm_calibration.json")
                       .read_text(encoding="utf-8"))
        return {"calibrated": bool(c.get("calibrated")),
                "auc": (c.get("metrics") or {}).get("auc")}
    except Exception:
        return {"calibrated": False, "auc": None}


_CALIB_ST = _calibration_court_terme()


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


def _route_forme(fund, category, F):
    """Forme de marge du routage — MIROIR STRICT de route.py, par DELEGATION.

    Rend les kwargs a passer a `build_dcf_from_fundamentals`, car la FORME compte
    autant que le niveau : un CYCLIQUE se valorise sur sa marge de cycle DES
    L'ANNEE 1 (le cycle se moyenne dans les deux sens — `margin_override`), tandis
    qu'un redressement (mature en perte, marge normalisee de la route standard) et
    une jeune pousse CONVERGENT vers leur cible (`marge_terminale`) : les premieres
    annees portent la marge courante, et le retard se paie dans l'actualisation.
    Simuler une forme differente de celle de la valorisation affichee, c'est
    ecraser l'upside avec un autre modele.
    """
    from quantbench.valuation.route import marge_de_cycle, marge_normalisee, sect
    if category == "cyclique":
        return {"margin_override": marge_de_cycle(F)}
    if category == "mature_deficitaire":
        return {"marge_terminale": marge_de_cycle(F)}
    if category == "jeune/deficitaire":
        om = fund.get("operating_margin")
        cible = sect(fund, "marge", 0.10)
        if om is not None and om > cible:
            cible = om
        return {"marge_terminale": cible}
    if category == "standard":
        return {"marge_terminale": marge_normalisee(fund, F)}
    return {}


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
                fund, **_route_forme(fund, cat, F))
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
    # Miroir de la dilution appliquee par _finalise (approche treasury stock).
    _dil = fund.get("facteur_dilution") or 1.0
    if shares:
        shares = shares * _dil
    if mcap:
        mcap = mcap * _dil
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
            base, _ = build_dcf_from_fundamentals(fund, rf=rf,
                                                  **_route_forme(fund, category, F))
            # SOURCE UNIQUE des lois ET des correlations. La copie locale ne
            # transmettait aucune correlation : la matrice restait l'identite et la
            # copule gaussienne annoncee par la documentation ne tournait pas. Deux
            # erreurs independantes se compensent, deux erreurs correlees
            # s'additionnent — la simulation SOUS-ESTIMAIT donc la dispersion, c'est-
            # a-dire exactement la grandeur qu'elle existe pour mesurer.
            from quantbench.valuation.build_universal import lois_de_tirage
            lois, correlations = lois_de_tirage(base)
            eq = monte_carlo_dcf(base, lois, n=n, correlations=correlations,
                                 current_market_cap=mcap, seed=42)["equity_values"]
            eq = np.maximum(eq, 0.0)                    # responsabilité limitée : équité ≥ 0
            # Pondération de la catégorie (miroir de route.value_young et de la
            # ponderation par le defaut commune a toutes les routes cote entreprise)
            # LES PONDERATIONS SONT CELLES DU ROUTAGE, APPELEES et non recopiees.
            # Ce bloc portait encore, le jour meme de leur correction, la survie
            # plancherisee a 0,30 et mesuree sur le RESULTAT NET, et une valeur de
            # liquidation posee a la moitie des fonds propres — l'erreur exacte que
            # `valeur_de_liquidation` corrige : appliquer la decote a l'equite revient
            # a supposer que les dettes la subissent au benefice de l'actionnaire.
            from quantbench.valuation.route import (probabilite_de_survie,
                                                    valeur_de_liquidation)
            if category == "jeune/deficitaire":
                surv = probabilite_de_survie(fund)
                liq = valeur_de_liquidation(fund)
                eq = np.maximum(eq * surv + liq * (1.0 - surv), 0.0)
            elif category == "detresse":
                from quantbench.forensics.scores import default_probability
                z = (forensic or {}).get("scores", {}).get("altman_z")
                pdef = default_probability(z)
                liq = valeur_de_liquidation(fund)
                eq = np.maximum(eq * (1.0 - pdef) + liq * pdef, 0.0)
            else:
                # MIROIR OBLIGATOIRE du routage : toutes les routes cote entreprise
                # sont desormais ponderees par la probabilite de DEFAUT tiree du
                # Z''-EMS (la lettre de Damodaran — il pondere Delta a BBB- et Las
                # Vegas Sands beneficiaire ; le critere est le fardeau de dette,
                # jamais le deficit de l'exercice). Sans ce miroir, la mediane
                # simulee ecrase la ponderation — l'angle mort qui faisait repasser
                # General Motors de -70 % a -100 %. L'ancienne « probabilite de
                # realisation » (part des exercices ayant atteint la marge cible)
                # n'existait pas chez Damodaran et a ete retiree du routage.
                from quantbench.forensics.scores import Z_SAIN, default_probability
                from quantbench.valuation.route import _NO_ALTMAN, _Z_POIDS_PLEIN
                sec = (fund.get("sector") or "").lower()
                z = (forensic or {}).get("scores", {}).get("altman_z")
                if (z is not None and z < Z_SAIN
                        and not any(x in sec for x in _NO_ALTMAN)):
                    facteur = (1.0 if z <= _Z_POIDS_PLEIN
                               else (Z_SAIN - z) / (Z_SAIN - _Z_POIDS_PLEIN))
                    pdef = default_probability(z) * facteur
                    if pdef > 0.0:
                        liq = valeur_de_liquidation(fund)
                        eq = np.where(eq > 0, eq * (1.0 - pdef) + liq * pdef, eq)
                        eq = np.maximum(eq, 0.0)
    except Exception:
        return None
    # MIROIR de la decote d'illiquidite appliquee par _finalise : sans lui, la
    # mediane simulee ecraserait la decote — meme angle mort que les ponderations.
    try:
        from quantbench.valuation.build_universal import decote_illiquidite
        d = decote_illiquidite(fund)
        if d > 0.0:
            eq = eq * (1.0 - d)
    except Exception:
        pass
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
    with _etape("etats financiers"):
        entry = fmp.statements(symbol)             # 3 appels par-ticker
    if len(set(entry["income"]) & set(entry["balance"])) < 1:   # ≥1 an suffit (actif net)
        # Abandon SILENCIEUX auparavant : environ six cents lignes disparaissaient
        # sans laisser de trace, dont les certificats canadiens (Eli Lilly, Micron,
        # General Electric) et la cotation de gre a gre de TSMC. Une absence de
        # donnee est un fait a AFFICHER, pas a taire.
        return None, {"__rejet__": ["comptes indisponibles chez le fournisseur"],
                      "ticker": symbol}
    with _etape("profil"):
        prof = fmp.profile(symbol)
    desc = {"description": prof.get("description"), "industry": prof.get("industry")}
    F = fmp.financials_from_fmp(entry)
    fund = fmp.fundamentals_from_fmp(symbol, sr, entry, desc)
    # Plus d'exigence de CA : les sociétés pré-revenu (biotech, mines, holdings) sont
    # valorisées sur l'actif net. Il faut juste un prix (pour l'upside).
    if not fund or not fund.get("price"):
        return None, {"__rejet__": ["cours indisponible"], "ticker": symbol}
    # VALIDATION DES ENTREES : une donnee non verifiee ne donne pas une valorisation
    # approximative mais une valorisation FAUSSE. On ne valorise que ce dont on
    # peut repondre, et on RETOURNE le motif quand ce n'est pas le cas.
    # FRAICHEUR : mesuree et PUBLIEE, jamais un motif de rejet. Des comptes anciens
    # sont les derniers comptes CONNUS ; Damodaran valorise dessus en signalant leur
    # date. Au-dela de trente mois on tente d'abord la reconstitution sur douze mois
    # glissants, puis on affiche la date retenue quoi qu'il arrive.
    date_comptes, age_mois = fraicheur_des_comptes(entry)
    fund["date_des_comptes"] = date_comptes
    fund["age_des_comptes_mois"] = age_mois
    motifs = valider(fund, F, entry)
    reparations = []
    declencheurs = list(motifs)
    if age_mois is not None and age_mois > 30:
        declencheurs.append("comptes perimes")
    if declencheurs:
        # On tente de CORRIGER avant de renoncer : capitalisation de la cotation
        # d'origine, identite du bilan, douze mois glissants, devise pivot.
        reparations = reparer(symbol, fund, F, entry, declencheurs)
        if reparations:
            motifs = valider(fund, F, entry)          # revalidation apres reparation
            for r in reparations:
                if "douze mois glissants" in r:
                    fund["date_des_comptes"] = fund.get("date_ttm") or date_comptes
                    fund["age_des_comptes_mois"] = 0
    sh, mc0 = fund.get("shares"), fund.get("market_cap")
    # LE NOMBRE D'ACTIONS N'EST PAS UNE QUESTION DE TAILLE MAIS DE COHERENCE.
    # Le seuil de 100 000 titres eliminait des societes parfaitement saines au
    # flottant minuscule : LICT (19 188 actions a 11 775 $, 226 M$ de
    # capitalisation), Beaver Coal, Merchants' National, CIBL. Toutes affichent un
    # nombre d'actions ENTIER — signe que capitalisation et cours se reconcilient.
    # Il rendait toutefois un vrai service en arretant les capitalisations
    # corrompues, qui donnent une base implicite FRACTIONNAIRE : 16 316,3 actions
    # pour une societe qui en a depose 36 177 712. C'est cette fracture, et non la
    # taille, qui signale la donnee fausse.
    if not sh or (sh < 100_000 and abs(sh - round(sh)) > max(1e-6 * sh, 1e-3)):
        motifs.append("nombre d'actions implausible (base implicite fractionnaire)")
    if not mc0 or mc0 <= 0:
        motifs.append("capitalisation indisponible")
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

    # Une regle ecartait tout "service public" cotant sous le tiers de ses fonds
    # propres, au motif qu'il s'agissait d'obligations de filiales (Entergy
    # Mississippi, Entergy New Orleans). Le raisonnement etait une OPINION DE
    # VALORISATION — "un regule se traite entre 1 et 2,5 fois ses fonds propres" —
    # executee de surcroit en abandon SILENCIEUX, sans motif ni trace. Ce qui
    # distingue ces lignes n'est pas leur decote mais leur NATURE : ce sont des
    # emissions obligataires, reconnaissables a leur libelle, et c'est la que le
    # filtre appartient.
    # Le secteur accompagne l'analyse : le Z d'Altman n'a pas de sens pour une
    # financiere, une fonciere ou un service public, et la fiche l'affichait
    # pourtant alors que la valorisation refusait de s'en servir.
    forensic = analyze(symbol, financials=F,
                       secteur=fund.get("sector")) if F else None
    with _etape("valorisation"):
        val = value_stock(symbol, fund=fund, forensic=forensic, F=F)
    if not val.get("ok"):
        return None, {"__rejet__": [val.get("reason") or "valorisation impossible"],
                      "ticker": symbol, "reparations_tentees": reparations}
    # MONTE CARLO : IL MESURE L'INCERTITUDE, IL NE PRODUIT PLUS LA VALEUR.
    #
    # Sa mediane ECRASAIT l'upside du routage sur les deux tiers de l'univers. Le
    # probleme n'est pas que la mediane simulee differe de la valeur deterministe —
    # mediane(f(X)) n'est pas f(mediane X), c'est une propriete et non une erreur.
    # Le probleme est structurel : `run_mc` est une SECONDE IMPLEMENTATION de la
    # valorisation, et c'est son resultat qui etait publie.
    # Elle portait d'ailleurs encore, ce jour meme, trois regles corrigees le matin
    # dans le routage : la survie plancherisee a 0,30 et mesuree sur le RESULTAT NET,
    # et une valeur de liquidation posee a la moitie des fonds propres — l'erreur
    # exacte que `valeur_de_liquidation` corrige et nomme dans sa docstring.
    # Une regle n'a droit qu'a une seule ecriture, et c'est le routage qui la porte.
    #
    # La simulation reste publiee : bande de dispersion, percentiles et probabilite
    # de sous-valorisation. C'est son objet — mesurer l'incertitude autour de la
    # valeur, pas la remplacer.
    with _etape("monte carlo"):
        mc = run_mc(fund, val.get("category"), F=F, forensic=forensic,
                    method=val.get("method"))
    # UN SEUL appel pour les cours ET le volume : la reponse porte deja les deux,
    # et nous jetions le second. La liquidite est pourtant la seule mesure directe
    # de la capacite a REVENDRE — un titre qu'on ne peut pas sortir a un prix
    # raisonnable est risque, quelle que soit la solidite de ses comptes.
    with _etape("historique de cours"):
        serie = fmp.history_ohlcv(symbol)
    signal = st_predict([x["close"] for x in serie])
    fund["volume_dollars_median"] = fmp.volume_dollars_median(serie)
    with _etape("actualites"):
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
                           **_route_forme(fund, val.get("category"), F))
        except Exception:
            proj = None
    methodo = _methodologie(fund, val, F)
    cik = fund.get("cik")
    with _etape("documents SEC"):
        ard = (annual_report_docs(cik) if cik
               else {"ars_pdf": None, "tenk": None, "documents": []})
    # UN CIK NUL N'EST PAS UN CIK. Le fournisseur renvoie la sentinelle
    # "0000000000" pour toute societe hors perimetre SEC — tout le TSX, la
    # quasi-totalite des lignes de gre a gre : 88 fiches sur 387. La garde
    # `if cik` ne s'en apercevait pas, une chaine de zeros etant vraie en Python,
    # et la fiche publiait un lien vers une recherche EDGAR VIDE. Pour 86 de ces
    # 88 societes, c'etait le SEUL lien du bloc « documents ».
    if cik and not str(cik).strip("0"):
        cik = None
    filing = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K"
              if cik else None)
    # NOTE DE RISQUE. Elle repond a une question DIFFERENTE de l'upside : non pas
    # "combien vaut cette societe" mais "quelle est la probabilite de perdre
    # durablement sa mise". Elle lit un calibrage GELE, donc n'exige aucune statistique
    # d'univers et laisse les cinq shards independants.
    fund["exchange"] = sr.get("exchange")
    try:
        risque = noter_risque(fund, F, motifs, reparations)
    except Exception:                                  # noqa: BLE001
        risque = None

    profile = {
        "ticker": symbol, "name": fund.get("name"), "sector": fund.get("sector"),
        "industry": fund.get("industry"), "summary": fund.get("summary"),
        "exchange": sr.get("exchange"), "valuation": val,
        "fundamentals": {k: fund.get(k) for k in (
            "price", "market_cap", "shares", "beta", "revenue", "ebit", "net_income",
            "total_debt", "cash", "book_equity", "operating_margin", "roe")},
        "risque": risque,
        "forensics": forensic, "statements": _statements(F), "news": news,
        "projection": proj, "methodologie": methodo,
        "reparations_donnees": reparations,
        "results_summary": _results_summary(F),
        # L'AUC mesuree accompagne le signal : la fiche la LIT au lieu de la
        # recopier. Elle ne coute rien — un flottant par fiche.
        "shortterm": ({**signal, "auc_calibration": _CALIB_ST["auc"]}
                      if signal else signal),
        "montecarlo": mc, "documents": ard.get("documents", []),
        "report_url": ard.get("tenk"), "ars_pdf_url": ard.get("ars_pdf"),
        "filing_url": filing, "pdf_url": None,
    }
    if with_pdf:
        with _etape("rapport PDF"):
            ok_pdf = financial_summary_pdf(profile, str(PDF / f"{symbol}.pdf"))
        if ok_pdf:
            profile["pdf_url"] = f"pdf/{symbol}.pdf"
    (US / f"{symbol}.json").write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
    f = forensic or {}
    row = {"ticker": symbol, "name": fund.get("name"), "sector": fund.get("sector"),
           "exchange": sr.get("exchange"), "category": val.get("category"),
           "method": val.get("method"), "price": val.get("price"),
           "market_cap": val.get("market_cap"), "value_per_share": val.get("value_per_share"),
           "upside": val.get("upside"), "confidence": val.get("confidence"),
           "note_risque": (risque or {}).get("grade"),
           "score_risque": (risque or {}).get("score"),
           "regime_risque": (risque or {}).get("regime"),
           "date_comptes": fund.get("date_des_comptes"),
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


def _archiver_les_notes(all_rows):
    """Archive MENSUELLE des notes de risque — la seule voie honnete vers un calibrage.

    Le probleme est structurel : nos fondamentaux ne sont pas historises. Correler les
    notes d'AUJOURD'HUI aux mouvements de cours PASSES ne mesurerait donc rien — les
    comptes utilises n'existaient pas a la date ou le cours a bouge, et la mesure
    serait un biais de survie a l'envers. Aucune astuce ne contourne cela.

    Reste une voie, lente mais propre : ecrire ce que le modele pense AUJOURD'HUI, et
    mesurer dans douze mois ce qui est arrive. Un fichier par mois — plusieurs par
    mois n'apporteraient rien, une note de risque ne bougeant pas d'une semaine a
    l'autre. `scripts/mesurer_les_notes.py` confrontera deux archives des qu'un an les
    separera, et c'est cette mesure, et elle seule, qui autorisera a ponderer les
    dimensions autrement qu'uniformement."""
    dossier = US / "_notes_risque"
    dossier.mkdir(exist_ok=True)
    fichier = dossier / f"{datetime.now(timezone.utc):%Y-%m}.json"
    notes = {r["ticker"]: [r.get("note_risque"), r.get("score_risque"), r.get("price")]
             for r in all_rows if r.get("note_risque")}
    if not notes:
        return

    # ON GARDE L'ARCHIVE LA PLUS COMPLETE DU MOIS, pas la premiere ecrite.
    # La regle precedente — "le fichier existe, on ne touche a rien" — protegeait
    # contre l'ecrasement, mais elle rendait toute archive partielle DEFINITIVE.
    # Un build interrompu, un essai local sur cent vingt titres, un shard perdu :
    # le mois entier restait fige sur cet echantillon, et la mesure a douze mois
    # aurait porte dessus sans que rien ne le signale. Une mesure plus large du
    # MEME mois est strictement meilleure ; une plus etroite ne l'est jamais.
    if fichier.exists():
        try:
            ancien = json.loads(fichier.read_text(encoding="utf-8")).get("n", 0)
        except Exception:
            ancien = 0
        if ancien >= len(notes):
            print(f"  archive {fichier.name} conservee ({ancien} notes, "
                  f"la nouvelle en compte {len(notes)})")
            return
        print(f"  archive {fichier.name} elargie : {ancien} -> {len(notes)} notes")
    fichier.write_text(json.dumps(
        {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
         "champs": ["grade", "score", "cours"], "n": len(notes), "notes": notes},
        ensure_ascii=False), encoding="utf-8")


def _write_aggregates(all_rows, universe_size, fail):
    """Écrit _screener.json + _shortterm.json depuis l'ensemble des lignes."""
    _archiver_les_notes(all_rows)
    clean = [r for r in all_rows if r["upside"] is not None and np.isfinite(r["upside"])]
    invalid = [r for r in all_rows if r["upside"] is None or not np.isfinite(r["upside"])]
    clean.sort(key=lambda r: -(r["upside"] if r["upside"] is not None else -9))
    # LA COUVERTURE REELLE, ET NON UN ZERO ECRIT A LA MAIN. Le champ `n_suspect`
    # valait 0 en dur et `suspects` une liste vide : la page affichait donc
    # « 0 ecartes (donnees suspectes) » alors que 5 080 titres sur 16 830 — 30 %
    # de l'univers — n'avaient produit aucune valorisation. Les compteurs
    # existaient, etaient justes, et n'etaient affiches nulle part.
    # Les deux causes sont DISTINCTES et doivent le rester : `n_fail` compte les
    # societes dont la donnee n'est pas verifiable (comptes absents, bilan qui ne
    # boucle pas), `n_invalid` celles qui ont ete valorisees mais dont l'upside
    # n'est pas un nombre fini. Les confondre masquerait laquelle progresse.
    non_couverts = int(fail) + len(invalid)
    (US / "_screener.json").write_text(json.dumps(
        {"n_ok": len(clean), "n_invalid": len(invalid), "n_fail": fail,
         "n_non_couverts": non_couverts,
         "part_couverte": (round(len(clean) / universe_size, 4)
                           if universe_size else None),
         "universe": universe_size, "updated": _now_et(), "rows": clean},
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


class ShardManquant(RuntimeError):
    """Un shard est absent, tronque, ou n'est pas celui qu'on attendait."""


def combine_shards(n_shards, tolerance=0.02):
    """Job de combinaison : fusionne les _shard_*.json (produits par les jobs parallèles)
    en _screener.json + _shortterm.json. Les fiches par-titre + PDF sont déjà en place
    (téléchargées depuis les artefacts de chaque shard).

    LA FUSION REFUSE DE PUBLIER UN UNIVERS INCOMPLET. Avant, un shard absent
    produisait un avertissement noye dans dix mille lignes de journal, puis un site
    ampute d'un cinquieme de sa couverture — environ 3 300 societes disparues sans
    que rien n'echoue. La barriere qualite ne le voyait pas : elle raisonne en
    RATIOS (valorisees / univers), et un shard manquant retire autant au numerateur
    qu'au denominateur. Le defaut etait donc invisible par construction.

    Quatre verifications, chacune contre un mode de panne observable :
      1. les `n_shards` fichiers sont la          -> artefact perdu ou job echoue ;
      2. chacun declare le meme decoupage         -> melange de deux builds ;
      3. chacun s'annonce sous l'index de son nom -> artefact renomme ou duplique ;
      4. chacun a traite ses tickers assignes     -> shard interrompu en cours de route.
    """
    all_rows, universe, fail = [], 0, 0
    manques = []
    for i in range(n_shards):
        p = US / f"_shard_{i}.json"
        if not p.exists():
            manques.append(f"shard {i} : fichier absent")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))

        declare_n = d.get("n_shards")
        if declare_n is not None and declare_n != n_shards:
            manques.append(f"shard {i} : construit pour un decoupage en "
                           f"{declare_n}, fusionne en {n_shards}")
        declare_i = d.get("shard")
        if declare_i is not None and declare_i != i:
            manques.append(f"shard {i} : le fichier declare etre le shard {declare_i}")

        # Un shard interrompu ecrit quand meme son fichier : seul l'ecart entre les
        # tickers assignes et les tickers acheves le trahit.
        assignes = d.get("assignes")
        acheves = d.get("acheves")
        if assignes is not None and acheves is not None and assignes:
            reste = 1.0 - acheves / len(assignes)
            if reste > tolerance:
                manques.append(f"shard {i} : {acheves}/{len(assignes)} titres traites "
                               f"({reste*100:.1f} % jamais atteints)")

        all_rows.extend(d.get("rows", []))
        universe += d.get("universe", 0)
        fail += d.get("fail", 0)

    # Un ticker present deux fois signifie que deux shards se recouvrent : la
    # societe apparaitrait en double dans le screener et ses statistiques
    # (medianes, quantiles de risque) seraient calculees sur un univers fausse.
    vus = {}
    for r in all_rows:
        t = r.get("ticker")
        if t:
            vus[t] = vus.get(t, 0) + 1
    doublons = sorted(t for t, k in vus.items() if k > 1)
    if doublons:
        manques.append(f"{len(doublons)} tickers presents dans plusieurs shards "
                       f"(ex. {', '.join(doublons[:5])})")

    if manques:
        raise ShardManquant(
            "Fusion refusee — l'univers serait incomplet et le site publierait "
            "silencieusement moins de societes qu'il n'en couvre :\n  "
            + "\n  ".join(manques)
            + "\nLe site conserve ses dernieres donnees completes.")

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

    _rapport_chrono(time.time() - t0, workers)
    _rapport_bande_passante(len(syms), shard[1] if shard else 1)

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
        # Le shard SIGNE son travail : son index, le decoupage sous lequel il a
        # tourne, et la liste exacte des tickers qui lui avaient ete assignes. Sans
        # cette signature, la fusion ne peut pas distinguer un shard complet d'un
        # shard interrompu, ni un artefact de la veille d'un artefact du jour.
        (US / f"_shard_{i}.json").write_text(json.dumps(
            {"rows": rows, "universe": len(syms), "fail": fail,
             "shard": i, "n_shards": n, "assignes": syms,
             "acheves": done + fail, "date": _now_et()}, ensure_ascii=False),
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

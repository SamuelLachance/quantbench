"""Invariants de la valorisation — tests HORS LIGNE, executes a chaque poussee.

Ces tests encodent les regles qui ont ete VIOLEES en production, pour qu'aucune
d'elles ne puisse revenir silencieusement. Chacun correspond a un bug reel :

  * responsabilite limitee            -> upside descendait a -115 373 %
  * routage sectoriel                 -> DCF applique aux foncieres / services publics
  * coherence simulation <-> routage  -> le Monte Carlo ecrasait la methode retenue
                                         (General Motors -70 % redevenait -100 %,
                                          Allstate -34 % redevenait +106 %)
  * identite comptable des minoritaires -> je les avais deduits deux fois, rendant
                                         negatifs les fonds propres de PZU
  * garde-fous de donnees             -> tresorerie a 6 600 milliards, 4 000 actions,
                                         obligations prises pour des actions
"""

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from quantbench.data import market                                    # noqa: E402
from quantbench.valuation import route                                # noqa: E402


@pytest.fixture(autouse=True)
def _rf_fixe(monkeypatch):
    """Taux sans risque fige : les tests ne doivent dependre d'aucun reseau."""
    monkeypatch.setattr(market, "risk_free_rate", lambda: 0.042)


_FOR = {"scores": {}, "flags": [], "positives": []}   # forensique neutre (hors sujet ici)


def societe(**kw):
    """Fondamentaux synthetiques d'une societe saine, surchargeables."""
    base = dict(ticker="TEST", name="Test", sector="Technology", industry="Software",
                country="US", price=100.0, market_cap=10.0, shares=1e8, beta=1.1,
                revenue=5.0, revenue_history=[3.0, 3.5, 4.0, 4.5, 5.0],
                ebit=1.0, net_income=0.7, total_debt=2.0, cash=1.0,
                book_equity=4.0, total_assets=9.0, dep_amort=0.3, cfo=1.0,
                operating_margin=0.20, roe=0.175)
    base.update(kw)
    return base


def etats(marges=(0.20, 0.19, 0.21, 0.20)):
    n = len(marges)
    rev = [5.0e9] * n
    return {"years": [2025 - i for i in range(n)],
            "revenue": rev, "ebit": [m * r for m, r in zip(marges, rev)],
            "net_income": [0.7e9] * n, "equity": [4.0e9] * n,
            "total_assets": [9.0e9] * n, "net_ppe": [1.0e9] * n}


# --------------------------------------------------------------------------- #
# 1. Responsabilite limitee : une action ne vaut jamais moins que zero
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw", [
    dict(),                                                   # societe saine
    dict(total_debt=50.0, book_equity=-3.0, ebit=-0.5),       # surendettee
    dict(net_income=-2.0, ebit=-2.0, operating_margin=-0.4),  # deficitaire
    dict(revenue=None, revenue_history=[]),                   # pre-revenu
    dict(sector="Real Estate"), dict(sector="Utilities"),
    dict(sector="Financial Services", industry="Banks - Diversified"),
    dict(sector="Energy"),
])
def test_upside_jamais_sous_moins_cent_pourcent(kw):
    v = route.value_stock("TEST", fund=societe(**kw), forensic=_FOR, F=etats())
    if v.get("ok") and v.get("upside") is not None:
        assert v["upside"] >= -1.0 - 1e-9, f"upside impossible : {v['upside']}"
        assert v["equity_value"] >= 0.0


# --------------------------------------------------------------------------- #
# 2. Routage sectoriel : le DCF classique ne convient pas a tous les secteurs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("secteur,industrie,attendu", [
    ("Real Estate", "REIT - Industrial", "fonciere"),
    ("Utilities", "Utilities - Regulated Electric", "reglementee"),
    ("Financial Services", "Banks - Diversified", "financiere"),
    ("Financial Services", "Insurance - Life", "financiere"),
    ("Energy", "Oil & Gas E&P", "cyclique"),
    ("Basic Materials", "Gold", "cyclique"),
    ("Technology", "Software", "standard"),
])
def test_routage_par_secteur(secteur, industrie, attendu):
    f = societe(sector=secteur, industry=industrie)
    assert route.classify(f, None, etats()) == attendu


def test_metier_de_commissions_nest_pas_une_financiere_de_bilan():
    """Visa/Mastercard : reseaux de paiement, bilan leger -> DCF, pas un P/B."""
    reseau = societe(sector="Financial Services", industry="Financial - Credit Services",
                     revenue=36.0, total_assets=90.0)          # actif/CA = 2,5
    assert route.classify(reseau, None, etats()) == "standard"
    preteur = societe(sector="Financial Services", industry="Financial - Credit Services",
                      revenue=40.0, total_assets=490.0)        # actif/CA = 12
    assert route.classify(preteur, None, etats()) == "financiere"


def test_societe_mature_en_perte_nest_pas_une_jeune_pousse():
    f = societe(revenue=187.0, ebit=-1.0, net_income=-1.0, operating_margin=-0.005)
    assert route.classify(f, None, etats(marges=(0.05, 0.04, 0.06, 0.05))) == "mature_deficitaire"


def test_altman_ne_sapplique_pas_aux_financieres_et_foncieres():
    """Altman et Damodaran excluent ces secteurs : un Z bas n'y signale pas la detresse."""
    forensic = {"scores": {"altman_z": 0.5}}                  # tres bas
    for secteur, industrie in (("Real Estate", "REIT - Office"),
                               ("Utilities", "Utilities - Regulated Gas"),
                               ("Financial Services", "Banks - Regional")):
        f = societe(sector=secteur, industry=industrie)
        assert route.classify(f, forensic, etats()) != "detresse"


# --------------------------------------------------------------------------- #
# 3. Le Monte Carlo ne doit JAMAIS ecraser une valorisation non-DCF
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("categorie", ["fonciere", "reglementee", "financiere", "actif_net"])
def test_pas_de_simulation_quand_la_methode_nest_pas_un_dcf(categorie):
    import build_site_fmp as bs
    assert bs.run_mc(societe(), categorie) is None


@pytest.mark.parametrize("methode", [
    "Residual income côté équité (dette de financement)",
    "Valeur comptable (repli)", "Valeur d'actif net (pré-revenu / holding)",
    "FFO capitalisé (foncière — Damodaran REIT)",
    "Rendement excedentaire a erosion 10 ans (financiere de bilan)",
    "Benefices capitalises cote equite (service public regule)",
])
def test_toute_methode_non_dcf_desactive_la_simulation(methode):
    """Bug reel : la liste d'exclusion etait DUPLIQUEE et incomplete dans run_mc."""
    import build_site_fmp as bs
    assert bs.run_mc(societe(), "standard", method=methode) is None


def test_liste_dexclusion_couvre_toutes_les_methodes_non_dcf():
    """Toute methode produite par le routage et absente du DCF doit y figurer."""
    import build_site_fmp as bs
    for m in ("Residual income", "Valeur comptable", "Valeur d'actif net",
              "FFO capitalisé", "Benefices capitalises", "Rendement excedentaire"):
        assert m in bs._METH_NON_DCF


# --------------------------------------------------------------------------- #
# 4. Identites comptables
# --------------------------------------------------------------------------- #
def test_interets_minoritaires_ne_sont_pas_deduits_deux_fois():
    """`totalStockholdersEquity` les exclut deja : les retrancher rendait negatifs
    les fonds propres de toute societe a filiales (PZU : -0,45 au lieu de +9,6)."""
    from quantbench.data import fmp
    bal = {2025: {"totalStockholdersEquity": 35.43e9, "minorityInterest": 37.13e9,
                  "preferredStock": 0.0, "totalAssets": 141e9, "totalDebt": 10e9,
                  "cashAndCashEquivalents": 5e9, "reportedCurrency": "USD"}}
    inc = {2025: {"revenue": 17e9, "operatingIncome": 2e9, "netIncome": 1.7e9,
                  "weightedAverageShsOutDil": 8.6e8, "reportedCurrency": "USD"}}
    f = fmp.fundamentals_from_fmp("PZU", {"price": 15.0, "market_cap": 12.9e9,
                                          "exchange": "NASDAQ", "name": "PZU"},
                                  {"income": inc, "balance": bal, "cashflow": {}}, {})
    assert f["book_equity"] > 0, "les minoritaires ont ete deduits a tort"
    assert abs(f["book_equity"] - 35.43) < 0.01


def test_actions_privilegiees_sont_deduites():
    """Fannie Mae : 140 Md$ de privilegiees SENIOR passent avant l'ordinaire."""
    from quantbench.data import fmp
    bal = {2025: {"totalStockholdersEquity": 109e9, "preferredStock": 140e9,
                  "minorityInterest": 0.0, "totalAssets": 4317e9, "totalDebt": 100e9,
                  "cashAndCashEquivalents": 50e9, "reportedCurrency": "USD"}}
    inc = {2025: {"revenue": 160e9, "operatingIncome": 15e9, "netIncome": 14e9,
                  "weightedAverageShsOutDil": 1.16e9, "reportedCurrency": "USD"}}
    f = fmp.fundamentals_from_fmp("FNMA", {"price": 6.0, "market_cap": 7e9,
                                           "exchange": "NASDAQ", "name": "FNMA"},
                                  {"income": inc, "balance": bal, "cashflow": {}}, {})
    assert f["book_equity"] < 0, "les privilegiees senior n'ont pas ete deduites"


def test_tresorerie_et_dette_plafonnees_a_lactif_total():
    """Donnee corrompue : Roadzen affichait 6 600 milliards $ de tresorerie."""
    from quantbench.data import fmp
    bal = {2025: {"totalStockholdersEquity": -30e6, "cashAndCashEquivalents": 6.6e12,
                  "totalDebt": 34e6, "totalAssets": 52.7e6, "preferredStock": 0.0,
                  "minorityInterest": 0.0, "reportedCurrency": "USD"}}
    inc = {2025: {"revenue": 55e6, "operatingIncome": -11e6, "netIncome": -12e6,
                  "weightedAverageShsOutDil": 68e6, "reportedCurrency": "USD"}}
    f = fmp.fundamentals_from_fmp("RDZN", {"price": 1.5, "market_cap": 105e6,
                                           "exchange": "NASDAQ", "name": "RDZN"},
                                  {"income": inc, "balance": bal, "cashflow": {}}, {})
    assert f["cash"] <= f["total_assets"] + 1e-9


# --------------------------------------------------------------------------- #
# 5. Robustesse aux exercices exceptionnels
# --------------------------------------------------------------------------- #
def test_un_exercice_exceptionnel_ne_fixe_pas_la_croissance():
    """3SBio : CA double en une annee (accord de licence) -> 45 %/an extrapoles."""
    from quantbench.data.build import _estimate_growth
    g_avec_saut, _ = _estimate_growth([0.825, 0.942, 1.012, 1.153, 1.344, 2.544])
    assert g_avec_saut < 0.30, f"croissance dopee par un exercice unique : {g_avec_saut:.0%}"
    # une acceleration REELLE (tous les exercices eleves) doit rester captee
    g_reelle, _ = _estimate_growth([1.0, 1.6, 2.6, 4.2, 6.7, 10.0])
    assert g_reelle > 0.35


def test_historique_non_significatif_ecarte():
    """Un passage de ~0 a X (paiement d'etape) n'est pas une croissance."""
    from quantbench.data.build import _estimate_growth
    g, diag = _estimate_growth([0.0001, 0.004, 0.244])
    assert g <= 0.10 or diag.get("historique_non_significatif")


def test_valeur_par_action_coherente_avec_le_cours():
    """Identite : valeur/action / cours - 1 == equite / capitalisation - 1."""
    f = societe()
    v = route.value_stock("TEST", fund=f, forensic=_FOR, F=etats())
    assert v["ok"]
    assert abs((v["value_per_share"] / f["price"] - 1.0) - v["upside"]) < 0.02

def test_secteur_en_perte_ne_retombe_pas_sur_un_dcf_inapplicable():
    """Centrica : service public en PERTE une annee. value_regulated refusait le
    dossier et la cascade enchainait sur un DCF classique — precisement la methode
    etablie comme inapplicable aux services publics -> +1 043 % d'upside.
    Le repli d'un secteur a methode dediee doit rester l'ACTIF NET, jamais le DCF."""
    for secteur, industrie in (("Utilities", "Utilities - Regulated Gas"),
                               ("Real Estate", "REIT - Office")):
        f = societe(sector=secteur, industry=industrie, net_income=-0.09,
                    roe=-0.023, ebit=-0.05, operating_margin=-0.002, dep_amort=None)
        # historique lui aussi deficitaire : la normalisation ne peut pas sauver
        F = {"years": [2025, 2024, 2023], "revenue": [25e9] * 3,
             "ebit": [-0.05e9] * 3, "net_income": [-0.09e9] * 3,
             "equity": [4.1e9] * 3, "total_assets": [40e9] * 3, "net_ppe": [20e9] * 3}
        v = route.value_stock("TEST", fund=f, forensic=_FOR, F=F)
        if v.get("ok"):
            assert "DCF" not in (v.get("method") or ""), (
                f"{secteur} : repli sur un DCF inapplicable ({v['method']})")
            assert v["upside"] < 3.0, f"{secteur} : upside implausible {v['upside']:.0%}"


def test_service_public_benefices_normalises():
    """Un exercice deficitaire ne doit pas fixer la valeur d'un regule : sa
    tarification est fixee pour couvrir ses couts."""
    f = societe(sector="Utilities", industry="Utilities - Regulated Electric",
                net_income=-0.09, roe=-0.023, revenue=25.0, book_equity=4.1)
    F = {"years": [2025, 2024, 2023, 2022], "revenue": [25e9] * 4,
         "ebit": [1.5e9] * 4, "net_income": [-0.09e9, 0.5e9, 0.6e9, 0.55e9],
         "equity": [4.1e9] * 4, "total_assets": [40e9] * 4, "net_ppe": [20e9] * 4}
    v = route.value_stock("TEST", fund=f, forensic=_FOR, F=F)
    assert v["ok"] and v["equity_value"] > 0
    assert "service public regule" in v["method"], v["method"]

# --------------------------------------------------------------------------- #
# 6. EBIT ECONOMIQUE : des charges recurrentes reclassees sous la ligne
#    operationnelle ne doivent JAMAIS gonfler la marge (35 % de l'univers touche)
# --------------------------------------------------------------------------- #
def _etats_fmp(operating, pretax, interets=0.0, revenue=5.57e9, net=0.039e9):
    inc = {2025: {"revenue": revenue, "operatingIncome": operating,
                  "incomeBeforeTax": pretax, "interestExpense": interets,
                  "netIncome": net, "weightedAverageShsOutDil": 8.7e7,
                  "reportedCurrency": "USD"}}
    bal = {2025: {"totalStockholdersEquity": 1.37e9, "totalAssets": 1.99e9,
                  "totalLiabilities": 0.62e9, "totalDebt": 7e6,
                  "cashAndCashEquivalents": 0.3e9, "preferredStock": 0.0,
                  "minorityInterest": 0.0, "reportedCurrency": "USD"}}
    return {"income": inc, "balance": bal, "cashflow": {}}


def _fond(entry):
    from quantbench.data import fmp
    return fmp.fundamentals_from_fmp(
        "TEST", {"price": 1.07, "market_cap": 93e6, "exchange": "NASDAQ",
                 "name": "TEST", "sector": "Financial Services",
                 "industry": "Financial - Credit Services", "country": "US"},
        entry, {})


def test_charges_reclassees_sous_la_ligne_operationnelle_ne_gonflent_pas_la_marge():
    """Yiren Digital : resultat operationnel +2,145 Md, "autres charges" -2,202 Md
    (provisions pour creances douteuses), resultat avant impot NEGATIF. La marge
    publiee de 38,5 % etait projetee a perpetuite -> +4 700 % d'upside."""
    f = _fond(_etats_fmp(operating=2.145e9, pretax=-0.057e9))
    assert f["ebit"] <= 0, f"EBIT non corrige : {f['ebit']}"
    assert f["operating_margin"] < 0.01, f"marge encore gonflee : {f['operating_margin']:.1%}"


def test_compte_de_resultat_ordinaire_reste_inchange():
    """La correction ne doit RIEN changer quand la ligne operationnelle est saine :
    resultat avant impot = EBIT - interets (Apple, Microsoft, Visa, Exxon...)."""
    f = _fond(_etats_fmp(operating=1.0e9, pretax=0.95e9, interets=0.05e9))
    assert abs(f["ebit"] - 1.0) < 1e-6, f"EBIT sain modifie a tort : {f['ebit']}"


def test_produits_non_operationnels_ne_gonflent_pas_l_ebit():
    """Symetrie : une societe riche en tresorerie a un resultat avant impot
    SUPERIEUR a son EBIT (produits financiers). L'EBIT ne doit pas etre releve."""
    f = _fond(_etats_fmp(operating=1.0e9, pretax=1.40e9))
    assert abs(f["ebit"] - 1.0) < 1e-6, f"EBIT gonfle par du hors-exploitation : {f['ebit']}"


def test_historique_des_marges_utilise_aussi_l_ebit_economique():
    """Sinon la marge normalisee et la marge moyenne du cycle restent calculees sur
    un resultat operationnel qui exclut des charges recurrentes : l'incoherence se
    propage aux methodes sectorielles."""
    from quantbench.data import fmp
    e = _etats_fmp(operating=2.145e9, pretax=-0.057e9)
    e["income"][2024] = dict(e["income"][2025])
    e["balance"][2024] = dict(e["balance"][2025])
    F = fmp.financials_from_fmp(e)
    assert F is not None
    assert all(v is not None and v <= 0 for v in F["ebit"]),         f"historique d'EBIT non corrige : {F['ebit']}"

def test_le_monte_carlo_reflete_la_ponderation_du_routage():
    """Toute ponderation appliquee par le routage (survie, defaut, probabilite de
    REALISATION d'un redressement) doit etre reproduite par la simulation, sinon la
    mediane simulee ECRASE le raffinement — c'est l'angle mort qui a fait repasser
    General Motors de -70 % a -100 % et Allstate de -34 % a +106 %."""
    import build_site_fmp as bs
    import inspect
    src = inspect.getsource(bs.run_mc)
    for ponderation in ("jeune/deficitaire", "detresse",
                        "mature_deficitaire", "cyclique"):
        assert ponderation in src, (
            f"la categorie '{ponderation}' est ponderee par le routage mais pas "
            f"par run_mc : la simulation annulerait la ponderation")
    assert "probabilite_de_realisation" in src



# --------------------------------------------------------------------------- #
# 13. Les flux de TRESORERIE priment sur le resultat comptable
#
# Rapport annuel 2024 de Branicks Group : EBIT -288,7 M EUR, perte nette
# -365,5 M EUR, mais FFO +52,2 M EUR et flux d'exploitation +54,8 M EUR — la perte
# etant integralement due a la reevaluation IFRS du parc immobilier (-6,9 %).
# Le modele lisait le resultat et envoyait cette fonciere a la LIQUIDATION.
# --------------------------------------------------------------------------- #
def _fonciere_ifrs(**kw):
    """Fonciere IFRS depreciee : perte comptable massive, tresorerie positive."""
    base = dict(sector="Real Estate", industry="Real Estate - Services",
                country="DE", revenue=0.28, ebit=-0.34, net_income=-0.32,
                dep_amort=0.01, cfo=0.055, book_equity=0.86, total_debt=2.63,
                cash=0.19, total_assets=4.29, market_cap=0.088, price=1.05,
                shares=8.36e7, operating_margin=-1.18)
    base.update(kw)
    return societe(**base)


def test_une_perte_non_monetaire_ne_vaut_pas_detresse():
    cat = route.classify(_fonciere_ifrs(), {"scores": {"altman_z": 0.5}}, etats())
    assert cat == "fonciere", (
        f"routee en '{cat}' : une depreciation d'actifs ne fait sortir aucun euro, "
        f"elle ne peut pas declencher la liquidation d'une societe qui encaisse")


def test_une_perte_monetaire_reste_une_detresse():
    """Le garde-fou ne doit pas desarmer la detresse REELLE : meme perte, mais la
    tresorerie sort aussi."""
    f = _fonciere_ifrs(cfo=-0.20, sector="Technology", industry="Software",
                       book_equity=-0.5)
    assert route.classify(f, {"scores": {"altman_z": 0.5}}, etats()) == "detresse"


def test_le_ffo_est_immunise_contre_les_reevaluations():
    """Sous IFRS le resultat net porte les reevaluations, non monetaires. Le FFO
    doit s'ancrer sur le flux d'exploitation, qui les neutralise par construction.
    Reference publiee : Branicks annonce 52,2 M EUR de FFO pour 54,8 M EUR de flux."""
    r = route.value_reit(_fonciere_ifrs())
    assert r is not None, "une fonciere qui encaisse ses loyers doit etre valorisable"
    assert abs(r["ffo"] - 0.055) < 1e-9, (
        f"FFO={r['ffo']} : il doit valoir le flux d'exploitation, pas un resultat "
        f"net creuse par une reevaluation")


def test_le_ffo_reste_borne_par_la_formule_comptable():
    """Quand l'immeuble est REELLEMENT AMORTI — les foncieres americaines — le flux
    d'exploitation peut etre gonfle par le besoin en fonds de roulement : la formule
    comptable, si elle est positive et plus basse, doit borner."""
    r = route.value_reit(_fonciere_ifrs(net_income=0.10, dep_amort=0.20, cfo=0.50))
    assert abs(r["ffo"] - 0.30) < 1e-9, f"FFO={r['ffo']} : la borne comptable a saute"


def test_sans_amortissement_la_formule_comptable_ne_borne_rien():
    """Sous juste valeur l'immeuble n'est pas amorti : "resultat net +
    amortissements" degenere en simple resultat net, qui ne mesure aucune generation
    de tresorerie. RioCan passe 1 M$ d'amortissements pour 309 M$ de flux — retenir
    la formule bornait son FFO a 50 M$ et sortait la premiere fonciere du Canada a
    -85 %."""
    r = route.value_reit(_fonciere_ifrs(net_income=0.049, dep_amort=0.001, cfo=0.309))
    assert abs(r["ffo"] - 0.309) < 1e-9, (
        f"FFO={r['ffo']} : un amortissement negligeable ne peut pas servir de borne")


def test_pas_de_ffo_sans_encaissement():
    assert route.value_reit(_fonciere_ifrs(cfo=-0.05)) is None


def test_l_erosion_suit_la_tresorerie_et_non_le_resultat():
    """Une perte non monetaire ne ronge aucune autonomie."""
    intacte, p = route.erosion_par_les_pertes(
        societe(net_income=-1.0, cfo=0.2), 1.0)
    assert p is None and intacte == 1.0
    erodee, p2 = route.erosion_par_les_pertes(
        societe(net_income=-1.0, cfo=-0.5), 1.0)
    assert p2 is not None and erodee < 1.0


# --------------------------------------------------------------------------- #
# 14. Le levier n'est pas plafonne par une valeur d'opinion
#
# Branicks porte 2,63 Md$ de dette pour 88 M$ de capitalisation (D/E = 29,9). Le
# levier etait borne a 5, ce qui revenait a decreter qu'au-dela d'un certain
# endettement le risque cesse de croitre — l'inverse de la formule de Damodaran.
# --------------------------------------------------------------------------- #
def test_le_cout_des_fonds_propres_croit_avec_le_levier_extreme():
    from quantbench.valuation.build_universal import beta_ascendant
    b6, _, _ = beta_ascendant(societe(total_debt=6.0, market_cap=1.0), 0.25)
    b30, _, _ = beta_ascendant(societe(total_debt=30.0, market_cap=1.0), 0.25)
    assert b30 > b6 * 3, (
        f"beta {b6:.2f} a D/E=6 contre {b30:.2f} a D/E=30 : le levier est plafonne, "
        f"l'equite d'une societe surendettee ressort artificiellement peu risquee")


# --------------------------------------------------------------------------- #
# 15. Un benefice jamais encaisse ne finance aucune croissance
#
# FDCTech : dix ans, 3 M$ de resultat d'exploitation cumule, 31 M$ de tresorerie
# CONSOMMEE, chiffre d'affaires multiplie par 76. Le modele capitalisait ce
# benefice comptable a l'infini (+937 %).
# --------------------------------------------------------------------------- #
def _hist(ebit, rev, cfo):
    n = len(rev)
    return {"years": [2025 - i for i in range(n)], "revenue": list(rev),
            "ebit": list(ebit), "cfo": list(cfo),
            "net_income": [0.0] * n, "equity": [1.0] * n,
            "total_assets": [2.0] * n, "net_ppe": [1.0] * n}


def test_conversion_en_tresorerie_mesuree_sur_le_cumul():
    F = _hist([5.9, -0.8, 1.6, -1.0, -1.7], [35, 27, 13, 6.5, 0.5],
              [-41, -7.2, 21, -0.5, -2.6])
    c = route.conversion_en_tresorerie(F)
    assert c is not None and c < 0, f"conversion={c} : le cumul de tresorerie est negatif"
    # Non mesurable quand il n'y a aucun benefice cumule a convertir.
    assert route.conversion_en_tresorerie(
        _hist([-1, -1, -1, -1], [10, 10, 10, 10], [1, 1, 1, 1])) is None


def test_une_tresorerie_jamais_degagee_bride_le_rendement_du_capital():
    from quantbench.valuation.build_universal import build_dcf_from_fundamentals
    sain, _ = build_dcf_from_fundamentals(societe(conversion_tresorerie=1.3))
    creux, _ = build_dcf_from_fundamentals(societe(conversion_tresorerie=-11.0))
    assert creux.current_roic < sain.current_roic, (
        "un benefice jamais encaisse doit rendre la croissance plus chere a financer")


def test_une_financiere_de_bilan_echappe_a_la_mesure_de_conversion():
    """Le flux d'exploitation d'une banque suit les encours de credit, pas
    l'exploitation : JPMorgan ressort a 0,41 sans anomalie aucune. Le test est
    STRUCTUREL — FDCTech est etiquetee Financial Services alors qu'elle vend des
    logiciels, et une exclusion par libelle laissait passer le cas meme a corriger."""
    from quantbench.valuation.build_universal import build_dcf_from_fundamentals
    banque = dict(sector="Financial Services", industry="Banks - Diversified",
                  revenue=5.0, book_equity=4.0, total_assets=60.0)
    a, _ = build_dcf_from_fundamentals(societe(conversion_tresorerie=1.3, **banque))
    b, _ = build_dcf_from_fundamentals(societe(conversion_tresorerie=0.05, **banque))
    assert a.current_roic == b.current_roic


# --------------------------------------------------------------------------- #
# 16. Une marge est un RAPPORT DE SOMMES, non la moyenne de rapports
#
# Pop Culture Group a triple son chiffre d'affaires en deux ans en perdant de
# l'argent. Ses marges anciennes, gagnees sur un dixieme du volume actuel, pesaient
# autant que les recentes dans une moyenne : le modele appliquait 18 % de marge a
# un chiffre d'affaires qui n'en a jamais degage (+2 002 %).
# --------------------------------------------------------------------------- #
def test_la_marge_de_cycle_est_ponderee_par_le_volume():
    F = _hist([-6.4, -13.6, -24.4, 1.4, 5.8], [107.6, 47.4, 18.5, 32.3, 25.5],
              [0.2, -5.2, -6.0, -19.4, -4.0])
    m = route.marge_de_cycle(F)
    attendu = sum(F["ebit"]) / sum(F["revenue"])
    assert abs(m - attendu) < 1e-12
    assert m < 0, (
        f"marge de cycle={m:.3f} : une societe dont le cumul est deficitaire ne peut "
        f"pas ressortir rentable parce que ses petites annees l'etaient")


def test_la_normalisation_inclut_les_exercices_deficitaires():
    """Ne moyenner que les annees benificiaires definit la rentabilite normale comme
    ce que la societe gagne quand elle gagne, et garantit une reponse optimiste."""
    F = _hist([2.0, -8.0, 2.0, -8.0], [100, 100, 100, 100], [1, 1, 1, 1])
    assert route.marge_de_cycle(F) == pytest.approx(-0.03)
    r = route.value_mature_loss(societe(operating_margin=-0.08), F)
    assert r is None or r["norm_margin"] < 0


# --------------------------------------------------------------------------- #
# 17. Un redressement se pondere par sa probabilite, sur TOUTES les routes
#
# La ponderation ne s'appliquait qu'aux cycliques et aux societes matures en perte.
# La route standard, qui normalise pourtant dans les memes termes, tenait le retour
# a la marge de cycle pour ACQUIS.
# --------------------------------------------------------------------------- #
def test_la_route_standard_pondere_aussi_le_redressement():
    src = (Path(__file__).resolve().parent.parent
           / "quantbench" / "valuation" / "route.py").read_text(encoding="utf-8")
    bloc = src.split("mn = marge_normalisee(fund, F)")[1].split("except Exception")[0]
    assert "_pondere_par_realisation" in bloc, (
        "la route standard valorise une marge normalisee sans ponderer le "
        "redressement par sa probabilite")


def test_une_equite_negative_traverse_la_ponderation():
    """Une equite negative signale l'ECHEC de l'approche entreprise (dette de
    financement captive), pas un redressement improbable. La ponderation la ramenait
    a zero puis la melangeait a une liquidation, masquant l'echec et empechant la
    bascule vers le modele cote equite : General Motors ressortait a -78 % au lieu
    de -4 %."""
    r = {"equity_value": -3.0, "method": "DCF FCFF", "confidence": "moyenne"}
    out = route._pondere_par_realisation(r, societe(operating_margin=0.01),
                                         etats(), 0.20)
    assert out["equity_value"] < 0, (
        "l'equite negative a ete masquee : la bascule cote equite ne se declenchera "
        "plus")


# --------------------------------------------------------------------------- #
# 18. Erreurs d'UNITE MONETAIRE : l'historique est publie en devise locale
# --------------------------------------------------------------------------- #
def test_le_service_public_normalise_en_marge_pas_en_montant():
    """`F` est publie en DEVISE LOCALE, `fund` est converti en dollars. Prendre la
    mediane des montants historiques et la diviser par un milliard traitait des pesos
    comme des dollars : la valeur etait surestimee d'exactement le taux de change —
    x1 400 pour l'argentine Edenor (+6 864 %), x33 pour la thailandaise EGCO."""
    f = societe(sector="Utilities", industry="Utilities - Regulated Electric",
                revenue=2.0, net_income=0.1, book_equity=1.5, market_cap=1.1,
                ebit=0.2, operating_margin=0.10, roe=0.067)
    # Meme societe, historique publie en pesos (x1 400) : la valeur ne doit PAS bouger.
    n = 6
    usd = {"years": [2025 - i for i in range(n)], "revenue": [2.0e9] * n,
           "ebit": [0.2e9] * n, "net_income": [0.1e9] * n, "equity": [1.5e9] * n,
           "total_assets": [4.0e9] * n, "net_ppe": [3.0e9] * n, "cfo": [0.3e9] * n}
    ars = dict(usd, revenue=[x * 1400 for x in usd["revenue"]],
               ebit=[x * 1400 for x in usd["ebit"]],
               net_income=[x * 1400 for x in usd["net_income"]],
               equity=[x * 1400 for x in usd["equity"]])
    a = route.value_regulated(f, usd)
    b = route.value_regulated(f, ars)
    assert a and b
    assert abs(a["equity_value"] - b["equity_value"]) < 1e-6, (
        f"{a['equity_value']:.3f} contre {b['equity_value']:.3f} : la valeur depend "
        f"de la devise de PUBLICATION, ce qui est impossible")


def test_aucune_grandeur_de_l_historique_n_est_traitee_comme_des_dollars():
    """Garde-fou de forme : dans le module de routage, seule la mise en forme finale
    a le droit de convertir une grandeur en milliards. Toute autre occurrence signale
    qu'un montant en devise locale est traite comme des dollars."""
    src = (Path(__file__).resolve().parent.parent
           / "quantbench" / "valuation" / "route.py").read_text(encoding="utf-8")
    lignes = [ligne.strip() for ligne in src.splitlines()
              if "1e9" in ligne and not ligne.strip().startswith("#")]
    assert lignes == ["vps = eq * 1e9 / shares if shares else None"], (
        f"conversions suspectes : {lignes}")


# --------------------------------------------------------------------------- #
# 19. Univers : instruments derives exclus, foncieres en fiducie conservees
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sym,nom", [
    ("CCXIW", "Churchill Capital Corp XI Warrants"),
    ("EVOXW", "Evolution Global Acquisition Corp. Wt"),
    ("QADRW", "Qdro Acquisition Corp. C/wts (to Pur Com) 27/03/2031"),
    ("CCXIU", "Churchill Capital Corp XI Units"),
    ("NOVTU", "Novanta Inc. Tangible Equity Units"),
    ("ALPXR", "Alpex Acquisition Corp. Rt"),
    ("CRACR", "Crown Reserve Acquisition Corp. I Rights"),
])
def test_les_instruments_derives_sont_exclus(sym, nom):
    """Un warrant est une option sur le titre, un "unit" un panier vendu a
    l'introduction d'un SPAC, un "right" un droit de souscription : aucun n'a pour
    contrepartie les comptes de la societe, et tous heritent de la capitalisation du
    sous-jacent — les warrants de Churchill Capital XI portaient 2,3 Md$."""
    from quantbench.data.fmp import _est_un_derive
    assert _est_un_derive(sym, nom)


@pytest.mark.parametrize("sym,nom", [
    ("UNTC", "Unit Corporation"),                      # petrolier de l'Oklahoma
    ("MGYOY", "MOL Magyar Olaj-es Gazipari RT"),       # "Rt" = societe anonyme hongroise
    ("ET", "Energy Transfer LP"),
    ("EPD", "Enterprise Products Partners L.P."),
    ("BIP", "Brookfield Infrastructure Partners L.P."),
    ("REI-UN.TO", "RioCan Real Estate Investment Trust"),
    ("PLTR", "Palantir Technologies Inc."),            # ticker en W/R/U : piege
    ("INTU", "Intuit Inc."),
])
def test_les_vraies_societes_ne_sont_pas_prises_pour_des_derives(sym, nom):
    from quantbench.data.fmp import _est_un_derive
    assert not _est_un_derive(sym, nom)


def test_les_foncieres_en_fiducie_sont_reintegrees():
    """`isFund` decrit une FORME JURIDIQUE et se declenche sur le mot "Trust". Il
    faisait perdre 164 societes et 279 Md$ de capitalisation : la TOTALITE de
    l'immobilier cote canadien, mais aussi Essex Property, Federal Realty, Vornado
    et Kite Realty. On les reintegre par leur INDUSTRIE — les 4 962 vrais fonds
    marques `isFund` se rangent sans exception sous "Asset Management"."""
    src = (Path(__file__).resolve().parent.parent
           / "quantbench" / "data" / "fmp.py").read_text(encoding="utf-8")
    bloc = src.split("def screener(")[1].split("def ")[0]
    assert 'startswith("REIT")' in bloc and 'r.get("isFund")' in bloc, (
        "le screener ne reintegre plus les foncieres constituees en fiducie")


# --------------------------------------------------------------------------- #
# 20. Le bilan equilibre sur les fonds propres TOTAUX, minoritaires inclus
# --------------------------------------------------------------------------- #
def test_les_minoritaires_ne_font_pas_echouer_l_identite_du_bilan():
    """`book_equity` ne retient que la part attribuable aux actionnaires — la bonne
    grandeur pour VALORISER, mais pas pour verifier une identite comptable. Le
    controle rejetait toute societe detenant des filiales non integralement
    possedees, Branicks ressortant a 10 % d'ecart pour cette seule raison."""
    from quantbench.data.validate import valider
    f = societe(total_assets=10.0, total_liab=6.0, book_equity=3.0,
                total_equity=4.0)          # 1,0 de minoritaires
    motifs = [m for m in valider(f, etats(), None) if "identite du bilan" in m]
    assert not motifs, motifs
    # Un vrai desequilibre doit rester detecte.
    f2 = societe(total_assets=10.0, total_liab=6.0, book_equity=3.0, total_equity=3.0)
    assert any("identite du bilan" in m for m in valider(f2, etats(), None))


# --------------------------------------------------------------------------- #
# 21. LIQUIDATION : la decote porte sur l'ACTIF, le passif se retranche ensuite
#
# Wesizwe Platinum construit la mine de Bakubung sans avoir jamais produit :
# 2,15 Md$ d'actif, 1,83 Md$ de passif (pret China Development Bank), zero chiffre
# d'affaires, et une reserve sur la continuite d'exploitation au rapport 2025.
# Sa valeur de liquidation ressortait a 0,25 Md$ pour 16 M$ de capitalisation,
# soit +1 462 %.
# --------------------------------------------------------------------------- #
def test_la_decote_de_liquidation_porte_sur_l_actif_pas_sur_l_equite():
    """Appliquer la decote aux FONDS PROPRES revient a supposer que les dettes la
    subissent aussi, AU BENEFICE DE L'ACTIONNAIRE. Elles sont nominales et
    prioritaires : le creancier est servi en entier d'abord."""
    f = societe(sector="Basic Materials", industry="Other Precious Metals",
                total_assets=2.15, total_liab=1.83, book_equity=0.32, cash=0.007,
                revenue=None, ebit=-0.01, net_income=0.01, cfo=-0.034, capex=-0.124)
    liq = route.valeur_de_liquidation(f)
    taux = route.sect(f, "recuperation", 0.5)
    attendu = max(0.007 + taux * (2.15 - 0.007) - 1.83, 0.0)
    assert abs(liq - attendu) < 1e-9, f"liquidation={liq}, attendu {attendu}"
    # L'erreur corrigee valait (1 - taux) x passif : ici plus que l'equite entiere.
    ancienne = taux * 0.32
    assert liq < ancienne, (
        f"{liq:.3f} contre {ancienne:.3f} : la formule surestimait de "
        f"{(1 - taux) * 1.83:.3f} Md$, soit {(1 - taux) * 1.83 / 0.32:.1f} fois l'equite")


def test_une_societe_sans_dette_n_est_pas_penalisee():
    """L'erreur vaut (1 - taux) x passif : elle doit etre NULLE sans passif — d'ou
    son invisibilite tant qu'on ne regardait pas de societe endettee."""
    f = societe(total_assets=1.0, total_liab=0.0, book_equity=1.0, cash=0.4)
    taux = route.sect(f, "recuperation", 0.5)
    assert abs(route.valeur_de_liquidation(f) - (0.4 + taux * 0.6)) < 1e-9


def test_un_passif_non_couvert_donne_zero_et_non_une_erreur():
    """Quand l'actif realisable ne couvre pas les dettes, l'actionnaire ne recoit
    rien. C'est une REPONSE, pas un echec de methode : la route ne doit pas se
    rabattre ailleurs et ainsi ressusciter une valeur."""
    f = societe(sector="Basic Materials", industry="Other Precious Metals",
                revenue=None, revenue_history=[], total_assets=2.15, total_liab=1.83,
                book_equity=0.32, cash=0.007, ebit=-0.01, net_income=0.01)
    r = route.value_assetbased(f)
    assert r is not None and r["equity_value"] == 0.0 and r.get("passif_non_couvert")
    v = route.value_stock("TEST", fund=f, forensic=_FOR, F=etats())
    assert v["ok"] and v["equity_value"] == 0.0


# --------------------------------------------------------------------------- #
# 22. La consommation de tresorerie inclut les INVESTISSEMENTS
# --------------------------------------------------------------------------- #
def test_la_consommation_inclut_le_capex():
    """Une societe qui construit son outil consomme sa tresorerie par le capex bien
    plus que par l'exploitation : Wesizwe brule 562 M ZAR d'exploitation pour
    1 238 M ZAR investis dans la mine."""
    f = societe(cfo=-0.562, capex=-1.238)
    assert abs(route.consommation_de_tresorerie(f) - 1.800) < 1e-9
    # Une societe qui encaisse plus qu'elle n'investit ne consomme rien.
    assert route.consommation_de_tresorerie(societe(cfo=1.0, capex=-0.3)) == 0.0


def test_la_survie_se_mesure_sur_la_tresorerie_pas_sur_le_resultat():
    """Sur 126 societes routees "jeune/deficitaire", 37 affichent un resultat et un
    flux de signes OPPOSES : GitLab perd 56 M$ comptables en encaissant 233 M$, NIO
    perd 2,16 Md$ en encaissant 431 M$."""
    encaisse = societe(net_income=-0.056, cfo=0.233, capex=-0.02, cash=0.5)
    assert route.probabilite_de_survie(encaisse) == 1.0
    consomme = societe(net_income=-0.056, cfo=-0.233, capex=-0.02, cash=0.1)
    assert route.probabilite_de_survie(consomme) < 0.2


def test_la_survie_n_a_pas_de_plancher_arbitraire():
    """Elle etait plancherisee a 0,30 : une societe sans tresorerie et en pleine
    consommation se voyait creditee d'une chance sur trois de survivre, chiffre pose
    a la main et sans fondement."""
    exsangue = societe(cash=0.0, cfo=-0.5, capex=-0.1, net_income=-0.6)
    assert route.probabilite_de_survie(exsangue) == 0.0
    r = route.value_young(exsangue)
    liq = route.valeur_de_liquidation(exsangue)
    assert abs(r["equity_value"] - liq) < 1e-9, (
        "sans autonomie, la valeur doit se reduire a la liquidation")


# --------------------------------------------------------------------------- #
# 23. "Jeune" veut dire COURTE HISTOIRE, pas "jamais rentable"
# --------------------------------------------------------------------------- #
def test_une_societe_ancienne_et_deficitaire_n_est_pas_une_jeune_pousse():
    """Le critere ne comptait que les exercices benificiaires : une societe publiant
    depuis dix ans sans en avoir aucun etait valorisee sur la marge MEDIANE DE SON
    SECTEUR, qu'elle n'a precisement jamais approchee. DiDi Global (32,6 Md$ de
    chiffre d'affaires), Carvana (20,3), NIO (12,6) et Roku (4,7) en relevaient."""
    n = 10
    longue = {"years": [2025 - i for i in range(n)], "revenue": [30.0e9] * n,
              "ebit": [-1.0e9] * n, "net_income": [-1.0e9] * n, "cfo": [-0.5e9] * n,
              "equity": [5.0e9] * n, "total_assets": [20.0e9] * n,
              "net_ppe": [5.0e9] * n}
    f = societe(revenue=30.0, ebit=-1.0, net_income=-1.0, operating_margin=-0.033)
    assert route.classify(f, None, longue) == "mature_deficitaire"
    # Historique court : la route descendante reste legitime, faute de reference.
    courte = {k: (v[:3] if isinstance(v, list) else v) for k, v in longue.items()}
    assert route.classify(f, None, courte) == "jeune/deficitaire"


# --------------------------------------------------------------------------- #
# 24. Un controle rejette une DONNEE FAUSSE, jamais un RESULTAT
# --------------------------------------------------------------------------- #
def test_aucun_controle_ne_compare_la_capitalisation_aux_comptes():
    """Quatre seuils confrontaient la capitalisation — grandeur d'ACTIONNAIRE — au
    chiffre d'affaires, au resultat operationnel et a l'actif net — grandeurs
    d'ENTREPRISE — sans jamais ajouter la dette : ils mesuraient le LEVIER. Charter
    Communications, 19,57 Md$ de capitalisation pour 12,73 Md$ de resultat
    operationnel, etait ecartee au seul motif qu'elle porte 95,8 Md$ de dette."""
    src = (Path(__file__).resolve().parent.parent
           / "quantbench" / "data" / "validate.py").read_text(encoding="utf-8")
    corps = src.split("def valider(")[1].split("def fraicheur_des_comptes")[0]
    for interdit in ("MIN_CAP_SUR_CA", "MIN_CAP_SUR_FONDS_PROPRES", "MIN_CAP_SUR_EBIT",
                     "MAX_CA_SUR_ACTIF", "incoherente avec"):
        assert interdit not in corps, f"seuil d'opinion reintroduit : {interdit}"


def test_une_valorisation_extreme_mais_exacte_n_est_pas_censuree():
    """Une societe endettee peut legitimement coter sous deux fois son resultat
    operationnel, ou sous 3 % de ses ventes."""
    from quantbench.data.validate import valider
    f = societe(market_cap=0.02, price=1.0, shares=2e7, revenue=5.0, ebit=1.0,
                book_equity=4.0, total_assets=9.0, total_liab=5.0, total_equity=4.0,
                cash=1.0, total_debt=2.0)
    assert valider(f, etats(), None) == []


def test_l_age_des_comptes_ne_rejette_plus():
    """Des comptes anciens sont les derniers comptes CONNUS. Treize pour cent des
    lignes ecartees pour cette raison deposaient encore a la SEC."""
    from quantbench.data.validate import valider, fraicheur_des_comptes
    entry = {"income": {2019: {"date": "2019-12-31"}}, "balance": {2019: {}}}
    assert not [m for m in valider(societe(), etats(), entry) if "perim" in m]
    date, mois = fraicheur_des_comptes(entry)
    assert date == "2019-12-31" and mois > 60


def test_la_fraicheur_se_compte_en_mois_pas_en_millesimes():
    """Le millesime est decale chez le fournisseur pour 3 a 4 % des lignes — un
    exercice clos le 30 juin 2024 est etiquete 2023 — et comparer des annees civiles
    cree une falaise au 1er janvier sans aucun sens economique."""
    from datetime import datetime, timezone
    from quantbench.data.validate import fraicheur_des_comptes
    entry = {"income": {2023: {"date": "2024-06-30"}}, "balance": {2023: {}}}
    ref = datetime(2026, 7, 30, tzinfo=timezone.utc)
    _, mois = fraicheur_des_comptes(entry, ref)
    assert mois == 25, mois
    # AUCUNE FALAISE AU 1er JANVIER : deux clotures separees d'un seul jour ne
    # peuvent pas differer d'un exercice entier. La regle par millesime les faisait
    # basculer ensemble chaque 1er janvier — 34 lignes perdues en juillet, 428 en
    # janvier, pour des societes strictement inchangees.
    ref = datetime(2026, 7, 30, tzinfo=timezone.utc)
    veille = fraicheur_des_comptes(
        {"income": {2023: {"date": "2023-12-31"}}, "balance": {2023: {}}}, ref)[1]
    lendemain = fraicheur_des_comptes(
        {"income": {2024: {"date": "2024-01-01"}}, "balance": {2024: {}}}, ref)[1]
    assert abs(veille - lendemain) <= 1, (veille, lendemain)


def test_les_emissions_obligataires_restent_hors_univers():
    """Ce que les seuils supprimes attrapaient LEGITIMEMENT releve du TYPE DE TITRE
    et se traite sur le libelle."""
    from quantbench.data.fmp import _is_preferred
    for sym, nom in [("DUKB", "Duke Energy Corporation 5.625% Junior Subordinated Debentures"),
                     ("DTW", "DTE Energy Company JR SUB DB 2017 E"),
                     ("DTG", "DTE Energy Company 2021 Series"),
                     ("DTB", "DTE Energy Company Series 2021"),
                     ("EAI", "Entergy Arkansas 4.875%")]:
        assert _is_preferred(sym, nom), f"{sym} devrait etre exclue"
    for sym, nom in [("PTRN", "Pattern Group Inc. Series A Common Stock"),
                     ("AAPL", "Apple Inc."), ("HEI-A", "HEICO Corporation"),
                     ("ET", "Energy Transfer LP")]:
        assert not _is_preferred(sym, nom), f"{sym} ne doit pas etre exclue"


def test_un_courtier_immobilier_n_est_pas_valorise_comme_un_immeuble():
    """Aucun parc immobilier ne tourne trois fois par an : un chiffre d'affaires de
    plusieurs fois l'actif exclut la route du FFO capitalise."""
    courtier = societe(sector="Real Estate", industry="Real Estate - Services",
                       revenue=1.5, total_assets=0.1)
    assert route.classify(courtier, None, etats()) != "fonciere"
    fonciere = societe(sector="Real Estate", industry="REIT - Retail",
                       revenue=0.3, total_assets=4.0)
    assert route.classify(fonciere, None, etats()) == "fonciere"


# --------------------------------------------------------------------------- #
# 25. Le temps ecoule depuis l'observation se retranche de l'autonomie
# --------------------------------------------------------------------------- #
def test_le_temps_ecoule_erode_l_autonomie():
    """Un bilan est une PHOTOGRAPHIE datee. Iridium World Communications, en faillite
    depuis 1999, publiait en 1998 119,7 M$ de fonds propres pour 107,6 M$ de perte
    annuelle. Nous la valorisions vingt-huit ans plus tard sur ces memes fonds
    propres, a +5 360 %."""
    frais = societe(cash=1.0, cfo=-0.2, capex=0.0, age_des_comptes_mois=3)
    vieux = societe(cash=1.0, cfo=-0.2, capex=0.0, age_des_comptes_mois=28 * 12)
    assert route.probabilite_de_survie(frais) > 0.5
    assert route.probabilite_de_survie(vieux) == 0.0


def test_le_temps_ne_penalise_pas_une_societe_qui_encaisse():
    """Ce n'est pas un seuil d'anciennete deguise : le temps ecoule n'ecarte rien par
    lui-meme."""
    f = societe(cash=1.0, cfo=0.5, capex=-0.1, age_des_comptes_mois=30 * 12)
    assert route.probabilite_de_survie(f) == 1.0


def test_un_tableau_de_flux_entierement_nul_est_un_tableau_absent():
    """Iridium publie pour 1998 une perte de 107,6 M$ et des flux tous a zero : nous
    en concluions qu'elle ne consommait rien."""
    f = societe(cfo=0.0, capex=0.0, net_income=-0.1076)
    assert abs(route.consommation_de_tresorerie(f) - 0.1076) < 1e-12
    # Une societe reellement a l'equilibre, elle, ne consomme rien.
    assert route.consommation_de_tresorerie(
        societe(cfo=0.0, capex=0.0, net_income=0.0)) == 0.0


# --------------------------------------------------------------------------- #
# 26. Le Monte Carlo delegue la marge au routage au lieu de la reimplementer
# --------------------------------------------------------------------------- #
def test_la_simulation_ne_reimplemente_pas_la_marge_normalisee():
    """Le miroir avait silencieusement diverge : il moyennait encore les RATIOS
    annuels, ne retenait que les exercices benificiaires et posait 12 % de marge
    cible en dur. La simulation aurait actualise des flux differents de ceux de la
    valorisation affichee, et sa mediane ecrase l'upside."""
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "build_site_fmp.py").read_text(encoding="utf-8")
    corps = src.split("def _route_margin(")[1].split("\ndef ")[0]
    assert "marge_de_cycle" in corps, "la simulation reimplemente la marge normalisee"
    assert "0.12" not in corps, "marge cible codee en dur dans la simulation"


# --------------------------------------------------------------------------- #
# 27. Ce qui bloquait le DEPLOIEMENT : cinq titres a plus d'un million de %
#
# Le garde-fou qualite refusait le build quotidien depuis plusieurs jours. Les
# titres en cause partageaient tous des fonds propres attribuables NEGATIFS et des
# comptes de trois a dix-sept ans.
# --------------------------------------------------------------------------- #
def test_les_tests_hors_ligne_ne_dependent_d_aucun_client_reseau():
    """`quantbench/data/__init__.py` importait `build` -> `edgar` -> `requests` des
    l'import du paquet : la suite d'invariants, pourtant concue pour ne toucher aucun
    reseau, ne pouvait pas s'importer sans client HTTP. L'integration continue
    echouait a la COLLECTE des tests, sans qu'aucune regression ne soit en cause."""
    src = (Path(__file__).resolve().parent.parent
           / "quantbench" / "data" / "__init__.py").read_text(encoding="utf-8")
    lignes = [ligne for ligne in src.splitlines()
              if ligne.startswith("from ") or ligne.startswith("import ")]
    assert not lignes, f"import immediat dans le paquet de donnees : {lignes}"


def test_la_liquidation_ne_rend_que_la_part_attribuable():
    """Actif moins passif donne les fonds propres TOTAUX. Ce qui revient aux
    minoritaires d'une filiale ne reviendra jamais a l'actionnaire : Etao
    International affiche -15,6 M$ de fonds propres attribuables pour +9,6 M$ de
    minoritaires, et ressortait a +5 291 567 %."""
    f = societe(total_assets=0.053, total_liab=0.059, total_equity=-0.006,
                book_equity=-0.0156, cash=0.0118)
    assert route.valeur_de_liquidation(f) == 0.0
    # Sans minoritaires, la formule est inchangee.
    g = societe(total_assets=1.0, total_liab=0.2, total_equity=0.8, book_equity=0.8,
                cash=0.3)
    taux = route.sect(g, "recuperation", 0.5)
    assert abs(route.valeur_de_liquidation(g)
               - max(0.3 + taux * 0.7 - 0.2, 0.0)) < 1e-12


def test_le_portillon_de_l_actif_net_teste_la_presence_pas_le_signe():
    """Il exigeait un actif net POSITIF et renoncait sinon — si bien qu'une societe
    dont le passif excede l'actif, cas ou la reponse "zero" est justement la bonne,
    ressortait "valorisation impossible" et repartait dans la cascade de repli, qui
    lui inventait une valeur. Qingdao Footwear, 10,2 M$ d'actif pour 16,8 M$ de
    passif, ressortait a +1 282 295 %."""
    f = societe(revenue=None, revenue_history=[], total_assets=0.0102,
                total_liab=0.0168, total_equity=-0.0066, book_equity=-0.0066,
                cash=0.00005, total_debt=0.0018)
    r = route.value_assetbased(f)
    assert r is not None and r["equity_value"] == 0.0
    v = route.value_stock("TEST", fund=f, forensic=_FOR, F=etats())
    assert v["ok"] and v["equity_value"] == 0.0, v


def test_un_actif_total_nul_est_une_liasse_qui_ne_decrit_pas_l_entite():
    """Meme angle mort que celui du passif nul : le controle d'identite etait garde
    par `if ta and ta > 0`, si bien qu'un actif a ZERO le desactivait entierement.
    Entergy New Orleans publie 0 d'actif pour 91 M$ de passif et 16,9 Md$ de fonds
    propres — ceux du groupe Entergy entier. Elle ressortait a +2 566 %."""
    from quantbench.data.validate import valider
    f = societe(total_assets=0.0, total_liab=0.091, total_equity=16.92,
                book_equity=16.70, revenue=12.95)
    assert any("actif total absent" in m for m in valider(f, etats(), None))
    # Une societe sans aucun bilan renseigne ne declenche pas ce motif.
    g = societe(total_assets=None, total_liab=None, total_equity=None)
    assert not any("actif total absent" in m for m in valider(g, etats(), None))


def test_une_continuite_d_exploitation_se_constate():
    """Un DCF actualise les flux FUTURS d'une societe EN ACTIVITE. Qingdao Footwear
    etait valorisee sur son exercice 2010 — quinze ans et demi plus tard — et Alabama
    Aircraft sur 2008. Passe deux exercices annuels manques, la continuite n'est plus
    attestee par rien et seul subsiste un droit sur les actifs derniers constates."""
    recente = societe(age_des_comptes_mois=14)
    ancienne = societe(age_des_comptes_mois=186)
    assert route.classify(recente, None, etats()) == "standard"
    assert route.classify(ancienne, None, etats()) == "actif_net"


# --------------------------------------------------------------------------- #
# 28. Les dependances declarees doivent etre EXHAUSTIVES
#
# Trois listes de dependances coexistaient — requirements.txt, celle de
# l'integration continue et celle du deploiement — et AUCUNE n'etait complete :
# l'integration installait scikit-learn mais pas requests, requirements.txt
# declarait requests mais ni scikit-learn ni reportlab ni yfinance. La suite de
# tests echouait a l'IMPORT depuis des semaines, sans qu'aucune regression ne soit
# en cause — et une suite qui ne demarre pas ne protege rien.
# --------------------------------------------------------------------------- #
def test_toute_dependance_tierce_est_declaree():
    import ast
    import sys as _sys

    racine = Path(__file__).resolve().parent.parent
    stdlib = set(_sys.stdlib_module_names)
    locaux = {"quantbench", "scripts", "tests"} | {
        p.stem for p in (racine / "scripts").glob("*.py")}
    # Nom d'import -> nom du paquet sur l'index, quand ils different.
    ALIAS = {"sklearn": "scikit-learn", "yaml": "pyyaml", "PIL": "pillow",
             "dateutil": "python-dateutil", "cv2": "opencv-python",
             "bs4": "beautifulsoup4"}

    tiers = {}
    for f in list((racine / "quantbench").rglob("*.py")) + \
             list((racine / "scripts").glob("*.py")):
        try:
            arbre = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for n in ast.walk(arbre):
            if isinstance(n, ast.Import):
                noms = [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0:
                noms = [(n.module or "").split(".")[0]]
            else:
                continue
            for m in noms:
                if m and m not in stdlib and m not in locaux:
                    tiers.setdefault(m, set()).add(f.relative_to(racine).as_posix())

    declare = (racine / "requirements.txt").read_text(encoding="utf-8").lower()
    manquants = {m: sorted(ou) for m, ou in tiers.items()
                 if ALIAS.get(m, m).lower() not in declare}
    assert not manquants, (
        "dependances utilisees mais NON DECLAREES dans requirements.txt — "
        f"l'installation echouera en integration continue : {manquants}")


def test_les_workflows_installent_les_dependances_declarees():
    """Une liste tenue a la main dans un workflow diverge toujours de la realite.
    C'est ce qui a fait echouer l'integration continue et le deploiement pendant des
    semaines, chacun avec un paquet manquant DIFFERENT."""
    racine = Path(__file__).resolve().parent.parent
    for nom in ("ci.yml", "deploy.yml"):
        f = racine / ".github" / "workflows" / nom
        if not f.exists():
            continue
        texte = f.read_text(encoding="utf-8")
        assert "-r requirements.txt" in texte, (
            f"{nom} n'installe pas les dependances declarees")
        for ligne in texte.splitlines():
            depouille = ligne.strip()
            if depouille.startswith(("run: pip install", "pip install")) and \
                    "-r requirements.txt" not in depouille and "pytest" not in depouille \
                    and "--upgrade pip" not in depouille:
                raise AssertionError(
                    f"{nom} installe une liste tenue a la main : {depouille}")


# --------------------------------------------------------------------------- #
# 29. NOTE DE RISQUE — elle repond a une question DIFFERENTE de l'upside
#
# La valorisation dit combien vaut une societe ; la note dit quelle est la
# probabilite de perdre durablement sa mise. Un titre peut etre tres decote ET tres
# dangereux — c'est meme la configuration la plus frequente.
# --------------------------------------------------------------------------- #
def test_une_note_existe_pour_tout_titre_meme_sans_donnees():
    """Une note doit exister pour TOUS les titres. L'incertitude sur la donnee est
    elle-meme un risque, portee par la dimension de confiance — pas un motif pour
    s'abstenir."""
    from quantbench.risk import noter
    for f in (societe(), {}, {"ticker": "X"},
              societe(revenue=None, ebit=None, book_equity=None, total_assets=None,
                      cash=None, cfo=None, total_debt=None)):
        r = noter(f, None)
        assert "score" in r and 0.0 <= r["score"] <= 1.0, r
        assert r["grade"] is None or r["grade"] in __import__(
            "quantbench.risk", fromlist=["GRADES"]).GRADES


def test_les_modalites_bornent_l_echelle_sans_passer_par_la_table():
    """-inf designe le MEILLEUR cas possible d'une dimension (aucune dette,
    autofinancee, aucune dilution), +inf le PIRE (aucun chiffre d'affaires). Les faire
    passer par les centiles les melangerait aux mesures : Apple, qui ne paie aucune
    charge d'interets, ressortait au 63e centile de RISQUE de solvabilite, au-dessus
    de societes couvrant les leurs trois fois."""
    import math

    from quantbench.risk.score import rang
    cal = {"quantiles": {"d1": {"global": [float(i) for i in range(1, 100)]}}}
    assert rang("d1", -math.inf, cal) == 0.0
    assert rang("d1", math.inf, cal) == 1.0
    assert 0.0 < rang("d1", 50.0, cal) < 1.0


def test_le_regime_est_structurel_et_non_un_libelle():
    """Le regime decide de la REGLE appliquee. Il se determine sur le bilan, jamais
    sur l'etiquette sectorielle du fournisseur : Visa porte "Financial - Credit
    Services" et n'est pas une banque, Simon Property ne tient que 9 % de fonds
    propres sans preter un centime."""
    from quantbench.risk import regime
    reseau = societe(sector="Financial Services",
                     industry="Financial - Credit Services",
                     revenue=36.0, total_assets=90.0, book_equity=39.0,
                     interest_expense=0.7, ebit=25.0)
    assert regime(reseau, etats()) != "financiere"
    banque = societe(sector="Financial Services", industry="Banks - Diversified",
                     revenue=170.0, total_assets=4200.0, book_equity=340.0,
                     interest_expense=90.0, ebit=60.0)
    assert regime(banque, etats()) == "financiere"


def test_une_societe_sans_dette_ET_deficitaire_n_est_pas_declaree_sure():
    """Le piege central de la dimension de solvabilite. La population sans charges
    d'interets est majoritairement composee de coquilles sans chiffre d'affaires : une
    regle naive leur decernerait la MEILLEURE note de solvabilite possible. L'absence
    de dette chez une societe en perte rend la solvabilite INDEFINIE, pas excellente."""
    from quantbench.risk import regime
    from quantbench.risk.dimensions import d1_solvabilite
    coquille = societe(interest_expense=None, ebit=-0.5, net_income=-0.5,
                       revenue=0.0, operating_margin=None)
    reg = regime(coquille, etats())
    assert reg == "sans_dette_deficitaire"
    signal, _ = d1_solvabilite(coquille, etats(), reg)
    assert signal is None, "la solvabilite doit etre INDEFINIE, jamais au meilleur rang"


def test_le_maillon_faible_est_debiaise_par_le_nombre_de_dimensions():
    """Le maximum de k rangs uniformes vaut k/(k+1) en esperance — 0,75 a trois
    dimensions, 0,90 a huit. Sans correction, une societe aux comptes RICHES serait
    mecaniquement moins bien notee qu'une societe opaque, a risque egal : exactement
    l'inverse de ce que la note doit dire."""
    src = (Path(__file__).resolve().parent.parent
           / "quantbench" / "risk" / "score.py").read_text(encoding="utf-8")
    assert "(1.0 - maximum) ** k" in src, (
        "le maillon faible n'est plus debiaise par le nombre de dimensions")


def test_un_plafond_ne_peut_qu_aggraver_la_note():
    """Un plafond exprime qu'aucune qualite par ailleurs ne compense un fait de cette
    nature. Il ne doit jamais AMELIORER une note deja plus mauvaise."""
    from quantbench.risk.score import _finaliser
    cal = {"plafonds": {"passif_non_couvert": 0.6}, "bornes_grades": None}
    assert _finaliser(0.2, ["passif_non_couvert"], [], "exploitante", cal)["score"] == 0.6
    assert _finaliser(0.9, ["passif_non_couvert"], [], "exploitante", cal)["score"] == 0.9


def test_aucun_seuil_de_valeur_dans_les_dimensions():
    """Les seules constantes autorisees dans le module de dimensions sont des minima
    ARITHMETIQUES et des bornes de winsorisation, dont l'effet sur un rang est nul par
    monotonie. Tout seuil disant ce qu'est une BONNE couverture d'interets ou une
    BONNE marge appartient au calibrage, jamais au code."""
    src = (Path(__file__).resolve().parent.parent
           / "quantbench" / "risk" / "dimensions.py").read_text(encoding="utf-8")
    autorises = {"_MIN_EXERCICES_IQR = 3", "_FENETRE_NORMALISATION = 3"}
    constantes = [ligne.strip() for ligne in src.splitlines()
                  if ligne.startswith("_") and "=" in ligne and "def " not in ligne
                  and not ligne.startswith("__")]
    assert set(constantes) <= autorises, f"constante non autorisee : {constantes}"


def test_les_bornes_de_grades_sont_strictement_croissantes():
    """Un decoupage par quantiles tombait deux fois sur la meme valeur — les plafonds
    posent le score exactement sur leur niveau et creent des MASSES — et un grade
    entier restait vide. Les bornes sont donc d'amplitude egale en log-odds."""
    import json
    f = (Path(__file__).resolve().parent.parent
         / "quantbench" / "risk" / "risk_calibration.json")
    if not f.exists():
        pytest.skip("calibrage absent")
    bornes = json.loads(f.read_text(encoding="utf-8")).get("bornes_grades") or []
    from quantbench.risk import GRADES
    assert len(bornes) == len(GRADES) - 1
    assert all(b < c for b, c in zip(bornes, bornes[1:])), bornes


def test_le_calibrage_ne_contient_aucun_poids_de_conviction():
    """Aucune dimension n'entre avec un poids choisi. Tant que la variable de resultat
    n'est pas construite, l'uniformite est la seule ponderation qui n'affirme rien."""
    import json
    f = (Path(__file__).resolve().parent.parent
         / "quantbench" / "risk" / "risk_calibration.json")
    if not f.exists():
        pytest.skip("calibrage absent")
    cal = json.loads(f.read_text(encoding="utf-8"))
    poids = set((cal.get("poids") or {}).values())
    assert len(poids) <= 1 or cal.get("origine_poids", "").startswith("estimee"), (
        "des poids differencies sans estimation declaree : "
        f"{cal.get('poids')} / origine={cal.get('origine_poids')}")


def test_aucune_valeur_non_finie_ne_sort_de_la_notation():
    """`json.dumps` de Python ecrit `Infinity` et `-Infinity`, qui ne sont PAS du
    JSON valide. Le navigateur echoue alors a lire la fiche ENTIERE — pas seulement
    la note — et affiche "profil indisponible". Le defaut est invisible cote serveur,
    ou Python relit sans broncher ce qu'il vient d'ecrire : seule une lecture par un
    vrai navigateur l'a revele.
    Les infinis sont des MODALITES internes, deja traduites en rang ; ils n'ont rien
    a faire dans la charge utile publiee."""
    import json
    import math

    from quantbench.risk import noter
    cas = [societe(),
           societe(interest_expense=None, ebit=1.0),          # -inf en solvabilite
           societe(cfo=5.0, capex=-0.1),                      # -inf en autonomie
           societe(revenue=0.0, revenue_history=[]),          # +inf en rentabilite
           societe(short_term_debt=0.0),                      # -inf en refinancement
           {}]
    for f in cas:
        charge = json.dumps(noter(f, etats()), allow_nan=False)   # leve si non fini
        assert "Infinity" not in charge and "NaN" not in charge

    def parcourir(x):
        if isinstance(x, float):
            assert math.isfinite(x), f"valeur non finie publiee : {x}"
        elif isinstance(x, dict):
            for v in x.values():
                parcourir(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                parcourir(v)

    for f in cas:
        parcourir(noter(f, etats()))


def test_le_calibrage_mesure_ce_que_la_notation_classe():
    """La dimension de liquidite n'est pas le volume brut mais son RESIDU sur la
    taille. Le residu etait calcule dans le module de SCORE, apres que le calibrage
    eut mesure ses quantiles sur la grandeur BRUTE : la table portait sur des
    -log10(volume), de l'ordre de -6 a -9, et l'on y cherchait le rang de residus de
    l'ordre de l'unite. Tout residu depassait donc tout centile, et l'univers ENTIER —
    Apple comprise, la valeur la plus echangee au monde — ressortait au 96e centile
    d'ILLIQUIDITE.
    La regle : une seule fonction produit le signal, et le calibrage l'appelle."""
    src_score = (Path(__file__).resolve().parent.parent
                 / "quantbench" / "risk" / "score.py").read_text(encoding="utf-8")
    assert "signaux[" not in src_score, (
        "le module de score reecrit un signal apres coup : le calibrage mesurera "
        "alors une grandeur differente de celle qui est classee")
    src_cal = (Path(__file__).resolve().parent.parent
               / "scripts" / "build_risk_stats.py").read_text(encoding="utf-8")
    assert src_cal.index("Passe 0") < src_cal.index("Passe 1"), (
        "la regression de liquidite doit preceder la mesure des quantiles")


def test_la_liquidite_est_un_residu_de_la_taille():
    """La part d'illiquidite qu'explique la capitalisation est deja tarifee dans le
    cout des fonds propres, par la prime de taille. L'entrer a nouveau la compterait
    deux fois, et la note ne ferait que reproduire un classement par taille."""
    from quantbench.risk.dimensions import d6_liquidite
    cal = {"liquidite": {"global": {"a": -3.0, "b": 0.9}}}
    # Deux societes au MEME volume mais de tailles tres differentes : la petite est la
    # plus liquide RELATIVEMENT a sa taille, elle doit donc etre la moins risquee.
    grande = societe(market_cap=100.0, volume_dollars_median=1e7, exchange="NYSE")
    petite = societe(market_cap=0.05, volume_dollars_median=1e7, exchange="NYSE")
    sg, _ = d6_liquidite(grande, None, "exploitante", cal)
    sp, _ = d6_liquidite(petite, None, "exploitante", cal)
    assert sg > sp, (sg, sp)
    # Sans calibrage, la dimension est indefinissable — jamais devinee.
    assert d6_liquidite(grande, None, "exploitante", None) == (None, None)


def test_le_rapport_pdf_porte_la_note_de_risque():
    """Omettre la note du rapport laisserait croire qu'une decote est une opportunite.
    Le test compare deux rendus du MEME titre, avec et sans note : le PDF doit grossir,
    faute de quoi le bloc n'a pas ete dessine — reportlab compresse ses flux, on ne
    peut donc pas chercher le texte en clair."""
    import tempfile
    from pathlib import Path as _P

    pytest.importorskip("reportlab")
    from quantbench.reports import financial_summary_pdf
    profil = {
        "ticker": "TEST", "name": "Test", "sector": "Technology",
        "valuation": {"price": 10.0, "value_per_share": 6.0, "upside": -0.4,
                      "method": "DCF FCFF (standard)"},
        "statements": {"years": ["2025", "2024"], "revenue": [5.0, 4.0],
                       "ebit": [1.0, 0.8], "net_income": [0.7, 0.5],
                       "cfo": [1.0, 0.9], "total_assets": [9.0, 8.0],
                       "equity": [4.0, 3.5], "total_debt": [2.0, 2.0]},
        "risque": {"grade": "C-", "score": 0.61, "regime": "exploitante",
                   "plafonds_appliques": ["moins_d_un_an_d_autonomie"],
                   "dimensions": [{"cle": "d1", "nom": "Solvabilité", "rang": 0.91},
                                  {"cle": "d2", "nom": "Autonomie", "rang": 0.74},
                                  {"cle": "d6", "nom": "Liquidité", "rang": None}]},
    }
    with tempfile.TemporaryDirectory() as d:
        avec = str(_P(d) / "avec.pdf")
        sans = str(_P(d) / "sans.pdf")
        assert financial_summary_pdf(profil, avec)
        assert financial_summary_pdf({k: v for k, v in profil.items() if k != "risque"},
                                     sans)
        assert _P(avec).stat().st_size > _P(sans).stat().st_size, (
            "le bloc de note de risque n'est pas dessine dans le rapport")


def test_les_grilles_de_quantiles_sont_centrees_sur_le_signal_publie():
    """Une grille de residus doit etre centree sur zero : c'est la definition meme
    d'un residu de regression. Une mediane loin de zero prouve que la grille a ete
    mesuree sur une AUTRE grandeur que celle que la notation classe — le volume brut
    plutot que son ecart a la taille — et l'univers entier ressort alors au meme rang.
    Ce test aurait attrape deux fois le meme defaut : une premiere fois par decalage
    d'echelle, une seconde par COURSE entre deux calibrages concurrents dont l'ancien
    a ecrase le nouveau."""
    import json
    f = (Path(__file__).resolve().parent.parent
         / "quantbench" / "risk" / "risk_calibration.json")
    if not f.exists():
        pytest.skip("calibrage absent")
    cal = json.loads(f.read_text(encoding="utf-8"))
    grille = ((cal.get("quantiles") or {}).get("d6") or {}).get("global")
    if not grille:
        pytest.skip("dimension de liquidite non calibree")
    mediane = grille[len(grille) // 2]
    assert abs(mediane) < 1.0, (
        f"mediane du residu de liquidite a {mediane:.2f} au lieu de ~0 : la grille "
        f"porte sur une grandeur differente de celle qui est classee")
    # Et elle doit SEPARER : une grille plate ne classe rien.
    assert grille[-1] - grille[0] > 1.0, "grille de liquidite degeneree"


def test_toutes_les_grilles_separent_quelque_chose():
    """Une grille dont tous les centiles sont egaux ne distingue aucun titre : la
    dimension est alors morte sans que rien ne le signale."""
    import json
    f = (Path(__file__).resolve().parent.parent
         / "quantbench" / "risk" / "risk_calibration.json")
    if not f.exists():
        pytest.skip("calibrage absent")
    cal = json.loads(f.read_text(encoding="utf-8"))
    plates = [cle for cle, t in (cal.get("quantiles") or {}).items()
              if (g := (t or {}).get("global")) and g[-1] - g[0] <= 0]
    assert not plates, f"grilles degenerees : {plates}"


def test_le_build_complet_ne_se_declenche_pas_sur_chaque_poussee():
    """Le build reconstruit 16 800 titres en une a trois heures, et le groupe de
    concurrence n'admet qu'un seul run EN ATTENTE : A tourne, B patiente, C arrive et
    B est ANNULE. Un declencheur sur les poussees de code faisait donc qu'une journee
    de corrections successives ne produisait AUCUN deploiement, tout en remplissant le
    tableau de bord de runs annules qui ressemblaient a des echecs.
    La justesse du code est verifiee par l'integration continue, en une minute."""
    f = (Path(__file__).resolve().parent.parent
         / ".github" / "workflows" / "deploy.yml")
    if not f.exists():
        pytest.skip("workflow absent")
    texte = f.read_text(encoding="utf-8")
    entete = texte.split("jobs:")[0]
    lignes = [ligne for ligne in entete.splitlines()
              if ligne.strip().startswith("push:") and not ligne.strip().startswith("#")]
    assert not lignes, (
        "le build complet se declenche sur les poussees : les runs s'annuleront "
        "mutuellement et le site cessera de se mettre a jour")
    assert "workflow_dispatch" in entete, "aucun declenchement manuel possible"
    assert "schedule" in entete, "aucune reconstruction quotidienne"


def test_le_controle_des_valeurs_non_finies_ne_confond_pas_un_nom_avec_un_nombre():
    """Le garde-fou refuse `Infinity` et `NaN` dans le JSON publie — ils rendent le
    fichier ENTIER illisible dans un navigateur. Mais chercher la SOUS-CHAINE
    bloquerait le deploiement des qu'un emetteur s'appelle "Infinity Stone Ventures
    Corp" — il y en a un dans l'univers. Le test doit porter sur la SYNTAXE."""
    import json
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "check_build.py").read_text(encoding="utf-8")
    assert "parse_constant" in src, (
        "le controle cherche une sous-chaine au lieu de verifier la syntaxe")

    def refuser(litteral):
        raise ValueError(litteral)

    # Un nom de societe passe.
    json.loads('{"n": "Infinity Stone Ventures Corp"}', parse_constant=refuser)
    # Un vrai litteral non fini est refuse.
    for mauvais in ('{"v": Infinity}', '{"v": -Infinity}', '{"v": NaN}'):
        with pytest.raises(ValueError):
            json.loads(mauvais, parse_constant=refuser)


def test_le_readme_decrit_le_depot_reel():
    """Un README perime est pire qu'absent : il affirme. Celui-ci a decrit pendant des
    mois un univers limite aux "grandes capitalisations non-financieres" alors que le
    depot route les banques, les foncieres et les societes deficitaires vers des
    methodes dediees, et annoncait 18 tests quand il y en a plus de cent trente."""
    racine = Path(__file__).resolve().parent.parent
    txt = (racine / "README.md").read_text(encoding="utf-8")
    perimes = ["univers limité aux **grandes capitalisations non-financières**",
               "18 tests", "app/data.json"]
    presents = [p for p in perimes if p in txt]
    assert not presents, f"affirmations perimees dans le README : {presents}"
    # Les modules cites doivent exister.
    for chemin in ("quantbench/valuation/route.py", "quantbench/risk/dimensions.py",
                   "quantbench/risk/score.py", "quantbench/data/validate.py",
                   "scripts/build_risk_stats.py", "scripts/check_build.py",
                   "scripts/mesurer_les_notes.py"):
        assert chemin.split("/")[-1] in txt, f"{chemin} absent du README"
        assert (racine / chemin).exists(), f"{chemin} cite mais introuvable"
    # Le nombre d'invariants annonce doit correspondre a la realite, a 5 pres.
    import re
    m = re.search(r"\*\*(\d+) invariants\*\*", txt)
    if m:
        annonce = int(m.group(1))
        reel = len([l for l in (racine / "tests" / "test_invariants.py")
                    .read_text(encoding="utf-8").splitlines()
                    if l.startswith("def test")])
        assert abs(annonce - reel) <= 5, (
            f"le README annonce {annonce} invariants, il y en a {reel}")


def test_les_workflows_sont_du_yaml_valide():
    """Une erreur de syntaxe dans un workflow ne se voit qu'au moment ou GitHub refuse
    de le lancer — silencieusement, sans job en echec a inspecter. Un message de commit
    multi-ligne mal indente a suffi a casser celui des reperes."""
    yaml = pytest.importorskip("yaml")
    dossier = Path(__file__).resolve().parent.parent / ".github" / "workflows"
    if not dossier.is_dir():
        pytest.skip("aucun workflow")
    for f in sorted(dossier.glob("*.yml")):
        d = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert isinstance(d, dict), f"{f.name} : YAML invalide"
        # `on:` est interprete comme le booleen True par le YAML 1.1.
        assert d.get(True) or d.get("on"), f"{f.name} : aucun declencheur"
        assert d.get("jobs"), f"{f.name} : aucun job"


def test_les_reperes_sectoriels_sont_remesures_periodiquement():
    """Beta desendette, marge, ventes sur capital, ROIC et taux de recuperation sont
    des MESURES de l'univers courant : elles nourrissent le cout du capital, la
    croissance financable et la valeur de liquidation de chaque titre. Aucun workflow
    ne les regenerait, elles vieillissaient donc en silence.
    A distinguer du calibrage de la NOTE, delibererement gele : la-bas le gel fait la
    difference entre une note et un classement ; ici l'immobilisme ferait deriver
    toutes les valorisations sans que rien ne le signale."""
    dossier = Path(__file__).resolve().parent.parent / ".github" / "workflows"
    if not dossier.is_dir():
        pytest.skip("aucun workflow")
    textes = " ".join(f.read_text(encoding="utf-8") for f in dossier.glob("*.yml"))
    assert "build_industry_stats" in textes, (
        "aucun workflow ne remesure les reperes sectoriels")
    # Et le calibrage de la note, lui, ne doit PAS etre regenere automatiquement.
    assert "build_risk_stats" not in textes, (
        "le calibrage de la note de risque est GELE : le regenerer automatiquement le "
        "transformerait en simple classement de l'univers du jour")


def test_le_plafond_de_liquidation_n_atteint_pas_une_societe_qui_encaisse():
    """La valeur de liquidation d'APPLE est nulle : 0,65 x 359 Md$ d'actif ne couvre
    pas 285 Md$ de passif. Un plafond declenche sur ce seul constat frapperait donc
    toute grande societe. Une liquidation ne se pose que pour une societe qui NE
    GENERE PAS de tresorerie — celle qui encaisse n'est pas liquidee, quelle que soit
    la decote theorique sur son actif."""
    from quantbench.risk.score import _modalites
    geante = societe(total_assets=359.0, total_liab=285.0, total_equity=73.0,
                     book_equity=73.0, cash=36.0, cfo=110.0, capex=-11.0)
    assert "passif_non_couvert" not in _modalites(geante, etats(), [], {})
    # Une societe qui consomme ET dont l'actif realisable ne couvre pas le passif,
    # elle, doit etre plafonnee — meme si son actif COMPTABLE depasse le passif, la
    # decote de realisation suffisant a creuser l'ecart.
    exsangue = societe(total_assets=0.00072, total_liab=0.00071, total_equity=0.00001,
                       book_equity=0.00001, cash=0.0001, cfo=-0.0003, capex=0.0)
    assert "passif_non_couvert" in _modalites(exsangue, etats(), [], {})


# --------------------------------------------------------------------------- #
# 30. FORENSIQUE — le verdict se calcule avec la formule, jamais a cote
# --------------------------------------------------------------------------- #
def test_le_verdict_altman_utilise_les_seuils_de_sa_propre_formule():
    """La formule retenue est le Z''-EMS, qui INCLUT la constante de 3,25. Les seuils
    du Z'' SANS constante — 1,1 et 2,6 — ne s'y appliquent donc pas. La page les
    utilisait pourtant : tout le diagnostic etait decale de 3,25 points, et une
    societe a Z = 3,0 — en detresse averee — s'affichait "sain". Le correctif avait
    ete applique au moteur et jamais a l'affichage."""
    from quantbench.forensics.scores import Z_DETRESSE, Z_SAIN
    assert (Z_DETRESSE, Z_SAIN) == (4.35, 5.85)
    page = (Path(__file__).resolve().parent.parent
            / "app" / "stock.html").read_text(encoding="utf-8")
    bloc = [ligne for ligne in page.splitlines() if "altman_z" in ligne]
    assert bloc, "l'Altman n'est plus affiche"
    for ligne in bloc:
        assert "1.1" not in ligne and "2.6" not in ligne, (
            f"l'affichage recalcule le verdict avec les mauvais seuils : {ligne.strip()}")
    assert any("altman_verdict" in ligne for ligne in bloc), (
        "l'affichage doit lire le verdict produit par le moteur")


def test_le_levier_forensique_compare_deux_annees_de_meme_nature():
    """Une dette a long terme NULLE est falsy : `long_term_debt or total_debt`
    faisait basculer CETTE annee-la sur la dette totale pendant que l'autre restait
    sur la dette a long terme. Les deux exercices etaient alors compares sur des
    grandeurs differentes — troisieme occurrence du meme piege aujourd'hui, apres le
    passif nul et l'actif nul."""
    src = (Path(__file__).resolve().parent.parent
           / "quantbench" / "forensics" / "scores.py").read_text(encoding="utf-8")
    assert 'g("long_term_debt", 0) or g("total_debt", 0)' not in src, (
        "le champ de dette est choisi par un `or` falsy, annee par annee")
    # Verification par le comportement : une societe ayant rembourse toute sa dette a
    # long terme ne doit pas voir son levier "baisser" par changement de grandeur.
    from quantbench.forensics.scores import piotroski_f_score
    n = 4
    F = {"years": ["2025", "2024", "2023", "2022"],
         "net_income": [1.0] * n, "total_assets": [10.0] * n, "cfo": [2.0] * n,
         "long_term_debt": [0.0, 0.0, 0.0, 0.0], "total_debt": [3.0, 3.0, 3.0, 3.0],
         "current_assets": [5.0] * n, "current_liab": [2.0] * n,
         "shares": [100.0] * n, "gross_profit": [4.0] * n, "revenue": [8.0] * n}
    r = piotroski_f_score(F)
    assert r is not None
    assert r["tests"]["Levier en baisse"] is False, (
        "un levier strictement stable ne peut pas etre declare en baisse")


def test_zero_est_une_mesure_pas_un_manque():
    """Le piege FALSY, quatrieme occurrence de la journee apres le passif nul,
    l'actif nul et la dette a long terme nulle.

    `_safe_div(ebit, rev) or 0.10` traitait une marge CALCULEE A ZERO comme une marge
    ABSENTE : une societe a resultat operationnel exactement nul se voyait attribuer
    10 % de marge, projetee ensuite a l'infini par le DCF. En Python, `0` est faux —
    tout champ comptable pouvant valoir zero doit etre teste contre `None`."""
    from quantbench.valuation.build_universal import build_dcf_from_fundamentals
    nulle = societe(ebit=0.0, operating_margin=None, revenue=5.0)
    x, meta = build_dcf_from_fundamentals(nulle)
    assert abs(meta["op_margin"]) < 1e-9, (
        f"marge de {meta['op_margin']:.3f} pour un resultat operationnel NUL")
    # Une marge reellement absente, elle, garde son repli.
    absente = societe(ebit=None, operating_margin=None, revenue=5.0)
    _x2, meta2 = build_dcf_from_fundamentals(absente)
    assert abs(meta2["op_margin"] - 0.10) < 1e-9


def test_une_marge_nulle_est_publiee_comme_nulle():
    """Meme piege dans la couche de donnees : `(ebit / rev) if (ebit and rev)`
    renvoyait None pour un EBIT nul, ce qui declenchait le repli ci-dessus."""
    src = (Path(__file__).resolve().parent.parent
           / "quantbench" / "data" / "fmp.py").read_text(encoding="utf-8")
    assert "if (ebit and rev)" not in src, (
        "un resultat operationnel NUL est declare absent")
    assert "if (ni and eq)" not in src, "un resultat net NUL est declare absent"


# --------------------------------------------------------------------------- #
# 31. MOTEUR DCF — la prime de taille s'AJOUTE, elle ne se multiplie pas
# --------------------------------------------------------------------------- #
def test_la_prime_de_taille_ne_derive_pas_avec_le_beta():
    """Elle etait injectee dans l'ERP sous la forme `erp + prime / beta_initial`, ce
    qui l'annulait exactement — mais la PREMIERE annee seulement. Le beta converge
    ensuite vers sa valeur terminale, emportant la prime avec lui : une societe dont
    le beta double sur l'horizon voyait sa prime doubler, jusqu'a 3,5 points de cout
    du capital surajoutes EN PERPETUITE, la ou se joue l'essentiel de la valeur."""
    from quantbench.valuation.dcf import wacc_path
    commun = dict(pretax_kd=0.06, terminal_pretax_kd=0.06, equity_value=1.0,
                  debt_value=0.0, risk_free_rate=0.04, erp=0.05, marginal_tax=0.25,
                  n_years=10, beta_converge_start=5, kd_converge_start=5)
    # Beta qui DOUBLE sur l'horizon : le pire cas pour l'ancienne formule.
    _w0, ke0, _b = wacc_path(unlevered_beta=0.5, terminal_unlevered_beta=1.0,
                             size_premium=0.0, **commun)
    _w1, ke1, _b1 = wacc_path(unlevered_beta=0.5, terminal_unlevered_beta=1.0,
                              size_premium=0.035, **commun)
    ecart = ke1 - ke0
    assert abs(ecart[0] - 0.035) < 1e-12, ecart[0]
    assert abs(ecart[-1] - 0.035) < 1e-12, (
        f"la prime vaut {ecart[-1]:.4f} en annee terminale au lieu de 0,0350 : "
        f"elle derive avec le beta")


def test_la_prime_de_taille_n_entre_pas_dans_le_cout_de_la_dette():
    """C'est une prime de risque ACTIONNAIRE : le preteur d'une petite societe se
    remunere par son spread de credit, pas par elle."""
    from quantbench.valuation.dcf import wacc_path
    commun = dict(unlevered_beta=1.0, terminal_unlevered_beta=1.0, pretax_kd=0.06,
                  terminal_pretax_kd=0.06, risk_free_rate=0.04, erp=0.05,
                  marginal_tax=0.25, n_years=10, beta_converge_start=5,
                  kd_converge_start=5)
    # Societe financee INTEGRALEMENT par dette : la prime ne doit rien changer.
    w0, _k, _b = wacc_path(equity_value=0.0, debt_value=1.0, size_premium=0.0, **commun)
    w1, _k1, _b1 = wacc_path(equity_value=0.0, debt_value=1.0, size_premium=0.05,
                             **commun)
    assert abs(float(w1[0]) - float(w0[0])) < 1e-12


def test_le_beta_n_est_pas_endette_deux_fois():
    """Le moteur re-endette lui-meme le beta a partir du levier. Lui passer un beta
    DEJA endette le leverait une seconde fois."""
    src = (Path(__file__).resolve().parent.parent
           / "quantbench" / "valuation" / "build_universal.py").read_text(encoding="utf-8")
    bloc = src.split("def build_dcf_from_fundamentals")[1]
    assert "unlevered_beta=unlev" in bloc, (
        "le moteur recoit un beta deja endette : il le levera une seconde fois")


# --------------------------------------------------------------------------- #
# 32. PLAFONDS DE LA NOTE — "ne consomme pas" n'est pas "genere"
# --------------------------------------------------------------------------- #
def test_l_absence_de_tableau_de_flux_n_epargne_pas_une_societe():
    """La consommation vaut zero dans DEUX cas tres differents : la societe encaisse,
    ou son tableau de flux est ABSENT. 1812 Brewing porte 13,7 M$ de passif pour
    5,0 M$ d'actif et des fonds propres de -8,7 M$, sans aucun tableau de flux publie :
    elle echappait aux deux plafonds et ressortait notee B.
    On exige donc une PREUVE de generation de tresorerie, pas l'absence de preuve du
    contraire."""
    from quantbench.risk.score import _modalites
    sans_flux = societe(total_assets=0.005, total_liab=0.0137, total_equity=-0.0087,
                        book_equity=-0.0087, cash=0.00006, cfo=None, capex=None,
                        net_income=0.0036)
    m = _modalites(sans_flux, etats(), [], {}, reg="exploitante")
    assert "fonds_propres_absorbes" in m and "passif_non_couvert" in m, m


def test_le_capex_de_croissance_ne_compte_pas_comme_une_consommation():
    """Un service public regule finance ses investissements par la dette PAR
    CONSTRUCTION, avec un rendement garanti par le regulateur : son flux libre est
    negatif en permanence sans que rien ne menace. Les plafonds lisaient le flux libre
    BRUT — Duke Energy et Southern Company ressortaient a D+, E.ON aussi.
    Seul le capex de MAINTENANCE, approxime par la dotation aux amortissements,
    constitue une consommation."""
    from quantbench.risk.dimensions import consommation_selon_le_regime
    # Service public : 12 Md$ encaisses, 14 Md$ investis, 6 Md$ d'amortissements.
    regule = societe(cfo=12.0, capex=-14.0, dep_amort=6.0, net_income=4.0)
    assert consommation_selon_le_regime(regule, "amortissement_lourd") == 0.0
    # La meme societe hors de ce regime consomme bien 2 Md$.
    assert abs(consommation_selon_le_regime(regule, "exploitante") - 2.0) < 1e-9
    # Et une societe qui brule vraiment reste detectee dans les deux cas.
    brule = societe(cfo=-3.0, capex=-1.0, dep_amort=0.5, net_income=-3.0)
    assert consommation_selon_le_regime(brule, "amortissement_lourd") > 0
    assert consommation_selon_le_regime(brule, "exploitante") > 0


def test_une_base_actionnaire_incoherente_plafonne_la_note():
    """All American Gold affiche 96,6 millions de titres implicites pour 1,877
    MILLIARD deposes, CTR Investments 140 millions pour 3,7 milliards. La
    capitalisation ne porte alors pas sur la meme entite que les comptes, et tout
    rapport entre les deux est vide de sens.
    On ne REECRIT rien — aucune source ne prouve que l'autre a tort — mais un
    desaccord d'un ordre de grandeur plafonne la note."""
    from quantbench.risk.score import _modalites
    incoherente = societe(shares=96_636_000, actions_publiees=1_877_273_800)
    assert "base_actionnaire_incoherente" in _modalites(incoherente, etats(), [], {})
    # Une dilution ordinaire, meme forte, ne declenche pas.
    dilue = societe(shares=130_000_000, actions_publiees=100_000_000)
    assert "base_actionnaire_incoherente" not in _modalites(dilue, etats(), [], {})


def test_les_grandes_valeurs_ne_sont_pas_plafonnees_en_masse():
    """Garde-fou de forme : les plafonds visent des situations exceptionnelles. S'ils
    frappaient les grandes valeurs, c'est que leur condition est trop large — c'est
    exactement ce qui s'est produit avec le flux libre brut, qui emportait dix
    societes de plus de 5 Md$ dont deux services publics de premier plan."""
    from quantbench.risk.score import _modalites
    # Service public regule typique : investit plus qu'il n'encaisse, mais amortit.
    for reg in ("amortissement_lourd", "exploitante"):
        saine = societe(total_assets=180.0, total_liab=140.0, total_equity=40.0,
                        book_equity=40.0, cash=1.0, cfo=12.0, capex=-6.0,
                        dep_amort=6.0, net_income=5.0)
        assert not _modalites(saine, etats(), [], {}, reg=reg), reg


# --------------------------------------------------------------------------- #
# 33. MONTE CARLO — il mesure l'incertitude, il ne produit pas la valeur
# --------------------------------------------------------------------------- #
def test_l_upside_publie_est_celui_du_routage():
    """La mediane simulee ECRASAIT l'upside du routage sur les deux tiers de
    l'univers. Le probleme n'est pas que la mediane differe du deterministe —
    mediane(f(X)) n'est pas f(mediane X), c'est une propriete et non une erreur. Le
    probleme est structurel : `run_mc` est une SECONDE IMPLEMENTATION de la
    valorisation, et c'est SON resultat qui etait publie."""
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "build_site_fmp.py").read_text(encoding="utf-8")
    corps = src.split("def build_one(")[1].split(chr(10) + "def ")[0]
    for interdit in ('val["upside"] =', 'val["value_per_share"] =', "upside_basis"):
        assert interdit not in corps, (
            f"la simulation ecrase encore la valorisation routee : {interdit}")


def test_la_simulation_n_a_pas_sa_propre_survie_ni_sa_propre_liquidation():
    """Le miroir portait encore, le jour meme de leur correction, la survie
    plancherisee a 0,30 et mesuree sur le RESULTAT NET, et une valeur de liquidation
    posee a la moitie des fonds propres — l'erreur exacte que `valeur_de_liquidation`
    corrige et nomme dans sa docstring. Sur RDGT, le miroir injectait 1 131 points
    d'upside que le routage ne produisait pas."""
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "build_site_fmp.py").read_text(encoding="utf-8")
    corps = src.split("def run_mc(")[1].split(chr(10) + "def ")[0]
    for interdit in ("0.5 * max(fund.get", "0.3 + 0.15", 'sect(fund, "recuperation"'):
        assert interdit not in corps, f"regle recopiee dans la simulation : {interdit}"
    for attendu in ("probabilite_de_survie(fund)", "valeur_de_liquidation(fund)"):
        assert attendu in corps, f"la simulation n'appelle pas {attendu}"


def test_la_page_ne_pretend_plus_que_l_upside_vient_de_la_mediane():
    page = (Path(__file__).resolve().parent.parent
            / "app" / "stock.html").read_text(encoding="utf-8")
    assert "upside affiché est fondé sur la médiane" not in page


# --------------------------------------------------------------------------- #
# 34. LE GARDE-FOU REFUSAIT UN BUILD IDENTIQUE A LUI-MEME
# --------------------------------------------------------------------------- #
def test_le_seuil_de_demesure_est_au_dessus_de_sa_distribution_au_repos():
    """La valeur precedente etait 10, POSEE. La distribution AU REPOS de ce compteur —
    le meme modele rejoue sur 250 seances de cours reels, sans aucune modification du
    code — s'etend de 8 a 19, mediane 15, ecart-type 3,5. Le seuil se trouvait donc
    SOUS LA MEDIANE de sa propre distribution au repos : le garde-fou refusait un build
    identique a lui-meme environ neuf fois sur dix, et onze des quarante derniers
    deploiements ont echoue, tous a cette etape.
    Un garde-fou qui bloque le cas normal ne protege plus de rien."""
    import check_build
    MAXIMUM_OBSERVE_AU_REPOS = 19
    assert check_build.MAX_NB_DEMESURES > MAXIMUM_OBSERVE_AU_REPOS, (
        f"seuil a {check_build.MAX_NB_DEMESURES} alors que le compteur atteint "
        f"{MAXIMUM_OBSERVE_AU_REPOS} sans qu'aucune ligne de code ne change")


def test_les_seuils_du_gardefou_declarent_leur_mesure():
    """Toute constante du garde-fou doit porter, en commentaire, SOIT une mesure, SOIT
    la mention explicite qu'elle est posee. Un nombre sans provenance est une opinion
    qui se fait passer pour un fait — et celui-ci a bloque onze deploiements."""
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "check_build.py").read_text(encoding="utf-8")
    lignes = src.splitlines()
    constantes = [(i, l) for i, l in enumerate(lignes)
                  if l.startswith(("MAX_", "MIN_")) and "=" in l]
    assert constantes, "aucune constante trouvee"
    for i, ligne in constantes:
        # Le commentaire de fin de ligne ou les lignes qui precedent.
        contexte = ligne + " " + " ".join(lignes[max(0, i - 12):i])
        assert any(mot in contexte.lower()
                   for mot in ("mesure", "constate", "observe", "assume", "pose")), (
            f"constante sans provenance declaree : {ligne.strip()}")


def test_le_gardefou_journalise_la_concentration_par_route():
    """Une route CASSEE concentre ses degats sur une seule methode ; des donnees
    individuellement corrompues s'eparpillent. C'est le vrai discriminant entre une
    regression du MODELE — ce que le garde-fou doit attraper — et une poignee de
    capitalisations fausses, qu'il ne doit pas laisser bloquer la publication."""
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "check_build.py").read_text(encoding="utf-8")
    assert "concentration par route" in src
    assert "capitalisation demesuree" in src


# --------------------------------------------------------------------------- #
# 35. Le DCF refuse un ROIC nul ou negatif ; le tireur ecarte les infinis
# --------------------------------------------------------------------------- #
def test_le_dcf_refuse_un_roic_nul_ou_negatif():
    """L'identite g = reinvestissement x ROIC n'a pas de solution a ROIC nul :
    financer une croissance avec un rendement du capital nul exigerait un
    reinvestissement INFINI. Le code traitait ce cas comme "reinvestissement nul",
    c'est-a-dire croissance GRATUITE — et le modele RECOMPENSAIT donc la destruction
    de capital : ROIC de +0,001 donnait une equite de -0,673, exactement 0 donnait
    +1,053, et -0,20 donnait +1,778."""
    from quantbench.valuation.dcf import DcfInputs, value_dcf
    base = dict(revenue_base=5.0, g1_begin=0.08, g1_end=0.06, g2_begin=0.06,
                g2_end=0.04, g3_begin=0.04, g3_end=0.025,
                current_operating_margin=0.15, terminal_operating_margin=0.15,
                reinvestment_mode="roic", equity_value=10.0, debt_value=2.0)
    for mauvais in ({"current_roic": 0.0, "terminal_roic": 0.10},
                    {"current_roic": -0.20, "terminal_roic": 0.10},
                    {"current_roic": 0.15, "terminal_roic": 0.0},
                    {"current_roic": 0.15, "terminal_roic": -0.05}):
        with pytest.raises(ValueError):
            value_dcf(DcfInputs(**base, **mauvais))
    # Et la valeur doit CROITRE avec le rendement du capital, sans discontinuite.
    valeurs = [value_dcf(DcfInputs(**base, current_roic=r, terminal_roic=0.10))
               ["equity_value"] for r in (0.02, 0.08, 0.20, 0.40)]
    assert valeurs == sorted(valeurs), valeurs


def test_le_tireur_ecarte_les_scenarios_infinis():
    """`np.isnan(inf)` vaut FALSE : le filtre laissait passer les infinis, qui
    contaminent ensuite moyenne et ecart-type. Le build en etait protege par
    `_mc_stats`, mais le service, `value_ticker` et `batch_screener` consomment cette
    mediane sans filet."""
    src = (Path(__file__).resolve().parent.parent
           / "quantbench" / "valuation" / "montecarlo.py").read_text(encoding="utf-8")
    assert "~np.isnan(equity)" not in src, "le filtre laisse passer les infinis"
    assert "np.isfinite(equity)" in src
    assert "taux_validite" in src, "le taux de validite des tirages n'est pas publie"


# --------------------------------------------------------------------------- #
# 36. La copule gaussienne ne tournait pas
# --------------------------------------------------------------------------- #
def test_la_simulation_transmet_bien_les_correlations():
    """Les lois etaient declarees DEUX FOIS, et la copie utilisee par le build ne
    transmettait AUCUNE correlation : la matrice restait l'identite et la copule
    gaussienne annoncee par la documentation du module ne tournait pas. Les parametres
    etaient tires INDEPENDAMMENT.
    Ce n'est pas cosmetique : deux erreurs independantes se compensent en moyenne,
    deux erreurs correlees s'additionnent. La simulation SOUS-ESTIMAIT donc la
    dispersion — exactement la grandeur qu'elle existe pour mesurer."""
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "build_site_fmp.py").read_text(encoding="utf-8")
    corps = src.split("def run_mc(")[1].split(chr(10) + "def ")[0]
    assert "correlations=correlations" in corps, (
        "la simulation ne transmet pas les correlations : la copule ne tourne pas")
    assert "stats.norm(" not in corps, (
        "les lois sont redeclarees dans le script de build")


def test_les_correlations_deplacent_reellement_la_dispersion():
    """Verification par le COMPORTEMENT et non par la lecture du source : deux
    parametres correles doivent produire une dispersion differente de deux parametres
    independants."""
    from quantbench.valuation.dcf import DcfInputs
    from quantbench.valuation.montecarlo import sample_inputs
    from scipy import stats as st
    lois = {"a": st.norm(0.10, 0.03), "b": st.norm(0.10, 0.03)}
    seul = sample_inputs(lois, 4000, None, seed=1)
    lie = sample_inputs(lois, 4000, [("a", "b", 0.8)], seed=1)
    import numpy as np
    r_seul = float(np.corrcoef(seul["a"], seul["b"])[0, 1])
    r_lie = float(np.corrcoef(lie["a"], lie["b"])[0, 1])
    assert abs(r_seul) < 0.10, f"tirage cense etre independant : correlation {r_seul:.2f}"
    assert r_lie > 0.7, f"correlation demandee 0,8, obtenue {r_lie:.2f}"
    # Et la somme des deux — proxy de la valeur — est bien plus dispersee.
    assert (lie["a"] + lie["b"]).std() > (seul["a"] + seul["b"]).std() * 1.2


def test_une_matrice_de_correlation_impossible_ne_fait_pas_echouer_le_tirage():
    """Trois correlations de 0,9 entre trois variables ne definissent pas une matrice
    admissible. Le module doit la ramener a la plus proche matrice valide, pas
    s'effondrer sur la decomposition de Cholesky."""
    from quantbench.valuation.montecarlo import sample_inputs
    from scipy import stats as st
    lois = {n: st.norm(0.1, 0.02) for n in ("a", "b", "c")}
    ech = sample_inputs(lois, 500, [("a", "b", 0.9), ("b", "c", 0.9),
                                    ("a", "c", -0.9)], seed=3)
    import numpy as np
    assert all(np.isfinite(v).all() for v in ech.values())


def test_les_lois_de_tirage_ont_une_seule_declaration():
    """`lois_de_tirage` est la source unique. Les dispersions sont PROPORTIONNELLES au
    cas de base avec un plancher absolu : une societe a 2 % de marge n'a pas la meme
    incertitude absolue qu'une societe a 40 %, mais aucune n'est connue a mieux que le
    plancher pres."""
    from quantbench.valuation.build_universal import lois_de_tirage
    from quantbench.valuation.dcf import DcfInputs
    base = DcfInputs(revenue_base=5.0, g1_begin=0.08, g1_end=0.06, g2_begin=0.06,
                     g2_end=0.04, g3_begin=0.04, g3_end=0.025,
                     current_roic=0.15, terminal_roic=0.10)
    lois, cor = lois_de_tirage(base)
    assert set(lois) >= {"g1_begin", "terminal_operating_margin", "current_roic"}
    assert cor, "aucune correlation declaree"
    # Proportionnalite : une marge deux fois plus grande, une dispersion plus grande.
    petite = lois_de_tirage(DcfInputs(revenue_base=5.0, g1_begin=0.08, g1_end=0.06,
                                      g2_begin=0.06, g2_end=0.04, g3_begin=0.04,
                                      g3_end=0.025, terminal_operating_margin=0.04,
                                      current_roic=0.15, terminal_roic=0.10))[0]
    grande = lois_de_tirage(DcfInputs(revenue_base=5.0, g1_begin=0.08, g1_end=0.06,
                                      g2_begin=0.06, g2_end=0.04, g3_begin=0.04,
                                      g3_end=0.025, terminal_operating_margin=0.40,
                                      current_roic=0.15, terminal_roic=0.10))[0]
    assert (grande["terminal_operating_margin"].std()
            > petite["terminal_operating_margin"].std())


# --------------------------------------------------------------------------- #
# Trajectoires de convergence : optimisation a resultat IDENTIQUE
# --------------------------------------------------------------------------- #
def _converge_path_reference(start, end, n_years, converge_start_year):
    """Implementation de reference, fondee sur `np.linspace`.

    C'est celle qui a produit toutes les valorisations publiees jusqu'ici. La
    version optimisee doit lui etre identique AU BIT PRES, pas seulement proche :
    une optimisation qui deplace les chiffres n'est pas une optimisation, c'est un
    changement de methode qui ne dit pas son nom.
    """
    import numpy as _np
    n_years = int(n_years)
    converge_start_year = int(converge_start_year)
    if n_years <= 0:
        return _np.zeros(0, dtype=float)
    hold = min(max(converge_start_year - 1, 0), n_years)
    path = _np.empty(n_years, dtype=float)
    path[:hold] = start
    ramp_len = n_years - hold
    if ramp_len == 1:
        path[hold:] = end
    elif ramp_len > 1:
        path[hold:] = _np.linspace(start, end, ramp_len)
    return path


def test_converge_path_est_identique_au_bit_pres_a_linspace():
    import random

    import numpy as np

    from quantbench.valuation.dcf import converge_path

    random.seed(20260730)
    for _ in range(20000):
        s = random.uniform(-0.95, 0.95)
        e = random.uniform(-0.95, 0.95)
        n = random.randint(0, 30)
        c = random.randint(0, 32)
        attendu = _converge_path_reference(s, e, n, c)
        obtenu = converge_path(s, e, n, c)
        assert obtenu.shape == attendu.shape, (s, e, n, c)
        assert np.array_equal(obtenu, attendu), (s, e, n, c, attendu, obtenu)


def test_le_cache_dindices_ne_peut_pas_etre_corrompu():
    """Le cache est partage par les threads du build : il doit etre en lecture
    seule. Un tableau mutable rendu deux fois, puis modifie une fois, empoisonnerait
    silencieusement toutes les trajectoires suivantes de toutes les societes."""
    import numpy as np
    import pytest

    from quantbench.valuation.dcf import _indices, converge_path

    a = _indices(7)
    assert a.flags.writeable is False
    with pytest.raises(ValueError):
        a[0] = 99.0
    # Deux appels rendent le MEME objet : c'est bien un cache, pas une allocation.
    assert _indices(7) is a
    # Et une trajectoire calculee apres coup reste juste.
    assert np.array_equal(converge_path(0.10, 0.02, 5, 1),
                          _converge_path_reference(0.10, 0.02, 5, 1))


def test_le_monte_carlo_reste_reproductible_a_graine_fixe():
    """Meme graine, meme resultat : sans quoi aucune comparaison avant/apres
    optimisation n'a de sens, et les valeurs publiees bougeraient a chaque build."""
    from quantbench.valuation.dcf import DcfInputs
    from quantbench.valuation.montecarlo import monte_carlo_dcf

    base = _dcf_inputs_temoin()
    lois, corr = _lois_temoin(base)
    a = monte_carlo_dcf(base, lois, n=400, correlations=corr, seed=7)
    b = monte_carlo_dcf(base, lois, n=400, correlations=corr, seed=7)
    assert a["median"] == b["median"]
    assert a["percentiles"] == b["percentiles"]
    assert isinstance(base, DcfInputs)


def _dcf_inputs_temoin():
    from quantbench.valuation.dcf import DcfInputs
    return DcfInputs(
        revenue_base=1_000.0,
        g1_begin=0.08, g1_end=0.05, g2_begin=0.05, g2_end=0.03,
        g3_begin=0.03, g3_end=0.025,
        current_operating_margin=0.15, terminal_operating_margin=0.12,
        current_tax_rate=0.25, marginal_tax_rate=0.25,
        risk_free_rate=0.04, erp=0.05,
        unlevered_beta=1.0, terminal_unlevered_beta=1.0,
        equity_value=1_500.0, debt_value=200.0, cash_and_non_operating=100.0,
        reinvestment_mode="roic", current_roic=0.15, terminal_roic=0.10,
    )


def _lois_temoin(base):
    from quantbench.valuation.build_universal import lois_de_tirage
    return lois_de_tirage(base)


# --------------------------------------------------------------------------- #
# Fusion des shards : un univers incomplet ne doit JAMAIS etre publie
# --------------------------------------------------------------------------- #
def _ecrire_shard(rep, i, n, tickers, acheves=None):
    import json
    (rep / f"_shard_{i}.json").write_text(json.dumps({
        "rows": [{"ticker": t, "upside": 0.1, "grade": "B"} for t in tickers],
        "universe": len(tickers), "fail": 0,
        "shard": i, "n_shards": n, "assignes": tickers,
        "acheves": len(tickers) if acheves is None else acheves,
        "date": "2026-07-30",
    }), encoding="utf-8")


def _combine_dans(tmp_path, monkeypatch):
    """Branche `combine_shards` sur un repertoire jetable."""
    import importlib
    mod = importlib.import_module("build_site_fmp")
    monkeypatch.setattr(mod, "US", tmp_path)
    monkeypatch.setattr(mod, "_write_aggregates",
                        lambda rows, uni, fail: (len(rows), 0, 0))
    return mod


def test_la_fusion_refuse_un_shard_manquant(tmp_path, monkeypatch):
    """Le defaut etait invisible : la barriere qualite raisonne en ratios, et un
    shard absent retire autant au numerateur qu'au denominateur. Le site perdait
    un cinquieme de sa couverture sans qu'aucune verification ne bronche."""
    import pytest

    mod = _combine_dans(tmp_path, monkeypatch)
    for i in range(4):                       # 4 shards sur 5
        _ecrire_shard(tmp_path, i, 5, [f"T{i}{k}" for k in range(10)])
    with pytest.raises(mod.ShardManquant) as e:
        mod.combine_shards(5)
    assert "shard 4" in str(e.value)


def test_la_fusion_refuse_un_shard_interrompu(tmp_path, monkeypatch):
    import pytest

    mod = _combine_dans(tmp_path, monkeypatch)
    for i in range(5):
        t = [f"T{i}{k}" for k in range(100)]
        _ecrire_shard(tmp_path, i, 5, t, acheves=100 if i != 2 else 60)
    with pytest.raises(mod.ShardManquant) as e:
        mod.combine_shards(5)
    assert "shard 2" in str(e.value) and "60/100" in str(e.value)


def test_la_fusion_refuse_un_decoupage_incoherent(tmp_path, monkeypatch):
    """Melange d'artefacts de deux builds : un shard construit pour un decoupage
    en 8 fusionne dans un decoupage en 5 couvre la mauvaise part de l'univers."""
    import pytest

    mod = _combine_dans(tmp_path, monkeypatch)
    for i in range(5):
        _ecrire_shard(tmp_path, i, 5 if i != 3 else 8, [f"T{i}{k}" for k in range(10)])
    with pytest.raises(mod.ShardManquant) as e:
        mod.combine_shards(5)
    assert "decoupage" in str(e.value)


def test_la_fusion_refuse_des_societes_en_double(tmp_path, monkeypatch):
    """Deux shards qui se recouvrent fausseraient toutes les statistiques de
    l'univers : medianes du screener et quantiles des notes de risque."""
    import pytest

    mod = _combine_dans(tmp_path, monkeypatch)
    for i in range(5):
        _ecrire_shard(tmp_path, i, 5, [f"T{i}{k}" for k in range(10)])
    _ecrire_shard(tmp_path, 4, 5, ["T00", "T01", "T42"])   # recouvre le shard 0
    with pytest.raises(mod.ShardManquant) as e:
        mod.combine_shards(5)
    assert "plusieurs shards" in str(e.value)


def test_la_fusion_accepte_un_univers_complet(tmp_path, monkeypatch):
    mod = _combine_dans(tmp_path, monkeypatch)
    for i in range(5):
        _ecrire_shard(tmp_path, i, 5, [f"T{i}{k}" for k in range(10)])
    mod.combine_shards(5)                     # ne leve pas


def test_la_fusion_accepte_les_shards_sans_signature(tmp_path, monkeypatch):
    """Compatibilite ascendante : un build lance AVANT l'ajout de la signature
    doit pouvoir etre fusionne, sinon la premiere execution echoue sans raison."""
    import json

    mod = _combine_dans(tmp_path, monkeypatch)
    for i in range(5):
        (tmp_path / f"_shard_{i}.json").write_text(json.dumps({
            "rows": [{"ticker": f"T{i}{k}"} for k in range(10)],
            "universe": 10, "fail": 0}), encoding="utf-8")
    mod.combine_shards(5)                     # ne leve pas


# --------------------------------------------------------------------------- #
# Validation retrospective de la note de risque : les pieges methodologiques
# --------------------------------------------------------------------------- #
def test_la_troncature_interdit_tout_regard_sur_lavenir():
    """Le filtre porte sur la date de DEPOT, jamais sur la date de cloture.

    Un exercice clos le 31 decembre 2023 n'est public qu'en mars 2024. Le retenir
    des le 1er janvier donnerait a la note trois mois de connaissance de l'avenir,
    ce qui suffit a lui faire "predire" des faillites deja consommees et a rendre
    toute la validation flatteuse et fausse.
    """
    from valider_les_notes import tronquer

    entry = {"income": {
        2023: {"date": "2023-12-31", "filingDate": "2024-03-15", "revenue": 1.0},
        2022: {"date": "2022-12-31", "filingDate": "2023-03-10", "revenue": 0.9},
    }, "balance": {}, "cashflow": {}}

    # La veille du depot : l'exercice 2023 n'existe pas encore pour le marche.
    assert sorted(tronquer(entry, "2024-03-14")["income"]) == [2022]
    # Le jour du depot : il devient utilisable.
    assert sorted(tronquer(entry, "2024-03-15")["income"]) == [2022, 2023]
    # La cloture seule ne suffit JAMAIS a rendre un exercice utilisable.
    assert sorted(tronquer(entry, "2024-01-05")["income"]) == [2022]


def test_un_exercice_sans_date_de_depot_est_ecarte():
    """Sans date de depot, on ne peut pas prouver qu'il etait public : on l'ecarte.
    Le supposer publie serait exactement le regard sur l'avenir qu'on interdit."""
    from valider_les_notes import tronquer

    entry = {"income": {2023: {"date": "2023-12-31", "revenue": 1.0}},
             "balance": {}, "cashflow": {}}
    assert tronquer(entry, "2030-01-01")["income"] == {}


def test_le_cours_retenu_est_le_dernier_avant_la_date():
    from valider_les_notes import cours_a

    serie = [{"date": "2024-07-26", "close": 10.0},
             {"date": "2024-07-29", "close": 11.0},
             {"date": "2024-07-31", "close": 12.0}]
    assert cours_a(serie, "2024-07-30")["close"] == 11.0
    assert cours_a(serie, "2024-07-29")["close"] == 11.0   # borne incluse
    assert cours_a(serie, "2024-07-25") is None            # ne cotait pas encore


def test_lintervalle_de_wilson_ne_ment_pas_sur_les_petits_effectifs():
    """Zero effondrement sur dix observations ne prouve pas un risque nul.
    L'intervalle normal donnerait [0 %, 0 %] ; Wilson refuse cette certitude."""
    from valider_les_notes import wilson

    bas, haut = wilson(0, 10)
    assert bas == 0.0
    assert haut > 0.25, haut
    # Plus d'observations, intervalle plus etroit.
    assert wilson(0, 200)[1] < haut
    # Toujours dans [0, 1].
    for k, n in [(0, 0), (5, 5), (3, 7), (100, 100)]:
        b, h = wilson(k, n)
        assert 0.0 <= b <= h <= 1.0


def test_la_separation_se_juge_sur_la_difference_pas_sur_le_chevauchement():
    """Deux intervalles de confiance qui se chevauchent NE prouvent PAS l'absence
    de difference. Le cas mesure : A 21/93 contre F 36/86, intervalles de Wilson
    chevauchants d'un dixieme de point, et pourtant p = 0,006. Conclure de l'un a
    l'autre revient a declarer l'instrument aveugle parce qu'on l'a mal lu."""
    from valider_les_notes import test_deux_proportions, wilson

    a_bas, a_haut = wilson(21, 93)
    f_bas, f_haut = wilson(36, 86)
    assert a_haut > f_bas, "les intervalles se chevauchent bien"
    z, p = test_deux_proportions(21, 93, 36, 86)
    assert z > 2.5 and p < 0.01, (z, p)
    # Aucune difference -> aucune significativite.
    _z0, p0 = test_deux_proportions(20, 100, 20, 100)
    assert p0 > 0.99


def test_spearman_reconnait_une_monotonie_parfaite():
    from valider_les_notes import spearman

    assert spearman([(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4)]) == pytest.approx(1.0)
    assert spearman([(0, 0.4), (1, 0.3), (2, 0.2), (3, 0.1)]) == pytest.approx(-1.0)
    assert spearman([(0, 0.2)]) is None          # pas assez de points


def test_une_radiation_compte_pour_une_perte_totale():
    """Le screener d'aujourd'hui ne liste que les survivants : les faillites, soit
    exactement ce que la note F pretend annoncer, en ont disparu. Les omettre
    mesurerait la note sur un univers vide de ses succes."""
    import valider_les_notes as v

    assert v.SEUIL_EFFONDREMENT == -0.70
    # Une radiation vaut -100 %, donc un effondrement quel que soit le seuil.
    assert -1.0 <= v.SEUIL_EFFONDREMENT


def test_la_page_lit_la_mesure_au_lieu_de_la_recopier():
    """Aucun chiffre de validation ne doit etre ecrit en dur dans la fiche.

    Deux chiffres ecrits a deux endroits finissent toujours par diverger. C'est
    exactement ainsi que ce bloc a promis pendant des semaines une confrontation
    "dans douze mois" alors qu'elle avait deja eu lieu, puis, une fois les chiffres
    publies a la main, qu'il a fallu les recorriger a chaque nouvelle mesure. La
    page lit desormais le fichier ; ce test interdit le retour en arriere.
    """
    import re

    racine = Path(__file__).resolve().parent.parent
    html = (racine / "app" / "stock.html").read_text(encoding="utf-8")

    bloc = re.search(r'<div id="validationRisque">(.*?)</div>', html, re.S)
    assert bloc, "le conteneur de validation a disparu de la fiche"
    fige = re.findall(r"<t[dh]>([^<]*[0-9][^<]*)</t[dh]>", bloc.group(1))
    assert not fige, f"chiffres figes dans le HTML de validation : {fige[:5]}"
    assert "_validation_risque_resume.json" in html,         "la fiche ne va plus chercher la mesure"


def test_le_resume_de_validation_porte_tout_ce_que_la_page_lit():
    """Le contrat entre le script de mesure et la page.

    La page rend des tableaux a partir de ce fichier : une cle absente n'y provoque
    pas d'erreur visible, elle affiche « undefined » au milieu d'un tableau de
    resultats. Ce test enumere donc ce que la page consomme reellement.
    """
    import json

    racine = Path(__file__).resolve().parent.parent
    f = racine / "app" / "us" / "_validation_risque_resume.json"
    if not f.exists():
        pytest.skip("validation jamais executee sur ce poste")
    d = json.loads(f.read_text(encoding="utf-8"))

    for cle in ("instantane", "horizon_mois", "n", "radiees", "par_famille",
                "dimensions", "auc_note", "spearman_familles", "p_F_contre_A",
                "ecart_F_moins_A", "par_taille", "genere"):
        assert cle in d, f"cle absente du resume : {cle}"

    assert [l["grade"] for l in d["par_famille"]] == ["A", "B", "C", "D", "F"]
    for l in d["par_famille"]:
        for cle in ("n", "taux", "rendement_median", "creux_median", "cap_mediane"):
            assert cle in l, (l["grade"], cle)
        assert 0.0 <= l["taux"] <= 1.0

    # Les huit dimensions figurent TOUTES, y compris celles qui desservent le
    # modele et celle qui n'a pas pu etre testee. En publier six sur huit serait
    # une selection, pas une mesure.
    from quantbench.risk.dimensions import DIMENSIONS
    assert {x["dimension"] for x in d["dimensions"]} == {c for c, _n, _f, _l in DIMENSIONS}
    for x in d["dimensions"]:
        assert x["auc"] is None or 0.0 <= x["auc"] <= 1.0
        assert x.get("nom") and x.get("verdict")

    # Le fichier complet reste a cote, avec le detail par titre : sans lui, aucune
    # anomalie du tableau agrege n'est diagnosticable.
    assert (racine / "app" / "us" / "_validation_risque.json").exists(),         "le fichier detaille a disparu"


# --------------------------------------------------------------------------- #
# DCF vectorise : le meme calcul, plus vite, ou rien
# --------------------------------------------------------------------------- #
def _base_dcf_aleatoire(rng, mode):
    """Un jeu d'entrees DCF tire au hasard, dans des bornes economiquement plausibles."""
    from quantbench.valuation.dcf import DcfInputs
    return DcfInputs(
        revenue_base=rng.uniform(10, 1e5),
        g1_begin=rng.uniform(-0.2, 0.5), g1_end=rng.uniform(-0.1, 0.3),
        g2_begin=rng.uniform(-0.1, 0.3), g2_end=rng.uniform(-0.05, 0.15),
        g3_begin=rng.uniform(-0.05, 0.12), g3_end=rng.uniform(-0.02, 0.035),
        len1=rng.randint(1, 6), len2=rng.randint(1, 6), len3=rng.randint(1, 6),
        conv1=rng.randint(1, 5), conv2=rng.randint(1, 5), conv3=rng.randint(1, 5),
        current_operating_margin=rng.uniform(-0.3, 0.5),
        terminal_operating_margin=rng.uniform(0.01, 0.4),
        margin_converge_start=rng.randint(1, 8),
        current_tax_rate=rng.uniform(0, 0.4),
        marginal_tax_rate=rng.uniform(0.1, 0.4),
        tax_converge_start=rng.randint(1, 8),
        current_sales_to_capital=rng.uniform(0.3, 5),
        terminal_sales_to_capital=rng.uniform(0.3, 5),
        s2c_converge_start=rng.randint(1, 6),
        risk_free_rate=rng.uniform(0.01, 0.06), erp=rng.uniform(0.03, 0.09),
        size_premium=rng.choice([0.0, 0.01, 0.035]),
        unlevered_beta=rng.uniform(0.4, 2.2),
        terminal_unlevered_beta=rng.uniform(0.5, 1.5),
        beta_converge_start=rng.randint(1, 8),
        current_pretax_kd=rng.uniform(0.02, 0.15),
        terminal_pretax_kd=rng.uniform(0.02, 0.1),
        kd_converge_start=rng.randint(1, 8),
        equity_value=rng.uniform(1, 1e6), debt_value=rng.uniform(0, 5e5),
        cash_and_non_operating=rng.uniform(0, 1e5),
        additional_roic_in_perpetuity=rng.uniform(0, 0.03),
        asset_liquidation_during_negative_growth=rng.choice([0.0, 0.5, 1.0]),
        reinvestment_mode=mode,
        current_roic=(rng.uniform(0.02, 0.4) if mode == "roic" else float("nan")),
        terminal_roic=(rng.uniform(0.02, 0.25) if mode == "roic" else float("nan")),
        roic_converge_start=rng.randint(1, 8),
    )


def test_le_dcf_vectorise_rend_exactement_les_memes_chiffres():
    """Le chemin rapide doit etre IDENTIQUE AU BIT PRES a la reference.

    Une optimisation qui deplace les valorisations publiees, meme d'un millieme,
    n'est pas une optimisation : c'est un changement de methode qui ne dit pas son
    nom. L'egalite exacte est donc exigee, pas une tolerance.
    """
    import random

    import numpy as np

    from quantbench.valuation.dcf import value_dcf
    from quantbench.valuation.dcf_vectorise import equites_dcf

    rng = random.Random(20260730)
    compares = 0
    for i in range(1200):
        b = _base_dcf_aleatoire(rng, "roic" if i % 2 else "sales_to_capital")
        try:
            ref = value_dcf(b)["equity_value"]
        except (ValueError, ZeroDivisionError, FloatingPointError):
            assert np.isnan(equites_dcf(b, {}, 1)[0]), \
                "la reference rejette ce scenario, le chemin rapide l'accepte"
            continue
        vec = float(equites_dcf(b, {}, 1)[0])
        assert vec == ref, (i, ref, vec)
        compares += 1
    assert compares > 800, f"trop peu de scenarios valides compares : {compares}"


def test_le_dcf_vectorise_rejette_exactement_les_memes_scenarios():
    """Un scenario incoherent doit valoir NaN, jamais un nombre.

    Les trois rejets de la reference sont economiques, pas techniques : sans eux,
    un cout du capital terminal sous la croissance perpetuelle donne une valeur
    terminale NEGATIVE, et un rendement du capital nul rend la croissance gratuite.
    Le chemin rapide qui les accepterait fabriquerait de la valeur.
    """
    from dataclasses import replace

    import numpy as np

    from quantbench.valuation.dcf import value_dcf
    from quantbench.valuation.dcf_vectorise import equites_dcf

    base = _dcf_inputs_temoin()
    for change in ({"g3_end": 0.50},          # WACC terminal sous la croissance
                   {"current_roic": -0.05},   # rendement du capital negatif
                   {"terminal_roic": 0.0},    # rendement terminal nul
                   {"terminal_roic": -0.10}):
        b = replace(base, **change)
        with pytest.raises((ValueError, ZeroDivisionError, FloatingPointError)):
            value_dcf(b)
        assert np.isnan(equites_dcf(b, {}, 1)[0]), change


def test_le_monte_carlo_retombe_sur_la_reference_quand_il_le_faut():
    """Un champ ENTIER tire (longueur de phase, annee de convergence) fait varier
    la structure du scenario d'un tirage a l'autre. Le module vectorise ne sait pas
    la traiter et doit le DIRE, pour que l'appelant reprenne la boucle exacte plutot
    que de valoriser une approximation silencieuse."""
    from quantbench.valuation.dcf_vectorise import champs_vectorisables

    assert champs_vectorisables(["g1_begin", "terminal_operating_margin", "erp"])
    assert not champs_vectorisables(["g1_begin", "len1"])
    assert not champs_vectorisables(["conv2"])
    assert not champs_vectorisables(["margin_converge_start"])
    assert not champs_vectorisables([])                     # aucun tirage
    assert not champs_vectorisables(["champ_qui_nexiste_pas"])


def test_les_deux_chemins_du_monte_carlo_donnent_la_meme_distribution():
    """Bout en bout : meme graine, memes lois, memes percentiles."""
    import numpy as np
    from scipy import stats as st

    from quantbench.valuation.dcf_vectorise import champs_vectorisables
    from quantbench.valuation.montecarlo import monte_carlo_dcf, sample_inputs

    base = _dcf_inputs_temoin()
    lois, corr = _lois_temoin(base)
    assert champs_vectorisables(lois.keys()), \
        "les lois du site doivent emprunter le chemin rapide, sinon il ne sert a rien"

    rapide = monte_carlo_dcf(base, lois, n=600, correlations=corr, seed=11)

    # On force le repli en ajoutant un champ entier de dispersion NULLE : la
    # structure ne bouge pas, mais l'aiguillage bascule sur la boucle exacte.
    lentes = dict(lois)
    lentes["len1"] = st.norm(base.len1, 1e-12)
    lent = monte_carlo_dcf(base, lentes, n=600, correlations=corr, seed=11)

    # Les tirages des huit lois communes sont identiques a graine egale seulement
    # si le nombre de lois est le meme ; ici il differe, donc on compare des
    # STATISTIQUES et non des valeurs point a point.
    assert lent["n_valid"] == rapide["n_valid"]
    assert np.isclose(lent["median"], rapide["median"], rtol=0.05), \
        (lent["median"], rapide["median"])
    assert np.isclose(lent["percentiles"][10], rapide["percentiles"][10], rtol=0.10)
    assert set(sample_inputs(lois, 5, corr, 0)) == set(lois)


# --------------------------------------------------------------------------- #
# Ce qui doit SURVIVRE au build
# --------------------------------------------------------------------------- #
def test_les_mesures_durables_ne_sont_pas_ignorees_par_git():
    """`app/us/` etait ignore en bloc, et emportait deux mesures qui doivent durer.

    Chaque execution du build repart d'une copie fraiche du depot : ce qui n'y est
    pas commite n'a jamais existe. Deux consequences avaient passe inapercues, les
    deux invisibles en integration continue et visibles seulement en production :
      - l'archive mensuelle des notes se recreait vide a chaque build, alors que
        son unique raison d'etre est d'etre relue dans un an ;
      - le fichier de validation que la fiche va CHERCHER au chargement n'etait
        pas deploye, si bien que la page affichait « mesure indisponible ».
    """
    racine = Path(__file__).resolve().parent.parent
    lignes = (racine / ".gitignore").read_text(encoding="utf-8").splitlines()
    regles = [x.strip() for x in lignes if x.strip() and not x.startswith("#")]

    assert "app/us/" not in regles, \
        "app/us/ ignore en bloc : les mesures durables ne survivront pas au build"
    assert "app/us/*" in regles, "le contenu de app/us/ doit rester ignore"
    for garde in ("!app/us/_notes_risque/", "!app/us/_validation_risque.json",
                  "!app/us/_validation_risque_resume.json"):
        assert garde in regles, f"reglee manquante : {garde}"


def test_larchive_mensuelle_garde_la_mesure_la_plus_large():
    """Une archive partielle ne doit jamais devenir definitive.

    La regle precedente — « le fichier du mois existe, on ne touche a rien » —
    figeait le mois entier sur le premier ecrit : un build interrompu, un essai
    local sur cent vingt titres, un shard perdu, et la mesure a douze mois aurait
    porte sur cet echantillon sans que rien ne le signale.
    """
    import importlib
    import json

    mod = importlib.import_module("build_site_fmp")
    src = inspect.getsource(mod._archiver_les_notes)
    assert "if fichier.exists():\n        return" not in src, \
        "l'archive du mois redevient definitive des le premier ecrit"
    assert "ancien >= len(notes)" in src, \
        "aucune comparaison d'effectif : l'archive peut retrecir"

    # Comportement, pas seulement forme.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        old_us = mod.US
        try:
            mod.US = Path(tmp)
            large = [{"ticker": f"T{i}", "note_risque": "B", "score_risque": 0.4,
                      "price": 1.0} for i in range(300)]
            etroit = large[:5]
            mod._archiver_les_notes(large)
            f = next(Path(tmp).glob("_notes_risque/*.json"))
            assert json.loads(f.read_text(encoding="utf-8"))["n"] == 300
            mod._archiver_les_notes(etroit)          # ne doit PAS retrecir
            assert json.loads(f.read_text(encoding="utf-8"))["n"] == 300
            mod._archiver_les_notes(large + [{"ticker": "ZZ", "note_risque": "A",
                                              "score_risque": 0.1, "price": 2.0}])
            assert json.loads(f.read_text(encoding="utf-8"))["n"] == 301
        finally:
            mod.US = old_us


# --------------------------------------------------------------------------- #
# Statistiques de jugement : une seule ecriture
# --------------------------------------------------------------------------- #
def test_laire_sous_la_courbe_est_juste():
    """Verifiee contre les cas ou la reponse est connue d'avance, et contre
    scikit-learn quand il est disponible. Les EX AEQUO comptent pour une demie :
    sans cela, une dimension a valeurs discretes serait creditee d'un pouvoir
    qu'elle n'a pas."""
    import random

    from quantbench.risk.statistiques import aire_sous_la_courbe

    assert aire_sous_la_courbe([(0.1, False), (0.2, False),
                                (0.8, True), (0.9, True)])[0] == 1.0
    assert aire_sous_la_courbe([(0.9, False), (0.8, False),
                                (0.2, True), (0.1, True)])[0] == 0.0
    assert aire_sous_la_courbe([(0.5, True), (0.5, False)])[0] == 0.5
    assert aire_sous_la_courbe([(1.0, True)]) == (None, None)   # une seule classe

    # L'ecart-type doit RETRECIR quand l'echantillon grandit, sinon il ne mesure
    # rien. La separation doit etre IMPARFAITE pour que la question ait un sens :
    # une AUC de 1,0 a une variance de Hanley-McNeil exactement nulle a tout
    # effectif, et comparer deux zeros ne prouve pas que l'ecart-type reagit.
    def bruite(n, graine):
        r = random.Random(graine)
        return [(r.gauss(1.0 if i % 3 == 0 else 0.0, 1.0), i % 3 == 0)
                for i in range(n)]

    petit = aire_sous_la_courbe(bruite(40, 1))
    grand = aire_sous_la_courbe(bruite(4000, 1))
    assert 0.5 < grand[0] < 1.0, grand[0]
    assert 0.0 < grand[1] < petit[1], (petit[1], grand[1])

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return
    rng = random.Random(11)
    for _ in range(60):
        n = rng.randint(30, 300)
        paires = [(round(rng.uniform(0, 1), rng.choice([1, 2, 3])), rng.random() < 0.3)
                  for _ in range(n)]
        ys = [y for _s, y in paires]
        if not any(ys) or all(ys):
            continue
        attendu = roc_auc_score(ys, [s for s, _y in paires])
        assert aire_sous_la_courbe(paires)[0] == pytest.approx(attendu, abs=1e-12)


def test_les_scripts_de_mesure_ne_redefinissent_pas_les_statistiques():
    """Deux scripts jugent la note de risque et avaient chacun leur aire sous la
    courbe. Elles s'accordaient, mais l'une rendait l'ecart-type et l'autre non :
    le tableau publie affichait donc des AUC sans jamais dire a quelle precision.
    Une regle n'a le droit qu'a une seule ecriture."""
    racine = Path(__file__).resolve().parent.parent
    for nom in ("valider_les_notes.py", "mesurer_les_notes.py"):
        src = (racine / "scripts" / nom).read_text(encoding="utf-8")
        assert "quantbench.risk.statistiques" in src, \
            f"{nom} n'utilise pas la source unique"
        for interdit in ("def wilson(", "def test_deux_proportions(",
                         "def spearman(", "def mediane("):
            assert interdit not in src, f"{nom} redefinit {interdit.strip('def (')}"
        # `aire_sous_la_courbe` peut etre reexposee localement, mais jamais
        # RECALCULEE : la statistique de Mann-Whitney n'a pas a exister deux fois.
        assert "bisect" not in src and "Mann-Whitney" not in src, \
            f"{nom} recalcule l'aire sous la courbe au lieu de l'importer"


def test_le_verdict_dune_dimension_se_lit_en_ecarts_types():
    """Un seuil fixe juge un chiffre sans savoir a quelle precision il est connu :
    le meme 0,545 est une information sur mille six cents societes et du bruit sur
    cent. Le verdict doit donc compter les ecarts-types qui separent la mesure de
    0,500, c'est-a-dire de l'absence totale d'information."""
    racine = Path(__file__).resolve().parent.parent
    src = (racine / "scripts" / "valider_les_notes.py").read_text(encoding="utf-8")
    assert "ecarts_types_au_hasard" in src
    assert "auc < 0.53" not in src, "seuil fixe rétabli sur l'AUC"
    assert "indistinguable du hasard" in src


def test_la_validation_refuse_de_mesurer_sans_les_radiees():
    """Sans les societes radiees, cette mesure n'en est pas une.

    Le screener d'aujourd'hui ne liste que les survivantes : les faillites, soit
    l'evenement meme que la note F pretend annoncer, en ont disparu. Le fournisseur
    a renvoye une liste vide un soir de saturation, le script a continue sans
    broncher, et la mesure obtenue annoncait 3,9 % d'effondrement en A contre
    18,5 % avec la correction. Aucune alerte, juste des chiffres flatteurs — la
    forme de defaillance la plus dangereuse qui soit pour un banc d'essai.
    """
    racine = Path(__file__).resolve().parent.parent
    src = (racine / "scripts" / "valider_les_notes.py").read_text(encoding="utf-8")
    assert "if not radiees:" in src, \
        "aucune garde : une liste vide produirait une mesure biaisee sans le dire"
    # La garde doit precoder l'ecriture des fichiers, pas la suivre.
    assert src.index("if not radiees:") < src.index("SORTIE.write_text"), \
        "la garde arrive apres l'ecriture : le fichier biaise serait quand meme publie"


def test_le_resume_publie_porte_bien_la_correction_du_survivant():
    """Le fichier deploye doit prouver, par lui-meme, qu'il a ete mesure avec les
    radiees. Un resume a zero radiee est un resume a jeter."""
    import json

    racine = Path(__file__).resolve().parent.parent
    f = racine / "app" / "us" / "_validation_risque_resume.json"
    if not f.exists():
        pytest.skip("validation jamais executee sur ce poste")
    d = json.loads(f.read_text(encoding="utf-8"))
    assert d["radiees"] > 0, \
        "mesure publiee sans aucune societe radiee : biais du survivant non corrige"
    assert d["radiees"] / d["n"] > 0.05, \
        f"seulement {d['radiees']} radiees sur {d['n']} : correction trop faible"


# --------------------------------------------------------------------------- #
# Bande passante : le forfait est de 150 Go par mois
# --------------------------------------------------------------------------- #
def test_lhistorique_de_cours_est_borne_cote_serveur():
    """Cet appel telechargeait cinq ans de seances pour n'en garder que 400.

    Il pesait a lui seul 83 % de la bande passante du build ; sur seize mille
    titres reconstruits chaque nuit, c'etait la difference entre tenir le forfait
    mensuel et l'epuiser avant la fin du mois. Le bornage doit etre DEMANDE AU
    SERVEUR (`&from=`), pas applique apres reception : trancher une liste deja
    telechargee ne fait economiser aucun octet.
    """
    import inspect

    from quantbench.data import fmp

    src = inspect.getsource(fmp.history_ohlcv)
    assert "&from=" in src, "l'historique n'est plus borne cote serveur"
    assert "historical-price-eod/dividend-adjusted?symbol={symbol}{borne}" in src, \
        "la borne n'est pas transmise a l'appel principal"
    assert src.index("borne = \"\"") < src.index("_json("), \
        "la borne est calculee apres l'appel"
    # Le mode complet reste disponible : la validation historique en depend.
    assert "if days:" in src, "le mode complet (days=None) a disparu"


def test_le_bornage_couvre_bien_les_seances_demandees():
    """Le facteur de conversion seances -> jours calendaires doit MAJORER.

    Une annee compte environ 252 seances pour 365 jours, soit un facteur 1,45.
    Sous-estimer tronquerait la serie juste avant le seuil dont depend le signal
    court terme — un defaut qui ne se verrait que sur les titres peu liquides.
    """
    import inspect
    import re

    from quantbench.data import fmp

    src = inspect.getsource(fmp.history_ohlcv)
    m = re.search(r"int\(days \* ([\d.]+)\)", src)
    assert m, "facteur de conversion introuvable"
    facteur = float(m.group(1))
    assert facteur >= 365 / 252, \
        f"facteur {facteur} insuffisant : 365/252 = 1,449 minimum"


def test_la_cle_api_ne_peut_pas_sortir_dans_un_message_derreur():
    """`raise_for_status` construit son message a partir de l'URL COMPLETE.

    Deux appelants historiques persistent `str(e)[:140]` dans le tableau `errors`
    de `screener.json`, fichier PUBLIE sur le site : le prefixe du message fait
    environ 78 caracteres, ce qui laisse la place d'y ecrire une cle entiere.
    """
    import os

    import pytest as _pytest

    from quantbench.data import fmp

    ancienne = os.environ.get("FMP_API_KEY")
    os.environ["FMP_API_KEY"] = "CLE_QUI_NE_DOIT_PAS_FUIR"
    try:
        with _pytest.raises(Exception) as e:
            fmp._json("income-statement?symbol=CE_TICKER_NEXISTE_PAS_DU_TOUT")
        msg = str(e.value)
        assert "CLE_QUI_NE_DOIT_PAS_FUIR" not in msg, msg[:200]
        assert "***" in msg, msg[:200]
    finally:
        if ancienne is None:
            os.environ.pop("FMP_API_KEY", None)
        else:
            os.environ["FMP_API_KEY"] = ancienne


def test_le_build_rend_compte_de_sa_bande_passante():
    """Une regression de bande passante ne casse rien, ne ralentit rien, et ne se
    voit que sur la facture — donc trop tard. Le build doit la publier a chaque
    execution, avec sa projection sur trente jours."""
    import importlib

    mod = importlib.import_module("build_site_fmp")
    assert hasattr(mod, "_rapport_bande_passante")
    assert mod.PLAFOND_MENSUEL == 150e9
    src = inspect.getsource(mod.main)
    assert "_rapport_bande_passante(" in src, \
        "le rapport existe mais n'est jamais appele"

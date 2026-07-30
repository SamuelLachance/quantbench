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

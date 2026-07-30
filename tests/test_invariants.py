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

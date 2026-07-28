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

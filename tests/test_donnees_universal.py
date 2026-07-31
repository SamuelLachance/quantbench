"""Couche de donnees `quantbench/data/universal.py` — tests HORS LIGNE.

Ce module n'avait AUCUN test alors qu'il est la porte d'entree de `route.py`
(`from ..data.universal import get_fundamentals`, appelee des que
`value_stock(ticker)` est invoquee sans `fund`). C'est aussi le seul endroit du
depot ou trois conversions se croisent sur une meme ligne : devise du COURS,
devise des COMPTES, et passage a l'unite « milliard de dollars ». Les trois bugs
d'unites deja rencontres ailleurs dans QuantBench y trouveraient exactement le
meme terrain :

  * taux de change fige a 1        -> une societe canadienne valorisee 39 % trop cher
  * conversion en milliards oubliee -> une capitalisation lue comme 2,8e12 milliards
  * piege du zero « falsy »        -> un poste comptable NUL pris pour une donnee absente

Chaque test ci-dessous fige l'une de ces trois frontieres. Aucun ne touche le
reseau : `yfinance` est remplace par un faux module et `market.fx_to_usd` par une
table de taux fixe — un appel HTTP reel fait echouer le test.
"""

import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantbench.data import universal                                 # noqa: E402


# --------------------------------------------------------------------------- #
# Bouchons : ni reseau, ni yfinance, ni Yahoo
# --------------------------------------------------------------------------- #
class _FauxTicker:
    """Imite `yf.Ticker` : `.info`, `.income_stmt`, `.balance_sheet`.

    `leve` permet de simuler le cas REEL du rate-limiting Yahoo, ou l'acces aux
    etats financiers jette au lieu de renvoyer None.
    """

    def __init__(self, info, inc, bal, leve=()):
        self.info = info
        self._inc, self._bal, self._leve = inc, bal, leve

    @property
    def income_stmt(self):
        if "inc" in self._leve:
            raise RuntimeError("Yahoo a repondu 429")
        return self._inc

    @property
    def balance_sheet(self):
        if "bal" in self._leve:
            raise RuntimeError("Yahoo a repondu 429")
        return self._bal


@pytest.fixture(autouse=True)
def _hors_ligne(monkeypatch):
    """Interdit tout appel reseau et fixe un taux de change par defaut.

    Sans ce garde-fou, un test qui oublierait de bouchonner `fx_to_usd`
    interrogerait Yahoo pour de vrai et deviendrait vert ou rouge selon la
    connexion du jour.
    """
    def _interdit(*a, **k):
        raise AssertionError("appel reseau interdit dans les tests")

    monkeypatch.setattr(requests, "get", _interdit)
    monkeypatch.setattr(requests.Session, "request", _interdit)
    monkeypatch.setattr(universal.market, "fx_to_usd",
                        lambda d: 1.0 if d == "USD" else None)


@pytest.fixture
def taux(monkeypatch):
    """Fixe la table des taux de change vue par le module."""
    def _fixer(table):
        monkeypatch.setattr(universal.market, "fx_to_usd", lambda d: table.get(d))
    return _fixer


@pytest.fixture
def yahoo(monkeypatch):
    """Installe un faux module `yfinance` (importe DANS la fonction testee)."""
    def _installer(info, inc=None, bal=None, leve=()):
        faux = types.ModuleType("yfinance")
        faux.Ticker = lambda symbole: _FauxTicker(info, inc, bal, leve)
        monkeypatch.setitem(sys.modules, "yfinance", faux)
    return _installer


def etat(lignes, annees=(2025, 2024, 2023)):
    """Etat financier au format yfinance : index = libelles, colonnes = exercices
    du plus RECENT au plus ancien (c'est l'ordre que renvoie yfinance, et c'est
    de lui que depend le sens de `revenue_history`)."""
    cols = [pd.Timestamp(f"{a}-12-31") for a in annees]
    return pd.DataFrame(lignes, index=cols).T


def info_us(**kw):
    """Societe americaine saine : une seule devise, tout est en USD."""
    base = {"currency": "USD", "financialCurrency": "USD",
            "shortName": "Test Corp", "sector": "Technology", "industry": "Software",
            "currentPrice": 187.5, "marketCap": 2.8e12, "sharesOutstanding": 1.4933e10,
            "beta": 1.15, "totalRevenue": 3.9e11, "ebitda": 1.4e11,
            "netIncomeToCommon": 9.7e10, "totalDebt": 1.1e11, "totalCash": 6.5e10,
            "dividendYield": 0.0044, "payoutRatio": 0.15}
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# 1. L'unite : milliards pour les montants, unite native pour le COURS
# --------------------------------------------------------------------------- #
def test_les_montants_sont_en_milliards_mais_pas_le_cours(yahoo):
    """Frontiere d'unites la plus facile a franchir par accident.

    Tout le reste du depot (build_universal, route, le Monte Carlo) raisonne en
    MILLIARDS de dollars, sauf le cours qui reste un prix par action. Le module
    divise donc `marketCap` par 1e9 mais surtout PAS `currentPrice`. Oublier la
    division rendait une capitalisation de 2 800 milliards de milliards ; l'ajouter
    au cours donnait une action a 1,9e-7 $ et un upside de plusieurs milliards de
    pour cent.
    """
    yahoo(info_us())
    f = universal.get_fundamentals("test")

    assert f["market_cap"] == pytest.approx(2800.0)      # 2,8e12 $ -> 2 800 Md$
    assert f["revenue"] == pytest.approx(390.0)
    assert f["ebitda"] == pytest.approx(140.0)
    assert f["net_income"] == pytest.approx(97.0)
    assert f["total_debt"] == pytest.approx(110.0)
    assert f["cash"] == pytest.approx(65.0)
    # Le cours, lui, reste un prix par action.
    assert f["price"] == pytest.approx(187.5)


def test_le_nombre_d_actions_reste_un_compte_brut(yahoo):
    """Le nombre d'actions n'est PAS en milliards — et c'est ce melange qui rend
    l'invariant utile : capitalisation (Md$) x 1e9 / actions doit redonner le cours.

    Un seul des deux champs converti par erreur decale la valeur par action d'un
    facteur 1e9, sans que le dict ait l'air anormal.
    """
    yahoo(info_us())
    f = universal.get_fundamentals("test")

    assert f["shares"] == pytest.approx(1.4933e10)
    assert f["market_cap"] * 1e9 / f["shares"] == pytest.approx(f["price"], rel=0.01)


def test_la_valeur_d_entreprise_est_homogene_en_milliards(yahoo):
    """VE = capitalisation + dette - tresorerie, les trois dans la MEME unite.

    Melanger un terme brut avec deux termes en milliards produit une VE aberrante
    qui contamine ensuite `ev_to_ebitda`, seul multiple publie pour les cycliques.
    """
    yahoo(info_us())
    f = universal.get_fundamentals("test")

    assert f["enterprise_value"] == pytest.approx(2800.0 + 110.0 - 65.0)
    assert f["ev_to_ebitda"] == pytest.approx(f["enterprise_value"] / f["ebitda"])


# --------------------------------------------------------------------------- #
# 2. Le change : jamais fige a 1
# --------------------------------------------------------------------------- #
def test_le_taux_de_change_est_reellement_applique(yahoo, taux):
    """Le defaut « taux fige a 1 » est indolore en apparence : le dict reste
    plausible, seules les valeurs sont fausses de 30 a 40 %.

    Une societe canadienne cotee 50 CAD ne vaut pas 50 USD. Ce test echoue si le
    taux disparait ou est remplace par 1,0 par defaut.
    """
    taux({"CAD": 0.72, "USD": 1.0})
    yahoo(info_us(currency="CAD", financialCurrency="CAD", currentPrice=50.0,
                  marketCap=8e9, sharesOutstanding=1.6e8, totalRevenue=4e9,
                  netIncomeToCommon=3e8, ebitda=8e8, totalDebt=1e9, totalCash=2e8))
    f = universal.get_fundamentals("bce.to")

    assert f["fx_price_to_usd"] == 0.72 and f["fx_financial_to_usd"] == 0.72
    assert f["price"] == pytest.approx(36.0)
    assert f["market_cap"] == pytest.approx(5.76)
    assert f["revenue"] == pytest.approx(2.88)
    assert f["currency_ok"] is True
    # Et l'invariant d'unites tient toujours apres conversion.
    assert f["market_cap"] * 1e9 / f["shares"] == pytest.approx(f["price"], rel=0.01)


def test_deux_devises_distinctes_pour_le_cours_et_pour_les_comptes(yahoo, taux):
    """Le piege Yahoo : `currency` (cotation) et `financialCurrency` (comptes)
    DIFFERENT souvent — minieres canadiennes cotees en CAD qui publient en USD,
    ADR, cotations en pence.

    Appliquer un seul taux aux deux groupes est le bug le plus vraisemblable ici :
    il passerait les tests mono-devise ci-dessus sans broncher. On force donc des
    taux differents et on verifie que chaque montant utilise le SIEN.
    """
    taux({"CAD": 0.72, "USD": 1.0})
    yahoo(info_us(currency="CAD", financialCurrency="USD", currentPrice=50.0,
                  marketCap=8e9, sharesOutstanding=1.6e8, totalRevenue=4e9,
                  netIncomeToCommon=3e8, ebitda=8e8, totalDebt=1e9, totalCash=2e8))
    f = universal.get_fundamentals("abx.to")

    assert f["price_currency"] == "CAD" and f["financial_currency"] == "USD"
    assert f["fx_price_to_usd"] == 0.72 and f["fx_financial_to_usd"] == 1.0
    # Cours et capitalisation : devise de COTATION.
    assert f["price"] == pytest.approx(36.0)
    assert f["market_cap"] == pytest.approx(5.76)
    # Compte de resultat et bilan : devise des COMPTES, non touches par le CAD.
    assert f["revenue"] == pytest.approx(4.0)
    assert f["net_income"] == pytest.approx(0.30)
    assert f["total_debt"] == pytest.approx(1.0)


def test_les_comptes_suivent_la_devise_du_cours_quand_yahoo_ne_la_donne_pas(yahoo, taux):
    """`financialCurrency` manque frequemment. Le repli doit etre la devise de
    COTATION, jamais le dollar implicite : un repli sur USD laisserait des comptes
    en pesos ou en yens convertis au taux 1 — exactement le defaut « fige a 1 »,
    mais sous une autre forme.
    """
    taux({"CAD": 0.72, "USD": 1.0})
    donnees = info_us(currency="CAD", currentPrice=50.0, marketCap=8e9,
                      sharesOutstanding=1.6e8, totalRevenue=4e9)
    donnees.pop("financialCurrency")
    yahoo(donnees)
    f = universal.get_fundamentals("t.to")

    assert f["financial_currency"] == "CAD"
    assert f["revenue"] == pytest.approx(2.88)


def test_une_devise_de_comptes_inconnue_invalide_la_fiche(yahoo, taux):
    """Cas le plus dangereux : le COURS est convertible, les COMPTES ne le sont pas.

    Sans garde-fou, un chiffre d'affaires de 4e9 unites exotiques serait publie tel
    quel, donc lu comme 4 milliards de dollars. Le contrat est double : les montants
    concernes valent None, ET `currency_ok` passe a False pour que `route.py` rejette
    le titre au lieu de le valoriser de travers.
    """
    taux({"USD": 1.0})                       # ZWL absente de la table
    yahoo(info_us(currency="USD", financialCurrency="ZWL", totalRevenue=4e9,
                  netIncomeToCommon=3e8))
    f = universal.get_fundamentals("zzz")

    assert f["currency_ok"] is False
    assert f["fx_financial_to_usd"] is None
    assert f["revenue"] is None and f["net_income"] is None
    assert f["price"] == pytest.approx(187.5)          # le cours, lui, reste lisible


def test_une_devise_inconnue_fait_refuser_le_titre_par_la_valorisation(yahoo, taux):
    """Le drapeau ne sert a rien s'il n'est pas lu : on verifie la CONSEQUENCE en
    production, c'est-a-dire le refus de `route.value_stock`. C'est le seul test
    qui relie ce module au pipeline qui le consomme.
    """
    from quantbench.valuation import route

    taux({})                                  # aucune devise connue
    yahoo(info_us(currency="XYZ", financialCurrency="XYZ"))
    f = universal.get_fundamentals("zzz")

    assert f["currency_ok"] is False
    v = route.value_stock("ZZZ", fund=f)
    assert v["ok"] is False and "devise" in v["reason"]


# --------------------------------------------------------------------------- #
# 3. Le zero comptable n'est pas une donnee absente
# --------------------------------------------------------------------------- #
def test_des_capitaux_propres_nuls_ne_sont_pas_une_donnee_manquante(yahoo):
    """Piege du zero « falsy » : `if not valeur` confond 0 et None.

    Des capitaux propres exactement nuls sont une information comptable (societe
    dont les pertes ont exactement absorbe le capital), pas une absence. Les
    effacer en None ferait basculer le titre vers une autre route de valorisation
    au lieu de le signaler.
    """
    bal = etat({"Stockholders Equity": [0.0, 4.0e9, 3.5e9],
                "Cash And Cash Equivalents": [0.0, 1e9, 1e9]})
    yahoo(info_us(), bal=bal)
    f = universal.get_fundamentals("test")

    assert f["book_equity"] == 0.0
    assert f["book_equity"] is not None
    # Et un denominateur nul ne doit produire ni exception ni infini.
    assert f["price_to_book"] is None
    assert f["roe"] is None


def test_un_beta_nul_et_un_dividende_nul_survivent(yahoo):
    """Meme piege, cote `info` : beta 0 (valeur de regression plate) et rendement
    du dividende 0 (societe qui ne distribue pas) sont des faits, pas des trous.
    Les transformer en None ferait retomber le beta sur une valeur par defaut de
    1,0 en aval — un cout du capital invente.
    """
    yahoo(info_us(beta=0.0, dividendYield=0.0, payoutRatio=0.0))
    f = universal.get_fundamentals("test")

    assert f["beta"] == 0.0
    assert f["dividend_yield"] == 0.0
    assert f["payout_ratio"] == 0.0


def test_un_poste_exactement_nul_ne_doit_pas_declencher_le_repli(yahoo):
    """Le zero « falsy » attaque ici la ligne de repli, pas le test d'absence.

    `_first` savait deja distinguer 0 de None — mais le resultat etait aussitot
    passe a `or`, qui, lui, ne le sait pas. Une societe a resultat net nul, sans
    dette et sans chiffre d'affaires (biotechnologie avant son premier produit)
    voyait ces trois postes remplaces par None des lors que les etats financiers
    etaient indisponibles. Corrige par `_sinon`, qui teste la presence et non la
    verite. C'est la CINQUIEME occurrence de ce piege dans le depot.
    """
    yahoo(info_us(netIncomeToCommon=0.0, totalDebt=0.0, totalRevenue=0.0))
    f = universal.get_fundamentals("test")

    assert f["net_income"] == 0.0
    assert f["total_debt"] == 0.0
    assert f["revenue"] == 0.0


# --------------------------------------------------------------------------- #
# 4. Lecture des etats financiers
# --------------------------------------------------------------------------- #
def test_l_historique_de_chiffre_d_affaires_va_du_plus_ancien_au_plus_recent(yahoo):
    """yfinance range ses colonnes du plus RECENT au plus ancien ; le reste du
    depot attend l'inverse pour mesurer une croissance.

    Perdre le retournement ne casse rien visiblement : une societe en croissance
    de +15 % par an ressort simplement en decroissance de -13 %, et son DCF avec.
    """
    inc = etat({"Total Revenue": [5e9, 4e9, 3e9]})
    yahoo(info_us(), inc=inc)
    f = universal.get_fundamentals("test")

    assert f["revenue_history"] == [pytest.approx(3.0), pytest.approx(4.0),
                                    pytest.approx(5.0)]
    assert f["revenue_history"] == sorted(f["revenue_history"])


def test_seule_la_valeur_annuelle_la_plus_recente_est_retenue(yahoo):
    """`_row` doit lire la colonne la plus recente, et ignorer les exercices vides
    plutot que de renvoyer NaN — un NaN se propage silencieusement jusqu'a la marge
    operationnelle affichee.
    """
    inc = etat({"Operating Income": [np.nan, 1.2e9, 1.0e9],
                "Pretax Income": [1.5e9, 1.4e9, 1.3e9],
                "Tax Provision": [0.3e9, 0.28e9, 0.26e9]})
    yahoo(info_us(), inc=inc)
    f = universal.get_fundamentals("test")

    assert f["ebit"] == pytest.approx(1.2)          # 2025 est vide -> on prend 2024
    assert f["pretax_income"] == pytest.approx(1.5)
    assert f["tax"] == pytest.approx(0.3)
    assert f["ebit"] == f["ebit"]                   # non-NaN


def test_les_libelles_alternatifs_sont_essayes_dans_l_ordre(yahoo):
    """Yahoo renomme ses lignes selon le referentiel comptable. Le module essaie
    plusieurs libelles ; si le premier disparait, le suivant doit prendre le relais
    — sinon des pans entiers de l'univers perdent leur EBIT sans erreur.
    """
    inc = etat({"EBIT": [1.2e9, 1.1e9, 1.0e9]})            # pas d'"Operating Income"
    bal = etat({"Common Stock Equity": [4.0e9, 3.8e9, 3.6e9]})
    yahoo(info_us(), inc=inc, bal=bal)
    f = universal.get_fundamentals("test")

    assert f["ebit"] == pytest.approx(1.2)
    assert f["book_equity"] == pytest.approx(4.0)


def test_une_ligne_absente_vaut_None_et_non_zero(yahoo):
    """Une ligne introuvable est une IGNORANCE, pas un zero : un EBIT absent
    converti en 0 ferait passer une societe rentable pour deficitaire et la
    routerait vers la valorisation de detresse.
    """
    yahoo(info_us(), inc=etat({"Total Revenue": [5e9, 4e9, 3e9]}))
    f = universal.get_fundamentals("test")

    assert f["ebit"] is None                        # ni `info` ni l'etat ne le portent
    assert f["book_equity"] is None                 # aucun bilan fourni
    # Et l'ignorance se propage en ignorance, pas en zero.
    assert f["operating_margin"] is None
    assert f["price_to_book"] is None
    # Le chiffre d'affaires, lui, est bien la : l'absence est CIBLEE, pas globale.
    assert f["revenue"] == pytest.approx(390.0)


def test_un_etat_financier_indisponible_ne_fait_pas_tomber_la_fiche(yahoo):
    """yfinance jette (429, HTML de captcha, schema change) au lieu de renvoyer None.

    Sur un batch de plusieurs milliers de titres, laisser remonter l'exception fait
    perdre le ticker entier alors que `info` suffit a le valoriser. Ce test echoue
    si les `try/except` autour de `income_stmt` / `balance_sheet` disparaissent.
    """
    yahoo(info_us(), leve=("inc", "bal"))
    f = universal.get_fundamentals("test")

    assert f["revenue"] == pytest.approx(390.0)     # repli sur `info`
    assert f["ebit"] is None and f["book_equity"] is None
    assert f["currency_ok"] is True


def test_un_info_vide_ne_leve_pas(yahoo):
    """Yahoo renvoie parfois `info = None` sur un ticker radie. Le module doit
    rendre une fiche entierement vide plutot que d'exploser au milieu du batch."""
    yahoo(None)
    f = universal.get_fundamentals("dead")

    assert f["ticker"] == "DEAD" and f["name"] == "DEAD"
    assert f["currency_ok"] is False
    assert f["price"] is None and f["market_cap"] is None


def test_les_valeurs_NaN_de_yahoo_sont_ignorees(yahoo):
    """`info` contient regulierement des NaN, qui ne sont ni None ni utilisables :
    un NaN garde comme cours se propage jusqu'a un upside NaN, qui compare faux
    partout sans jamais lever."""
    yahoo(info_us(currentPrice=float("nan"), regularMarketPrice=201.0))
    f = universal.get_fundamentals("test")

    assert f["price"] == pytest.approx(201.0)
    assert f["price"] == f["price"]


# --------------------------------------------------------------------------- #
# 5. Ratios : recalcules en USD, jamais repris de yfinance
# --------------------------------------------------------------------------- #
def test_les_ratios_sont_recalcules_et_non_recopies_de_yahoo(yahoo, taux):
    """Les ratios publies par Yahoo melangent les deux devises (capitalisation en
    monnaie de cotation / benefice en monnaie des comptes). Les recopier reintroduit
    precisement le bug de change que le reste du module elimine.

    On glisse donc des ratios FAUX dans `info` : ils ne doivent apparaitre nulle part.
    """
    taux({"CAD": 0.72, "USD": 1.0})
    yahoo(info_us(currency="CAD", financialCurrency="USD", currentPrice=50.0,
                  marketCap=8e9, sharesOutstanding=1.6e8, totalRevenue=4e9,
                  netIncomeToCommon=3e8, ebitda=8e8,
                  trailingPE=999.0, priceToBook=888.0, profitMargins=0.777,
                  returnOnEquity=0.666),
           bal=etat({"Stockholders Equity": [2e9, 1.9e9, 1.8e9]}))
    f = universal.get_fundamentals("abx.to")

    assert f["trailing_pe"] == pytest.approx(5.76 / 0.30)
    assert f["price_to_book"] == pytest.approx(5.76 / 2.0)
    assert f["net_margin"] == pytest.approx(0.30 / 4.0)
    assert f["roe"] == pytest.approx(0.30 / 2.0)
    for cle, faux in (("trailing_pe", 999.0), ("price_to_book", 888.0),
                      ("net_margin", 0.777), ("roe", 0.666)):
        assert f[cle] != pytest.approx(faux), f"{cle} recopie de yfinance"


def test_aucun_ratio_ne_divise_par_zero(yahoo):
    """Un denominateur nul doit donner None, pas une exception ni un infini : un
    `inf` traverse les JSON du site en `Infinity` et casse le rendu de la fiche."""
    yahoo(info_us(totalRevenue=0.0, netIncomeToCommon=0.0, ebitda=0.0),
          inc=etat({"Total Revenue": [0.0, 0.0, 0.0]}),
          bal=etat({"Stockholders Equity": [0.0, 0.0, 0.0]}))
    f = universal.get_fundamentals("test")

    for cle in ("operating_margin", "net_margin", "roe", "trailing_pe",
                "price_to_book", "ev_to_ebitda"):
        assert f[cle] is None, f"{cle} = {f[cle]}"


# --------------------------------------------------------------------------- #
# 6. Contrat de sortie et statut du module
# --------------------------------------------------------------------------- #
def test_le_contrat_de_sortie_est_complet_et_le_ticker_normalise(yahoo):
    """`route.py` et `build_universal.py` lisent ces cles par leur nom. En perdre
    une ne leve pas : le champ devient None et la valorisation change de route en
    silence."""
    yahoo(info_us())
    f = universal.get_fundamentals("aapl")

    assert f["ticker"] == "AAPL"
    attendues = {
        "ticker", "name", "sector", "industry", "price_currency",
        "financial_currency", "fx_price_to_usd", "fx_financial_to_usd",
        "currency_ok", "price", "market_cap", "shares", "beta", "revenue",
        "revenue_history", "ebit", "ebitda", "net_income", "pretax_income", "tax",
        "total_debt", "cash", "book_equity", "enterprise_value",
        "operating_margin", "net_margin", "roe", "trailing_pe", "price_to_book",
        "ev_to_ebitda", "dividend_yield", "payout_ratio",
    }
    assert attendues <= set(f), f"cles disparues : {attendues - set(f)}"


def test_le_nom_retombe_sur_le_ticker(yahoo):
    """Un titre sans `shortName` doit garder un libelle affichable ; `None` se
    retrouverait tel quel dans le titre de la fiche publiee."""
    donnees = info_us(longName="Test Corporation Inc.")
    donnees.pop("shortName")
    yahoo(donnees)
    assert universal.get_fundamentals("test")["name"] == "Test Corporation Inc."

    donnees.pop("longName")
    yahoo(donnees)
    assert universal.get_fundamentals("test")["name"] == "TEST"


def test_ce_module_reste_le_repli_de_la_valorisation(yahoo):
    """STATUT : ce module n'est exerce par AUCUN script de build (tous passent
    `fund=` explicitement, alimente par FMP ou la SEC). Il n'est pas mort pour
    autant : `route.py` l'importe au chargement — une erreur d'import y casserait
    tout le pipeline — et l'appelle des que `value_stock` est invoquee sans
    fondamentaux, ce que fait toute utilisation interactive.

    Ce test fige ce lien. S'il rougit, c'est que le module est devenu du code
    reellement mort : il faut alors le supprimer, pas le maintenir.
    """
    from quantbench.valuation import route

    src = inspect.getsource(route)
    assert "from ..data.universal import get_fundamentals" in src
    assert "fund = fund or get_fundamentals(ticker)" in inspect.getsource(route.value_stock)

    # Et c'est bien CETTE fonction que `route` appelle, pas une homonyme.
    assert route.get_fundamentals is universal.get_fundamentals
    yahoo(info_us())
    assert route.get_fundamentals("test")["ticker"] == "TEST"

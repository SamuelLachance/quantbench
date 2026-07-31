"""Donnees SEC — tests HORS LIGNE de `quantbench/data/edgar.py` et
`quantbench/data/sec_fundamentals.py`.

Ces deux modules tournent dans le build quotidien deploye et n'avaient AUCUN test.
Ils partagent un trait dangereux : ils echouent SANS BRUIT. Un User-Agent refuse
par la SEC, un CIK sentinelle, un `except Exception` trop large — rien ne casse,
rien ne remonte, et seize mille fiches perdent leurs documents officiels pendant
que chaque job du jour se declare vert.

Ce que ces tests fixent :

  * User-Agent SEC        -> un secret d'integration continue NON DEFINI vaut la
                             CHAINE VIDE, pas une variable absente : `os.environ.get(cle,
                             defaut)` ne rend alors jamais le defaut, le User-Agent
                             part vide et la SEC repond 403.
  * repli du User-Agent   -> la chaine de repli d'origine, parenthesee, etait elle
                             aussi refusee en 403. Un repli qui echoue n'est pas un
                             repli.
  * documents SEC         -> `annual_report_docs` avale toute exception : la
                             structure rendue en cas d'echec doit rester celle que
                             l'appelant lit (`ars_pdf`, `tenk`, `documents`), sans
                             quoi l'echec se transforme en KeyError chez lui.
  * CIK degenere          -> le fournisseur rend la sentinelle "0000000000" pour
                             toute societe hors perimetre SEC.
  * series annuelles      -> un exercice partiel pris pour un exercice complet, un
                             tag abandonne pris pour la serie a jour, une
                             retraitement ignore : autant de chiffres faux, jamais
                             d'erreur.
  * alignement des annees -> les ratios forensiques comparent t et t-1 ; des
                             metriques tirees d'exercices differents produisent des
                             scores faux et plausibles.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantbench.data import edgar                                     # noqa: E402
from quantbench.data import sec_fundamentals as sf                    # noqa: E402


# --------------------------------------------------------------------------- #
# Outillage : faux reseau, faux faits XBRL, caches vides
# --------------------------------------------------------------------------- #
class Reponse:
    """Reponse HTTP minimale. `boom_json` imite un corps non-JSON (la SEC rend du
    HTML sur ses 404), `boom_statut` imite un refus (403/404) vu par
    raise_for_status."""

    def __init__(self, payload=None, boom_json=False, boom_statut=None):
        self._payload = payload
        self._boom_json = boom_json
        self._boom_statut = boom_statut
        self.status_code = 200 if boom_statut is None else boom_statut
        self.raise_for_status_appele = False

    def json(self):
        if self._boom_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload

    def raise_for_status(self):
        self.raise_for_status_appele = True
        if self._boom_statut is not None:
            raise RuntimeError(f"HTTP {self._boom_statut} de la SEC")


class FauxRequests:
    """Substitut du module `requests`, pose sur le module teste. On remplace le
    MODULE dans l'espace de noms du module teste, jamais `requests.get` lui-meme :
    le vrai client reste intact pour le reste de la suite."""

    def __init__(self, reponse):
        self._reponse = reponse
        self.appels = []

    def get(self, url, headers=None, timeout=None, **kw):
        self.appels.append({"url": url, "headers": headers, "timeout": timeout})
        if callable(self._reponse):
            return self._reponse(url)
        return self._reponse

    @property
    def dernier(self):
        return self.appels[-1]


def ligne_duree(annee, val, filed=None, form="10-K", debut=None, fin=None):
    """Fait XBRL de FLUX (chiffre d'affaires, resultat) : porte start ET end."""
    return {"form": form, "start": debut or f"{annee}-01-01",
            "end": fin or f"{annee}-12-31", "val": val,
            "filed": filed or f"{annee + 1}-02-15"}


def ligne_instant(annee, val, filed=None, form="10-K"):
    """Fait XBRL de BILAN (dette, tresorerie) : un point, pas de duree."""
    return {"form": form, "end": f"{annee}-12-31", "val": val,
            "filed": filed or f"{annee + 1}-02-15"}


def faits(tags, ns="us-gaap", unite="USD"):
    """Reponse companyfacts reduite : {tag: [lignes]} dans un espace de noms."""
    return {"facts": {ns: {t: {"units": {unite: lignes}}
                           for t, lignes in tags.items()}}}


def fusionner(*blocs):
    out = {"facts": {}}
    for b in blocs:
        for ns, tags in b["facts"].items():
            out["facts"].setdefault(ns, {}).update(tags)
    return out


def depots(*lignes):
    """Reponse submissions/CIK##########.json : listes PARALLELES, depot le plus
    recent d'abord — c'est l'ordre que la SEC garantit."""
    return {"filings": {"recent": {
        "form": [x[0] for x in lignes],
        "primaryDocument": [x[1] for x in lignes],
        "accessionNumber": [x[2] for x in lignes],
        "filingDate": [x[3] for x in lignes]}}}


@pytest.fixture(autouse=True)
def _caches_sec_vides():
    """Les quatre fonctions reseau sont memoisees (lru_cache). Sans purge, un test
    servirait la reponse bouchonnee du precedent — et la memoisation survivrait a
    la fin du fichier, empoisonnant le reste de la suite."""
    def vider():
        for module, noms in ((edgar, ("_ticker_map", "get_facts")),
                             (sf, ("submission_meta", "annual_report_docs"))):
            for nom in noms:
                fn = getattr(module, nom, None)
                if hasattr(fn, "cache_clear"):
                    fn.cache_clear()
    vider()
    yield
    vider()


@pytest.fixture
def recharge_sec():
    """Recharge edgar + sec_fundamentals avec une valeur donnee de
    QUANTBENCH_SEC_UA, puis remet les deux modules dans leur etat d'origine.

    `_UA` est evalue A L'IMPORT : sans rechargement, aucune variable
    d'environnement posee par un test n'a le moindre effet, et le test passerait
    quoi qu'il arrive."""
    ancien = os.environ.get("QUANTBENCH_SEC_UA")

    def _recharge(valeur):
        if valeur is None:
            os.environ.pop("QUANTBENCH_SEC_UA", None)
        else:
            os.environ["QUANTBENCH_SEC_UA"] = valeur
        importlib.reload(edgar)
        importlib.reload(sf)          # il recopie edgar._UA a l'import
        return edgar, sf

    yield _recharge

    if ancien is None:
        os.environ.pop("QUANTBENCH_SEC_UA", None)
    else:
        os.environ["QUANTBENCH_SEC_UA"] = ancien
    importlib.reload(edgar)
    importlib.reload(sf)


# --------------------------------------------------------------------------- #
# 1. Le User-Agent SEC : le defaut qui vidait les fiches sans faire echouer un job
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("valeur", ["", None])
def test_un_secret_non_defini_ne_donne_pas_un_user_agent_vide(recharge_sec, valeur):
    """Un secret GitHub non defini devient la CHAINE VIDE, pas une variable
    absente. `os.environ.get(cle, defaut)` ne rend donc JAMAIS le defaut dans
    l'integration continue : le User-Agent partait vide et la SEC repondait 403,
    sans qu'aucun job n'echoue — `annual_report_docs` avale l'exception."""
    ed, _ = recharge_sec(valeur)
    ua = ed._UA["User-Agent"]
    assert ua, "User-Agent vide : la SEC repondra 403 sur chaque appel"
    assert ua.strip(), "User-Agent fait de blancs : la SEC le refuse aussi"


def test_un_user_agent_explicite_est_transmis_tel_quel(recharge_sec):
    """Le repli ne doit pas ecraser le contact reel. La SEC demande un identifiant
    joignable : le secret, quand il existe, est la seule chose qui la satisfait."""
    ed, _ = recharge_sec("QuantBench/1.0 samuel@example.com")
    assert ed._UA["User-Agent"] == "QuantBench/1.0 samuel@example.com"


def test_le_repli_evite_la_forme_que_la_sec_refuse(recharge_sec):
    """Le repli d'origine — « QuantBench research tool (contact via GitHub
    issues) » — etait lui-meme refuse en 403, alors que la meme chaine sans la
    parenthese passait. Un repli qui echoue n'est pas un repli : le corriger sans
    le tester laissait la porte ouverte a sa reintroduction."""
    ed, _ = recharge_sec(None)
    repli = ed._UA["User-Agent"]
    assert "(" not in repli and ")" not in repli, \
        f"repli parenthese, refuse en 403 par la SEC : {repli!r}"
    assert "QuantBench" in repli, "le repli n'identifie plus l'outil aupres de la SEC"


def test_le_correctif_du_user_agent_atteint_aussi_sec_fundamentals(recharge_sec):
    """`sec_fundamentals` fait ses PROPRES appels HTTP (submissions, documents) et
    recopie `edgar._UA` a l'import. Corriger le User-Agent dans edgar seul aurait
    laisse la moitie des appels SEC partir avec l'ancien en-tete."""
    ed, secf = recharge_sec("")
    assert secf._UA["User-Agent"], "sec_fundamentals appelle la SEC sans User-Agent"
    assert secf._UA == ed._UA, "les deux modules n'envoient pas le meme en-tete"


def test_tout_appel_a_la_sec_porte_le_user_agent_et_un_delai_de_garde(monkeypatch):
    """Les quatre points d'entree reseau doivent porter l'en-tete ET un timeout.
    Sans en-tete c'est un 403 ; sans timeout, un shard du build quotidien reste
    pendu jusqu'au plafond de six heures et meurt sans rien publier."""
    faux_ed = FauxRequests(Reponse({"0": {"cik_str": 320193, "ticker": "AAPL"}}))
    faux_sf = FauxRequests(Reponse({"sic": "3571", "name": "Apple Inc."}))
    monkeypatch.setattr(edgar, "requests", faux_ed)
    monkeypatch.setattr(sf, "requests", faux_sf)

    edgar.get_cik("AAPL")
    monkeypatch.setattr(edgar, "requests",
                        FauxRequests(Reponse({"facts": {}})), raising=True)
    edgar.get_facts("0000320193")
    sf.submission_meta("0000320193")

    appels = faux_ed.appels + edgar.requests.appels + faux_sf.appels
    assert len(appels) == 3
    for appel in appels:
        ua = (appel["headers"] or {}).get("User-Agent")
        assert ua and ua.strip(), f"appel sans User-Agent : {appel['url']}"
        assert appel["timeout"] and appel["timeout"] > 0, \
            f"appel sans delai de garde : {appel['url']}"


# --------------------------------------------------------------------------- #
# 2. Resolution du CIK : dix chiffres, sinon 404
# --------------------------------------------------------------------------- #
def test_le_cik_est_complete_a_dix_chiffres(monkeypatch):
    """Les URL SEC sont `CIK##########.json`, sur DIX chiffres. Un CIK non
    complete (320193 au lieu de 0000320193) ne leve rien : il rend un 404, donc
    une fiche sans comptes et sans documents."""
    monkeypatch.setattr(edgar, "requests", FauxRequests(
        Reponse({"0": {"cik_str": 320193, "ticker": "AAPL"},
                 "1": {"cik_str": 1045810, "ticker": "NVDA"}})))
    assert edgar.get_cik("AAPL") == "0000320193"
    assert edgar.get_cik("NVDA") == "0001045810"
    assert all(len(edgar.get_cik(t)) == 10 for t in ("AAPL", "NVDA"))


def test_le_ticker_est_insensible_a_la_casse(monkeypatch):
    """L'univers arrive en minuscules d'une source, en majuscules d'une autre. Une
    resolution sensible a la casse aurait fait passer la moitie de l'univers pour
    « entreprise non americaine »."""
    monkeypatch.setattr(edgar, "requests", FauxRequests(
        Reponse({"0": {"cik_str": 320193, "ticker": "aapl"}})))
    assert edgar.get_cik("AAPL") == edgar.get_cik("aapl") == "0000320193"


def test_un_ticker_absent_de_l_index_leve_une_erreur_explicite(monkeypatch):
    """Un ticker hors perimetre SEC doit lever KeyError EN NOMMANT le ticker :
    l'appelant s'en sert pour router la societe vers l'autre source de donnees.
    Rendre None ou une chaine vide l'aurait envoye chercher CIKNone.json."""
    monkeypatch.setattr(edgar, "requests", FauxRequests(
        Reponse({"0": {"cik_str": 320193, "ticker": "AAPL"}})))
    with pytest.raises(KeyError) as err:
        edgar.get_cik("SHOP.TO")
    assert "SHOP.TO" in str(err.value)


def test_un_refus_de_la_sec_ne_se_deguise_pas_en_ticker_introuvable(monkeypatch):
    """Distinction VITALE : « ticker inconnu » se rattrape (societe non
    americaine), « la SEC nous refuse » ne se rattrape pas. Sans
    `raise_for_status`, un 403 d'index rendrait un dictionnaire vide et TOUT
    l'univers passerait pour non americain — un build entier sans un seul CIK,
    et pas une erreur."""
    # Corps lisible ET statut 403 : seul `raise_for_status` distingue les deux.
    monkeypatch.setattr(edgar, "requests", FauxRequests(
        Reponse({"0": {"cik_str": 320193, "ticker": "AAPL"}}, boom_statut=403)))
    with pytest.raises(Exception) as err:
        edgar.get_cik("AAPL")
    assert not isinstance(err.value, KeyError), \
        "un refus de la SEC est confondu avec un ticker inconnu"
    assert "403" in str(err.value)


# --------------------------------------------------------------------------- #
# 3. Series annuelles : les chiffres faux ne levent aucune exception
# --------------------------------------------------------------------------- #
def test_seuls_les_rapports_annuels_alimentent_la_serie():
    """Le filtre de forme n'est pas redondant avec le filtre de duree. Un 10-Q
    porte des periodes GLISSANTES de douze mois — que la fenetre 350-380 jours
    laisse passer — et surtout des postes de bilan INSTANTANES, que le mode
    'instant' n'examine pas du tout. Sans le filtre de forme, la tresorerie de fin
    d'exercice devient celle du 31 mars, et le DCF tourne sur un bilan de
    trimestre sans qu'aucune erreur ne soit levee."""
    f = faits({"Revenues": [
        ligne_duree(2023, 100.0),
        ligne_duree(2024, 120.0),
        # douze mois glissants publies dans un 10-Q : duree parfaitement annuelle
        {"form": "10-Q", "start": "2024-04-01", "end": "2025-03-31",
         "val": 131.0, "filed": "2025-04-30"}]})
    assert edgar.annual_series(f, "Revenues") == [("2023-12-31", 100.0),
                                                  ("2024-12-31", 120.0)]

    bilan = faits({"CashAndCashEquivalentsAtCarryingValue": [
        ligne_instant(2024, 50.0),
        {"form": "10-Q", "end": "2025-03-31", "val": 12.0, "filed": "2025-04-30"}]})
    assert edgar.annual_series(bilan, edgar.TAGS["cash"], "instant") == [
        ("2024-12-31", 50.0)], "un bilan de trimestre est entre dans la serie annuelle"


@pytest.mark.parametrize("jours,retenu", [
    (364, True),      # exercice civil
    (371, True),      # exercice de 53 semaines (retail, Apple)
    (350, True), (380, True),                      # bornes admises
    (349, False), (381, False),                    # juste au-dela
    (92, False),      # trimestre depose dans un 10-K
    (730, False),     # cumul de deux exercices
])
def test_un_exercice_partiel_ne_passe_pas_pour_un_exercice_complet(jours, retenu):
    """Le 10-K contient aussi des periodes qui ne sont pas l'exercice : trimestres,
    cumuls, periodes de transition. Retenue comme annuelle, une periode de 92
    jours fait chuter le chiffre d'affaires de 75 % — et le DCF valorise une
    societe qui n'existe pas."""
    from datetime import date, timedelta
    fin = date(2024, 12, 31)
    debut = fin - timedelta(days=jours)
    f = faits({"Revenues": [{"form": "10-K", "start": debut.isoformat(),
                             "end": fin.isoformat(), "val": 100.0,
                             "filed": "2025-02-15"}]})
    serie = edgar.annual_series(f, "Revenues")
    assert bool(serie) is retenu, f"periode de {jours} jours mal jugee : {serie}"


def test_le_depot_le_plus_recent_l_emporte_sur_la_meme_cloture():
    """Une meme cloture est publiee plusieurs fois : d'abord dans son 10-K, puis en
    comparatif dans les suivants — parfois RETRAITEE. Garder la premiere valeur
    rencontree, c'est publier un chiffre que l'emetteur a lui-meme corrige."""
    f = faits({"Revenues": [
        ligne_duree(2023, 100.0, filed="2024-02-15"),
        ligne_duree(2023, 97.5, filed="2026-02-15"),      # retraitement
        ligne_duree(2023, 99.0, filed="2025-02-15")]})
    assert edgar.annual_series(f, "Revenues") == [("2023-12-31", 97.5)]


def test_les_tags_candidats_sont_fusionnes_et_non_classes():
    """Les emetteurs CHANGENT de tag en cours de route (NVDA passe de
    RevenueFromContractWithCustomer... a Revenues). Prendre le premier tag non
    vide de la liste de preference rendait la serie de l'ANCIEN tag : des comptes
    perimes de plusieurs exercices, complets et credibles."""
    f = faits({
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            ligne_duree(2021, 60.0), ligne_duree(2022, 70.0)],
        "Revenues": [ligne_duree(2023, 100.0), ligne_duree(2024, 120.0)]})
    serie = edgar.annual_series(f, edgar.TAGS["revenue"])
    assert [d for d, _ in serie] == ["2021-12-31", "2022-12-31",
                                     "2023-12-31", "2024-12-31"]
    assert edgar.latest(f, edgar.TAGS["revenue"]) == 120.0, \
        "la valeur la plus recente vient du tag abandonne"


def test_un_poste_de_bilan_se_lit_en_instantane():
    """Un poste de bilan n'a pas de duree : filtre comme un flux, il disparait
    entierement — dette a zero, tresorerie a zero, et une valeur d'entreprise
    fausse dans le meme sens pour tout l'univers."""
    f = faits({"CashAndCashEquivalentsAtCarryingValue": [
        ligne_instant(2023, 40.0), ligne_instant(2024, 50.0)]})
    assert edgar.annual_series(f, edgar.TAGS["cash"], "instant") == [
        ("2023-12-31", 40.0), ("2024-12-31", 50.0)]
    assert edgar.annual_series(f, edgar.TAGS["cash"], "duration") == []


def test_les_faits_hors_us_gaap_sont_lus():
    """Le nombre d'actions vit dans l'espace de noms `dei`, pas `us-gaap`. Ne
    chercher que dans us-gaap le rendait introuvable — donc pas de
    capitalisation, donc aucun upside calculable."""
    f = faits({"EntityCommonStockSharesOutstanding": [ligne_instant(2024, 1.5e10)]},
              ns="dei", unite="shares")
    assert edgar.latest(f, edgar.TAGS["shares"], "instant") == 1.5e10


def test_latest_rend_le_defaut_sur_une_serie_vide():
    """`latest` sert de garde-fou pour la tresorerie (defaut 0.0). S'il levait sur
    une serie vide, une societe sans poste de tresorerie balise ferait tomber sa
    fiche entiere."""
    vide = faits({})
    assert edgar.latest(vide, "Revenues") is None
    assert edgar.latest(vide, "Revenues", default=0.0) == 0.0
    assert edgar.annual_series(vide, edgar.TAGS["revenue"]) == []


def test_la_dette_totale_somme_les_composantes_a_defaut_du_poste_global():
    """Beaucoup d'emetteurs ne balisent pas `LongTermDebt` mais ses composantes.
    Sans le repli, ils ressortaient SANS DETTE : valeur d'entreprise egale a la
    valeur des fonds propres, et un upside gonfle de tout leur endettement."""
    global_ = faits({"LongTermDebt": [ligne_instant(2024, 95.0)],
                     "LongTermDebtNoncurrent": [ligne_instant(2024, 80.0)]})
    assert edgar.total_debt(global_) == 95.0, \
        "le poste global doit primer sur la somme des composantes"

    composantes = faits({
        "LongTermDebtNoncurrent": [ligne_instant(2024, 80.0)],
        "LongTermDebtCurrent": [ligne_instant(2024, 10.0)],
        "ShortTermBorrowings": [ligne_instant(2024, 5.0)]})
    assert edgar.total_debt(composantes) == 95.0

    assert edgar.total_debt(faits({})) == 0.0


# --------------------------------------------------------------------------- #
# 4. Documents SEC : `annual_report_docs` avale tout — la structure doit tenir
# --------------------------------------------------------------------------- #
_STRUCTURE_VIDE = {"ars_pdf": None, "tenk": None, "documents": []}


@pytest.mark.parametrize("reponse", [
    "reseau",                                        # requests.get leve
    Reponse(boom_json=True),                         # 404 SEC : corps HTML
    Reponse({}),                                     # JSON sans "filings"
    Reponse({"filings": {}}),                        # sans "recent"
    Reponse({"filings": {"recent": {}}}),            # sans "form"
    Reponse({"filings": {"recent": {"form": ["10-K"]}}}),   # sans primaryDocument
    Reponse(None),                                   # corps JSON nul
])
def test_un_echec_sec_ne_propage_jamais_d_exception(monkeypatch, reponse):
    """`annual_report_docs` est appelee pour CHAQUE fiche du build quotidien. Une
    exception qui remonte fait perdre la fiche entiere — valorisation comprise —
    pour un simple bloc de liens documentaires. Le contrat est : elle ne leve
    jamais, et rend toujours la meme structure."""
    if reponse == "reseau":
        def get(url, **kw):
            raise OSError("connexion reinitialisee par le pair")
        faux = type("F", (), {"get": staticmethod(get)})()
    else:
        faux = FauxRequests(reponse)
    monkeypatch.setattr(sf, "requests", faux)
    assert sf.annual_report_docs("0000320193") == _STRUCTURE_VIDE


def test_la_structure_de_repli_est_celle_que_l_appelant_lit(monkeypatch):
    """L'appelant fait `ard.get("tenk")`, `ard.get("ars_pdf")` et
    `ard.get("documents", [])`, et la page fait `docs.length`. Un repli qui
    omettrait `documents`, ou le rendrait a None, deplacerait simplement la panne
    d'un module a l'autre — la ou plus personne ne l'attrape."""
    monkeypatch.setattr(sf, "requests", FauxRequests(Reponse(boom_json=True)))
    echec = sf.annual_report_docs("0000320193")

    sf.annual_report_docs.cache_clear()
    monkeypatch.setattr(sf, "requests", FauxRequests(Reponse(depots(
        ("10-K", "aapl-20240928.htm", "0000320193-24-000123", "2024-11-01")))))
    nominal = sf.annual_report_docs("0000320193")

    assert set(echec) == set(nominal) == {"ars_pdf", "tenk", "documents"}
    assert isinstance(echec["documents"], list)
    assert echec["ars_pdf"] is None and echec["tenk"] is None


@pytest.mark.parametrize("cik", ["0000000000", "0", "", "00"])
def test_un_cik_nul_ne_fait_pas_tomber_la_fiche(monkeypatch, cik):
    """Le fournisseur rend la sentinelle « 0000000000 » pour toute societe hors
    perimetre SEC — tout le TSX, la quasi-totalite du gre a gre : 88 fiches sur
    387. Une chaine de zeros etant VRAIE en Python, la garde `if cik` ne la voit
    pas et l'appel part quand meme.

    Ce que ce test verrouille : cet appel condamne d'avance ne fait tomber
    aucune fiche. Il ne tient aujourd'hui que parce que la SEC refuse
    CIK0000000000.json — `int(cik)`, lui, est HORS du try (voir le rapport)."""
    monkeypatch.setattr(sf, "requests", FauxRequests(Reponse(boom_json=True)))
    assert sf.annual_report_docs(cik) == _STRUCTURE_VIDE


def test_un_cik_vide_part_sur_le_fil_comme_la_sentinelle(monkeypatch):
    """`"".zfill(10)` vaut « 0000000000 » : un CIK vide devient la sentinelle et
    consomme un appel SEC (~10 req/s partages par tout le build). Le test fige le
    fait observable ; le rapport dit ou la garde manque."""
    faux = FauxRequests(Reponse(boom_json=True))
    monkeypatch.setattr(sf, "requests", faux)
    sf.annual_report_docs("")
    assert faux.dernier["url"].endswith("CIK0000000000.json")


def test_les_liens_sec_suivent_le_format_des_archives(monkeypatch):
    """Deux formats coexistent et ne sont PAS interchangeables : l'API veut un CIK
    sur dix chiffres, l'archive veut le CIK sans zeros et le numero d'accession
    SANS TIRETS. Se tromper de format rend un lien 404 sur une fiche par
    ailleurs impeccable — l'erreur la moins visible qui soit."""
    faux = FauxRequests(Reponse(depots(
        ("10-K", "aapl-20240928.htm", "0000320193-24-000123", "2024-11-01"))))
    monkeypatch.setattr(sf, "requests", faux)
    out = sf.annual_report_docs("0000320193")

    assert faux.dernier["url"] == \
        "https://data.sec.gov/submissions/CIK0000320193.json"
    assert out["tenk"] == ("https://www.sec.gov/Archives/edgar/data/320193/"
                           "000032019324000123/aapl-20240928.htm")
    assert "-" not in out["tenk"].rsplit("/", 2)[1], \
        "numero d'accession avec tirets : lien 404"


def test_seul_un_ars_en_pdf_devient_le_rapport_glossy(monkeypatch):
    """Le bloc « rapport annuel officiel » de la page annonce un PDF. Un ARS
    depose en HTML derriere une pastille « PDF » est un lien qui ment ; et le
    premier 10-K rencontre est le plus recent, la SEC listant ses depots du plus
    recent au plus ancien."""
    faux = FauxRequests(Reponse(depots(
        ("ARS", "resume-annuel.htm", "0000320193-25-000200", "2025-01-10"),
        ("10-K", "recent-10k.htm", "0000320193-24-000123", "2024-11-01"),
        ("ARS", "rapport-glossy.pdf", "0000320193-24-000124", "2024-11-02"),
        ("10-K", "vieux-10k.htm", "0000320193-23-000100", "2023-11-01"))))
    monkeypatch.setattr(sf, "requests", faux)
    out = sf.annual_report_docs("0000320193")

    assert out["ars_pdf"].endswith("rapport-glossy.pdf")
    assert out["tenk"].endswith("recent-10k.htm"), "le 10-K retenu n'est pas le dernier"
    assert all(d["is_pdf"] == d["url"].lower().endswith(".pdf")
               for d in out["documents"])


def test_la_liste_des_documents_est_bornee_et_filtree(monkeypatch):
    """Deux erreurs opposees a eviter : publier les 1 000 depots recents d'une
    grande capitalisation dans une fiche (formulaires 4, SC 13G, ATS-N...), ou
    laisser passer un formulaire sans libelle, qui s'affiche a l'utilisateur sous
    son code brut. Le bloc se limite aux formes documentaires connues, plafonnees
    a huit."""
    # Les formes a ecarter sont EN TETE : chez une grande capitalisation, les
    # depots les plus recents sont des formulaires 4 (declarations de dirigeants).
    # Placees en fin de liste, elles seraient masquees par le seul plafond.
    lignes = [("4", f"form4-{i}.xml", f"0000320193-24-{i:06d}", "2024-06-10")
              for i in range(9)]
    lignes += [("SC 13G", "sc13g.htm", "0000320193-24-000900", "2024-06-09")]
    lignes += [("8-K", f"comm{i}.htm", f"0000320193-24-{i + 100:06d}", "2024-06-01")
               for i in range(30)]
    faux = FauxRequests(Reponse(depots(*lignes)))
    monkeypatch.setattr(sf, "requests", faux)
    out = sf.annual_report_docs("0000320193")

    assert len(out["documents"]) == 8
    assert {d["form"] for d in out["documents"]} == {"8-K"}
    for d in out["documents"]:
        assert set(d) == {"form", "label", "date", "url", "is_pdf"}
        assert d["label"] == sf._FORM_LABEL[d["form"]]
        assert d["date"] == "2024-06-01"


def test_un_depot_sans_document_principal_est_ignore(monkeypatch):
    """Certains depots n'ont pas de document principal (chaine vide). Construire
    l'URL quand meme donne un lien qui se termine par « / » : il pointe sur
    l'index du depot, pas sur le rapport, et la pastille annonce autre chose que
    ce qu'on ouvre."""
    faux = FauxRequests(Reponse(depots(
        ("10-K", "", "0000320193-24-000123", "2024-11-01"),
        ("10-K", "vrai-10k.htm", "0000320193-23-000100", "2023-11-01"))))
    monkeypatch.setattr(sf, "requests", faux)
    out = sf.annual_report_docs("0000320193")

    assert len(out["documents"]) == 1
    assert out["tenk"].endswith("vrai-10k.htm")


# --------------------------------------------------------------------------- #
# 5. Metadonnees d'emetteur : secteur et libelles
# --------------------------------------------------------------------------- #
def test_les_champs_absents_des_metadonnees_prennent_un_defaut_du_bon_type(monkeypatch):
    """`sicDescription` alimente le champ « industrie » de la fiche et `exchanges`
    est parcouru par l'appelant. Rendre None la ou une chaine ou une liste est
    attendue transforme une metadonnee manquante en TypeError a l'affichage."""
    monkeypatch.setattr(sf, "requests", FauxRequests(Reponse({"name": "Truc Inc."})))
    meta = sf.submission_meta("0000320193")

    assert set(meta) == {"sic", "sicDescription", "name", "exchanges", "tickers"}
    assert meta["sicDescription"] == "" and meta["sic"] is None
    assert meta["exchanges"] == [] and meta["tickers"] == []


def test_un_refus_http_sur_les_metadonnees_ne_passe_pas_inapercu(monkeypatch):
    """Contrairement aux documents, les metadonnees portent le SECTEUR — donc la
    METHODE de valorisation. Les rendre vides en silence sur un 403 ferait
    valoriser une banque en DCF classique. Ici, l'erreur doit remonter."""
    # Le corps du 403 reste lisible : sans `raise_for_status`, la fonction rendrait
    # des metadonnees d'apparence normale tirees d'une page de refus.
    monkeypatch.setattr(sf, "requests", FauxRequests(
        Reponse({"name": "Refus", "sic": "6021"}, boom_statut=403)))
    with pytest.raises(Exception):
        sf.submission_meta("0000320193")


@pytest.mark.parametrize("sic,attendu", [
    (6021, "Financial Services"), ("6021", "Financial Services"),
    (6000, "Financial Services"), (6799, "Financial Services"),
    (5999, "Other"), (6800, "Other"),
    (1311, "Energy"), (1300, "Energy"), (1399, "Energy"),
    (2911, "Energy"), (2900, "Energy"), (2999, "Energy"),
    (1000, "Basic Materials"), (1299, "Basic Materials"),
    (1400, "Basic Materials"), (1499, "Basic Materials"),
    (3310, "Basic Materials"), (3399, "Basic Materials"),
    (7372, "Other"), (3571, "Other"),
])
def test_le_code_sic_decide_du_secteur(sic, attendu):
    """Le secteur commande la METHODE (route.classify) : une banque au DCF FCFF ou
    un petrolier valorise comme un logiciel donnent des upsides a trois chiffres.
    Les bornes comptent autant que le centre — 6799 est une financiere, 6800 ne
    l'est plus."""
    assert sf.sic_to_sector(sic) == attendu


@pytest.mark.parametrize("sic", [None, "", "N/A", "abc", [], {}])
def test_un_code_sic_illisible_ne_fait_pas_tomber_le_build(sic):
    """Le code SIC manque ou arrive sale pour une partie de l'univers. `int(sic)`
    non garde leve TypeError ou ValueError en plein milieu d'un shard de 3 400
    titres — pour un champ purement descriptif."""
    assert sf.sic_to_sector(sic) == "Unknown"


# --------------------------------------------------------------------------- #
# 6. Comptes annuels : l'alignement des exercices
# --------------------------------------------------------------------------- #
def _comptes(annees=(2022, 2023, 2024), **extra):
    tags = {
        "Revenues": [ligne_duree(a, 100.0 + a) for a in annees],
        "NetIncomeLoss": [ligne_duree(a, 10.0 + a) for a in annees],
        "Assets": [ligne_instant(a, 500.0 + a) for a in annees],
    }
    tags.update(extra)
    return faits(tags)


def test_les_metriques_sont_alignees_sur_les_memes_exercices():
    """CRUCIAL. Toutes les metriques doivent decrire les MEMES exercices : les
    ratios forensiques comparent l'indice 0 a l'indice 1. Si le chiffre
    d'affaires porte 2024-2022 et l'actif 2023-2021, le M-Score compare 2024 a
    2022 et le Z-Score melange deux exercices — des scores faux, plausibles, et
    parfaitement silencieux."""
    F = sf.get_financials(_comptes(
        AssetsCurrent=[ligne_instant(2023, 200.0), ligne_instant(2024, 210.0)],
        LiabilitiesCurrent=[ligne_instant(2023, 100.0), ligne_instant(2024, 90.0)],
        Liabilities=[ligne_instant(2024, 300.0)]))

    n = len(F["years"])
    assert n == 3
    for cle, valeurs in F.items():
        assert len(valeurs) == n, f"{cle} n'est pas aligne sur les exercices"
    # Un exercice non publie pour une metrique reste un TROU, pas un decalage.
    assert F["current_assets"] == [210.0, 200.0, None]
    assert F["total_liab"] == [300.0, None, None]


def test_l_exercice_le_plus_recent_est_en_tete():
    """Toute la chaine aval lit l'indice 0 comme « le dernier exercice » :
    `fiscal_year = F["years"][0]`, et la forensique compare 0 a 1. Une serie
    triee dans l'autre sens inverserait toutes les tendances — croissance lue
    comme declin — sans jamais lever."""
    F = sf.get_financials(_comptes())
    assert F["years"] == ["2024-12-31", "2023-12-31", "2022-12-31"]
    assert F["revenue"][0] > F["revenue"][1], "la serie n'est pas anti-chronologique"


@pytest.mark.parametrize("cas,tags", [
    ("un seul exercice commun", {
        "Revenues": [ligne_duree(2024, 100.0)],
        "NetIncomeLoss": [ligne_duree(2024, 10.0)],
        "Assets": [ligne_instant(2024, 500.0)]}),
    ("aucune intersection", {
        "Revenues": [ligne_duree(2023, 100.0), ligne_duree(2024, 120.0)],
        "NetIncomeLoss": [ligne_duree(2021, 10.0), ligne_duree(2022, 11.0)],
        "Assets": [ligne_instant(2023, 500.0), ligne_instant(2024, 520.0)]}),
    ("actif total absent", {
        "Revenues": [ligne_duree(2023, 100.0), ligne_duree(2024, 120.0)],
        "NetIncomeLoss": [ligne_duree(2023, 10.0), ligne_duree(2024, 12.0)]}),
    ("rien du tout", {}),
])
def test_des_comptes_insuffisants_rendent_none_et_non_une_coquille(cas, tags):
    """Renoncer explicitement vaut mieux que rendre une structure a un element :
    la forensique indexe [1] et l'appelant lit `F["years"][0]`. Une coquille
    aurait produit des scores calcules sur un seul exercice, donc des tendances
    inventees."""
    assert sf.get_financials(faits(tags)) is None, cas


def test_toutes_les_cles_lues_par_la_forensique_sont_produites():
    """La forensique attrape TypeError, ZeroDivisionError et IndexError — mais PAS
    KeyError. Une metrique retiree de `_F_TAGS` ne rendrait donc pas un score
    None : elle ferait remonter une KeyError depuis le M-Score jusqu'a la fiche."""
    F = sf.get_financials(_comptes())
    lues = {"revenue", "cogs", "gross_profit", "ebit", "sga", "net_income",
            "total_assets", "current_assets", "current_liab", "net_ppe",
            "receivables", "inventory", "long_term_debt", "total_debt",
            "retained_earnings", "equity", "total_liab", "shares", "cfo",
            "dep_amort", "working_capital", "years"}
    assert lues <= set(F), f"cles absentes de la structure : {sorted(lues - set(F))}"


def test_le_fonds_de_roulement_manquant_reste_inconnu_et_non_zero():
    """Le fonds de roulement pese 6,56 dans le Z''-Score d'Altman — le plus gros
    coefficient. Le mettre a zero faute de donnee n'est pas neutre : c'est
    affirmer un fonds de roulement NUL, et pousser vers la detresse une societe
    dont on ignore simplement l'actif courant."""
    F = sf.get_financials(_comptes(
        AssetsCurrent=[ligne_instant(2024, 210.0), ligne_instant(2023, 200.0)],
        LiabilitiesCurrent=[ligne_instant(2024, 90.0)]))
    assert F["working_capital"][0] == 120.0
    assert F["working_capital"][1] is None, \
        "un poste manquant est devenu un fonds de roulement chiffre"
    assert F["working_capital"][2] is None


# --------------------------------------------------------------------------- #
# 7. Fondamentaux : unites et denominateurs
# --------------------------------------------------------------------------- #
@pytest.fixture
def societe_sec(monkeypatch):
    """Cable une societe SEC complete sans reseau : CIK, faits XBRL, metadonnees,
    prix et beta."""
    etat = {"prix": 200.0, "beta": 1.25, "sic": 3571}

    def _monter(tags=None, **kw):
        etat.update(kw)
        base = {
            "Revenues": [ligne_duree(2023, 100e9), ligne_duree(2024, 120e9)],
            "OperatingIncomeLoss": [ligne_duree(2024, 30e9)],
            "NetIncomeLoss": [ligne_duree(2024, 24e9)],
            "StockholdersEquity": [ligne_instant(2024, 60e9)],
            "LongTermDebt": [ligne_instant(2024, 90e9)],
            "CashAndCashEquivalentsAtCarryingValue": [ligne_instant(2024, 30e9)],
        }
        if tags:
            base.update(tags)
        f = fusionner(
            faits(base),
            faits({"EntityCommonStockSharesOutstanding": [ligne_instant(2024, 15e9)]},
                  ns="dei", unite="shares"))
        monkeypatch.setattr(sf.edgar, "get_cik", lambda t: "0000320193")
        monkeypatch.setattr(sf.edgar, "get_facts", lambda c: f)
        monkeypatch.setattr(sf, "submission_meta", lambda c: {
            "sic": etat["sic"], "sicDescription": "Electronic Computers",
            "name": "Apple Inc.", "exchanges": ["Nasdaq"], "tickers": ["AAPL"]})

        def prix(t):
            if etat["prix"] is None:
                raise RuntimeError("cotation indisponible")
            return etat["prix"]

        def beta(t):
            if etat["beta"] is None:
                raise RuntimeError("historique insuffisant")
            return etat["beta"]

        monkeypatch.setattr(sf.market, "latest_price", prix)
        monkeypatch.setattr(sf.market, "levered_beta", beta)
        return sf.get_fundamentals("aapl")

    return _monter


def test_les_montants_sont_en_milliards_et_les_actions_en_unites(societe_sec):
    """La SEC publie en unites, la valorisation raisonne en MILLIARDS — sauf le
    nombre d'actions et le cours, qui restent en unites et en dollars. C'est
    exactement la ou un facteur 1e9 se loge sans se voir : la capitalisation est
    le seul endroit ou les deux echelles se rencontrent."""
    f = societe_sec()
    assert f["revenue"] == 120.0 and f["ebit"] == 30.0 and f["net_income"] == 24.0
    assert f["total_debt"] == 90.0 and f["cash"] == 30.0 and f["book_equity"] == 60.0
    assert f["revenue_history"] == [100.0, 120.0], "historique non chronologique"
    assert f["shares"] == 15e9, "le nombre d'actions a ete converti en milliards"
    assert f["price"] == 200.0
    assert f["market_cap"] == pytest.approx(200.0 * 15e9 / 1e9)
    assert f["ticker"] == "AAPL"


def test_un_beta_indisponible_ne_fait_pas_tomber_la_fiche(societe_sec):
    """Le beta vient de Yahoo, hors SEC : une societe recemment cotee n'a pas
    assez d'historique. Laisser remonter l'erreur ferait perdre des comptes
    complets pour un parametre de cout du capital qui a un repli raisonnable."""
    f = societe_sec(beta=None)
    assert f["beta"] == 1.1
    assert f["revenue"] == 120.0, "les comptes SEC ont ete perdus avec le beta"


def test_les_ratios_ne_se_calculent_pas_sur_un_denominateur_nul(societe_sec):
    """Marge d'exploitation et rentabilite des fonds propres divisent par des
    grandeurs qui valent legitimement zero (societe pre-revenu, fonds propres
    absorbes). Une division par zero ici ferait tomber la fiche ; un zero rendu
    silencieusement mentirait sur la rentabilite."""
    f = societe_sec(tags={
        "Revenues": [ligne_duree(2023, 0.0), ligne_duree(2024, 0.0)],
        "StockholdersEquity": [ligne_instant(2024, 0.0)]})
    assert f["operating_margin"] is None and f["roe"] is None


def test_la_capitalisation_manque_plutot_que_de_valoir_zero(societe_sec):
    """Sans cours ni nombre d'actions, la capitalisation est INCONNUE. La rendre a
    zero la ferait passer pour une societe sans valeur de marche — donc un upside
    infini face a n'importe quelle valeur intrinseque."""
    f = societe_sec(tags={"EntityCommonStockSharesOutstanding": []})
    assert f["shares"] is None and f["market_cap"] is None


def test_le_secteur_provient_du_code_sic_et_non_du_libelle(societe_sec):
    """Le secteur route la METHODE de valorisation. Il doit venir du code SIC
    (structure), pas du libelle textuel (libre) : `sicDescription` alimente le
    champ « industrie », rien d'autre."""
    f = societe_sec(sic=6021)
    assert f["sector"] == "Financial Services"
    assert f["industry"] == "Electronic Computers"
    assert f["sic"] == 6021

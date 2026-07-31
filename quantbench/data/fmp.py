"""
quantbench.data.fmp
===================
Couche de données UNIQUE via Financial Modeling Prep (API « stable », palier
Ultimate) : univers (US + Canada), fondamentaux (bulk), profils, historique de
prix, news. Fiable, non-Yahoo, couverture complète. Tout est ramené en USD.

Clé via la variable d'environnement FMP_API_KEY (jamais en dur dans le dépôt).
"""

from __future__ import annotations

import csv
import functools
import io
import os
import re

import requests

from ..series import serie_continue

# Actions privilégiées / notes / débentures émises par banques & sociétés (quasi-
# obligataire, pas des actions ordinaires) : à exclure de l'univers. Le modèle
# excess-return les valorise comme des capitaux propres -> upside aberrant.
# Prudence : NE PAS attraper les ADR ("American Depositary Shares" = ordinaire) ni
# les sociétés dont le nom contient "Preferred" (ex. Preferred Bank).
_PREF_NAME = re.compile(
    r"\bPFD\b|\bPREF(ERRED)?\s+(STOCK|SHARES|SHS|SEC|SERIES|EQUITY)"
    r"|PERPETUAL\s+(RED\s+)?PREF|CUM\s+PERP|\bRATE\s+RESET\b|\bRST\s+PFD\b"
    r"|\b(SUB(ORDINATED)?|JR|JUNIOR|SR|SENIOR)\s+(NOTES?|DEBENT)"
    r"|\bDEBENTURES?\b|\bTRUST\s+PREF"
    # Coupon : le "%" ne doit PAS exiger d'espace apres lui — les libelles se
    # terminent souvent par le taux. Six lignes obligataires entraient dans
    # l'univers pour ce seul caractere, dont Duke Energy pour 17,5 Md$.
    r"|\d+(\.\d+)?\s*%"
    # "JR SUB" et "SUB DB" sans mot suivant : le motif exigeait NOTES ou DEBENT
    # apres, ce que le libelle de DTE Energy ("JR SUB DB 2017 E") ne porte pas.
    r"|\bJR\s+SUB\b|\bSUB\s+DB\b"
    # Millesime de serie obligataire, dans les DEUX ordres rencontres : "2021
    # Series" comme "Series 2021". Attention a ne pas retenir "Series" seul, qui
    # apparie "Pattern Group Inc. Series A Common Stock" — une action ordinaire.
    r"|\b(19|20)\d\d\s+SERIES\b|\bSERIES\s+(19|20)\d\d\b", re.I)
# Privilégiées par ticker : canadiennes (TD-PFA.TO, ENB-PN.TO) ET américaines
# (FITB-PM, OAK-PB). Motif -P + 1-2 lettres de série. Les actions de CLASSE
# (BRK-B, BF-B, HEI-A) utilisent -A/-B/-C, jamais -P -> non touchées.
_PREF_TICKER = re.compile(r"-P[A-Z]{1,2}(\.(TO|V))?$")


# Un nom d'emetteur se terminant par un COUPON nu ("Prudential Financial, Inc.
# 5.62", "KKR Group Finance Co. IX LLC 4.") designe une obligation cotee : FMP
# tronque le libelle et le signe "%" disparait, si bien que le motif de coupon
# habituel ne s'applique plus.
_COUPON_FIN = re.compile(r"\s\d{1,2}\.\d*\s*$")



def _is_preferred(symbol, name):
    return bool(_PREF_TICKER.search(symbol)
                or (name and (_PREF_NAME.search(name) or _COUPON_FIN.search(name))))


# Instruments DERIVES de l'action, cotes sous leur propre ticker : un warrant est
# une option sur le titre, un "unit" un panier action + warrant vendu a
# l'introduction d'un SPAC, un "right" un droit de souscription. Aucun n'a pour
# contrepartie les comptes de la societe, et les valoriser sur ses fondamentaux n'a
# aucun sens — d'autant qu'ils heritent de la capitalisation du sous-jacent : les
# warrants de Churchill Capital XI portaient 2,3 Md$.
# Le NOM est le seul signal fiable. Le suffixe de ticker en emporterait des
# centaines de vraies societes : Palantir (PLTR), Palo Alto (PANW), Intuit (INTU),
# Baker Hughes (BKR) finissent tous par W, R ou U.
# "Right" et "Unit" ne sont retenus qu'en FIN de libelle, ou le mot designe
# l'instrument et non l'activite — sauf en tete de nom, ou il fait partie de la
# raison sociale (Unit Corporation, petrolier de l'Oklahoma).
# Verifie sur l'univers : aucune societe en commandite n'est touchee. Elles se
# nomment toutes "... LP" ou "... L.P." (Energy Transfer, Enterprise Products,
# Brookfield Infrastructure) et jamais "Units", alors meme que leur equite EST
# constituee de parts.
_DERIVE = re.compile(r"(?i)\bwarrants?\b|\bwts?\b|\bc/wts?\b|\brights?\s*$|\bunits?\b")
# "Rt" abrege un DROIT en anglais mais designe la forme sociale en hongrois
# (Reszvenytarsasag) : MOL Magyar Olaj-es Gazipari RT est le petrolier national.
# On ne l'exclut donc que si le ticker porte AUSSI la marque du droit.
_RT_ABREGE = re.compile(r"(?i)\brts?\s*$")


def _est_un_derive(symbol, name):
    if not name:
        return False
    # "Unit" en tete de nom appartient a la raison sociale : Unit Corporation est
    # un producteur petrolier de l'Oklahoma, pas un panier de titres.
    if name.lower().startswith("unit "):
        return False
    if _DERIVE.search(name):
        return True
    return bool(_RT_ABREGE.search(name)
                and re.search(r"(-R|R)$", (symbol or "").split(".")[0]))


def _sane_beta(b, sector=None):
    """Beta FMP fiable ? Les nano-caps illiquides donnent des betas aberrants
    (ex. −29) -> cout du capital negatif -> DCF qui explose. Hors [0.1, 3.5] =
    regression bidon -> on retombe sur le beta MEDIAN DU SECTEUR (mesure sur
    l'univers reel) plutot que sur un 1,1 identique pour tous : un service public
    et un editeur de logiciels n'ont pas le meme risque systematique."""
    b = _num(b)
    if b is not None and 0.1 <= b <= 3.5:
        return b
    # Repli sur le beta d'activite MESURE du secteur, via la source unique de
    # reperes. (Ce repli ne sert plus qu'aux rares cas ou le beta ascendant lui-meme
    # est indisponible : le moteur utilise desormais le beta d'industrie re-endette.)
    try:
        from ..valuation.build_universal import _STATS
        s = _STATS.get("secteurs", {}).get(sector or "")
        if s and s.get("beta_desendette"):
            return float(s["beta_desendette"])
    except Exception:
        pass
    return 1.1

_BASE = "https://financialmodelingprep.com/stable"
_TIMEOUT = 120


def _key():
    k = os.environ.get("FMP_API_KEY")
    if not k:
        raise RuntimeError("FMP_API_KEY manquante (variable d'environnement).")
    return k


# COMPTEUR DE BANDE PASSANTE, par point d'entree de l'API.
#
# Le forfait autorise 150 Go par mois et le build quotidien en consommait assez
# pour epuiser le quota avant la fin du mois. Personne ne savait quel appel
# telechargeait quoi : optimiser sans mesurer aurait consiste a deviner.
#
# `_get` est le passage OBLIGE de tous les appels FMP : un seul endroit a
# instrumenter. On compte les octets reellement recus (`len(r.content)`), pas une
# estimation, et on agrege par famille d'appel — le symbole varie, la famille non.
_OCTETS = {}
_OCTETS_VERROU = __import__("threading").Lock()


def _famille(path):
    """Nom d'appel sans le symbole ni les parametres : la famille de l'endpoint."""
    return path.split("?")[0]


def octets_consommes():
    """{famille: (nombre d'appels, octets)}. Copie, pour lecture concurrente sure."""
    with _OCTETS_VERROU:
        return dict(_OCTETS)


def reinitialiser_les_octets():
    with _OCTETS_VERROU:
        _OCTETS.clear()


def _sans_la_cle(exc, cle):
    """Meme exception, cle remplacee par des etoiles dans son message.

    `raise_for_status` construit son message a partir de l'URL COMPLETE : la cle y
    figure en clair. Ce message voyage ensuite loin — deux appelants historiques
    persistent `str(e)[:140]` dans le tableau `errors` de `screener.json`, fichier
    PUBLIE sur le site. Le prefixe du message fait environ 78 caracteres, ce qui
    laisse la place d'y ecrire une cle entiere. Rien n'a fuite a ce jour, mais le
    chemin etait ouvert, et il s'ouvre aussi dans toute trace locale.
    """
    if not cle:
        return exc
    msg = str(exc).replace(cle, "***")
    nouvelle = type(exc)(msg) if not isinstance(exc, requests.HTTPError) else \
        requests.HTTPError(msg, response=getattr(exc, "response", None))
    return nouvelle


def _get(path, retries=3):
    cle = _key()
    url = f"{_BASE}/{path}" + ("&" if "?" in path else "?") + f"apikey={cle}"
    try:
        return _get_brut(url, path, retries)
    except Exception as exc:
        raise _sans_la_cle(exc, cle) from None


def _get_brut(url, path, retries=3):
    import time
    for attempt in range(retries):
        r = requests.get(url, timeout=_TIMEOUT)
        # Compte AVANT le test de statut : une reponse 429 ou 500 consomme aussi
        # du transfert, et une boucle de reessai en consomme plusieurs fois.
        n = len(r.content or b"")
        with _OCTETS_VERROU:
            k, o = _OCTETS.get(_famille(path), (0, 0))
            _OCTETS[_famille(path)] = (k + 1, o + n)
        if r.status_code == 429:                    # rate limit -> backoff
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def statements(symbol, limit=6):
    """3 états annuels par ticker (endpoints à débit élevé). Retourne
    {income:{year:row}, balance:{year:row}, cashflow:{year:row}}."""
    out = {"income": {}, "balance": {}, "cashflow": {}}
    for ep, kind in (("income-statement", "income"),
                     ("balance-sheet-statement", "balance"),
                     ("cash-flow-statement", "cashflow")):
        try:
            for row in _json(f"{ep}?symbol={symbol}&period=annual&limit={limit}"):
                y = row.get("fiscalYear") or str(row.get("date", ""))[:4]
                try:
                    out[kind][int(y)] = row
                except (TypeError, ValueError):
                    continue
        except Exception:
            continue
    return out


def profile(symbol):
    try:
        j = _json(f"profile?symbol={symbol}")
        return j[0] if j else {}
    except Exception:
        return {}


def _json(path):
    return _get(path).json()


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Change → USD (via FMP forex)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=32)
def fx_to_usd(currency: str):
    if not currency or currency.upper() == "USD":
        return 1.0
    c = currency.upper()
    for sym, inv in ((f"{c}USD", False), (f"USD{c}", True)):
        try:
            j = _json(f"quote?symbol={sym}")
            p = _num(j[0].get("price")) if j else None
            if p and p > 0:
                return (1.0 / p) if inv else p
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
# Univers (screener)
# --------------------------------------------------------------------------- #
def _date_des_comptes(inc, bal):
    """Date de CLOTURE du dernier exercice publie (chaine AAAA-MM-JJ) ou None."""
    for src in (inc, bal):
        if src:
            d = str((src or {}).get(max(src), {}).get("date") or "")[:10]
            if len(d) == 10:
                return d
    return None


def _age_en_mois(date_cloture, reference=None):
    """Age en MOIS depuis la cloture. Compte en mois et non en millesimes : le
    millesime d'exercice est decale chez le fournisseur pour 3 a 4 % des lignes — un
    exercice clos le 30 juin 2024 est etiquete 2023 — et comparer des annees civiles
    cree une falaise au 1er janvier sans aucun sens economique.

    `reference` (AAAA-MM-JJ) remplace la date du jour. Le defaut sert la production,
    ou "maintenant" est la bonne question. Toute reconstitution historique doit en
    revanche demander l'age QU'AVAIENT les comptes a la date etudiee : compter depuis
    aujourd'hui vieillit tout l'univers du meme nombre de mois, ce qui sature la
    dimension de confiance dans la donnee et lui fait perdre le pouvoir de distinguer
    des comptes frais de comptes perimes — puis fait conclure, a tort, qu'elle
    n'apporte rien."""
    if not date_cloture:
        return None
    from datetime import datetime, timezone
    try:
        c = datetime.strptime(date_cloture, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    if reference:
        try:
            n = datetime.strptime(str(reference)[:10], "%Y-%m-%d").replace(
                tzinfo=timezone.utc)
        except ValueError:
            n = datetime.now(timezone.utc)
    else:
        n = datetime.now(timezone.utc)
    return (n.year - c.year) * 12 + (n.month - c.month) - (n.day < c.day)


@functools.lru_cache(maxsize=1)
def societes_radiees():
    """Tickers RADIES, avec leur date — le seul fait qui atteste qu'une ligne n'est
    plus une societe cotee.

    C'est le pendant indispensable de la suppression du rejet sur l'age des comptes.
    Une societe ne se declare pas morte parce que ses comptes datent : elle se
    declare morte parce qu'elle a ete RADIEE, a une date certaine et verifiable.
    ITT Educational Services a ete liquidee en 2016 ; il n'existe aucune "bonne
    valorisation" de cette ligne, et son exclusion ne releve pas du controle de
    donnee mais de la DEFINITION DE L'UNIVERS.

    Garde-fou : le fournisseur marque aussi des societes bien vivantes, le plus
    souvent a quelques jours d'un changement de place ou de code. On n'exclut donc
    qu'au-dela d'un TRIMESTRE, delai apres lequel une radiation erronee aurait ete
    corrigee."""
    from datetime import datetime, timedelta, timezone
    limite = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
    out = {}
    for page in range(200):                       # ~9 500 lignes, 100 par page
        try:
            rows = _json(f"delisted-companies?page={page}&limit=100")
        except Exception:
            break
        if not rows:
            break
        for r in rows:
            sym, d = r.get("symbol"), str(r.get("delistedDate") or "")[:10]
            if sym and d and d < limite:
                # La place de cotation et le nom accompagnent la date. Ils ne
                # servent pas a l'univers courant — `screener` ne teste que
                # l'appartenance — mais ils sont indispensables des qu'on veut
                # REMETTRE ces societes dans une mesure : sans la place, on ignore
                # dans quelle monnaie elles cotaient ; sans le nom, on ne peut pas
                # ecarter les bons de souscription et les unites de SPAC, que
                # l'univers courant exclut et qu'un test doit exclure aussi.
                out[sym] = {"date": d, "exchange": r.get("exchange"),
                            "name": r.get("companyName")}
    return out


def screener(exchanges=("NASDAQ",)):
    """Sociétés tangibles (hors ETF/fonds) des bourses données. Retourne
    {symbol: {sector, industry, beta, price, market_cap, currency, exchange, name}}."""
    out = {}
    try:
        radiees = societes_radiees()
    except Exception:
        radiees = {}
    for ex in exchanges:
        try:
            rows = _json(f"company-screener?exchange={ex}&isEtf=false&isFund=false"
                         f"&isActivelyTrading=true&limit=15000")
        except Exception:
            continue
        # `isFund` decrit une FORME JURIDIQUE, pas une realite economique : il se
        # declenche sur le mot "Trust" dans la raison sociale. Or une foncière est
        # tres souvent constituee en fiducie tout en exploitant des immeubles,
        # publiant un compte de resultat et versant des loyers. Le drapeau nous
        # faisait perdre 164 societes et 279 Md$ de capitalisation — la TOTALITE de
        # l'immobilier cote canadien (RioCan, CAP REIT, SmartCentres, Granite,
        # Choice, First Capital, CT REIT, Allied) mais aussi des foncieres du
        # S&P 500 : Essex Property Trust, Federal Realty, Vornado, Kite Realty.
        # On les reintegre par leur INDUSTRIE, taxonomie que le fournisseur etablit
        # sur l'activite reelle. Verifie sur l'univers entier : les 4 962 vrais
        # fonds marques `isFund` se rangent sans exception sous "Asset Management",
        # jamais sous "REIT — ...". Aucun fonds n'est donc readmis.
        try:
            rows += [r for r in _json(f"company-screener?exchange={ex}&isEtf=false"
                                      f"&isActivelyTrading=true&limit=20000")
                     if r.get("isFund")
                     and str(r.get("industry") or "").upper().startswith("REIT")]
        except Exception:
            pass
        for r in rows:
            sym = r.get("symbol")
            if not sym:
                continue
            if _is_preferred(sym, r.get("companyName")):   # actions privilégiées / notes : exclues
                continue
            if _est_un_derive(sym, r.get("companyName")):       # warrants / units / droits
                continue
            if sym in radiees:                    # radiee a date certaine : hors univers
                continue
            # Le volume de la DERNIERE SEANCE d'une ligne de cotation ne dit rien de
            # l'existence de l'emetteur. Ce filtre n'etait ni necessaire — ITT
            # Educational, liquidee en 2016, affiche 66 400 titres echanges, et 2 314
            # des 9 510 lignes de gre a gre a volume non nul n'ont aucun compte
            # recent — ni suffisant : il ecartait 490 lignes et 1 725 Md$ de
            # capitalisation, dont 189 au-dessus du milliard. SoftBank (161,6 Md$),
            # Bank of Communications (81,3), Generali (71,8), Volvo (69,4),
            # Brookfield (56,8) et Ryanair (30,7) etaient absentes du site pour cette
            # seule raison. Une societe morte se constate sur un FAIT DATE — radiation
            # ou cessation de depot — jamais sur l'activite d'une seance.
            out[sym] = {"name": r.get("companyName"), "sector": r.get("sector"),
                        "industry": r.get("industry"), "beta": _num(r.get("beta")),
                        "price": _num(r.get("price")),
                        "market_cap": _num(r.get("marketCap")),
                        "country": r.get("country"),
                        "currency": None, "exchange": ex}
    return _garder_la_ligne_principale(out)


def _garder_la_ligne_principale(uni):
    """Une meme societe peut avoir PLUSIEURS lignes cotees : l'action ordinaire, mais
    aussi ses bons de souscription, ses droits, ses unites ou une ligne ADR. Toutes
    portent les MEMES comptes consolides alors que leur capitalisation ne couvre que
    la ligne — SBC Medical Group Holdings apparait en SBC (319 M$, l'action) et en
    SBCWW (27 M$, les warrants), ce dernier ressortant a +2 505 % d'upside.
    On ne conserve donc, par raison sociale, que la ligne principale. Les actions de
    CLASSE (Alphabet A et C, Berkshire A et B) sont de vraies actions ordinaires et
    ont des capitalisations du meme ordre : le seuil de 20 % les preserve."""
    par_nom = {}
    for sym, r in uni.items():
        nom = (r.get("name") or "").strip().lower()
        if nom:
            par_nom.setdefault(nom, []).append(sym)
    a_retirer = set()
    for nom, syms in par_nom.items():
        if len(syms) < 2:
            continue
        caps = {s: (uni[s].get("market_cap") or 0.0) for s in syms}
        cap_max = max(caps.values())
        if cap_max <= 0:
            continue
        for s, c in caps.items():
            if c < 0.20 * cap_max:          # ligne accessoire, pas l'action ordinaire
                a_retirer.add(s)
    for s in a_retirer:
        uni.pop(s, None)
    return uni


# --------------------------------------------------------------------------- #
# Fondamentaux en bulk (toutes sociétés, 1 CSV par année/état)
# --------------------------------------------------------------------------- #
def _bulk_csv(endpoint, year, keep):
    r = _get(f"{endpoint}?year={year}&period=annual")
    reader = csv.DictReader(io.StringIO(r.text))
    rows = {}
    for row in reader:
        s = row.get("symbol")
        if s and (keep is None or s in keep):
            rows[s] = row
    return rows


def bulk_statements(symbols, years):
    """{symbol: {income:{y:row}, balance:{y:row}, cashflow:{y:row}}} pour l'univers."""
    keep = set(symbols)
    data = {}
    eps = [("income-statement-bulk", "income"),
           ("balance-sheet-statement-bulk", "balance"),
           ("cash-flow-statement-bulk", "cashflow")]
    for y in years:
        for endpoint, kind in eps:
            try:
                rows = _bulk_csv(endpoint, y, keep)
            except Exception:
                continue
            for s, row in rows.items():
                e = data.setdefault(s, {"income": {}, "balance": {}, "cashflow": {}})
                e[kind][y] = row
    return data


# --------------------------------------------------------------------------- #
# Profils (descriptions) en bulk
# --------------------------------------------------------------------------- #
def profile_descriptions(symbols, max_parts=6):
    keep = set(symbols)
    out = {}
    for part in range(max_parts):
        try:
            r = _get(f"profile-bulk?part={part}")
        except Exception:
            break
        reader = csv.DictReader(io.StringIO(r.text))
        any_row = False
        for row in reader:
            any_row = True
            s = row.get("symbol")
            if s in keep and s not in out:
                out[s] = {"description": (row.get("description") or "")[:650],
                          "industry": row.get("industry")}
        if not any_row:
            break
    return out


# --------------------------------------------------------------------------- #
# Historique de prix (signal court terme) + news (par ticker)
# --------------------------------------------------------------------------- #
def history_ohlcv(symbol, days=400):
    """Serie quotidienne AJUSTEE : cours de cloture ET volume echange.

    Le volume figure DEJA dans la reponse que nous telechargeons pour les cours, et
    nous le jetions. L'exposer ne coute donc pas un appel de plus, alors qu'il porte
    la seule mesure directe de LIQUIDITE dont nous disposions — un titre qu'on ne
    peut pas revendre a un prix raisonnable est risque, quelle que soit la solidite
    de ses comptes.

    Cours AJUSTES des splits et dividendes : les cours bruts creent de faux signaux,
    un split 4:1 apparaissant comme une chute de 75 % et un detachement de dividende
    comme une baisse.

    La DATE de chaque seance est conservee : sans elle, la serie ne peut servir
    qu'a des mesures relatives (volatilite, volume median). Toute question de la
    forme "que valait ce titre a telle date" — c'est-a-dire toute validation
    historique d'une note ou d'une valorisation — exige de savoir a quel jour
    correspond chaque cours. `days=None` rend l'historique complet.

    LE BORNAGE EST DEMANDE AU SERVEUR, PAS APPLIQUE APRES COUP. Cet appel
    telechargeait cinq ans de seances — 210 ko par titre — pour n'en garder que
    les 400 dernieres, et il pesait a lui seul 83 % de toute la bande passante du
    build. Sur seize mille titres reconstruits chaque nuit, c'etait la difference
    entre tenir le forfait mensuel et l'epuiser avant la fin du mois.
    Mesure sur cinq grandes valeurs : 211 ko -> 50 ko, soit 4,2 fois moins.

    Le facteur 1,5 convertit des SEANCES en jours calendaires : une annee compte
    environ 252 seances pour 365 jours. On majore volontairement — mieux vaut
    quelques seances de trop que la serie tronquee juste avant le seuil dont
    depend le signal court terme."""
    borne = ""
    if days:
        from datetime import date as _date, timedelta as _td
        depuis = _date.today() - _td(days=int(days * 1.5) + 10)
        borne = f"&from={depuis.isoformat()}"
    try:
        j = _json(f"historical-price-eod/dividend-adjusted?symbol={symbol}{borne}")
        lignes = [{"date": d.get("date"), "close": _num(d.get("adjClose")),
                   "volume": _num(d.get("volume"))}
                  for d in j][::-1]                           # ancien -> recent
        out = [x for x in lignes if x["close"]]
        if out:
            return out[-days:] if days else out
    except Exception:
        pass
    try:                                                      # repli : cours brut
        j = _json(f"historical-price-eod/light?symbol={symbol}{borne}")
        lignes = [{"date": d.get("date"), "close": _num(d.get("price")),
                   "volume": _num(d.get("volume"))}
                  for d in j][::-1]
        out = [x for x in lignes if x["close"]]
        return out[-days:] if days else out
    except Exception:
        return []


def history_closes(symbol, days=400):
    """Cours seuls — adaptateur conserve pour le signal court terme."""
    return [x["close"] for x in history_ohlcv(symbol, days)]


def volume_dollars_median(serie, seances=60):
    """Volume quotidien MEDIAN en dollars, sur les dernieres seances.

    En dollars et non en titres : un million d'actions a 0,001 $ n'est pas un marche.
    Mediane et non moyenne : une seule seance exceptionnelle — annonce, entree dans un
    indice — ne doit pas faire passer pour liquide un titre qui ne s'echange jamais."""
    valeurs = sorted(x["close"] * x["volume"] for x in (serie or [])[-seances:]
                     if x.get("close") and x.get("volume") is not None)
    if not valeurs:
        return None
    n = len(valeurs)
    return valeurs[n // 2] if n % 2 else (valeurs[n // 2 - 1] + valeurs[n // 2]) / 2.0


def news(symbol, limit=8):
    try:
        j = _json(f"news/stock?symbols={symbol}&limit={limit}")
        return [{"title": n.get("title"), "publisher": n.get("publisher"),
                 "url": n.get("url"), "pubDate": n.get("publishedDate")} for n in j]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Capitalisation reelle d'un ADR : via la cotation D'ORIGINE
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4096)
def _capitalisation_origine(symbol: str, nom: str):
    """Capitalisation de la SOCIETE, en USD, retrouvee sur sa cotation d'origine.

    Pour un ADR de gre a gre, FMP renvoie la capitalisation de la seule LIGNE ADR
    enregistree — 1 132 899 titres pour PT Kalbe Farma, soit 8,7 M$ — alors que les
    etats financiers sont ceux du groupe entier (1,95 Md$ de chiffre d'affaires).
    Le rapport valeur/capitalisation etait donc multiplie par plusieurs centaines.
    La cotation d'origine (KLBF.JK) porte, elle, la vraie capitalisation.

    Retourne None si aucune correspondance credible n'est trouvee."""
    if not nom:
        return None
    try:
        cands = _json(f"search-name?query={requests.utils.quote(nom[:40])}&limit=8") or []
    except Exception:
        return None
    best = None
    for c in cands:
        sym = c.get("symbol") or ""
        if sym == symbol or "." not in sym:      # on cherche une place BOURSIERE etrangere
            continue
        try:
            j = _json(f"profile?symbol={sym}")
            pr = j[0] if j else None
        except Exception:
            continue
        if not pr:
            continue
        # meme societe ? on compare les debuts de raison sociale
        n1 = (pr.get("companyName") or "").lower()[:14]
        if not n1 or n1 not in nom.lower() and nom.lower()[:14] not in n1:
            continue
        cap = _num(pr.get("marketCap"))
        fx = fx_to_usd(pr.get("currency") or "USD") or 1.0
        if cap and cap > 0:
            usd = cap * fx
            if best is None or usd > best:
                best = usd
    return best


# --------------------------------------------------------------------------- #
# Conversion vers nos structures (forensics F + fundamentals) — tout en USD
# --------------------------------------------------------------------------- #
def financials_from_fmp(entry):
    """Structure compatible forensics.get_financials (ratios -> devise neutre,
    on garde la devise d'origine, cohérente par société)."""
    inc, bal, cf = entry.get("income", {}), entry.get("balance", {}), entry.get("cashflow", {})
    yrs = sorted(set(inc) & set(bal), reverse=True)
    if len(yrs) < 2:
        return None

    def col(src, field):
        return [_num(src.get(y, {}).get(field)) for y in yrs]

    def col_ebit():
        """Serie d'EBIT ECONOMIQUES : le resultat operationnel publie, borne par
        resultat avant impot + charges d'interets. Sans cela l'historique des marges
        (marge normalisee, cyclique) restait fonde sur un resultat operationnel qui
        exclut des charges recurrentes majeures — l'incoherence se propageait."""
        out = []
        for y in yrs:
            r = inc.get(y, {})
            e = _num(r.get("operatingIncome"))
            pt, it = _num(r.get("incomeBeforeTax")), _num(r.get("interestExpense"))
            if e is not None and pt is not None:
                eco = pt + (abs(it) if it else 0.0)
                e = min(e, eco)
            out.append(e)
        return out

    F = {"years": [str(y) for y in yrs],
         "revenue": col(inc, "revenue"), "cogs": col(inc, "costOfRevenue"),
         "gross_profit": col(inc, "grossProfit"), "ebit": col_ebit(),
         "sga": col(inc, "sellingGeneralAndAdministrativeExpenses"),
         "net_income": col(inc, "netIncome"), "shares": col(inc, "weightedAverageShsOutDil"),
         "total_assets": col(bal, "totalAssets"), "current_assets": col(bal, "totalCurrentAssets"),
         "current_liab": col(bal, "totalCurrentLiabilities"),
         "net_ppe": col(bal, "propertyPlantEquipmentNet"),
         "receivables": col(bal, "netReceivables"), "inventory": col(bal, "inventory"),
         "total_debt": col(bal, "totalDebt"), "long_term_debt": col(bal, "longTermDebt"),
         "retained_earnings": col(bal, "retainedEarnings"),
         "equity": col(bal, "totalStockholdersEquity"), "total_liab": col(bal, "totalLiabilities"),
         "cfo": col(cf, "operatingCashFlow"), "dep_amort": col(cf, "depreciationAndAmortization"),
         # Dividendes VERSES (tableau de flux). Le nom du champ a change d'une
         # version de l'API a l'autre : on essaie les trois formes connues. Sert a
         # la retention des financieres (g = ROE x retention, Damodaran) et au DDM.
         "dividends": [next((_num(cf.get(y, {}).get(k)) for k in
                             ("commonDividendsPaid", "netDividendsPaid",
                              "dividendsPaid")
                             if cf.get(y, {}).get(k) is not None), None)
                       for y in yrs],
         # Charges d'interets, en valeur absolue : la couverture doit se mesurer sur
         # la MEDIANE de plusieurs exercices, un instantane faisant perdre plusieurs
         # crans a un cyclique en creux puis les lui rendant l'annee suivante.
         "interest_expense": [None if v is None else abs(v)
                              for v in col(inc, "interestExpense")],
         "capex": col(cf, "capitalExpenditure")}
    ca, cl = F["current_assets"], F["current_liab"]
    F["working_capital"] = [(ca[i] - cl[i]) if (ca[i] is not None and cl[i] is not None)
                            else None for i in range(len(yrs))]
    return F


# Devise de cotation par place. CBOE y figure parce que Cboe Global Markets,
# membre du S&P 500, cote sur SA PROPRE bourse — la seule ligne que cette place
# porte chez le fournisseur. Sans elle, un titre de l'indice de reference
# manquait a l'univers, et rien ne le signalait.
_EX_CUR = {"NASDAQ": "USD", "NYSE": "USD", "AMEX": "USD", "CBOE": "USD",
           "TSX": "CAD", "TSXV": "CAD"}


def fundamentals_from_fmp(symbol, sr, entry, desc, reference=None):
    """Dict compatible universal.get_fundamentals — montants convertis en USD.

    `reference` fixe la date depuis laquelle l'age des comptes est compte.
    None = aujourd'hui, ce que veut la production ; une date, ce qu'exige
    toute reconstitution historique."""
    inc, bal = entry.get("income", {}), entry.get("balance", {})
    cf = entry.get("cashflow", {})
    yrs = sorted(set(inc) & set(bal), reverse=True)
    if not yrs:
        return None
    y = yrs[0]
    rep_cur = inc.get(y, {}).get("reportedCurrency") or "USD"
    # Taux de change : JAMAIS de repli silencieux a 1. Traiter des roupies
    # indonesiennes comme des dollars fausse tout d'un facteur 18 000 — on signale
    # l'indisponibilite et la societe sera ecartee par la validation.
    _fxs = fx_to_usd(rep_cur)
    fx_ko = (_fxs is None)
    fxs = _fxs if _fxs is not None else 1.0               # etats -> USD
    price_cur = _EX_CUR.get((sr.get("exchange") or "").upper(), "USD")
    _fxp = fx_to_usd(price_cur)
    fx_ko = fx_ko or (_fxp is None)
    fxp = _fxp if _fxp is not None else 1.0               # cours/capi -> USD
    B = 1e9
    g = lambda src, f: _num(src.get(y, {}).get(f))
    rev, ebit, ni = g(inc, "revenue"), g(inc, "operatingIncome"), g(inc, "netIncome")
    # EBIT ECONOMIQUE. Le resultat operationnel publie peut EXCLURE des charges
    # recurrentes majeures reclassees en "autres produits et charges" : chez Yiren
    # Digital, preteur chinois, cette ligne vaut -2,2 Md CNY et efface la TOTALITE
    # d'un resultat operationnel de 2,1 Md — les provisions pour creances douteuses,
    # qui sont le coeur du metier. Le DCF projetait une marge de 38,5 % a l'infini
    # sur une societe en PERTE avant impot, d'ou +4 700 % d'upside.
    # Identite : resultat avant impot + charges d'interets = EBIT economique. On
    # retient le plus PRUDENT des deux, ce qui ne change rien a une societe dont le
    # compte de resultat est ordinaire.
    pretax, interets = g(inc, "incomeBeforeTax"), g(inc, "interestExpense")
    ebit_publie, ecart_ebit = ebit, None
    if ebit is not None and pretax is not None:
        ebit_eco = pretax + (abs(interets) if interets else 0.0)
        # L'ECART entre resultat publie et resultat economique est une information
        # sur la QUALITE DE LA LIASSE, et il etait calcule puis jete. Un ecart nul
        # est le cas normal ; un ecart massif signale des charges recurrentes
        # reclassees hors du resultat operationnel, ou un bras financier consolide.
        if abs(ebit) > 1e-12:
            ecart_ebit = (ebit_eco - ebit) / abs(ebit)
        if ebit_eco < ebit:
            ebit = ebit_eco
    eq, debt = g(bal, "totalStockholdersEquity"), g(bal, "totalDebt")
    # Capitaux propres revenant a l'ACTIONNAIRE ORDINAIRE : on retranche les
    # actions privilegiees (creance prioritaire) et les interets minoritaires
    # (part des filiales non detenue). Sans cela, Fannie Mae affichait 109 Md$ de
    # fonds propres alors que 140 Md$ de privilegiees senior du Tresor passent
    # AVANT l'ordinaire -> valeur ordinaire en realite negative, upside +2400 %.
    # `totalStockholdersEquity` EXCLUT deja les interets minoritaires (verifie :
    # totalStockholdersEquity + minorityInterest == totalEquity). Les retrancher
    # une seconde fois rendait negatifs les fonds propres de toute societe a
    # filiales — l'assureur polonais PZU tombait a -0,45 Md$ au lieu de +9,6 Md$.
    # Seules les actions PRIVILEGIEES, creance prioritaire sur l'ordinaire, se
    # deduisent (Fannie Mae : 140 Md$ de privilegiees senior du Tresor).
    privilegiees = _num(bal.get(y, {}).get("preferredStock")) or 0.0
    minoritaires = _num(bal.get(y, {}).get("minorityInterest")) or 0.0
    if eq is not None:
        eq -= privilegiees
    # TRESORERIE REALISABLE : liquidites ET placements court terme. Une societe de
    # biotechnologie place l'essentiel de ses fonds en titres negociables plutot
    # qu'en depots — Revolution Medicines detient 0,38 Md$ de liquidites pour
    # 1,64 Md$ de placements court terme. N'en retenir que les liquidites
    # sous-estimait sa tresorerie de 80 % et lui appliquait a tort une decote.
    cash = g(bal, "cashAndShortTermInvestments") or g(bal, "cashAndCashEquivalents")
    tot_assets = g(bal, "totalAssets")
    # Garde-fou données FMP corrompues (ex. RDZN cash=6.6e12 pour 55 M$ de CA) :
    # trésorerie et dette ne peuvent PAS dépasser l'actif total -> on plafonne.
    if tot_assets and tot_assets > 0:
        if cash is not None:
            cash = min(cash, tot_assets)
        if debt is not None:
            debt = min(max(debt, 0.0), tot_assets)
    # SERIE CONTINUE D'EXERCICES. Elle etait batie sur les seules annees PRESENTES
    # dans `inc`, puis purgee de ses valeurs manquantes : ses trous disparaissaient
    # DEUX FOIS, et sans laisser de trace. Rien n'indiquait plus que deux elements
    # voisins pouvaient etre distants de deux ans, et `_estimate_growth` annualisait
    # alors une croissance pluriannuelle.
    _annees, _rev = serie_continue(
        ((yy, _num(inc.get(yy, {}).get("revenue"))) for yy in inc))
    rev_hist = [None if v is None else v * fxs / B for v in _rev]
    price, mcap = _num(sr.get("price")), _num(sr.get("market_cap"))
    # COHERENCE capitalisation <-> comptes. Une societe ne se traite pas a 3 % de
    # son chiffre d'affaires ni a 5 % de ses fonds propres : un tel ecart signale
    # que la capitalisation porte sur une LIGNE ADR et non sur la societe. On va
    # alors chercher la capitalisation sur sa cotation d'origine.
    if mcap and mcap > 0:
        rev_usd = (rev or 0) * fxs
        eq_usd = (eq or 0) * fxs
        mcap_usd = mcap * fxp
        ebit_usd = (ebit or 0) * fxs
        incoherent = ((rev_usd > 0 and mcap_usd < 0.03 * rev_usd)
                      or (eq_usd > 0 and mcap_usd < 0.05 * eq_usd)
                      # Une societe RENTABLE ne se traite pas sous deux fois son
                      # resultat operationnel : Apple vaut 37 fois le sien,
                      # Coca-Cola 28 fois. En dessous, la capitalisation ne porte
                      # pas sur la meme entite que les comptes.
                      or (ebit_usd > 0 and mcap_usd < 2.0 * ebit_usd))
        if incoherent:
            vraie = _capitalisation_origine(symbol, sr.get("name") or "")
            if vraie and vraie > mcap_usd:
                mcap, fxp = vraie, 1.0            # deja en USD
    # Nombre d'actions COHÉRENT avec le cours : capitalisation / cours donne la base
    # exacte sur laquelle le marché price le titre. Les actions déclarées aux comptes
    # peuvent porter sur une autre base (ADR = N actions ordinaires, multi-classes) ou
    # être corrompues -> la valeur par action ne serait pas comparable au cours.
    # Avec la base implicite : valeur/action ÷ cours − 1 == équité ÷ capi − 1 (identité).
    shares = (mcap / price) if (mcap and price and price > 0) else g(inc, "weightedAverageShsOutDil")
    b = lambda v: None if v is None else v * fxs / B
    return {
        "ticker": symbol, "name": sr.get("name") or symbol, "sector": sr.get("sector"),
        "industry": (desc or {}).get("industry") or sr.get("industry"),
        "summary": (desc or {}).get("description"),
        "price": price * fxp if price else None,
        "market_cap": mcap * fxp / B if mcap else None,
        "shares": shares, "beta": _sane_beta(sr.get("beta"), sr.get("sector")),
        # DESACCORD ENTRE DEUX SOURCES sur la base actionnaire : le nombre d'actions
        # DEDUIT du marche (capitalisation / cours) contre celui DEPOSE aux comptes.
        # On le MESURE sans jamais en tirer un rejet ni une reecriture : quand deux
        # sources se contredisent, aucune ne prouve que l'autre a tort. Mesure sur
        # l'univers : ecart median 3,4 %, neuvieme decile 64 % — un rachat d'actions
        # ou une emission suffit a l'expliquer, et la distribution est CONTINUE, donc
        # aucun seuil ne s'y lit. Les corrections tentees allaient d'ailleurs neuf
        # fois sur douze dans le mauvais sens, et toujours celui qui gonfle la valeur.
        # L'ecart est donc publie tel quel : c'est une information sur la CONFIANCE
        # que merite la donnee, pas un verdict sur la societe.
        "actions_publiees": g(inc, "weightedAverageShsOutDil") or g(inc, "weightedAverageShsOut"),
        # FACTEUR DE DILUTION : moyenne diluee / moyenne de base du MEME exercice.
        # La moyenne diluee applique la methode du rachat — l'approche « treasury
        # stock » de Damodaran pour les options des employes, la deuxieme de sa
        # hierarchie (la premiere, valoriser les options en Black-Scholes, exige un
        # strike et un encours que le flux ne porte pas). Le RAPPORT est sans unite
        # et insensible aux ratios ADR, contrairement aux niveaux. Borne a [1 ; 1,5] :
        # au-dela, ce n'est plus une dilution mais une donnee corrompue.
        "facteur_dilution": (lambda d, b_: min(max(d / b_, 1.0), 1.5)
                             if (d and b_ and b_ > 0) else 1.0)(
            g(inc, "weightedAverageShsOutDil"), g(inc, "weightedAverageShsOut")),
        # --- Champs nourrissant la NOTE DE RISQUE -----------------------------
        # Charges d'interets en VALEUR ABSOLUE : le fournisseur les publie tantot
        # positives tantot negatives selon l'emetteur, et une couverture d'interets
        # dont le denominateur change de signe est ininterpretable.
        "interest_expense": b(abs(g(inc, "interestExpense") or 0.0)) or None,
        # COUVERTURE DES INTERETS, mediane pluriannuelle — l'entree de la notation
        # synthetique de Damodaran (ratings.html) : le spread de credit se lit sur
        # EBIT / charge d'interets. Mediane et non dernier exercice, pour qu'un
        # cyclique en creux ne perde pas trois crans une annee et les reprenne la
        # suivante — la meme discipline que la dimension de solvabilite D1.
        "couverture_interets": (lambda cs: sorted(cs)[len(cs) // 2] if cs else None)(
            [_num(inc[yy].get("operatingIncome")) / abs(_num(inc[yy].get("interestExpense")))
             for yy in inc
             if _num(inc[yy].get("operatingIncome")) is not None
             and _num(inc[yy].get("interestExpense")) not in (None, 0)]),
        # Ecart entre resultat operationnel PUBLIE et resultat ECONOMIQUE : mesure
        # de la qualite de la liasse, calculee plus haut et jusqu'ici jetee.
        "ecart_ebit": ecart_ebit,
        "ebit_publie": b(ebit_publie),
        # Incorporels : la solvabilite d'une financiere se juge sur ses fonds
        # propres TANGIBLES, un ecart d'acquisition n'absorbant aucune perte.
        "goodwill_intangibles": b(g(bal, "goodwillAndIntangibleAssets")
                                  or ((g(bal, "goodwill") or 0.0)
                                      + (g(bal, "intangibleAssets") or 0.0))),
        # Mur de refinancement a douze mois.
        "short_term_debt": b(g(bal, "shortTermDebt")),
        "capital_lease_current": b(g(bal, "capitalLeaseObligationsCurrent")),
        "country": sr.get("country"),
        "currency_ok": not fx_ko, "fx_indisponible": fx_ko,
        "price_currency": price_cur, "financial_currency": rep_cur,
        "revenue": b(rev), "revenue_history": rev_hist, "ebit": b(ebit), "net_income": b(ni),
        "total_debt": b(debt), "cash": b(cash), "book_equity": b(eq),
        # Creances SENIOR sur l'ordinaire, aux bornes du perimetre consolide. Elles
        # servent au pont vers l'equite du DCF : les flux consolides portent 100 %
        # des filiales, la valeur qui en derive aussi — Damodaran retranche donc la
        # valeur des minoritaires et des privilegiees avant de parler d'equite
        # ordinaire (« subtract the market value of the minority interest »).
        "minoritaires": b(minoritaires) or 0.0,
        "preferred_equity": b(privilegiees) or 0.0,
        "total_assets": b(tot_assets),
        # Passif total CONVERTI en USD : indispensable pour verifier l'identite
        # actif = passif + fonds propres. Le comparer au passif BRUT (en devise
        # locale) faisait echouer l'identite pour toute societe non americaine —
        # l'ecart valait exactement le taux de change (534 % pour une chinoise).
        "total_liab": b(g(bal, "totalLiabilities")),
        # Fonds propres TOTAUX, minoritaires inclus. Le bilan equilibre sur eux :
        #     actif = passif + fonds propres TOTAUX
        # alors que `book_equity` ne retient, a juste titre, que la part revenant
        # aux actionnaires. Confronter l'identite a la part attribuable rejetait
        # toute societe detenant des filiales non integralement possedees — le
        # controle de coherence eliminait ainsi des groupes parfaitement sains,
        # Branicks ressortant a 10 % d'ecart pour cette seule raison.
        "total_equity": b(g(bal, "totalEquity")) or (
            (b(g(bal, "totalStockholdersEquity")) or 0.0)
            + (b(g(bal, "minorityInterest")) or 0.0)) or None,
        "operating_margin": (ebit / rev) if (ebit is not None and rev) else None,
        "roe": (ni / eq) if (ni is not None and eq) else None,
        # Amortissements + flux d'exploitation : nécessaires au FFO des foncières
        # (Damodaran : les REIT se valorisent sur FFO/NAV, pas sur le FCFF).
        "dep_amort": b(g(cf, "depreciationAndAmortization")),
        "cfo": b(g(cf, "operatingCashFlow")),
        # INVESTISSEMENTS : une societe qui construit son outil consomme sa
        # tresorerie par le capex bien plus que par l'exploitation. Wesizwe Platinum
        # brule 562 M ZAR d'exploitation pour 1 238 M ZAR d'investissement dans la
        # mine de Bakubung : juger son autonomie sur le seul flux d'exploitation
        # revenait a ignorer les deux tiers de la consommation.
        "capex": b(g(cf, "capitalExpenditure")),
        # DATE d'observation des comptes, portee par `fund` pour voyager partout.
        # Ce n'est pas un critere de rejet — des comptes anciens sont les derniers
        # comptes CONNUS — mais une information indispensable a la valorisation :
        # une tresorerie qui se consume continue de se consumer pendant que
        # l'information vieillit. Iridium World Communications, en faillite depuis
        # 1999, etait valorisee sur son bilan de 1998 vingt-huit ans plus tard.
        "date_des_comptes": _date_des_comptes(inc, bal),
        "age_des_comptes_mois": _age_en_mois(_date_des_comptes(inc, bal), reference),
        "cik": inc.get(y, {}).get("cik"), "sic": None,
    }


__all__ = ["screener", "bulk_statements", "profile_descriptions", "history_closes",
           "news", "fx_to_usd", "financials_from_fmp", "fundamentals_from_fmp"]


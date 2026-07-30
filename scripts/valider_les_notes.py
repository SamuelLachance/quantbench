"""La note de risque predit-elle quoi que ce soit ? Mesure retrospective.

La fiche de chaque societe affiche une note de A+ a F qui pretend classer sa
probabilite de "perdre durablement sa mise". Cette note a ete CALIBREE sur des
quantiles d'univers — c'est-a-dire sur la forme de la distribution, pas sur des
resultats. Une echelle jamais confrontee a ce qui s'est reellement passe n'est pas
une mesure : c'est une opinion presentee comme une mesure.

Ce script produit la confrontation. Il reconstruit la note telle qu'elle AURAIT
ete calculee il y a N mois, puis regarde ce que le titre a fait depuis.

Trois pieges, et ce qu'on en fait :

1. REGARD SUR L'AVENIR. Un instantane du 30 juillet 2024 ne peut utiliser que des
   comptes DEJA PUBLIES a cette date. On filtre sur `filingDate`, pas sur la date
   de cloture de l'exercice : l'exercice clos en decembre 2023 n'etait pas public
   avant mars 2024. Utiliser la date de cloture donnerait a la note trois mois de
   clairvoyance et gonflerait artificiellement son pouvoir predictif.

2. BIAIS DU SURVIVANT, et c'est le plus grave ici. Le screener d'aujourd'hui ne
   liste que les societes ENCORE COTEES. Les faillites — exactement les evenements
   que la note F pretend annoncer — en ont disparu. Un test mene sur les seuls
   survivants mesure la note sur un univers d'ou l'on a retire ses succes, et la
   fait paraitre inutile. On y ajoute donc explicitement les societes radiees
   depuis l'instantane, comptees pour ce qu'elles sont : une perte totale.

3. LE RESULTAT PEUT ETRE MAUVAIS. Si la note ne separe rien, ce script le dira. Un
   banc d'essai qui ne peut pas invalider ce qu'il teste ne sert a rien.

Usage :
    python scripts/valider_les_notes.py --horizon 24 --echantillon 900
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantbench.data import fmp                                    # noqa: E402
from quantbench.data.repair import reparer                         # noqa: E402
from quantbench.data.validate import valider                       # noqa: E402
from quantbench.risk import noter                                  # noqa: E402
from quantbench.valuation.route import value_stock                 # noqa: E402
from quantbench.risk.statistiques import (                         # noqa: E402
    aire_sous_la_courbe as _auc, mediane, spearman,
    test_deux_proportions, wilson)

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "app" / "us" / "_validation_risque.json"
# RESUME SEPARE, et c'est une decision d'architecture, pas une optimisation.
# La fiche affichait ces chiffres EN DUR dans son HTML. Deux chiffres ecrits a deux
# endroits divergent toujours : c'est ainsi que le bloc de methodologie a promis
# pendant des semaines une confrontation "dans douze mois" alors qu'elle avait eu
# lieu. La page lit desormais la mesure au lieu de la recopier, et aucune
# republication ne peut plus la desynchroniser.
# Fichier a part parce que le fichier complet pese 400 ko : il porte le detail par
# titre, indispensable pour diagnostiquer une anomalie, inutile a l'affichage.
RESUME = RACINE / "app" / "us" / "_validation_risque_resume.json"

# Un titre qui a perdu 70 % de sa valeur n'est pas "en baisse" : il exige un gain
# de 233 % pour revenir a son point de depart. C'est la definition operationnelle
# de "perdre durablement sa mise" que la note pretend classer.
SEUIL_EFFONDREMENT = -0.70

# LES PLACES QUE LE SITE COUVRE, et elles seules. La liste des societes radiees
# fournie par le fournisseur est MONDIALE : Tokyo, Seoul, Londres, Hong Kong. Les
# admettre dans le test a produit deux erreurs qui se cumulaient. D'abord une
# capitalisation de 198 millions de milliards de dollars, parce qu'une place
# inconnue retombait sur "USD" et qu'un cours en yens etait lu comme un cours en
# dollars — precisement le repli silencieux que le projet s'interdit partout
# ailleurs, et que j'avais reintroduit ici. Ensuite, et plus grave : valider une
# note sur des societes que le site ne valorise pas ne dit rien de cette note.
DEVISE_DE_LA_PLACE = {"NASDAQ": "USD", "NYSE": "USD", "AMEX": "USD",
                      "OTC": "USD", "TSX": "CAD", "TSXV": "CAD"}

# Profondeur d'historique dont dispose la production (`fmp.statements`).
PROFONDEUR_PRODUCTION = 6

FAMILLES = ["A", "B", "C", "D", "F"]
ORDRE = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"]


# --------------------------------------------------------------------------- #
# Instantane point-in-time
# --------------------------------------------------------------------------- #
def tronquer(entry, limite: str):
    """Ne garde que les exercices DEJA PUBLIES au `limite` (AAAA-MM-JJ).

    Le filtre porte sur la date de DEPOT, pas sur la date de cloture. Un exercice
    clos le 31 decembre est depose en mars ou avril : le retenir des le 1er janvier
    donnerait a la note trois a quatre mois de connaissance de l'avenir, ce qui
    suffit a lui faire "predire" des faillites deja consommees.
    """
    out = {}
    for bloc in ("income", "balance", "cashflow"):
        garde = {}
        for y, d in (entry.get(bloc) or {}).items():
            depot = d.get("filingDate") or d.get("fillingDate") or d.get("acceptedDate")
            if depot and str(depot)[:10] <= limite:
                garde[y] = d
        out[bloc] = garde
    return out


def cours_a(serie, jour: str):
    """Dernier cours cote AU PLUS TARD a `jour`. None si le titre ne cotait pas."""
    retenu = None
    for x in serie:
        d = x.get("date")
        if d and d[:10] <= jour:
            retenu = x
        else:
            break
    return retenu


# --------------------------------------------------------------------------- #
# Une societe
# --------------------------------------------------------------------------- #
def evaluer(sym, sr, jour_t0: str, radiees: dict, plafond_capi=None):
    """Note reconstituee a `jour_t0`, puis devenir du titre jusqu'a aujourd'hui."""
    try:
        entry = fmp.statements(sym, limit=14)
        if not entry or not entry.get("income"):
            return None
        # On tire quatorze exercices pour pouvoir en jeter, puis on ne garde que
        # les six derniers DEJA PUBLIES a la date — soit exactement la profondeur
        # dont dispose la production. Sans ce recadrage, une societe de vingt ans
        # d'anciennete offrirait ici quatorze exercices contre six en production,
        # et la dimension de confiance dans la donnee, qui lit cette profondeur,
        # serait mesuree sur une echelle qui n'existe nulle part.
        vieux = tronquer(entry, jour_t0)
        if len(vieux["income"]) < 2:
            return None                       # pas d'historique publie a la date
        recents = sorted(vieux["income"], reverse=True)[:PROFONDEUR_PRODUCTION]
        vieux = {b: {y: d for y, d in vieux[b].items() if y in recents}
                 for b in ("income", "balance", "cashflow")}

        serie = fmp.history_ohlcv(sym, days=None)
        if not serie:
            return None
        p0 = cours_a(serie, jour_t0)
        if not p0 or not p0.get("close"):
            return None

        F = fmp.financials_from_fmp(vieux)
        # `sr` est PURGE de son cours et de sa capitalisation d'aujourd'hui. Sans
        # cela, `fundamentals_from_fmp` deduit la base actionnaire de
        # capitalisation / cours COURANTS : un instantane de 2024 herite alors du
        # nombre d'actions de 2026, dilutions comprises. C'est un regard sur
        # l'avenir d'autant plus pernicieux qu'il frappe surtout les societes en
        # difficulte, celles qui emettent des actions pour survivre.
        sr_t0 = {k: v for k, v in (sr or {}).items()
                 if k not in ("price", "market_cap")}
        # L'age des comptes se compte DEPUIS L'INSTANTANE, pas depuis
        # aujourd'hui : sinon tout l'univers parait vieux de deux ans de plus.
        f = fmp.fundamentals_from_fmp(sym, sr_t0, vieux, {}, reference=jour_t0)
        if not f:
            return None

        # Cours et capitalisation DE L'EPOQUE, dans les memes unites que la
        # production : cours en dollars, capitalisation en MILLIARDS de dollars.
        # Ecrire ici le cours brut multiplie par le nombre d'actions donnait une
        # capitalisation un milliard de fois trop grande et non convertie — les
        # dimensions de taille et de liquidite de la note s'en trouvaient toutes
        # faussees, et la premiere mesure obtenue ne valait rien.
        px_brut = float(p0["close"])
        sh = f.get("shares")
        if not sh or sh <= 0:
            return None
        devise = DEVISE_DE_LA_PLACE.get((sr_t0.get("exchange") or "").upper())
        if devise is None:
            return None                       # hors de l'univers du site
        fxp = fmp.fx_to_usd(devise)
        if fxp is None:
            return None                       # jamais de repli silencieux a 1
        f["price"] = px_brut * fxp
        f["market_cap"] = px_brut * fxp * sh / 1e9
        f["exchange"] = sr_t0.get("exchange")

        # LIQUIDITE DE SORTIE A LA DATE DE L'INSTANTANE. Cette dimension entrait
        # « non testee » dans chaque mesure, faute de volumes — alors que la serie
        # datee, volumes compris, etait deja entre nos mains dix lignes plus haut.
        # Son absence du tableau publie se lisait a tort comme une absence de
        # signal. On tronque la serie a la date etudiee, exactement comme les
        # comptes : utiliser les volumes d'aujourd'hui serait un regard sur
        # l'avenir, et un titre devenu illiquide APRES coup passerait pour liquide.
        avant = [x for x in serie if x.get("date") and x["date"][:10] <= jour_t0]
        vol = fmp.volume_dollars_median(avant)
        if vol is not None:
            f["volume_dollars_median"] = vol * fxp

        # LA RECONSTITUTION N'A PAS DE SECONDE SOURCE. La production deduit la base
        # actionnaire du marche (capitalisation / cours) et ne se fie au nombre
        # d'actions DEPOSE qu'a defaut ; ici, faute de capitalisation d'epoque, on
        # n'a que le nombre depose, sans rien pour le contredire. Deux lignes l'ont
        # montre : Repco Nickel, cotee de gre a gre, porte un cours de 10 000 $ dans
        # sa serie, ce qui donne 18 468 Md$ pour 1,8 milliard d'actions.
        # Le plafond n'est pas pose : c'est la plus grosse capitalisation REELLEMENT
        # observee dans l'univers d'aujourd'hui. Rien ne peut la depasser d'un ordre
        # de grandeur sans que les deux entrees se contredisent.
        if plafond_capi and f["market_cap"] > plafond_capi:
            return {"ticker": sym, "ecarte": "capitalisation impossible"}

        motifs = valider(f, F, vieux)
        rep = reparer(sym, f, F, vieux, motifs) if motifs else []
        if rep:
            motifs = valider(f, F, vieux)
        n = noter(f, F, motifs, rep)
        grade = n.get("grade")
        if not grade:
            return None

        # LA VALORISATION SUBIT LE MEME EXAMEN QUE LA NOTE, et c'est la question
        # centrale du site : il annonce qu'une action est decotee de tant, sans
        # que rien n'ait jamais verifie que cela predise quoi que ce soit. La
        # valorisation est reconstruite sur le MEME instantane — memes comptes
        # deja deposes, meme cours d'epoque, meme base actionnaire — de sorte que
        # les deux mesures sont comparables entre elles.
        upside, methode = None, None
        try:
            val = value_stock(sym, fund=f, F=F)
            if val and val.get("value_per_share") and f.get("price"):
                upside = val["value_per_share"] / f["price"] - 1.0
                methode = val.get("method")
        except Exception:
            pass

        # --- Devenir ---------------------------------------------------------
        radiee = (radiees.get(sym) or {}).get("date")
        if radiee and str(radiee)[:10] > jour_t0:
            # Radiee APRES l'instantane : perte totale. C'est l'evenement meme que
            # la note pretend annoncer, et celui que le screener d'aujourd'hui a
            # efface. L'omettre reviendrait a noter un test sur ses seuls reussites.
            rendement, issue = -1.0, "radiee"
        else:
            pn = serie[-1]
            if not pn.get("close"):
                return None
            # Rapport de deux cours de la MEME serie : la devise se simplifie,
            # aucune conversion n'est necessaire ni souhaitable ici.
            rendement = float(pn["close"]) / px_brut - 1.0
            issue = "cotee"

        # Plus bas atteint depuis l'instantane : une note de risque parle de la
        # PERTE ENCOURUE, pas seulement du point d'arrivee.
        apres = [x["close"] for x in serie
                 if x.get("date") and x["date"][:10] > jour_t0 and x.get("close")]
        creux = (min(apres) / px_brut - 1.0) if apres else rendement

        # Le rang de CHAQUE dimension, pour savoir laquelle porte le signal. La
        # note les pese uniformement faute d'avoir jamais pu mesurer leur pouvoir
        # respectif ; ces rangs le rendent mesurable.
        dims = {d["cle"]: d["rang"] for d in (n.get("dimensions") or [])
                if d.get("rang") is not None}

        return {"ticker": sym, "grade": grade, "famille": grade[0], "dimensions": dims,
                "upside": upside, "methode": methode,
                "score": n.get("score"), "regime": n.get("regime"),
                "cap_t0": f["market_cap"], "rendement": rendement,
                "creux": creux, "issue": issue,
                "effondre": bool(rendement <= SEUIL_EFFONDREMENT)}
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Statistiques
# --------------------------------------------------------------------------- #



def agreger(res, cle):
    """Table par grade ou par famille."""
    paquets = {}
    for r in res:
        paquets.setdefault(r[cle], []).append(r)
    lignes = []
    for g, rs in paquets.items():
        k = sum(1 for r in rs if r["effondre"])
        bas, haut = wilson(k, len(rs))
        lignes.append({
            "grade": g, "n": len(rs),
            "effondrements": k,
            "taux": k / len(rs),
            "ic_bas": bas, "ic_haut": haut,
            "rendement_median": mediane([r["rendement"] for r in rs]),
            "creux_median": mediane([r["creux"] for r in rs]),
            "radiees": sum(1 for r in rs if r["issue"] == "radiee"),
            # La taille mediane du paquet : sans elle, on lirait "A" comme
            # "grande valeur solide". L'univers est majoritairement micro-cap et
            # de gre a gre — un A parmi des micro-capitalisations n'est pas Apple,
            # et son taux d'effondrement de 23 % sur deux ans decrit ce monde-la.
            "cap_mediane": mediane([r["cap_t0"] for r in rs if r["cap_t0"]]),
        })
    ordre = ORDRE if cle == "grade" else FAMILLES
    lignes.sort(key=lambda l: ordre.index(l["grade"]) if l["grade"] in ordre else 99)
    return lignes


def par_taille(res, n_paquets=3):
    """La note ajoute-t-elle quelque chose A TAILLE COMPARABLE ?

    C'est la question qui decide si cette note vaut d'exister. Elle correle
    fortement avec la capitalisation — 4,4 Md$ de mediane en A, 1 M$ en F — et une
    micro-capitalisation s'effondre plus souvent qu'une grande, ce que chacun sait
    sans lire un bilan. Si toute la separation observee disparait une fois la
    taille tenue constante, la note ne fait que renommer la capitalisation, et huit
    dimensions de calcul auront servi a redecouvrir une colonne du screener.

    On decoupe donc l'echantillon en paquets de taille de meme effectif, et on
    reprend la mesure DANS chaque paquet. La monotonie qui survit a ce decoupage
    est celle que la note apporte en propre.
    """
    avec = [r for r in res if r.get("cap_t0")]
    avec.sort(key=lambda r: r["cap_t0"])
    if len(avec) < n_paquets * 40:
        return []
    taille = len(avec) // n_paquets
    out = []
    for i in range(n_paquets):
        bout = avec[i * taille:(i + 1) * taille] if i < n_paquets - 1 else avec[i * taille:]
        lignes = agreger(bout, "famille")
        paires = [(FAMILLES.index(l["grade"]), l["taux"])
                  for l in lignes if l["grade"] in FAMILLES]
        out.append({
            "paquet": i + 1,
            "cap_min": bout[0]["cap_t0"], "cap_max": bout[-1]["cap_t0"],
            "n": len(bout), "familles": lignes, "spearman": spearman(paires),
        })
    return out


def aire_sous_la_courbe(paires):
    """AUC seule. L'ecart-type accompagne la mesure la ou elle est publiee."""
    auc, _se = _auc(paires)
    return auc


def pouvoir_des_dimensions(res):
    """AUC de chaque dimension, comparee a celle de la note SUR LE MEME sous-echantillon.

    La comparaison naive est faussee : la solvabilite n'est definissable que sur
    82 % des societes, la note sur 100 %. Confronter 0,645 mesure sur 1 339 titres
    a 0,651 mesure sur 1 631 revient a comparer deux epreuves differentes. Chaque
    dimension est donc opposee a la note RESTREINTE aux memes titres, et l'ecart
    qui en resulte est le seul qui reponde a la question posee : agreger huit
    dimensions vaut-il mieux que garder la meilleure ?
    """
    from quantbench.risk.dimensions import DIMENSIONS
    noms = {c: n for c, n, _f, _niv in DIMENSIONS}

    out = []
    for cle, nom, _f, _niv in DIMENSIONS:
        sous = [r for r in res if cle in (r.get("dimensions") or {})]
        paires = [(r["dimensions"][cle], r["effondre"]) for r in sous]
        auc, ecart_type = _auc(paires)
        if auc is None:
            # Une dimension jamais definissable dans cette reconstitution n'est pas
            # une dimension sans pouvoir : elle est NON TESTEE. La liquidite de
            # sortie en fait partie — elle exige une serie de volumes a la date de
            # l'instantane, que le calcul de note ne recoit pas ici. Le taire
            # laisserait lire son absence comme une absence de signal.
            out.append({"dimension": cle, "nom": noms[cle], "n": 0, "auc": None,
                        "ecart_type": None,
                        "auc_note_meme_sous_echantillon": None, "couverture": 0.0,
                        "verdict": "non testee dans cette reconstitution"})
            continue
        auc_note = aire_sous_la_courbe(
            [(r["score"], r["effondre"]) for r in sous if r.get("score") is not None])
        # LE VERDICT SE LIT EN ECARTS-TYPES, PAS EN CENTIEMES. Un seuil fixe a
        # 0,53 juge un chiffre sans savoir a quelle precision il est connu : le
        # meme 0,545 est une information sur mille six cents societes et du bruit
        # sur cent. On compte donc les ecarts-types qui separent la mesure de 0,50,
        # c'est-a-dire de l'absence totale d'information.
        sigmas = (auc - 0.5) / ecart_type if ecart_type else 0.0
        if sigmas < 2.0:
            verdict = "indistinguable du hasard"
        elif sigmas < 4.0:
            verdict = "signal faible"
        else:
            verdict = "porte du signal"
        out.append({"dimension": cle, "nom": noms[cle], "n": len(paires), "auc": auc,
                    "ecart_type": ecart_type, "ecarts_types_au_hasard": sigmas,
                    "auc_note_meme_sous_echantillon": auc_note,
                    "couverture": len(paires) / len(res), "verdict": verdict})
    out.sort(key=lambda d: -(d["auc"] if d["auc"] is not None else -1))
    return out


def pouvoir_de_la_valorisation(res, n_paquets=5):
    """L'upside annonce predit-il le rendement ? La question centrale du site.

    Il annonce qu'une action vaut tant de plus que son cours. Cette affirmation
    n'avait jamais ete confrontee a quoi que ce soit. On range donc les societes
    par upside croissant, en paquets d'effectif egal, et on regarde le rendement
    median de chaque paquet sur l'horizon.

    Deux precautions qui changent la lecture :
      - la MEDIANE, jamais la moyenne : un titre qui fait x20 emporterait la
        moyenne d'un paquet entier et ferait conclure a un pouvoir predictif que
        les autres titres du paquet dementent ;
      - on publie AUSSI le taux d'effondrement par paquet. Un upside eleve se
        rencontre surtout chez les societes en difficulte, dont le cours est bas
        parce que le marche doute — la decote et le risque de ruine vont donc
        ensemble, et lire l'un sans l'autre serait trompeur.
    """
    avec = [r for r in res if r.get("upside") is not None]
    if len(avec) < n_paquets * 30:
        return []
    avec.sort(key=lambda r: r["upside"])
    taille = len(avec) // n_paquets
    out = []
    for i in range(n_paquets):
        bout = avec[i * taille:(i + 1) * taille] if i < n_paquets - 1             else avec[i * taille:]
        k = sum(1 for r in bout if r["effondre"])
        bas, haut = wilson(k, len(bout))
        out.append({
            "paquet": i + 1, "n": len(bout),
            "upside_min": bout[0]["upside"], "upside_max": bout[-1]["upside"],
            "rendement_median": mediane([r["rendement"] for r in bout]),
            "creux_median": mediane([r["creux"] for r in bout]),
            "taux_effondrement": k / len(bout), "ic_bas": bas, "ic_haut": haut,
        })
    return out


# --------------------------------------------------------------------------- #
def main(horizon=24, echantillon=900, workers=14, graine=20260730):
    t0 = time.time()
    jour_t0 = (date.today() - timedelta(days=int(horizon * 30.44))).isoformat()
    print(f"Instantane au {jour_t0} (horizon {horizon} mois), "
          f"echantillon {echantillon}\n")

    uni = fmp.screener(["NASDAQ", "NYSE", "TSX", "TSXV", "OTC"])
    print(f"  univers cote aujourd'hui : {len(uni)}")
    radiees = fmp.societes_radiees()
    print(f"  societes radiees connues : {len(radiees)}")
    # SANS LES RADIEES, CETTE MESURE N'EN EST PAS UNE. Le screener d'aujourd'hui ne
    # liste que les survivantes ; les faillites, soit l'evenement meme que la note F
    # pretend annoncer, en ont disparu. Une passe sans elles ne mesure pas une note
    # trop indulgente, elle mesure un univers ampute de ses ruines — et elle le fait
    # SANS RIEN CASSER : le fournisseur a renvoye une liste vide (saturation de
    # l'API pendant un build), le script a continue, et la mesure obtenue affichait
    # 3,9 % d'effondrement en A contre 18,5 % avec la correction. Aucune alerte,
    # juste des chiffres flatteurs. On refuse donc de produire un fichier.
    if not radiees:
        print("\nECHEC : le fournisseur n'a renvoye AUCUNE societe radiee.\n"
              "  Sans elles, la mesure porterait sur un univers vide de ses "
              "faillites\n  et surestimerait la note. Rien n'est ecrit ; "
              "relancer plus tard.")
        return 2

    # Les radiees APRES l'instantane sont les evenements que le test doit voir.
    # On leur applique le MEME filtre que l'univers courant : bons de souscription,
    # unites de SPAC et droits ne sont pas des societes, et le site ne les valorise
    # pas. Les reintroduire ici gonflerait le taux d'effondrement de toutes les
    # notes indistinctement — une unite de SPAC liquidee n'apprend rien sur la
    # capacite de la note a lire des comptes.
    apres_t0 = {s: d for s, d in radiees.items()
                if str(d.get("date"))[:10] > jour_t0
                and (d.get("exchange") or "").upper() in DEVISE_DE_LA_PLACE
                and not fmp._est_un_derive(s, d.get("name"))
                and not fmp._is_preferred(s, d.get("name"))}
    print(f"  dont radiees depuis l'instantane (hors derives) : {len(apres_t0)}")

    random.seed(graine)
    vivantes = random.sample(sorted(uni), min(echantillon, len(uni)))
    mortes = sorted(apres_t0)
    if len(mortes) > echantillon // 3:
        mortes = random.sample(mortes, echantillon // 3)
    # Une societe radiee n'est plus dans le screener : on lui fabrique la fiche
    # minimale dont la note a besoin (place de cotation inconnue -> non renseignee).
    cibles = ([(s, uni[s]) for s in vivantes]
              + [(s, {"exchange": apres_t0[s].get("exchange"),
                      "name": apres_t0[s].get("name")}) for s in mortes])
    print(f"  a evaluer : {len(cibles)} ({len(vivantes)} cotees + {len(mortes)} radiees)\n")

    # Plafond MESURE sur l'univers du jour, majore d'un ordre de grandeur : au-dela,
    # les deux entrees de la reconstitution (cours d'epoque, actions deposees) se
    # contredisent, et aucune des deux ne prouve que l'autre a tort.
    plus_grosse = max((u.get("market_cap") or 0.0) for u in uni.values()) / 1e9
    plafond_capi = plus_grosse * 10.0
    print(f"  plus grosse capitalisation observee : {plus_grosse:.0f} Md$ "
          f"-> plafond de reconstitution {plafond_capi:.0f} Md$\n")

    res, ecartes = [], 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(evaluer, s, sr, jour_t0, radiees, plafond_capi)
                for s, sr in cibles]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            if r and r.get("ecarte"):
                ecartes += 1
            elif r:
                res.append(r)
            if i % 200 == 0:
                print(f"  {i}/{len(cibles)} (notes reconstituees : {len(res)})")

    if len(res) < 50:
        print(f"\nECHEC : seulement {len(res)} notes reconstituees, "
              f"aucune conclusion possible.")
        return 1

    par_grade = agreger(res, "grade")
    par_famille = agreger(res, "famille")

    print(f"\n{len(res)} societes notees retrospectivement "
          f"({sum(1 for r in res if r['issue']=='radiee')} radiees depuis)\n")
    print(f"  {'grade':6} {'n':>5} {'effondr.':>9} {'taux':>7} "
          f"{'IC 95 %':>16} {'rend. med.':>11} {'creux med.':>11} {'cap. med.':>10}")
    for l in par_famille:
        cap = l.get("cap_mediane")
        print(f"  {l['grade']:6} {l['n']:5} {l['effondrements']:9} "
              f"{l['taux']*100:6.1f}% [{l['ic_bas']*100:5.1f}%,{l['ic_haut']*100:5.1f}%] "
              f"{l['rendement_median']*100:10.1f}% {l['creux_median']*100:10.1f}% "
              f"{(cap*1000 if cap else 0):9.0f}M$")

    # Monotonie : le taux d'effondrement doit croitre de A vers F. C'est LA
    # propriete que la note revendique ; tout le reste est decoratif.
    paires = [(FAMILLES.index(l["grade"]), l["taux"])
              for l in par_famille if l["grade"] in FAMILLES]
    rho = spearman(paires)
    a = next((l for l in par_famille if l["grade"] == "A"), None)
    f_ = next((l for l in par_famille if l["grade"] == "F"), None)
    print()
    if rho is not None:
        print(f"  monotonie A -> F (Spearman sur les familles) : rho = {rho:+.3f}")
    z = pval = None
    if a and f_:
        sep = f_["taux"] - a["taux"]
        z, pval = test_deux_proportions(a["effondrements"], a["n"],
                                        f_["effondrements"], f_["n"])
        verdict = ("separation non etablie" if pval is None or pval >= 0.05
                   else f"separation etablie, p = {pval:.4f}")
        print(f"  ecart F - A : {sep*100:+.1f} points  ({verdict})")

    # La monotonie du CREUX est le second test, et le plus severe : le rendement
    # final peut etre sauve par un rebond, le plus bas traverse ne l'est pas.
    creux = [(FAMILLES.index(l["grade"]), -l["creux_median"])
             for l in par_famille if l["grade"] in FAMILLES]
    rho_creux = spearman(creux)
    if rho_creux is not None:
        print(f"  monotonie du plus-bas traverse : rho = {rho_creux:+.3f}")

    # --- A taille comparable ---------------------------------------------
    strates = par_taille(res)
    if strates:
        print("\n  a TAILLE COMPARABLE (paquets de capitalisation, effectifs egaux) :")
        for st in strates:
            print(f"    paquet {st['paquet']} — {st['cap_min']*1000:.0f} a "
                  f"{st['cap_max']*1000:.0f} M$, n={st['n']}, "
                  f"rho={st['spearman'] if st['spearman'] is None else format(st['spearman'], '+.3f')}")
            for l in st["familles"]:
                print(f"      {l['grade']:3} n={l['n']:4} effondrements "
                      f"{l['taux']*100:5.1f}%  creux med. {l['creux_median']*100:7.1f}%")

    # --- Quelle dimension porte le signal ? --------------------------------
    dims = pouvoir_des_dimensions(res)
    if dims:
        print("\n  POUVOIR DE CHAQUE DIMENSION (AUC ; 0,500 = aucune information) :")
        print(f"    {'dimension':28} {'AUC':>6} {'+/-':>6} {'sigmas':>7} "
              f"{'note ici':>9} {'couvre':>7}  verdict")
        for d in dims:
            if d["auc"] is None:
                print(f"    {d['nom']:28} {'—':>6} {'—':>6} {'—':>7} {'—':>9} "
                      f"{'—':>7}  {d['verdict']}")
                continue
            an = d["auc_note_meme_sous_echantillon"]
            print(f"    {d['nom']:28} {d['auc']:6.3f} {d['ecart_type']:6.3f} "
                  f"{d['ecarts_types_au_hasard']:7.1f} {an:9.3f} "
                  f"{d['couverture']*100:6.1f}%  {d['verdict']}")
        auc_note = aire_sous_la_courbe([(r["score"], r["effondre"]) for r in res
                                        if r.get("score") is not None])
        if auc_note is not None:
            print(f"    {'NOTE COMPLETE (tous titres)':28} {auc_note:6.3f}")

    # --- L'upside predit-il le rendement ? ---------------------------------
    paquets = pouvoir_de_la_valorisation(res)
    auc_upside = rho_upside = None
    if paquets:
        print("\n  POUVOIR DE LA VALORISATION (paquets d'upside, effectifs egaux) :")
        print(f"    {'paquet':7} {'n':>5} {'upside':>18} {'rend. med.':>11} "
              f"{'creux med.':>11} {'effondr.':>9}")
        for p in paquets:
            print(f"    {p['paquet']:7} {p['n']:5} "
                  f"{p['upside_min']*100:7.0f}% a {p['upside_max']*100:6.0f}% "
                  f"{p['rendement_median']*100:10.1f}% {p['creux_median']*100:10.1f}% "
                  f"{p['taux_effondrement']*100:8.1f}%")
        rho_upside = spearman([(p["paquet"], p["rendement_median"]) for p in paquets])
        print(f"    monotonie du rendement selon l'upside : rho = {rho_upside:+.3f}")
        # Mesure continue, sans decoupage : l'upside classe-t-il mieux qu'un tirage
        # a pile ou face les titres qui ont SURPERFORME la mediane de l'echantillon ?
        avec = [r for r in res if r.get("upside") is not None]
        med = mediane([r["rendement"] for r in avec])
        auc_upside, se_up = _auc([(r["upside"], r["rendement"] > med) for r in avec])
        if auc_upside is not None:
            sig = (auc_upside - 0.5) / se_up if se_up else 0.0
            print(f"    AUC de l'upside contre la mediane de rendement : "
                  f"{auc_upside:.3f} ± {se_up:.3f} ({sig:+.1f} ecarts-types)")

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(json.dumps({
        "instantane": jour_t0, "horizon_mois": horizon,
        "seuil_effondrement": SEUIL_EFFONDREMENT,
        "n": len(res), "radiees": sum(1 for r in res if r["issue"] == "radiee"),
        "ecartes_capitalisation_impossible": ecartes,
        "par_grade": par_grade, "par_famille": par_famille,
        "spearman_familles": rho, "spearman_creux": rho_creux,
        "par_taille": strates, "dimensions": dims,
        "valorisation": {"paquets": paquets, "spearman": rho_upside,
                         "auc_upside": auc_upside},
        "z_F_contre_A": z, "p_F_contre_A": pval,
        "genere": date.today().isoformat(),
        # Les lignes brutes : sans elles, toute anomalie du tableau agrege est
        # indiagnosticable. C'est ainsi qu'une capitalisation de 1,98e17 dollars
        # — cent mille fois la richesse mondiale — est restee invisible une passe.
        "titres": sorted(res, key=lambda r: -(r["cap_t0"] or 0)),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # RESUME SEPARE : c'est lui que la fiche va chercher au chargement. Le fichier
    # complet pese 400 ko parce qu'il porte le detail par titre — indispensable
    # pour diagnostiquer une anomalie, inutile a l'affichage, et hors de question
    # a telecharger sur chaque profil consulte.
    RESUME.write_text(json.dumps({
        "instantane": jour_t0, "horizon_mois": horizon,
        "seuil_effondrement": SEUIL_EFFONDREMENT,
        "n": len(res), "radiees": sum(1 for r in res if r["issue"] == "radiee"),
        "par_famille": [{k: l[k] for k in ("grade", "n", "taux", "ic_bas", "ic_haut",
                                           "rendement_median", "creux_median",
                                           "cap_mediane")}
                        for l in par_famille],
        "dimensions": [{k: x.get(k) for k in ("dimension", "nom", "auc",
                                              "ecart_type", "ecarts_types_au_hasard",
                                              "couverture", "verdict")} for x in dims],
        "auc_note": _auc([(r["score"], r["effondre"]) for r in res
                          if r.get("score") is not None])[0],
        "valorisation": {"paquets": paquets, "spearman": rho_upside,
                         "auc_upside": auc_upside},
        "spearman_familles": rho, "spearman_creux": rho_creux,
        "p_F_contre_A": pval,
        "ecart_F_moins_A": (f_["taux"] - a["taux"]) if (a and f_) else None,
        "par_taille": [{"paquet": st["paquet"], "cap_min": st["cap_min"],
                        "cap_max": st["cap_max"], "n": st["n"],
                        "spearman": st["spearman"]} for st in strates],
        "genere": date.today().isoformat(),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n  -> {SORTIE.relative_to(RACINE)} + {RESUME.name} "
          f"({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    kw = {}
    args = sys.argv[1:]
    while args:
        a = args.pop(0)
        if a == "--horizon":
            kw["horizon"] = int(args.pop(0))
        elif a == "--echantillon":
            kw["echantillon"] = int(args.pop(0))
        elif a == "--workers":
            kw["workers"] = int(args.pop(0))
    raise SystemExit(main(**kw))

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

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "app" / "us" / "_validation_risque.json"

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
def evaluer(sym, sr, jour_t0: str, radiees: dict):
    """Note reconstituee a `jour_t0`, puis devenir du titre jusqu'a aujourd'hui."""
    try:
        entry = fmp.statements(sym, limit=14)
        if not entry or not entry.get("income"):
            return None
        vieux = tronquer(entry, jour_t0)
        if len(vieux["income"]) < 2:
            return None                       # pas d'historique publie a la date

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
        f = fmp.fundamentals_from_fmp(sym, sr_t0, vieux, {})
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

        motifs = valider(f, F, vieux)
        rep = reparer(sym, f, F, vieux, motifs) if motifs else []
        if rep:
            motifs = valider(f, F, vieux)
        n = noter(f, F, motifs, rep)
        grade = n.get("grade")
        if not grade:
            return None

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

        return {"ticker": sym, "grade": grade, "famille": grade[0],
                "score": n.get("score"), "regime": n.get("regime"),
                "cap_t0": f["market_cap"], "rendement": rendement,
                "creux": creux, "issue": issue,
                "effondre": bool(rendement <= SEUIL_EFFONDREMENT)}
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Statistiques
# --------------------------------------------------------------------------- #
def wilson(k, n, z=1.96):
    """Intervalle de confiance de Wilson pour une proportion.

    Preferé a l'intervalle normal : avec dix observations dans un grade et zero
    effondrement, l'intervalle normal donne [0 %, 0 %], ce qui affirmerait une
    certitude que dix observations ne peuvent pas porter.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    e = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - e), min(1.0, c + e))


def test_deux_proportions(k1, n1, k2, n2):
    """Test z sur la difference de deux proportions. Renvoie (z, p bilateral).

    Il remplace une comparaison d'intervalles de confiance, qui donnait ici un
    verdict FAUX. Les intervalles de Wilson de A [15,3 % ; 32,1 %] et de F
    [32,0 % ; 52,4 %] se chevauchent de un dixieme de point, d'ou la conclusion
    "separation non etablie" — alors que le test direct de la difference donne
    p = 0,006. Le chevauchement de deux intervalles est un critere CONSERVATEUR :
    son absence prouve la difference, sa presence ne prouve rien. Conclure de l'un
    a l'autre revient a declarer un instrument aveugle parce qu'on l'a mal lu.
    """
    if n1 == 0 or n2 == 0:
        return (None, None)
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (None, None)
    z = (p2 - p1) / se
    # Fonction de repartition normale via erf : evite une dependance a scipy dans
    # un script destine a tourner en integration continue.
    p_bilateral = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (z, p_bilateral)


def mediane(xs):
    s = sorted(xs)
    if not s:
        return None
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def spearman(paires):
    """Correlation de rang entre le rang du grade et le taux d'effondrement."""
    if len(paires) < 3:
        return None
    xs = [p[0] for p in paires]
    ys = [p[1] for p in paires]

    def rangs(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            moy = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[o[k]] = moy
            i = j + 1
        return r

    rx, ry = rangs(xs), rangs(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else None


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

    # Les radiees APRES l'instantane sont les evenements que le test doit voir.
    # On leur applique le MEME filtre que l'univers courant : bons de souscription,
    # unites de SPAC et droits ne sont pas des societes, et le site ne les valorise
    # pas. Les reintroduire ici gonflerait le taux d'effondrement de toutes les
    # notes indistinctement — une unite de SPAC liquidee n'apprend rien sur la
    # capacite de la note a lire des comptes.
    apres_t0 = {s: d for s, d in radiees.items()
                if str(d.get("date"))[:10] > jour_t0
                and (d.get("exchange") or "").upper() in DEVISE_DE_LA_PLACE
                and not fmp._est_un_derive(s, d.get("name"))}
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

    res = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(evaluer, s, sr, jour_t0, radiees) for s, sr in cibles]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            if r:
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

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(json.dumps({
        "instantane": jour_t0, "horizon_mois": horizon,
        "seuil_effondrement": SEUIL_EFFONDREMENT,
        "n": len(res), "radiees": sum(1 for r in res if r["issue"] == "radiee"),
        "par_grade": par_grade, "par_famille": par_famille,
        "spearman_familles": rho, "spearman_creux": rho_creux,
        "par_taille": strates,
        "z_F_contre_A": z, "p_F_contre_A": pval,
        "genere": date.today().isoformat(),
        # Les lignes brutes : sans elles, toute anomalie du tableau agrege est
        # indiagnosticable. C'est ainsi qu'une capitalisation de 1,98e17 dollars
        # — cent mille fois la richesse mondiale — est restee invisible une passe.
        "titres": sorted(res, key=lambda r: -(r["cap_t0"] or 0)),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  -> {SORTIE.relative_to(RACINE)} ({time.time()-t0:.0f}s)")
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

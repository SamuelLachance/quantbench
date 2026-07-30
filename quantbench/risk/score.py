"""Agregation des dimensions en une note de A+ a F.

TROIS GELS, qui font la difference entre une NOTE et un CLASSEMENT :

  1. les tables de quantiles de chaque dimension sont mesurees une fois sur une
     population de reference datee, puis LUES par le build ;
  2. les bornes de grade sont mesurees en meme temps, sur la meme population ;
  3. les niveaux de plafond aussi.

Sans ces gels, tous les rangs seraient calcules sur la coupe du jour, leurs marges
seraient uniformes PAR CONSTRUCTION, et la distribution des notes serait invariante :
dans une crise ou toutes les couvertures d'interets s'effondrent ensemble, personne ne
serait degrade. Avec eux, la distribution des grades a le DROIT de se deplacer, ce qui
est le comportement attendu d'une notation et ce qui la rend surveillable.

Benefice technique decisif : aucune statistique de l'univers du jour n'est requise,
donc les cinq shards du build restent independants et aucune passe d'agregation
supplementaire n'est necessaire.
"""

from __future__ import annotations

import json
import math
import os

from .dimensions import DIMENSIONS, mesurer

# Treize grades. Le NOMBRE est une contrainte d'affichage assumee comme telle ; tout
# le reste — les bornes qui les separent — est mesure.
GRADES = ("A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F")

_ICI = os.path.dirname(os.path.abspath(__file__))
_FICHIER = os.path.join(_ICI, "risk_calibration.json")


def _charger():
    try:
        with open(_FICHIER, encoding="utf-8") as f:
            return json.load(f)
    except Exception:                                  # noqa: BLE001
        return None


_CAL = _charger()


def rang(cle, signal, cal, secteur=None, industrie=None, niveau="univers"):
    """Rang percentile du signal dans la table GELEE — la part de la population de
    reference qui fait STRICTEMENT mieux.

    Le percentile est UNILATERAL, et ce n'est pas un detail : avec un percentile a
    rang moyen, une societe irreprochable sur une dimension a mode propre — zero
    dilution, zero motif de validation, ecart de resultat nul — recevrait 0,50 alors
    qu'elle merite 0. Une societe parfaite ressortait ainsi a la moyenne."""
    if signal is None:
        return None
    # Les MODALITES bornent l'echelle sans passer par la table : -inf designe le
    # meilleur cas possible de la dimension (aucune dette, autofinancee, aucune
    # dilution), +inf le pire (aucun chiffre d'affaires). Les faire passer par les
    # centiles les melangerait aux MESURES : une societe sans charge d'interets se
    # retrouvait classee plus risquee qu'une societe couvrant les siennes trois fois.
    if signal == math.inf:
        return 1.0
    if signal == -math.inf:
        return 0.0
    tables = (cal or {}).get("quantiles", {}).get(cle) or {}
    grille = None
    if niveau == "secteur" and secteur:
        grille = tables.get("secteurs", {}).get(secteur)
    elif niveau == "industrie" and industrie:
        grille = (tables.get("industries", {}).get(industrie)
                  or tables.get("secteurs", {}).get(secteur or ""))
    if not grille:
        grille = tables.get("global")
    if not grille:
        return None
    # `grille` est la liste des centiles du signal, du plus faible au plus fort.
    n = len(grille)
    bas = 0
    for i, borne in enumerate(grille):
        if signal > borne:
            bas = i + 1
        else:
            break
    return min(1.0, max(0.0, bas / n))


def _residu_liquidite(fund, cal):
    """Volume echange corrige de ce que la TAILLE laisse attendre.

    La part d'illiquidite qu'explique la capitalisation est deja tarifee dans le cout
    des fonds propres, par la prime de taille. L'entrer telle quelle la compterait
    deux fois et la note ne ferait que reproduire un classement par taille."""
    vol, cap = fund.get("volume_dollars_median"), fund.get("market_cap")
    if not vol or vol <= 0 or not cap or cap <= 0:
        return None
    reg = ((cal or {}).get("liquidite", {}) or {}).get(fund.get("exchange") or "", None)
    if not reg:
        reg = ((cal or {}).get("liquidite", {}) or {}).get("global")
    if not reg:
        return None
    attendu = reg["a"] + reg["b"] * math.log10(cap * 1e9)
    return -(math.log10(vol) - attendu)          # signal oriente RISQUE


def _modalites(fund, F, motifs, mesures):
    """Faits VERIFIABLES qui plafonnent la note. Ce ne sont pas des jugements : chacun
    est une observation qu'aucune qualite par ailleurs ne compense."""
    from ..valuation.route import consommation_de_tresorerie, valeur_de_liquidation
    out = []
    ta, tl = fund.get("total_assets"), fund.get("total_liab")
    if ta and tl is not None and valeur_de_liquidation(fund) <= 0 and tl > ta:
        out.append("passif_non_couvert")
    be = fund.get("book_equity")
    conso = consommation_de_tresorerie(fund)
    if be is not None and be <= 0 and conso > 0:
        out.append("fonds_propres_absorbes")
    if conso > 0:
        cash = max(fund.get("cash") or 0.0, 0.0)
        if cash / conso < 1.0:
            out.append("moins_d_un_an_d_autonomie")
    for m in (motifs or []):
        if "identite du bilan" in m or "taux de change" in m:
            out.append("bilan_non_verifiable")
            break
    if not ((F or {}).get("years")):
        out.append("aucun_exercice_exploitable")
    return out


def noter(fund, F=None, motifs=None, reparations=None, cal=None):
    """Note de risque complete. Retourne toujours un resultat : une note doit exister
    pour TOUS les titres, y compris ceux dont les donnees sont pauvres — l'incertitude
    sur la donnee est elle-meme un risque, et c'est la dimension D7 qui la porte."""
    cal = cal if cal is not None else _CAL
    signaux = mesurer(fund, F, motifs, reparations,
                      (cal or {}).get('seuil_amortissement'))
    reg = signaux.pop("__regime__")
    if fund.get("volume_dollars_median"):
        residu = _residu_liquidite(fund, cal)
        if residu is not None:
            signaux["d6"] = (residu, "volume quotidien échangé, corrigé de la taille")

    poids = (cal or {}).get("poids") or {}
    secteur, industrie = fund.get("sector"), fund.get("industry")
    detail, total, somme_poids, rangs = [], 0.0, 0.0, []
    for cle, nom, _f, niveau in DIMENSIONS:
        signal, libelle = signaux.get(cle, (None, None))
        r = rang(cle, signal, cal, secteur, industrie, niveau)
        w = float(poids.get(cle, 1.0))
        detail.append({"cle": cle, "nom": nom, "libelle": libelle,
                       "signal": (None if signal is None or signal == math.inf
                                  else round(signal, 4)),
                       "rang": None if r is None else round(r, 3),
                       "poids": w, "niveau": niveau})
        if r is not None and w > 0:
            total += w * r
            somme_poids += w
            rangs.append(r)

    if not rangs:
        # Aucune dimension mesurable : la note EXISTE quand meme, et elle dit
        # exactement cela. C'est la branche de repli, plafonnee.
        return _finaliser(1.0, ["aucune_dimension_mesurable"], detail, reg, cal,
                          repli=True)

    moyenne = total / somme_poids
    # MAILLON FAIBLE, DEBIAISE. Une societe ne fait pas defaut "en moyenne", elle fait
    # defaut par son point le plus faible. Mais le maximum de k rangs uniformes vaut
    # k/(k+1) en esperance — 0,75 a trois dimensions, 0,90 a huit : une societe aux
    # comptes riches serait mecaniquement moins bien notee qu'une societe opaque, a
    # risque egal. On corrige par la p-valeur du maximum sous uniformite.
    k = len(rangs)
    maximum = max(rangs)
    debiaise = 1.0 - (1.0 - maximum) ** k
    lam = float((cal or {}).get("lambda_maillon_faible", 0.0))
    score = (1.0 - lam) * moyenne + lam * debiaise
    return _finaliser(score, _modalites(fund, F, motifs, signaux), detail, reg, cal)


def _finaliser(score, modalites, detail, reg, cal, repli=False):
    bornes = (cal or {}).get("bornes_grades")
    plafonds = (cal or {}).get("plafonds") or {}
    applique = []
    for m in modalites:
        niveau = plafonds.get(m)
        if niveau is not None and niveau > score:
            score = niveau
            applique.append(m)
        elif niveau is not None:
            applique.append(m)
    if repli:
        score = max(score, plafonds.get("aucune_dimension_mesurable", score))

    if bornes:
        i = sum(1 for b in bornes if score > b)
        grade = GRADES[min(i, len(GRADES) - 1)]
    else:
        grade = None                                  # calibrage absent
    return {
        "grade": grade,
        "score": round(float(score), 4),
        "regime": reg,
        "dimensions": detail,
        "plafonds_appliques": applique,
        "calibrage": (cal or {}).get("version"),
    }


__all__ = ["GRADES", "noter", "rang"]

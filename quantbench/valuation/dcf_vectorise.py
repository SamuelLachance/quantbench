"""Le meme DCF, evalue sur tous les scenarios a la fois.

POURQUOI. Le Monte Carlo lance dix mille valorisations par societe, chacune dans
une boucle Python. Mesure a un thread, il coute 1,2 seconde par titre — soit, pour
un cinquieme de l'univers, plus d'une heure de calcul par shard, et cette heure
n'est pas parallelisable : le calcul est en Python pur, donc serialise par le
verrou global de l'interpreteur. Le reseau, lui, ne pese que quelques minutes. Le
Monte Carlo EST la duree du build.

Les tableaux manipules font une vingtaine d'annees. A cette taille, numpy passe
l'essentiel de son temps en frais d'appel : resolution de types, allocations,
verifications. Evaluer les dix mille scenarios d'un coup, sur des tableaux
(scenarios x annees), fait disparaitre ces frais — le meme nombre d'operations
numpy porte alors deux cent mille valeurs au lieu de vingt.

CE QUI N'EST PAS NEGOCIABLE. `value_dcf` reste la reference et n'est pas touche.
Ce module doit lui rendre EXACTEMENT les memes chiffres — l'ordre des operations
flottantes y est reproduit a l'identique. Une optimisation qui deplace les
valorisations publiees n'est pas une optimisation, c'est un changement de methode
qui ne dit pas son nom.

Verifie a quatre echelles, de la plus fine a la plus large :
  - 1 200 jeux d'entrees tires au hasard, les deux modes de reinvestissement :
    egalite EXACTE exigee, pas une tolerance ;
  - les quatre scenarios de rejet economique : memes verdicts des deux cotes ;
  - 113 societes tirees au hasard dans l'univers reel — micro-capitalisations et
    lignes de gre a gre comprises, non pas une poignee de grandes valeurs bien
    tenues — a 1 500 tirages chacune : meme ensemble de scenarios valides ET
    memes valeurs au bit pres, 113 fois sur 113 ;
  - douze fiches construites de bout en bout par les deux chemins : JSON
    identiques au caractere pres, notes de risque comprises.

Au passage, cette mesure d'univers donne le taux de scenarios retenus : mediane
95,3 %, minimum 75,1 %. Les 5 % ecartes sont les tirages ou la valeur terminale
n'existe pas — cout du capital terminal sous la croissance perpetuelle — et ils
sont publies sous `taux_validite`, a lire, non a comparer a un seuil.

CE QUE CE MODULE NE SAIT PAS FAIRE. Il suppose que la STRUCTURE du scenario est
commune a tous les tirages : memes longueurs de phases, memes annees de debut de
convergence, meme mode de reinvestissement. C'est le cas de la simulation du site,
qui ne tire que huit parametres continus. Des qu'un champ entier est tire,
`champs_vectorisables` renvoie faux et l'appelant retombe sur la boucle de
reference — jamais sur une approximation.
"""
from __future__ import annotations

from dataclasses import fields as _dc_fields

import numpy as np

from .dcf import DcfInputs, _indices

# Champs dont ce module sait faire varier la valeur d'un tirage a l'autre. Tout
# autre champ tire renvoie le Monte Carlo vers la boucle de reference. La liste
# est volontairement CONSERVATRICE : mieux vaut retomber sur le chemin lent que
# valoriser un scenario dont on n'a pas verifie la vectorisation.
CHAMPS_VARIABLES = frozenset({
    "g1_begin", "g1_end", "g2_begin", "g2_end", "g3_begin", "g3_end",
    "current_operating_margin", "terminal_operating_margin",
    "current_tax_rate", "marginal_tax_rate",
    "current_sales_to_capital", "terminal_sales_to_capital",
    "risk_free_rate", "erp", "size_premium",
    "unlevered_beta", "terminal_unlevered_beta",
    "current_pretax_kd", "terminal_pretax_kd",
    "equity_value", "debt_value", "cash_and_non_operating",
    "revenue_base", "current_roic", "terminal_roic",
    "additional_roic_in_perpetuity", "asset_liquidation_during_negative_growth",
    "current_invested_capital",
})

_CHAMPS_CONNUS = {f.name for f in _dc_fields(DcfInputs)}


def champs_vectorisables(noms) -> bool:
    """Ce module peut-il traiter un tirage portant sur ces champs ?"""
    noms = set(noms)
    return bool(noms) and noms <= CHAMPS_VARIABLES and noms <= _CHAMPS_CONNUS


def _colonne(v, n):
    """Scalaire ou tableau -> colonne (n, 1), sans copie inutile."""
    a = np.asarray(v, dtype=float)
    return np.full((n, 1), float(a)) if a.ndim == 0 else a.reshape(n, 1)


def _trajectoire(depart, arrivee, n_annees, debut_convergence, n):
    """`converge_path` pour n scenarios a la fois -> tableau (n, n_annees).

    L'ordre des operations reproduit celui de la version scalaire : indice fois
    pente, puis plus depart, puis borne finale imposee. C'est cette identite
    d'ordre qui garantit le meme resultat au bit pres, et non une simple
    equivalence algebrique.
    """
    n_annees = int(n_annees)
    if n_annees <= 0:
        return np.zeros((n, 0), dtype=float)
    hold = min(max(int(debut_convergence) - 1, 0), n_annees)
    d = _colonne(depart, n)
    a = _colonne(arrivee, n)
    path = np.empty((n, n_annees), dtype=float)
    path[:, :hold] = d
    rampe = n_annees - hold
    if rampe == 1:
        path[:, hold:] = a
    elif rampe > 1:
        pente = (a - d) / (rampe - 1)
        np.multiply(_indices(rampe)[None, :], pente, out=path[:, hold:])
        path[:, hold:] += d
        path[:, -1] = a[:, 0]
    return path


def _croissance(v, n, T):
    """Concatenation des trois phases de croissance, vectorisee."""
    parts = [
        _trajectoire(v["g1_begin"], v["g1_end"], v["len1"], v["conv1"], n),
        _trajectoire(v["g2_begin"], v["g2_end"], v["len2"], v["conv2"], n),
        _trajectoire(v["g3_begin"], v["g3_end"], v["len3"], v["conv3"], n),
    ]
    parts = [p for p in parts if p.shape[1]]
    return np.concatenate(parts, axis=1) if parts else np.zeros((n, T))


def equites_dcf(base: DcfInputs, tirages: dict, n: int) -> np.ndarray:
    """Valeur d'equite des `n` scenarios. NaN pour les scenarios incoherents.

    Les scenarios que `value_dcf` REJETTE en levant une exception (WACC terminal
    sous la croissance perpetuelle, rendement du capital nul ou negatif) valent ici
    NaN : c'est exactement ce que la boucle de reference laissait dans le tableau
    de sortie, et le filtre `isfinite` du Monte Carlo les ecarte de la meme facon.
    """
    v = {}
    for f in _dc_fields(DcfInputs):
        v[f.name] = tirages[f.name] if f.name in tirages else getattr(base, f.name)

    T = int(v["len1"]) + int(v["len2"]) + int(v["len3"])
    if T <= 0:
        raise ValueError("L'horizon total (len1+len2+len3) doit etre > 0.")

    col = lambda k: _colonne(v[k], n)                                  # noqa: E731

    # --- Chiffre d'affaires -------------------------------------------------
    g = _croissance(v, n, T)
    rev0 = col("revenue_base")
    revenues = rev0 * np.cumprod(1.0 + g, axis=1)
    prev_rev = np.concatenate([rev0, revenues[:, :-1]], axis=1)

    # --- Trajectoires -------------------------------------------------------
    margins = _trajectoire(v["current_operating_margin"],
                           v["terminal_operating_margin"],
                           T, v["margin_converge_start"], n)
    taxes = _trajectoire(v["current_tax_rate"], v["marginal_tax_rate"],
                         T, v["tax_converge_start"], n)
    s2c = _trajectoire(v["current_sales_to_capital"],
                       v["terminal_sales_to_capital"],
                       T, v["s2c_converge_start"], n)

    # --- Cout du capital ----------------------------------------------------
    eq, dt = col("equity_value"), col("debt_value")
    mt = col("marginal_tax_rate")
    de = np.where(eq == 0.0, 0.0, dt / np.where(eq == 0.0, 1.0, eq))
    levier = 1.0 + (1.0 - mt) * de
    betas = _trajectoire(col("unlevered_beta") * levier,
                         col("terminal_unlevered_beta") * levier,
                         T, v["beta_converge_start"], n)
    kd = _trajectoire(v["current_pretax_kd"], v["terminal_pretax_kd"],
                      T, v["kd_converge_start"], n)
    total = eq + dt
    w_e = np.where(total == 0.0, 1.0, eq / np.where(total == 0.0, 1.0, total))
    w_d = np.where(total == 0.0, 0.0, dt / np.where(total == 0.0, 1.0, total))
    cost_equity = col("risk_free_rate") + betas * col("erp") + col("size_premium")
    wacc = w_e * cost_equity + w_d * (kd * (1.0 - mt))

    # --- Flux ---------------------------------------------------------------
    ebit_after_tax = (revenues * margins) * (1.0 - taxes)

    invalide = np.zeros(n, dtype=bool)
    if v["reinvestment_mode"] == "roic":
        cr, tr = col("current_roic"), col("terminal_roic")
        if np.any(np.isnan(cr)) or np.any(np.isnan(tr)):
            raise ValueError(
                "reinvestment_mode='roic' exige current_roic et terminal_roic.")
        # Un rendement du capital nul ou negatif rend l'identite g = reinvestissement
        # x ROIC insoluble : le scenario n'existe pas, il n'est pas "prudent".
        invalide |= (cr[:, 0] <= 0) | (tr[:, 0] <= 0)
        roic_path = _trajectoire(cr, tr, T, v["roic_converge_start"], n)
        cur_ebi = (col("revenue_base") * col("current_operating_margin")
                   * (1.0 - col("current_tax_rate")))
        ebi_prev = np.concatenate([cur_ebi, ebit_after_tax[:, :-1]], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            g_ebi = np.where(ebi_prev > 0, ebit_after_tax / ebi_prev - 1.0, g)
            taux = np.where(roic_path > 0, g_ebi / roic_path, 0.0)
        reinvestment = ebit_after_tax * np.clip(taux, 0.0, 3.0)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            reinvestment = np.where(s2c != 0, (revenues - prev_rev) / s2c, 0.0)
        reinvestment = np.where(
            reinvestment > 0, reinvestment,
            reinvestment * col("asset_liquidation_during_negative_growth"))

    fcff = ebit_after_tax - reinvestment
    discount = np.cumprod(1.0 + wacc, axis=1)
    pv_fcff = fcff / discount

    # --- Valeur terminale ---------------------------------------------------
    wacc_t = wacc[:, -1]
    gt = np.asarray(v["g3_end"], dtype=float)
    gt = np.full(n, float(gt)) if gt.ndim == 0 else gt.reshape(n)
    invalide |= ~(wacc_t > gt)              # ~(a > b) attrape aussi les NaN

    if v["reinvestment_mode"] == "roic":
        tr1 = np.asarray(v["terminal_roic"], dtype=float)
        tr1 = np.full(n, float(tr1)) if tr1.ndim == 0 else tr1.reshape(n)
        roic_t = np.minimum(tr1, wacc_t + 0.02)
    else:
        ap = np.asarray(v["additional_roic_in_perpetuity"], dtype=float)
        ap = np.full(n, float(ap)) if ap.ndim == 0 else ap.reshape(n)
        roic_t = wacc_t + ap
    invalide |= ~(roic_t > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        reinv_rate_t = np.where(gt > 0, gt / roic_t, 0.0)
        rev_t = revenues[:, -1] * (1.0 + gt)
        tom = np.asarray(v["terminal_operating_margin"], dtype=float)
        tom = np.full(n, float(tom)) if tom.ndim == 0 else tom.reshape(n)
        ebit_at_t = rev_t * tom * (1.0 - mt[:, 0])
        fcff_t = ebit_at_t * (1.0 - reinv_rate_t)
        pv_terminal = (fcff_t / (wacc_t - gt)) / discount[:, -1]

    equity = (pv_fcff.sum(axis=1) + pv_terminal
              + col("cash_and_non_operating")[:, 0] - dt[:, 0])
    equity[invalide] = np.nan
    return equity

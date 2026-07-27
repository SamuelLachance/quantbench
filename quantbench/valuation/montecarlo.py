"""
quantbench.valuation.montecarlo
===============================
Enveloppe Monte Carlo autour du DCF deterministe. On tire des scenarios sur les
hypotheses incertaines (croissance, marge, WACC, beta...), on valorise chacun,
et on obtient une DISTRIBUTION de la valeur intrinseque plutot qu'un point.

C'est la vraie valeur ajoutee vs. un DCF classique : au lieu d'une fausse
precision ("l'action vaut 142,37 $"), on repond "il y a X % de chances que la
valeur intrinseque depasse le cours actuel".

Correlations entre hypotheses gerees via une COPULE GAUSSIENNE (numpy/scipy) :
pas besoin d'OpenTURNS. Les marginales sont des lois scipy.stats gelees.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy import stats

from .dcf import DcfInputs, value_dcf


def _nearest_psd(corr: np.ndarray) -> np.ndarray:
    """Projette une matrice de correlation sur le cone semi-defini positif
    (clipping des valeurs propres) puis renormalise la diagonale a 1."""
    vals, vecs = np.linalg.eigh((corr + corr.T) / 2.0)
    vals = np.clip(vals, 1e-10, None)
    psd = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(psd), 1e-12, None))
    psd = psd / np.outer(d, d)
    np.fill_diagonal(psd, 1.0)
    return psd


def sample_inputs(distributions: dict, n: int,
                  correlations=None, seed: int = 0) -> dict:
    """Tire `n` scenarios pour chaque hypothese incertaine.

    distributions : {nom_de_champ_DcfInputs: loi scipy.stats gelee}
    correlations  : liste de (nomA, nomB, rho) entre paires d'hypotheses.
    Retourne {nom: ndarray(n)}.
    """
    rng = np.random.default_rng(seed)
    names = list(distributions)
    k = len(names)

    corr = np.eye(k)
    if correlations:
        idx = {name: i for i, name in enumerate(names)}
        for a, b, rho in correlations:
            if a in idx and b in idx:
                corr[idx[a], idx[b]] = corr[idx[b], idx[a]] = rho
        corr = _nearest_psd(corr)

    # Copule gaussienne : normales correlees -> uniformes -> quantiles des marginales
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, k)) @ chol.T
    u = stats.norm.cdf(z)
    return {name: np.asarray(distributions[name].ppf(u[:, i]), dtype=float)
            for i, name in enumerate(names)}


# Champs qui doivent rester entiers apres tirage
_INT_FIELDS = {"len1", "len2", "len3", "conv1", "conv2", "conv3",
               "margin_converge_start", "tax_converge_start",
               "s2c_converge_start", "beta_converge_start", "kd_converge_start"}


def monte_carlo_dcf(base: DcfInputs, distributions: dict, *,
                    n: int = 5000, correlations=None,
                    shares_outstanding: float | None = None,
                    current_market_cap: float | None = None,
                    seed: int = 0) -> dict:
    """Lance `n` valorisations DCF sur des scenarios tires.

    Renvoie la distribution des valeurs intrinseques d'equite + des statistiques
    utiles (percentiles, prob. de sous-valorisation vs. capitalisation actuelle).
    Les scenarios qui font echouer le DCF (ex. WACC terminal <= g) sont ignores.
    """
    samples = sample_inputs(distributions, n, correlations, seed)
    equity = np.full(n, np.nan)

    for d in range(n):
        overrides = {}
        for name, arr in samples.items():
            v = arr[d]
            overrides[name] = int(round(v)) if name in _INT_FIELDS else float(v)
        try:
            equity[d] = value_dcf(replace(base, **overrides))["equity_value"]
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue  # scenario incoherent -> ecarte

    valid = equity[~np.isnan(equity)]
    if valid.size == 0:
        raise RuntimeError("Aucun scenario Monte Carlo valide. Verifiez les lois.")

    pct_levels = [5, 10, 25, 50, 75, 90, 95]
    out = {
        "equity_values": valid,
        "n_valid": int(valid.size),
        "n_total": n,
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid)),
        "percentiles": {p: float(np.percentile(valid, p)) for p in pct_levels},
    }
    if current_market_cap is not None and current_market_cap > 0:
        out["prob_undervalued"] = float(np.mean(valid > current_market_cap))
        out["median_upside"] = float(np.median(valid) / current_market_cap - 1.0)
    if shares_outstanding is not None and shares_outstanding > 0:
        out["value_per_share"] = {
            "median": float(np.median(valid) / shares_outstanding),
            "p10": float(np.percentile(valid, 10) / shares_outstanding),
            "p90": float(np.percentile(valid, 90) / shares_outstanding),
        }
    return out


__all__ = ["monte_carlo_dcf", "sample_inputs"]

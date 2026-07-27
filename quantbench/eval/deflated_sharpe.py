"""
quantbench.eval.deflated_sharpe
===============================
Le Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) — la piece d'honnetete
du banc d'essai. Il corrige le Sharpe observe pour :
  1. le BIAIS DE SELECTION (on a essaye N configurations et garde la meilleure) ;
  2. la NON-NORMALITE des rendements (asymetrie, kurtosis).

Resultat mathematique cle (non perissable) : avec N essais independants de vrai
Sharpe nul, le meilleur Sharpe observe monte mecaniquement (E[max]≈3,26 pour
N=1000 si la variance des Sharpe d'essai vaut 1). Le DSR teste le Sharpe observe
CONTRE ce maximum attendu sous l'hypothese nulle.

Convention : les Sharpe passes ici sont PAR PERIODE (non annualises) ; `n` est le
nombre d'observations. Annualiser separement pour l'affichage.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def sharpe_per_period(returns) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) if r.size > 1 else 0.0
    return float(r.mean() / sd) if sd > 0 else 0.0


def annualized_sharpe(returns, periods_per_year: int = 252) -> float:
    return sharpe_per_period(returns) * np.sqrt(periods_per_year)


def probabilistic_sharpe_ratio(sr, n, skew=0.0, kurt=3.0, sr_benchmark=0.0) -> float:
    """P(vrai Sharpe > benchmark) etant donne le Sharpe observe `sr` (par periode),
    `n` observations, l'asymetrie et la kurtosis des rendements."""
    n = int(n)
    if n < 2:
        return float("nan")
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2))
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Sharpe maximum attendu (par periode) sous l'hypothese nulle, pour `n_trials`
    essais independants dont les Sharpe ont une variance `sr_variance`."""
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    N = int(n_trials)
    z1 = norm.ppf(1 - 1.0 / N)
    z2 = norm.ppf(1 - 1.0 / (N * np.e))
    return float(np.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(sr, sr_trials, n, skew=0.0, kurt=3.0) -> dict:
    """DSR : probabilite que le Sharpe observe soit reel apres correction du biais
    de selection (via la dispersion des Sharpe des `sr_trials` essais) et de la
    non-normalite. Retourne un dict avec le DSR et le benchmark deflate."""
    sr_trials = np.asarray(sr_trials, dtype=float)
    sr_trials = sr_trials[np.isfinite(sr_trials)]
    var = sr_trials.var(ddof=1) if sr_trials.size > 1 else 0.0
    bench = expected_max_sharpe(var, sr_trials.size)
    dsr = probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=bench)
    return {"dsr": dsr, "deflated_benchmark": bench,
            "n_trials": int(sr_trials.size), "sr_trials_var": float(var)}


def min_track_record_length(sr, skew=0.0, kurt=3.0, sr_benchmark=0.0, prob=0.95) -> float:
    """Nombre minimal d'observations pour etre confiant a `prob` que le vrai Sharpe
    depasse le benchmark (MinTRL, Bailey & Lopez de Prado)."""
    if sr <= sr_benchmark:
        return float("inf")
    denom = 1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2
    return float(1 + denom * (norm.ppf(prob) / (sr - sr_benchmark)) ** 2)


__all__ = ["sharpe_per_period", "annualized_sharpe", "probabilistic_sharpe_ratio",
           "expected_max_sharpe", "deflated_sharpe_ratio", "min_track_record_length"]

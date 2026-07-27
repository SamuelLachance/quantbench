"""
quantbench.eval.pbo
===================
Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado, Zhu, 2015)
par validation croisee combinatoire symetrique (CSCV).

Idee : on decoupe l'historique en S blocs, on forme toutes les combinaisons de
S/2 blocs "in-sample" (IS) / S/2 "out-of-sample" (OOS). Pour chaque combinaison,
on prend la strategie championne IS et on regarde son RANG OOS. Si la championne
IS finit souvent dans la moitie basse OOS, la selection est surajustee.

PBO = P(la meilleure strategie IS soit sous la mediane OOS). Eleve => overfitting.
Complementaire du Deflated Sharpe : le DSR juge UNE strategie ; le PBO juge le
PROCESSUS DE SELECTION lui-meme.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _sharpe_cols(sub: np.ndarray) -> np.ndarray:
    mu = sub.mean(axis=0)
    sd = sub.std(axis=0, ddof=1)
    return np.where(sd > 0, mu / sd, 0.0)


def probability_of_backtest_overfitting(returns_matrix, n_splits: int = 10) -> dict:
    """returns_matrix : (T observations, N strategies). Retourne {pbo, ...}."""
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return {"pbo": float("nan"), "n_combinations": 0, "n_strategies": 0}
    T, N = M.shape
    S = n_splits - (n_splits % 2)               # nombre pair de blocs
    S = max(S, 2)
    blocks = np.array_split(np.arange(T), S)

    logits = []
    for is_sel in combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[b] for b in is_sel])
        oos_idx = np.concatenate([blocks[b] for b in range(S) if b not in is_sel])
        sr_is = _sharpe_cols(M[is_idx])
        sr_oos = _sharpe_cols(M[oos_idx])
        best = int(np.argmax(sr_is))            # championne in-sample
        # rang relatif OOS de la championne (1 = meilleure, ->0 = pire)
        ranks = np.argsort(np.argsort(sr_oos))
        w = (ranks[best] + 1) / (N + 1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))

    logits = np.asarray(logits)
    return {
        "pbo": float(np.mean(logits <= 0)),     # sous la mediane OOS => overfit
        "n_combinations": int(logits.size),
        "n_strategies": int(N),
        "median_logit": float(np.median(logits)),
    }


__all__ = ["probability_of_backtest_overfitting"]

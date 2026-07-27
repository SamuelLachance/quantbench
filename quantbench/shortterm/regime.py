"""
quantbench.shortterm.regime
===========================
Detection de regimes de marche par melange gaussien (features : rendement + vol
realisee). C'est le substitut pragmatique honnete du HMM (Baum-Welch) : un GMM
est le modele d'emission d'un HMM sans la structure de transition. Suffisant pour
etiqueter les regimes (haussier / neutre / baissier) ; le HMM ajouterait la
persistance temporelle des etats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def detect_regimes(prices, n_regimes: int = 3, vol_lookback: int = 10, seed: int = 0):
    """Retourne (labels, current_label, names, proba_courante).

    names : {label -> 'baissier'|'neutre'|'haussier'} ordonne par rendement moyen.
    """
    p = np.asarray(prices, dtype=float)
    ret = np.diff(np.log(p))
    vol = pd.Series(ret).rolling(vol_lookback).std(ddof=0).bfill().values
    X = np.column_stack([ret, vol])
    X = X[np.isfinite(X).all(axis=1)]

    gm = GaussianMixture(n_components=n_regimes, covariance_type="full",
                         random_state=seed, n_init=3).fit(X)
    labels = gm.predict(X)

    mean_ret = [X[labels == k, 0].mean() if (labels == k).any() else 0.0
                for k in range(n_regimes)]
    order = np.argsort(mean_ret)
    if n_regimes == 3:
        names = {int(order[0]): "baissier", int(order[1]): "neutre",
                 int(order[2]): "haussier"}
    else:
        names = {int(k): f"regime_{r}" for r, k in enumerate(order)}

    current = int(labels[-1])
    proba = gm.predict_proba(X[-1:].reshape(1, -1))[0]
    return {"labels": labels, "current": current, "names": names,
            "current_name": names.get(current, str(current)),
            "current_proba": float(proba[current]),
            "mean_returns": {names.get(int(k), str(k)): round(float(mean_ret[k]), 5)
                             for k in range(n_regimes)}}


__all__ = ["detect_regimes"]

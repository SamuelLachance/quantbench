"""
quantbench.shortterm.signals
============================
Signaux court terme de retour a la moyenne (heritage stat-arb / Ornstein-Uhlenbeck).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_meanrev(prices, lookback=20, entry=1.0, exit=0.5):
    """Signal de mean-reversion sur le z-score du prix vs sa moyenne mobile.

    Entre SHORT quand z > entry (prix trop haut), LONG quand z < -entry, et sort
    quand |z| < exit. Machine a etats (pas de re-entree tant qu'on n'est pas sorti).
    Retourne (positions, zscores), positions dans {-1, 0, +1}.
    """
    p = pd.Series(np.asarray(prices, dtype=float))
    ma = p.rolling(lookback).mean()
    sd = p.rolling(lookback).std(ddof=0)
    z = (p - ma) / sd

    pos = np.zeros(len(p))
    state = 0
    zv = z.values
    for i in range(len(p)):
        zi = zv[i]
        if not np.isfinite(zi):
            pos[i] = 0
            continue
        if state == 0:
            if zi > entry:
                state = -1
            elif zi < -entry:
                state = 1
        elif abs(zi) < exit:
            state = 0
        pos[i] = state
    return pos, zv


def ou_half_life(prices) -> float:
    """Demi-vie de retour a la moyenne d'un processus d'Ornstein-Uhlenbeck, estimee
    par AR(1) sur le log-prix : dx_t = a + b*x_{t-1} + e. half-life = -ln(2)/b."""
    x = np.log(np.asarray(prices, dtype=float))
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("inf")
    dx = np.diff(x)
    b = np.polyfit(x[:-1], dx, 1)[0]
    return float(-np.log(2) / b) if b < 0 else float("inf")


__all__ = ["zscore_meanrev", "ou_half_life"]

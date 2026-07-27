"""
quantbench.backtest.engine
==========================
Backtest vectorise minimal, mais HONNETE :
  * delai d'une barre (la position decidee a t s'applique au rendement t->t+1) :
    aucun look-ahead ;
  * couts de transaction preleves sur le turnover (changement de position).
"""

from __future__ import annotations

import numpy as np


def backtest(prices, positions, cost_bps: float = 10.0) -> dict:
    """Retourne les rendements nets periode par periode + diagnostics.

    prices    : serie de prix (longueur T).
    positions : position visee a chaque date (-1..+1), longueur T. La position a
                l'indice t est APPLIQUEE au rendement de t a t+1 (pas de look-ahead).
    cost_bps  : cout aller (bps) applique au |changement de position|.
    """
    p = np.asarray(prices, dtype=float)
    pos = np.asarray(positions, dtype=float)
    if p.size < 2:
        return {"net": np.zeros(0), "gross": np.zeros(0), "turnover": 0.0}

    ret = np.diff(p) / p[:-1]              # rendement t->t+1, longueur T-1
    pos_held = pos[:-1]                    # position connue a t, appliquee sur ret[t]
    gross = pos_held * ret

    changes = np.abs(np.diff(np.concatenate([[0.0], pos_held])))
    cost = changes * (cost_bps / 1e4)
    net = gross - cost

    return {"net": net, "gross": gross,
            "turnover": float(changes.sum()),
            "n_trades": int((changes > 0).sum())}


def equity_curve(net_returns):
    return np.cumprod(1.0 + np.asarray(net_returns, dtype=float))


def max_drawdown(net_returns) -> float:
    eq = equity_curve(net_returns)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


__all__ = ["backtest", "equity_curve", "max_drawdown"]

"""
quantbench.shortterm.predict
============================
Signal directionnel court terme (horizon ~1 mois) inspiré de la discipline
Medallion : COMBINER plusieurs signaux faibles plutôt qu'un gros. Ici, sur
l'historique de prix quotidien :
  * retour à la moyenne court terme (z-score prix vs MM20 : survendu -> hausse)
  * momentum moyen terme (6 mois)
  * tendance (prix vs MM200)
  * repli récent (1 mois)

Le score composite est mappé sur une PROBABILITÉ DE HAUSSE, volontairement
BORNÉE à [0,30 ; 0,70] : sur données gratuites et marchés efficients, l'edge
directionnel est faible — on l'affiche honnêtement, sans fausse certitude.
"""

from __future__ import annotations

import numpy as np


def predict(prices, horizon: int = 21) -> dict | None:
    p = np.asarray([x for x in (prices or []) if x and x == x], dtype=float)
    if p.size < 60:
        return None
    last = float(p[-1])
    ma = lambda n: float(p[-n:].mean()) if p.size >= n else float(p.mean())
    ma20, ma200 = ma(20), ma(200)
    sd20 = float(p[-20:].std()) or 1.0
    z = (last - ma20) / sd20                        # >0 sur-acheté, <0 sur-vendu
    mom6 = last / float(p[-126]) - 1 if p.size >= 126 else last / float(p[0]) - 1
    mom1 = last / float(p[-21]) - 1 if p.size >= 21 else 0.0
    ret = np.diff(np.log(p))
    vol = float(ret[-21:].std() * np.sqrt(252)) if ret.size >= 21 else float(ret.std() * np.sqrt(252))
    trend = 1.0 if last > ma200 else -1.0

    # Composite de signaux faibles (poids modérés).
    score = (-0.60 * np.tanh(z / 1.5)          # mean-reversion court terme
             + 0.45 * np.tanh(mom6 * 3.0)      # momentum 6 mois
             + 0.25 * trend                    # tendance de fond
             - 0.20 * np.tanh(mom1 * 5.0))     # repli/anti-momentum très court

    p_up = 1.0 / (1.0 + np.exp(-1.2 * score))
    p_up = min(max(p_up, 0.30), 0.70)           # borne l'edge (honnêteté)
    return {
        "p_up": round(float(p_up), 3),
        "p_down": round(float(1 - p_up), 3),
        "score": round(float(score), 3),
        "horizon_days": horizon,
        "bias": "hausse" if p_up >= 0.53 else ("baisse" if p_up <= 0.47 else "neutre"),
        "momentum_6m": round(float(mom6), 4),
        "zscore": round(float(z), 2),
        "trend": "haussier" if trend > 0 else "baissier",
        "vol_annual": round(float(vol), 3),
    }


__all__ = ["predict"]

"""
quantbench.shortterm.predict
============================
Signal directionnel court terme (~21 jours de bourse), CALIBRE EMPIRIQUEMENT.

Principe non negociable : une probabilite affichee doit etre CALIBREE — si le
modele annonce 60 %, l'evenement doit survenir 60 % du temps. Ce module ne publie
donc `p_up` QUE si des coefficients estimes hors echantillon existent
(shortterm_calibration.json, produit par scripts/calibrate_shortterm.py) ET que
la calibration a demontre un pouvoir predictif. Sinon il ne renvoie que le SCORE
et ses composantes — jamais un pourcentage invente.

Caracteristiques mesurees sur des fenetres DISJOINTES (condition necessaire pour
qu'elles portent une information distincte, et non le meme mouvement compte deux
fois) :
  * renversement (t-20 -> t) : ecart du cours a sa moyenne 20 jours, standardise
    par la volatilite des RENDEMENTS -- et non par celle des niveaux, contaminee
    par la derive, qui bornait mecaniquement l'ancien z-score a +/-1,65 ;
  * momentum 6-1 (t-126 -> t-21) : EXCLUT le dernier mois, precisement celui que
    mesure le renversement ;
  * volatilite (EWMA RiskMetrics, lambda = 0,94) : anomalie de faible volatilite.

Le signal vise le rendement RELATIF au marche : une prevision de sur/sous-
performance, et non un pari deguise sur la direction de l'indice.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

_CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "shortterm_calibration.json")


def _load_calibration():
    try:
        with open(_CALIB_PATH, encoding="utf-8") as f:
            c = json.load(f)
        return c if c.get("calibrated") else None
    except Exception:
        return None


_CALIB = _load_calibration()
FEATURES = ("reversal", "momentum", "lowvol")


def ewma_vol(ret, lam: float = 0.94, n: int = 250):
    """Volatilite quotidienne EWMA (RiskMetrics). Bien plus stable qu'un
    ecart-type sur 21 points (erreur-type divisee par ~3) tout en restant reactif."""
    r = np.asarray(ret[-n:], dtype=float)
    if r.size < 20:
        return None
    w = lam ** np.arange(r.size)[::-1]
    var = float(np.sum(w * r ** 2) / np.sum(w))
    return math.sqrt(var) if var > 0 else None


def features(prices) -> dict | None:
    """Caracteristiques brutes standardisees. None si l'historique complet
    (200 seances) n'est pas disponible : pas de repli silencieux sur une fenetre
    plus courte, qui donnerait au meme titre un signal different selon la
    profondeur de son historique."""
    p = np.asarray([x for x in (list(prices) if prices is not None else [])
                    if x and x == x and x > 0], dtype=float)
    if p.size < 200:
        return None
    logp = np.log(p)
    ret = np.diff(logp)
    sd = ewma_vol(ret)
    if not sd:
        return None

    # Renversement : ecart a la moyenne 20 j, en unites d'ecart-type de l'ecart.
    # Signe : positif = SURVENDU.
    #
    # VARIANCE EXACTE, ET NON (n+1)/3. Pour une marche aleatoire, la variance de
    # l'ecart entre le dernier point et la moyenne des n derniers — CE POINT
    # COMPRIS — vaut (n-1)(2n-1) / (6n) fois la variance journaliere, soit 6,175
    # pour n = 20. La formule (n+1)/3 = 7,000 vaut pour une moyenne mobile qui
    # EXCLUT le point courant. L'ecart de 13 % sur la variance comprimait toutes
    # les valeurs de `reversal` de 6,08 %, et rendait fausse l'affirmation
    # « en unites d'ecart-type » que porte cette ligne.
    # MOYENNE DES LOGARITHMES, ET NON LOGARITHME DE LA MOYENNE. L'ecart entre les
    # deux est celui de l'inegalite de Jensen : le second majore toujours le
    # premier, d'autant plus que la serie est dispersee. Le numerateur en heritait
    # un biais qui grandit avec la volatilite — exactement la grandeur par laquelle
    # on le divise ensuite. Toute la construction se fait en logarithmes ; il n'y
    # avait aucune raison d'en sortir pour la moyenne.
    _n = 20.0
    n_eff = (_n - 1.0) * (2.0 * _n - 1.0) / (6.0 * _n)
    reversal = -float(logp[-1] - logp[-20:].mean()) / (sd * math.sqrt(n_eff))

    # Momentum 6-1 : de t-126 a t-21, fenetre DISJOINTE du renversement.
    #
    # DIVISE PAR LA VOLATILITE DE SA PROPRE FENETRE. Il l'etait par `sd`, la
    # volatilite EWMA globale — dont la demi-vie est de 11 seances, si bien que
    # 71 % de son poids porte sur les VINGT DERNIERES seances, c'est-a-dire
    # entierement HORS de la fenetre du momentum, et 29 % seulement dedans. Deux
    # titres au rendement et a la volatilite de fenetre identiques recevaient donc
    # des scores differant d'un facteur 2,6 selon la seule agitation recente — et
    # cette agitation recente est deja ce que le renversement mesure. Les deux
    # signaux, annonces comme portant sur des fenetres disjointes, partageaient en
    # fait leur denominateur.
    ret_fenetre = ret[-125:-20]
    sd_momentum = ewma_vol(ret_fenetre) or sd
    momentum = float(logp[-21] - logp[-126]) / (sd_momentum * math.sqrt(105.0))

    vol_annual = sd * math.sqrt(252.0)
    return {"reversal": float(np.clip(reversal, -5, 5)),
            "momentum": float(np.clip(momentum, -5, 5)),
            "lowvol": float(np.clip(-vol_annual, -3, 0)),
            "vol_annual": vol_annual}


def predict(prices, horizon: int = 21) -> dict | None:
    f = features(prices)
    if f is None:
        return None
    out = {"reversal": round(f["reversal"], 3),
           "momentum": round(f["momentum"], 3),
           "vol_annual": round(f["vol_annual"], 3),
           "horizon_days": horizon,
           "trend": "haussier" if f["momentum"] > 0 else "baissier"}

    if not _CALIB:
        # Aucune calibration validee : on publie le score BRUT (utilisable pour un
        # classement relatif) et AUCUNE probabilite. Mieux vaut pas de chiffre
        # qu'un chiffre faux.
        out["score"] = round(f["reversal"] * 0.5 + f["momentum"] * 0.5, 3)
        out["p_up"] = None
        out["p_down"] = None
        out["bias"] = "non calibre"
        out["calibrated"] = False
        return out

    b = _CALIB["coefficients"]
    z = float(b.get("intercept", 0.0)) + sum(float(b.get(k, 0.0)) * f[k] for k in FEATURES)
    # Echelle temporelle : la derive attendue croit en racine de l'horizon.
    z *= math.sqrt(horizon / float(_CALIB.get("horizon_days", 21)))
    p_up = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, z))))
    out.update({
        "score": round(z, 3),
        "p_up": round(p_up, 4),
        "p_down": round(1.0 - p_up, 4),
        "bias": "hausse" if p_up >= 0.52 else ("baisse" if p_up <= 0.48 else "neutre"),
        "calibrated": True,
        "calibration": _CALIB.get("metrics"),
    })
    return out


__all__ = ["predict", "features", "ewma_vol", "FEATURES"]

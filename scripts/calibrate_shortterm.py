"""Calibration EMPIRIQUE et hors echantillon du signal court terme.

Construit un panel point-in-time (date, titre, caracteristiques calculees
UNIQUEMENT avec l'information disponible a t, rendement realise a t+21), ajuste
une regression logistique sur la periode d'APPRENTISSAGE, puis mesure sa qualite
sur une periode de TEST posterieure jamais vue.

La cible est le rendement RELATIF au marche (surperformance vs SPY) : sans cela,
16 500 "probabilites independantes" ne seraient qu'un seul pari sur la direction
de l'indice.

Ne declare `calibrated: true` (et n'autorise donc l'affichage d'une probabilite)
QUE si le modele bat, hors echantillon, la prevision naive du taux de base
(score de Brier) ET discrimine (AUC > 0,52). Sinon le site n'affiche aucun
pourcentage : mieux vaut pas de chiffre qu'un chiffre faux.

Usage : FMP_API_KEY=... python scripts/calibrate_shortterm.py [--n 250] [--step 21]
"""

import concurrent.futures as cf
import json
import math
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from quantbench.data import fmp                                  # noqa: E402
from quantbench.shortterm.predict import features, FEATURES      # noqa: E402

OUT = os.path.join(os.path.dirname(_HERE), "quantbench", "shortterm",
                   "shortterm_calibration.json")
HORIZON = 21


def series(symbol, frm="2013-01-01"):
    """[(date, cours ajuste)] du plus ancien au plus recent."""
    try:
        j = fmp._json(f"historical-price-eod/dividend-adjusted?symbol={symbol}"
                      f"&from={frm}&to=2026-07-28")
        rows = [(d["date"], fmp._num(d.get("adjClose"))) for d in j if d.get("date")]
        rows = [(d, p) for d, p in rows if p and p > 0]
        return sorted(rows)
    except Exception:
        return []


def build_rows(symbol, mkt, step):
    """Observations point-in-time pour un titre."""
    s = series(symbol)
    if len(s) < 260 + HORIZON:
        return []
    dates = [d for d, _ in s]
    px = [p for _, p in s]
    out = []
    for i in range(200, len(px) - HORIZON, step):
        f = features(px[:i + 1])          # UNIQUEMENT l'information jusqu'a t
        if f is None:
            continue
        d0, d1 = dates[i], dates[i + HORIZON]
        m0, m1 = mkt.get(d0), mkt.get(d1)
        if not m0 or not m1:
            continue
        r = px[i + HORIZON] / px[i] - 1.0
        rm = m1 / m0 - 1.0
        out.append({"date": d0, "sym": symbol,
                    **{k: f[k] for k in FEATURES},
                    "excess": r - rm, "y": 1 if (r - rm) > 0 else 0})
    return out


def logistic_fit(X, y, l2=1.0, iters=400, lr=0.5):
    """Regression logistique (descente de gradient, penalisation L2) — evite une
    dependance a scikit-learn dans le build."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    w = np.zeros(Xb.shape[1])
    n = Xb.shape[0]
    for _ in range(iters):
        z = np.clip(Xb @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xb.T @ (p - y) / n
        grad[1:] += l2 * w[1:] / n
        w -= lr * grad
    return w


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(order.size, float)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2)
                 / (pos.size * neg.size))


def reliability(y, p, bins=8):
    y = np.asarray(y); p = np.asarray(p)
    qs = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for i in range(bins):
        m = (p >= qs[i]) & (p <= qs[i + 1] if i == bins - 1 else p < qs[i + 1])
        if m.sum() >= 30:
            out.append({"annonce": round(float(p[m].mean()), 4),
                        "realise": round(float(y[m].mean()), 4), "n": int(m.sum())})
    return out


def main(n_sym=250, step=21):
    print("Univers de calibration : membres actuels du S&P 500 (titres liquides).")
    syms = [r["symbol"] for r in fmp._json("sp500-constituent") if r.get("symbol")][:n_sym]
    mkt = dict(series("SPY"))
    if not mkt:
        print("Marche de reference indisponible."); return
    print(f"{len(syms)} titres, pas={step} seances, horizon={HORIZON}. Construction du panel…")

    rows = []
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(build_rows, s, mkt, step): s for s in syms}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            try:
                rows.extend(fut.result() or [])
            except Exception:
                pass
            if i % 50 == 0:
                print(f"  {i}/{len(syms)} titres ({len(rows)} observations)")
    if len(rows) < 3000:
        print(f"Panel insuffisant ({len(rows)})."); return

    rows.sort(key=lambda r: r["date"])
    # Separation TEMPORELLE stricte : on apprend sur le passe, on teste sur le futur.
    cut = int(len(rows) * 0.65)
    coupure = rows[cut]["date"]
    tr, te = rows[:cut], rows[cut:]
    Xtr = [[r[k] for k in FEATURES] for r in tr]; ytr = [r["y"] for r in tr]
    Xte = [[r[k] for k in FEATURES] for r in te]; yte = [r["y"] for r in te]

    w = logistic_fit(Xtr, ytr)
    zte = np.clip(np.hstack([np.ones((len(Xte), 1)), np.asarray(Xte, float)]) @ w, -30, 30)
    pte = 1.0 / (1.0 + np.exp(-zte))
    yte_a = np.asarray(yte, float)

    base = float(np.mean(ytr))                       # prevision naive : taux de base
    brier = float(np.mean((pte - yte_a) ** 2))
    brier_base = float(np.mean((base - yte_a) ** 2))
    skill = 1.0 - brier / brier_base if brier_base > 0 else 0.0
    a = auc(yte_a, pte)

    # Ecart de rendement excedentaire entre decile haut et decile bas (hors echantillon)
    ex = np.asarray([r["excess"] for r in te], float)
    q = np.quantile(pte, [0.1, 0.9])
    top, bot = ex[pte >= q[1]], ex[pte <= q[0]]
    spread = float(top.mean() - bot.mean()) if top.size and bot.size else float("nan")

    print(f"\n{'='*66}\nCALIBRATION HORS ECHANTILLON — signal court terme {HORIZON} j")
    print(f"panel : {len(rows)} observations ({len(tr)} apprentissage / {len(te)} test)")
    print(f"coupure temporelle : {coupure}  (test = tout ce qui suit)")
    print(f"taux de base (surperformance) : {base:.4f}")
    print(f"  score de Brier      : {brier:.5f}   (naif : {brier_base:.5f})")
    print(f"  gain vs naif        : {skill*100:+.3f} %")
    print(f"  AUC                 : {a:.4f}   (0,5 = aucune discrimination)")
    print(f"  ecart decile 10-1   : {spread*100:+.2f} % sur {HORIZON} j")
    print("\ncoefficients :", {k: round(float(v), 4) for k, v in
                               zip(("intercept",) + tuple(FEATURES), w)})
    rel = reliability(yte_a, pte)
    print("fiabilite (annonce -> realise) :")
    for b in rel:
        print(f"   {b['annonce']*100:5.1f} % -> {b['realise']*100:5.1f} %  (n={b['n']})")

    # Un edge n'est reconnu que s'il bat le naif ET discrimine.
    ok = bool(skill > 0.0005 and a > 0.52)
    payload = {
        "calibrated": ok,
        "horizon_days": HORIZON,
        "coefficients": {k: float(v) for k, v in
                         zip(("intercept",) + tuple(FEATURES), w)},
        "metrics": {"brier": round(brier, 5), "brier_naif": round(brier_base, 5),
                    "gain_vs_naif_pct": round(skill * 100, 3), "auc": round(a, 4),
                    "ecart_decile_pct": round(spread * 100, 3),
                    "n_apprentissage": len(tr), "n_test": len(te),
                    "coupure": coupure, "taux_base": round(base, 4)},
        "fiabilite": rel,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\nVERDICT : {'EDGE RETENU -> probabilites publiees' if ok else 'AUCUN EDGE -> aucune probabilite ne sera affichee (score/rang seulement)'}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    args = sys.argv[1:]
    kw = {"n_sym": 250, "step": 21}
    while args:
        a = args.pop(0)
        if a == "--n": kw["n_sym"] = int(args.pop(0))
        elif a == "--step": kw["step"] = int(args.pop(0))
    main(**kw)

"""Demonstration du volet COURT TERME + honnetete anti-overfitting.

On balaie une grille de strategies mean-reversion sur un actif liquide, on garde
la meilleure par Sharpe... puis le DEFLATED SHARPE (qui sait qu'on a essaye N
configurations) tranche : signal reel ou artefact du data-snooping ?

Usage :  python scripts/shortterm_demo.py [SYMBOLE]   (defaut BTC-USD)
"""

import sys

import numpy as np
from scipy import stats

from quantbench.data import market
from quantbench.backtest import backtest, max_drawdown
from quantbench.eval import (sharpe_per_period, annualized_sharpe,
                             probabilistic_sharpe_ratio, deflated_sharpe_ratio,
                             probability_of_backtest_overfitting)
from quantbench.shortterm import zscore_meanrev, ou_half_life, detect_regimes

LOOKBACKS = [10, 15, 20, 30, 40, 60]
ENTRIES = [1.0, 1.5, 2.0, 2.5]
EXITS = [0.25, 0.5, 0.75]
COST_BPS = 10.0


def main(symbol="BTC-USD"):
    ppy = 365 if "-USD" in symbol else 252
    prices = market._chart(symbol, "5y", "1d")
    prices = np.asarray(prices, dtype=float)
    print(f"\n=== Court terme : {symbol} ({len(prices)} barres quotidiennes) ===")

    reg = detect_regimes(prices, n_regimes=3)
    print(f"Regime courant : {reg['current_name'].upper()} "
          f"(proba {reg['current_proba']:.0%})  |  rendements moyens par regime : "
          f"{reg['mean_returns']}")
    print(f"Demi-vie OU (log-prix) : {ou_half_life(prices):.1f} barres")

    # --- Grid search mean-reversion ---
    trials, nets, best = [], [], None
    for lb in LOOKBACKS:
        for en in ENTRIES:
            for ex in EXITS:
                if ex >= en:
                    continue
                pos, _ = zscore_meanrev(prices, lb, en, ex)
                bt = backtest(prices, pos, cost_bps=COST_BPS)
                if bt["net"].size < 30 or bt["n_trades"] < 5:
                    continue
                srp = sharpe_per_period(bt["net"])
                trials.append(srp)
                nets.append(bt["net"])
                cand = {"lb": lb, "en": en, "ex": ex, "srp": srp, "bt": bt}
                if best is None or srp > best["srp"]:
                    best = cand

    if best is None:
        print("Aucune strategie exploitable.")
        return

    net = best["bt"]["net"]
    n = net.size
    sr_ann = best["srp"] * np.sqrt(ppy)
    skew = float(stats.skew(net))
    kurt = float(stats.kurtosis(net, fisher=False))
    psr0 = probabilistic_sharpe_ratio(best["srp"], n)
    dsr = deflated_sharpe_ratio(best["srp"], trials, n, skew, kurt)

    print(f"\n{len(trials)} configurations testees (data-snooping).")
    print(f"MEILLEURE : lookback={best['lb']} entree={best['en']} sortie={best['ex']} "
          f"| trades {best['bt']['n_trades']} | max DD {max_drawdown(net):.1%}")
    print(f"  Sharpe annualise (brut)          : {sr_ann:+.2f}")
    print(f"  PSR vs 0 (sans correction)       : {psr0:.1%}")
    print(f"  Sharpe max attendu sous H0        : {dsr['deflated_benchmark']*np.sqrt(ppy):+.2f} "
          f"(annualise, {dsr['n_trials']} essais)")
    print(f"  DEFLATED SHARPE (proba reel)     : {dsr['dsr']:.1%}")
    pbo = probability_of_backtest_overfitting(np.column_stack(nets), n_splits=10)
    print(f"  PBO (proba de surajustement)     : {pbo['pbo']:.1%} "
          f"({pbo['n_combinations']} combinaisons CSCV)")
    real = dsr["dsr"] > 0.95 and pbo["pbo"] < 0.5
    verdict = ("SIGNAL PROBABLEMENT REEL" if real
               else "PROBABLE ARTEFACT DU DATA-SNOOPING (garder prudence)")
    print(f"  VERDICT : {verdict}")
    print("\nLecon : un Sharpe brut flatteur ne survit pas toujours a la correction "
          "du nombre d'essais. C'est tout l'interet du banc d'essai honnete.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "BTC-USD")

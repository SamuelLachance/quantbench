"""Tests du module d'evaluation honnete et du backtest."""

import numpy as np

from quantbench.eval import (
    expected_max_sharpe, probabilistic_sharpe_ratio, deflated_sharpe_ratio,
    min_track_record_length, probability_of_backtest_overfitting,
)
from quantbench.backtest import backtest, max_drawdown


def test_expected_max_sharpe_matches_known_result():
    # Resultat de reference (Bailey & Lopez de Prado) : variance des Sharpe = 1,
    # N=1000 essais -> E[max Sharpe] ~ 3,26 sous l'hypothese nulle.
    assert abs(expected_max_sharpe(1.0, 1000) - 3.257) < 0.01


def test_expected_max_sharpe_monotone():
    assert expected_max_sharpe(1.0, 100) < expected_max_sharpe(1.0, 10000)
    assert expected_max_sharpe(0.5, 500) < expected_max_sharpe(2.0, 500)


def test_psr_bounds_and_monotonicity():
    a = probabilistic_sharpe_ratio(0.05, 500)
    b = probabilistic_sharpe_ratio(0.15, 500)
    assert 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0
    assert b > a                                   # meilleur Sharpe -> plus de confiance


def test_deflation_reduces_confidence():
    # Le DSR (corrige du biais de selection) doit etre <= PSR vs zero.
    sr = 0.12
    trials = np.random.default_rng(0).normal(0.0, 0.08, 80)
    psr0 = probabilistic_sharpe_ratio(sr, 750)
    res = deflated_sharpe_ratio(sr, trials, 750)
    assert res["deflated_benchmark"] > 0
    assert res["dsr"] <= psr0


def test_min_trl_finite_for_positive_sr():
    assert np.isfinite(min_track_record_length(0.1))
    assert min_track_record_length(-0.1) == np.inf


def test_backtest_no_lookahead_alignment():
    # prix +10% chaque periode, position longue constante -> ~10%/periode net.
    prices = [100.0, 110.0, 121.0]
    r = backtest(prices, [1.0, 1.0, 1.0], cost_bps=10.0)
    assert np.isclose(r["net"][1], 0.10, atol=1e-9)          # pas de cout, pas de leak
    assert np.isclose(r["net"][0], 0.10 - 10 / 1e4, atol=1e-9)  # cout d'entree
    assert r["n_trades"] == 1


def test_backtest_flat_is_zero():
    r = backtest([100, 90, 120, 80], [0, 0, 0, 0])
    assert np.allclose(r["net"], 0.0)
    assert max_drawdown(r["net"]) == 0.0


def test_pbo_high_for_non_stationary_edge():
    # Avantage NON stationnaire : chaque strategie est bonne dans une moitie et
    # nulle dans l'autre. La championne in-sample s'effondre OOS -> PBO eleve.
    rng = np.random.default_rng(1)
    T, N = 400, 8
    M = rng.normal(0, 0.005, size=(T, N))
    h = T // 2
    for j in range(N):
        if j < N // 2:
            M[:h, j] += 0.01          # bonne dans la 1re moitie
        else:
            M[h:, j] += 0.01          # bonne dans la 2e moitie
    res = probability_of_backtest_overfitting(M, n_splits=2)
    assert res["n_strategies"] == N
    assert res["pbo"] > 0.7


def test_pbo_low_for_genuine_edge():
    # Une strategie a vrai edge domine partout -> PBO faible.
    rng = np.random.default_rng(2)
    M = rng.normal(0, 0.01, size=(500, 10))
    M[:, 0] += 0.004                     # edge reel et stable sur la strat 0
    res = probability_of_backtest_overfitting(M, n_splits=8)
    assert res["pbo"] < 0.3

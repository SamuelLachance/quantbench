"""Tests de non-regression mathematique du moteur DCF.

Le test central est un controle par FORME FERMEE : en croissance nulle, sans
reinvestissement, a WACC constant, la valeur des actifs operationnels doit
egaler exactement une perpetuite FCFF / WACC.
"""

import numpy as np
import pytest

from quantbench.valuation.dcf import (
    DcfInputs, value_dcf, value_per_share, converge_path,
)


def _flat_zero_growth() -> DcfInputs:
    """Cas analytique : g=0 partout, WACC plat (pas de dette), marge & impot plats."""
    return DcfInputs(
        revenue_base=100.0,
        g1_begin=0.0, g1_end=0.0, g2_begin=0.0, g2_end=0.0,
        g3_begin=0.0, g3_end=0.0,
        len1=3, len2=4, len3=3, conv1=1, conv2=1, conv3=1,
        current_operating_margin=0.20, terminal_operating_margin=0.20,
        margin_converge_start=1,
        current_tax_rate=0.25, marginal_tax_rate=0.25, tax_converge_start=1,
        current_sales_to_capital=2.0, terminal_sales_to_capital=2.0,
        s2c_converge_start=1,
        risk_free_rate=0.02, erp=0.05,
        unlevered_beta=1.0, terminal_unlevered_beta=1.0, beta_converge_start=1,
        current_pretax_kd=0.05, terminal_pretax_kd=0.05, kd_converge_start=1,
        equity_value=1.0, debt_value=0.0, cash_and_non_operating=0.0,
        additional_roic_in_perpetuity=0.0,
    )


def test_closed_form_perpetuity():
    x = _flat_zero_growth()
    res = value_dcf(x)
    fcff = 100.0 * 0.20 * (1 - 0.25)          # = 15
    wacc = 0.02 + 1.0 * 0.05                   # = 0.07 (dette nulle -> WACC = cout des FP)
    expected = fcff / wacc                      # perpetuite = 214.2857...
    assert np.isclose(res["value_of_operating_assets"], expected, rtol=1e-9)
    assert np.isclose(res["equity_value"], expected, rtol=1e-9)
    # En croissance nulle, aucun reinvestissement.
    assert np.allclose(res["reinvestment"], 0.0)


def test_terminal_wacc_must_exceed_growth():
    x = _flat_zero_growth()
    x.g3_end = 0.10                             # croissance terminale > WACC (0.07)
    with pytest.raises(ValueError):
        value_dcf(x)


def test_cash_and_debt_bridge():
    # Le pont firm -> equity est une IDENTITE, vraie quel que soit le WACC
    # (ajouter de la dette change aussi le WACC via D/E, donc on teste l'identite,
    #  pas une valeur figee).
    x = _flat_zero_growth()
    x.cash_and_non_operating = 50.0
    x.debt_value = 30.0
    res = value_dcf(x)
    assert np.isclose(res["firm_value"],
                      res["value_of_operating_assets"] + 50.0, rtol=1e-12)
    assert np.isclose(res["equity_value"],
                      res["firm_value"] - 30.0, rtol=1e-12)


def test_higher_growth_increases_value():
    lo = _flat_zero_growth()
    hi = _flat_zero_growth()
    hi.g1_begin = hi.g1_end = hi.g2_begin = hi.g2_end = 0.05
    assert value_dcf(hi)["equity_value"] > value_dcf(lo)["equity_value"]


def test_value_per_share():
    x = _flat_zero_growth()
    vps = value_per_share(x, shares_outstanding=10.0)
    assert np.isclose(vps, (15.0 / 0.07) / 10.0, rtol=1e-9)


def test_roic_mode_matches_perpetuity_at_zero_growth():
    # A g=0, aucun reinvestissement quel que soit le mode -> meme perpetuite.
    x = _flat_zero_growth()
    x.reinvestment_mode = "roic"
    x.current_roic = 0.30
    x.terminal_roic = 0.10
    x.roic_converge_start = 3
    res = value_dcf(x)
    assert np.isclose(res["value_of_operating_assets"], 15.0 / 0.07, rtol=1e-9)
    assert np.allclose(res["reinvestment"], 0.0)


def test_roic_mode_beats_sales_to_capital_for_high_roic():
    # Avec de la croissance, un ROIC eleve reinvestit moins -> plus de FCFF -> plus de valeur.
    s2c = _flat_zero_growth()
    s2c.g1_begin = s2c.g1_end = s2c.g2_begin = s2c.g2_end = s2c.g3_begin = 0.06
    s2c.current_sales_to_capital = s2c.terminal_sales_to_capital = 1.0  # capital-lourd
    roic = DcfInputs(**{**s2c.__dict__})
    roic.reinvestment_mode = "roic"
    roic.current_roic = 0.40           # tres capital-leger
    roic.terminal_roic = 0.15
    assert value_dcf(roic)["equity_value"] > value_dcf(s2c)["equity_value"]


def test_roic_mode_requires_roic_values():
    x = _flat_zero_growth()
    x.reinvestment_mode = "roic"       # current_roic/terminal_roic restent NaN
    with pytest.raises(ValueError):
        value_dcf(x)


def test_converge_path_shapes():
    # Reste a 0.10 pendant 4 ans, puis converge vers 0.02 a l'annee 10.
    p = converge_path(0.10, 0.02, n_years=10, converge_start_year=5)
    assert len(p) == 10
    assert np.allclose(p[:4], 0.10)
    assert np.isclose(p[-1], 0.02)

"""
quantbench.valuation.dcf
========================
Valorisation intrinseque FCFF multi-phase (methode Damodaran), reecrite a partir
du script legacy : mathematiques corrigees, indexation robuste (numpy pur),
aucune dependance reseau, entierement testable hors-ligne.

Conventions
-----------
* Toutes les grandeurs monetaires sont dans la MEME unite (ex. milliards USD).
* Les taux (croissance, marge, WACC, impots...) sont des fractions decimales :
  0.08 == 8 %.
* Le "cycle 3" se termine sur la croissance terminale (`g3_end`), qui sert aussi
  de taux de croissance perpetuel dans la valeur terminale.

Corrections vs. le script d'origine
------------------------------------
* firm_future_value ne mettait pas au carre le mauvais terme -> supprime (dead code).
* Le rendement annualise applique correctement le signe : sign*((r)**(1/n) - 1).
* Indexation pandas fragile remplacee par des vecteurs numpy alignes par position.
* La valeur terminale est actualisee explicitement : PV(TV) = TV / facteur_actualisation[N].
* Garde-fou : WACC terminal doit etre strictement > croissance terminale.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import numpy as np


# --------------------------------------------------------------------------- #
# Helpers de trajectoires (convergence lineaire vers une valeur terminale)
# --------------------------------------------------------------------------- #
def converge_path(start: float, end: float, n_years: int,
                  converge_start_year: int) -> np.ndarray:
    """Trajectoire annuelle : reste a `start` jusqu'a `converge_start_year - 1`,
    puis converge lineairement vers `end` a l'annee `n_years`.

    Retourne un tableau de longueur `n_years` (annees 1..n_years).
    """
    n_years = int(n_years)
    converge_start_year = int(converge_start_year)
    if n_years <= 0:
        return np.zeros(0, dtype=float)
    hold = min(max(converge_start_year - 1, 0), n_years)
    path = np.empty(n_years, dtype=float)
    path[:hold] = start
    ramp_len = n_years - hold
    if ramp_len == 1:
        path[hold:] = end
    elif ramp_len > 1:
        path[hold:] = np.linspace(start, end, ramp_len)
    return path


def multi_phase_growth(cycles, lengths, converge_starts) -> np.ndarray:
    """Concatene les trajectoires de croissance de plusieurs cycles.

    `cycles` : liste de (begin, end) par cycle.
    Retourne un tableau de longueur sum(lengths).
    """
    parts = [converge_path(b, e, n, c)
             for (b, e), n, c in zip(cycles, lengths, converge_starts)]
    return np.concatenate(parts) if parts else np.zeros(0)


def wacc_path(unlevered_beta, terminal_unlevered_beta,
              pretax_kd, terminal_pretax_kd,
              equity_value, debt_value,
              risk_free_rate, erp, marginal_tax,
              n_years, beta_converge_start, kd_converge_start):
    """Trajectoire du cout du capital (WACC) et du cout des fonds propres.

    Beta leve = beta non-leve * (1 + (1 - t) * D/E). Beta et cout de la dette
    convergent vers leurs valeurs terminales sur l'horizon.
    """
    de_ratio = 0.0 if equity_value == 0 else debt_value / equity_value
    lever = lambda b: b * (1.0 + (1.0 - marginal_tax) * de_ratio)

    betas = converge_path(lever(unlevered_beta), lever(terminal_unlevered_beta),
                          n_years, beta_converge_start)
    kd = converge_path(pretax_kd, terminal_pretax_kd, n_years, kd_converge_start)

    total = equity_value + debt_value
    w_e = 1.0 if total == 0 else equity_value / total
    w_d = 0.0 if total == 0 else debt_value / total

    cost_equity = risk_free_rate + betas * erp
    after_tax_kd = kd * (1.0 - marginal_tax)
    wacc = w_e * cost_equity + w_d * after_tax_kd
    return wacc, cost_equity, betas


# --------------------------------------------------------------------------- #
# Entrees du modele (plates : tous scalaires -> Monte Carlo trivial)
# --------------------------------------------------------------------------- #
@dataclass
class DcfInputs:
    revenue_base: float

    # --- Croissance du chiffre d'affaires (3 phases) ---
    g1_begin: float
    g1_end: float
    g2_begin: float
    g2_end: float
    g3_begin: float
    g3_end: float                     # == croissance terminale perpetuelle
    len1: int = 3
    len2: int = 4
    len3: int = 3
    conv1: int = 1
    conv2: int = 1
    conv3: int = 1

    # --- Marge operationnelle ---
    current_operating_margin: float = 0.15
    terminal_operating_margin: float = 0.15
    margin_converge_start: int = 5

    # --- Impots ---
    current_tax_rate: float = 0.25
    marginal_tax_rate: float = 0.25
    tax_converge_start: int = 5

    # --- Reinvestissement (ratio ventes / capital) ---
    current_sales_to_capital: float = 2.0
    terminal_sales_to_capital: float = 2.0
    s2c_converge_start: int = 3

    # --- Cout du capital ---
    risk_free_rate: float = 0.03
    erp: float = 0.05
    unlevered_beta: float = 1.0
    terminal_unlevered_beta: float = 1.0
    beta_converge_start: int = 5
    current_pretax_kd: float = 0.05
    terminal_pretax_kd: float = 0.05
    kd_converge_start: int = 5

    # --- Structure du capital & ajustements ---
    equity_value: float = 1.0          # market cap (pour les poids D/E)
    debt_value: float = 0.0
    cash_and_non_operating: float = 0.0
    additional_roic_in_perpetuity: float = 0.0
    asset_liquidation_during_negative_growth: float = 0.0
    current_invested_capital: float = float("nan")   # NaN -> implicite

    # --- Reinvestissement : methode ---
    # "sales_to_capital" : reinvest = delta_CA / (ventes/capital)  [historique/legacy]
    # "roic"             : reinvest = EBI * (croissance / ROIC), ROIC declinant  [Damodaran]
    reinvestment_mode: str = "sales_to_capital"
    current_roic: float = float("nan")
    terminal_roic: float = float("nan")
    roic_converge_start: int = 5

    @property
    def n_years(self) -> int:
        return int(self.len1) + int(self.len2) + int(self.len3)


# --------------------------------------------------------------------------- #
# Valorisation deterministe
# --------------------------------------------------------------------------- #
def value_dcf(x: DcfInputs) -> dict:
    """Valorise une entreprise par actualisation des FCFF sur l'horizon
    explicite + valeur terminale. Retourne un dictionnaire de resultats.
    """
    n = x.n_years
    if n <= 0:
        raise ValueError("L'horizon total (len1+len2+len3) doit etre > 0.")
    terminal_growth = x.g3_end

    # --- Chiffre d'affaires ---
    g = multi_phase_growth([(x.g1_begin, x.g1_end),
                            (x.g2_begin, x.g2_end),
                            (x.g3_begin, x.g3_end)],
                           [x.len1, x.len2, x.len3],
                           [x.conv1, x.conv2, x.conv3])
    revenues = x.revenue_base * np.cumprod(1.0 + g)
    prev_rev = np.concatenate([[x.revenue_base], revenues[:-1]])

    # --- Trajectoires ---
    margins = converge_path(x.current_operating_margin,
                            x.terminal_operating_margin, n, x.margin_converge_start)
    taxes = converge_path(x.current_tax_rate, x.marginal_tax_rate,
                          n, x.tax_converge_start)
    s2c = converge_path(x.current_sales_to_capital, x.terminal_sales_to_capital,
                        n, x.s2c_converge_start)
    wacc, cost_equity, betas = wacc_path(
        x.unlevered_beta, x.terminal_unlevered_beta,
        x.current_pretax_kd, x.terminal_pretax_kd,
        x.equity_value, x.debt_value,
        x.risk_free_rate, x.erp, x.marginal_tax_rate,
        n, x.beta_converge_start, x.kd_converge_start)

    # --- Flux ---
    ebit = revenues * margins
    ebit_after_tax = ebit * (1.0 - taxes)

    if x.reinvestment_mode == "roic":
        # Reinvestissement lie au ROIC : pour financer une croissance g, il faut
        # reinvestir g/ROIC de l'EBI. Le ROIC decline vers sa valeur terminale
        # (erosion des rentes de situation). Favorise les entreprises capital-legeres.
        if np.isnan(x.current_roic) or np.isnan(x.terminal_roic):
            raise ValueError("reinvestment_mode='roic' exige current_roic et terminal_roic.")
        roic_path = converge_path(x.current_roic, x.terminal_roic, n, x.roic_converge_start)
        with np.errstate(divide="ignore", invalid="ignore"):
            reinv_rate = np.where(roic_path > 0, g / roic_path, 0.0)
        reinv_rate = np.clip(reinv_rate, 0.0, 0.95)
        reinvestment = ebit_after_tax * reinv_rate
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            reinvestment = np.where(s2c != 0, (revenues - prev_rev) / s2c, 0.0)
        # En croissance negative, on ne "recupere" du capital que selon le parametre
        # asset_liquidation_during_negative_growth (0 = pas de liquidation).
        reinvestment = np.where(reinvestment > 0, reinvestment,
                                reinvestment * x.asset_liquidation_during_negative_growth)
    fcff = ebit_after_tax - reinvestment

    discount = np.cumprod(1.0 + wacc)          # facteur d'actualisation cumule
    pv_fcff = fcff / discount

    # --- Valeur terminale (croissance perpetuelle, methode Damodaran) ---
    wacc_t = float(wacc[-1])
    if wacc_t <= terminal_growth:
        raise ValueError(
            f"WACC terminal ({wacc_t:.4f}) <= croissance terminale "
            f"({terminal_growth:.4f}) : valeur terminale non definie.")
    if x.reinvestment_mode == "roic" and not np.isnan(x.terminal_roic):
        roic_t = x.terminal_roic
    else:
        roic_t = wacc_t + x.additional_roic_in_perpetuity
    if terminal_growth <= 0 or roic_t == 0:
        reinv_rate_t = 0.0
    else:
        reinv_rate_t = terminal_growth / roic_t
    rev_t = revenues[-1] * (1.0 + terminal_growth)
    ebit_at_t = rev_t * x.terminal_operating_margin * (1.0 - x.marginal_tax_rate)
    fcff_t = ebit_at_t * (1.0 - reinv_rate_t)
    terminal_value = fcff_t / (wacc_t - terminal_growth)
    pv_terminal = terminal_value / discount[-1]

    # --- Agregation ---
    value_operating = float(pv_fcff.sum() + pv_terminal)
    firm_value = value_operating + x.cash_and_non_operating
    equity_value = firm_value - x.debt_value

    # --- Capital investi & ROIC (diagnostic) ---
    ic0 = (x.revenue_base / x.current_sales_to_capital
           if np.isnan(x.current_invested_capital) or x.current_sales_to_capital == 0
           else x.current_invested_capital)
    invested_capital = ic0 + np.cumsum(reinvestment)
    with np.errstate(divide="ignore", invalid="ignore"):
        roic = np.where(invested_capital != 0, ebit_after_tax / invested_capital, np.nan)

    return {
        "equity_value": equity_value,
        "firm_value": firm_value,
        "value_of_operating_assets": value_operating,
        "terminal_value": terminal_value,
        "pv_terminal_value": pv_terminal,
        "terminal_growth": terminal_growth,
        "wacc_terminal": wacc_t,
        # series annuelles (annees 1..n)
        "revenues": revenues,
        "growth": g,
        "margins": margins,
        "ebit_after_tax": ebit_after_tax,
        "reinvestment": reinvestment,
        "fcff": fcff,
        "wacc": wacc,
        "cost_of_equity": cost_equity,
        "pv_fcff": pv_fcff,
        "invested_capital": invested_capital,
        "roic": roic,
    }


def value_per_share(x: DcfInputs, shares_outstanding: float) -> float:
    """Valeur intrinseque par action (meme unite monetaire que revenue_base)."""
    if shares_outstanding <= 0:
        raise ValueError("shares_outstanding doit etre > 0.")
    return value_dcf(x)["equity_value"] / shares_outstanding


__all__ = ["DcfInputs", "value_dcf", "value_per_share",
           "converge_path", "multi_phase_growth", "wacc_path"]

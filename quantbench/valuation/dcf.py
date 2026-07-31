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
# Indices 0..n-1 en flottant, mis en cache par longueur. Le Monte Carlo appelle
# `converge_path` neuf fois par tirage, dix mille tirages par titre : `np.linspace`
# y consommait 40 % du temps total de simulation, non pas a calculer dix nombres
# mais a resoudre des types et a redimensionner. Les longueurs utiles se comptent
# sur les doigts (les horizons de projection), le cache tient donc en quelques
# tableaux. Ils ne sont JAMAIS ecrits : les threads du build les partagent.
_INDICES: dict[int, np.ndarray] = {}


def _indices(n: int) -> np.ndarray:
    a = _INDICES.get(n)
    if a is None:
        a = _INDICES[n] = np.arange(n, dtype=float)
        a.flags.writeable = False
    return a


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
        # Rampe lineaire ecrite directement dans la vue, sans passer par
        # `np.linspace`. L'ordre des operations reproduit exactement le sien
        # (i * pas, puis + depart, puis borne finale imposee) : le resultat est
        # identique au bit pres, ce qu'un test verifie sur des milliers de tirages.
        pente = (end - start) / (ramp_len - 1)
        np.multiply(_indices(ramp_len), pente, out=path[hold:])
        path[hold:] += start
        path[-1] = end
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
              n_years, beta_converge_start, kd_converge_start,
              size_premium=0.0):
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

    # PRIME DE TAILLE : elle s'AJOUTE au cout des fonds propres, elle ne se
    # multiplie pas par le beta. Elle etait auparavant injectee dans l'ERP sous la
    # forme `erp + prime / beta_initial`, ce qui l'annulait exactement — mais la
    # premiere annee SEULEMENT. Le beta CONVERGE ensuite vers sa valeur terminale :
    # une societe dont le beta double sur l'horizon voyait sa prime de taille doubler
    # avec lui, jusqu'a 3,5 points de cout du capital surajoutes en perpetuite, la ou
    # se joue l'essentiel de la valeur.
    # Elle n'entre pas dans le cout de la DETTE : c'est une prime de risque
    # ACTIONNAIRE, le preteur d'une petite societe se remunere par son spread.
    cost_equity = risk_free_rate + betas * erp + size_premium
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
    # Prime de TAILLE : s'ajoute au cout des fonds propres sans passer par le beta.
    size_premium: float = 0.0
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
    # Creances SENIOR sur l'actionnaire ordinaire : interets minoritaires (en valeur
    # de marche) et actions privilegiees (au nominal). Les flux actualises sont
    # CONSOLIDES — ils portent 100 % des filiales — donc la valeur qui en derive
    # aussi ; Damodaran les retranche explicitement dans son pont vers l'equite
    # (« Value of Equity = Firm Value - Debt - Minority Interests », dcfstabl p.207).
    minority_and_preferred: float = 0.0
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
        n, x.beta_converge_start, x.kd_converge_start, x.size_premium)

    # --- Flux ---
    ebit = revenues * margins
    ebit_after_tax = ebit * (1.0 - taxes)

    if x.reinvestment_mode == "roic":
        # Reinvestissement lie au ROIC : pour financer une croissance g, il faut
        # reinvestir g/ROIC de l'EBI. Le ROIC decline vers sa valeur terminale
        # (erosion des rentes de situation). Favorise les entreprises capital-legeres.
        if np.isnan(x.current_roic) or np.isnan(x.terminal_roic):
            raise ValueError("reinvestment_mode='roic' exige current_roic et terminal_roic.")
        # UN ROIC NUL OU NEGATIF N'EST PAS UNE VALEUR ADMISSIBLE.
        # L'identite g = reinvestissement x ROIC n'a pas de solution : financer une
        # croissance avec un rendement du capital nul exigerait un reinvestissement
        # INFINI. Le code traitait ce cas comme "reinvestissement nul", c'est-a-dire
        # croissance GRATUITE — la meme famille que les quatre pieges falsy du jour,
        # sauf qu'ici c'est le tireur Monte Carlo qui produit la valeur interdite.
        # Discontinuite mesuree : ROIC de +0,001 donne une equite de -0,673 ;
        # exactement 0 donne +1,053 ; -0,20 donne +1,778. Le modele RECOMPENSE la
        # destruction de capital.
        # Le moteur deterministe n'en produit jamais — les bornes de
        # `build_universal` garantissent au moins 0,02 — donc le cout de cette garde
        # est nul et son filet reel.
        if x.current_roic <= 0 or x.terminal_roic <= 0:
            raise ValueError(
                f"ROIC nul ou negatif (courant {x.current_roic:.4f}, terminal "
                f"{x.terminal_roic:.4f}) : la croissance n'y est pas finançable.")
        roic_path = converge_path(x.current_roic, x.terminal_roic, n, x.roic_converge_start)
        # Damodaran : taux de reinvestissement = croissance de l'EBIT APRES IMPOT / ROIC
        # (pas la croissance du CA — elles different quand marge/impot varient).
        cur_ebi = x.revenue_base * x.current_operating_margin * (1 - x.current_tax_rate)
        ebi_prev = np.concatenate([[cur_ebi], ebit_after_tax[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            g_ebi = np.where(ebi_prev > 0, ebit_after_tax / ebi_prev - 1.0, g)
            reinv_rate = np.where(roic_path > 0, g_ebi / roic_path, 0.0)
        # Damodaran : taux de reinvestissement = g / ROIC, SANS plafond a 1. Une
        # croissance superieure a ce que le ROIC finance exige plus que la totalite
        # des benefices -> FCFF NEGATIF (la croissance consomme du cash). Plafonner
        # a 0.95 fabriquait de la valeur : le modele accordait une forte croissance
        # sans en facturer le capital (ex. 45% de croissance a 9% de ROIC exige 490%
        # de reinvestissement, pas 95%). Plafond large uniquement pour eviter les
        # divergences numeriques sur des ROIC quasi nuls.
        reinv_rate = np.clip(reinv_rate, 0.0, 3.0)
        reinvestment = ebit_after_tax * reinv_rate

        # L'IDENTITE DE DAMODARAN N'A DE SENS QUE SUR UN BENEFICE POSITIF.
        # `reinvestissement = EBI x g / ROIC` multiplie un taux positif par le
        # benefice apres impot : quand celui-ci est NEGATIF, le reinvestissement
        # devient negatif, et `FCFF = EBI - reinvestissement` fait RENTRER de
        # l'argent. Mesure : a 30 % de croissance et -20 % de marge, un benefice de
        # -195 produisait un reinvestissement de -585 et un flux libre de +390.
        # Reinvestir, c'est SORTIR de l'argent — quel que soit le signe du benefice.
        #
        # Damodaran traite precisement ce cas par l'autre voie, celle des ventes
        # rapportees au capital : le reinvestissement suit la croissance du CHIFFRE
        # D'AFFAIRES et l'intensite capitalistique, grandeurs qui restent definies
        # quand le benefice ne l'est pas. On bascule donc annee par annee, avec le
        # meme traitement de la croissance negative que dans l'autre mode.
        with np.errstate(divide="ignore", invalid="ignore"):
            secours = np.where(s2c != 0, (revenues - prev_rev) / s2c, 0.0)
        secours = np.where(secours > 0, secours,
                           secours * x.asset_liquidation_during_negative_growth)
        reinvestment = np.where(ebit_after_tax > 0, reinvestment, secours)
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
        # Pas de rente excessive en perpetuite : ROIC terminal plafonne a WACC + 2%
        # (Damodaran : en regime permanent, ROIC tend vers le cout du capital).
        roic_t = min(x.terminal_roic, wacc_t + 0.02)
    else:
        roic_t = wacc_t + x.additional_roic_in_perpetuity
    if roic_t <= 0:
        # Meme raison qu'a l'horizon explicite : un rendement terminal nul ou negatif
        # rend le taux de reinvestissement indefini, et un roic_t negatif produisait
        # un taux NEGATIF, donc un flux terminal GONFLE.
        raise ValueError(f"ROIC terminal nul ou negatif ({roic_t:.4f}).")
    if terminal_growth <= 0:
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
    equity_value = firm_value - x.debt_value - x.minority_and_preferred

    # --- Capital investi & ROIC (diagnostic) ---
    # LE REPLI NE SERT QUE SI LA VALEUR EXPLICITE MANQUE. Ecrit avec un `or`, il
    # se declenchait AUSSI quand un capital investi explicite etait fourni mais que
    # le rapport ventes sur capital valait zero : le code ignorait alors la valeur
    # donnee pour diviser par ce meme zero, levant une ZeroDivisionError nue. Les
    # deux conditions doivent etre lues dans l'autre sens — on ne se replie que
    # faute de mieux, et le repli lui-meme doit etre possible.
    if not np.isnan(x.current_invested_capital):
        ic0 = x.current_invested_capital
    elif x.current_sales_to_capital:
        ic0 = x.revenue_base / x.current_sales_to_capital
    else:
        # Ni capital explicite, ni intensite capitalistique : le ROIC de
        # diagnostic n'est pas calculable, et l'affirmer serait pire que se taire.
        ic0 = float("nan")
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

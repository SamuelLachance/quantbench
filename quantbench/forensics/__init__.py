"""Analyse forensique des états financiers (Beneish, Altman, Piotroski, accruals)."""

from .scores import (get_financials, beneish_m_score, altman_z_score,
                     piotroski_f_score, accruals_ratio, analyze)

__all__ = ["get_financials", "beneish_m_score", "altman_z_score",
           "piotroski_f_score", "accruals_ratio", "analyze"]

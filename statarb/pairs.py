"""Pair discovery: fast Engle-Granger cointegration test, hedge ratio, Ornstein-Uhlenbeck half-life."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.adfvalues import mackinnonp


def ols(y: np.ndarray, x: np.ndarray):
    """y = a + b x. Returns (a, b, residuals)."""
    X = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return coef[0], coef[1], y - X @ coef


def adf_tstat(u: np.ndarray) -> float:
    """ADF t-stat on regression residuals: du_t = g*u_{t-1} + d*du_{t-1} + e (no constant, one lagged difference).
    This matches statsmodels.coint(..., maxlag=1, autolag=None); the tests verify it."""
    du = np.diff(u)
    Y = du[1:]
    X = np.column_stack([u[1:-1], du[:-1]])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    e = Y - X @ coef
    s2 = e @ e / (len(Y) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(coef[0] / np.sqrt(cov[0, 0]))


def eg_pvalue(ly: np.ndarray, lx: np.ndarray):
    """Engle-Granger two-step test. Returns (p-value, alpha, beta, residuals)."""
    a, b, u = ols(ly, lx)
    t = adf_tstat(u)
    return float(mackinnonp(t, regression="c", N=2)), a, b, u


def half_life(spread: np.ndarray) -> float:
    """Mean-reversion half-life in days from the AR(1) fit  ds_t = a + b s_{t-1}.  inf if not mean-reverting."""
    s = np.asarray(spread, float)
    a, b, _ = ols(np.diff(s), s[:-1])
    if b >= 0:
        return float("inf")
    return float(-np.log(2) / np.log1p(b))


@dataclass
class Pair:
    y: str
    x: str
    alpha: float
    beta: float
    pvalue: float
    half_life: float
    corr: float

    @property
    def name(self) -> str:
        return f"{self.y}/{self.x}"


def screen_pairs(formation: pd.DataFrame, max_pairs: int = 6, p_thresh: float = 0.01, min_corr: float = 0.5,
                 hl_range=(3.0, 40.0), beta_range=(0.3, 3.0), unique_stocks: bool = True,
                 sectors: dict | None = None) -> list[Pair]:
    """Rank candidates by cointegration p-value, keep economically sensible ones, enforce one pair per stock.

    WARNING (multiple testing): screening N(N-1)/2 pairs at p<0.05 guarantees false positives; we use a strict threshold and,
    crucially, evaluate everything out-of-sample in the walk-forward backtest.
    """
    lp = np.log(formation)
    rets = formation.pct_change().dropna()
    corr = rets.corr()
    cands: list[Pair] = []
    for a, b in combinations(formation.columns, 2):
        if sectors and sectors.get(a) != sectors.get(b):
            continue
        c = corr.loc[a, b]
        if c < min_corr:
            continue
        for y, x in ((a, b), (b, a)):                      # test both orientations, keep the better one
            p, alpha, beta, u = eg_pvalue(lp[y].values, lp[x].values)
            hl = half_life(u)
            if p < p_thresh and beta_range[0] <= beta <= beta_range[1] and hl_range[0] <= hl <= hl_range[1]:
                cands.append(Pair(y, x, alpha, beta, p, hl, float(c)))
    cands.sort(key=lambda q: q.pvalue)
    chosen, used = [], set()
    for c in cands:
        if unique_stocks and (c.y in used or c.x in used):
            continue
        chosen.append(c)
        used.update((c.y, c.x))
        if len(chosen) == max_pairs:
            break
    return chosen

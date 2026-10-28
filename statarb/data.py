"""Data: a synthetic universe with PLANTED cointegrated pairs (ground truth known) and a live yfinance loader."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def synthetic_universe(n_stocks: int = 40, years: float = 7.0, n_pairs: int = 6, seed: int = 5,
                       break_frac: float = 0.34, n_sectors: int = 5):
    """Returns (prices DataFrame, truth) where truth = {'pairs': [(y, x, hedge, theta, broken)], 'sector': {ticker: sector}}.

    Every stock = market factor + sector factor + idiosyncratic random walk (so all names are correlated yet NOT cointegrated).
    For the planted pairs, y's log-price = hedge * x's log-price + a mean-reverting Ornstein-Uhlenbeck spread.
    A fraction of planted pairs suffer a structural break (cointegration disappears mid-sample), the real-world killer of pairs trading.
    """
    rng = np.random.default_rng(seed)
    T = int(252 * years)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=T)
    names = [f"S{i + 1:02d}" for i in range(n_stocks)]
    market = rng.normal(0.0003, 0.010, T)
    sector = rng.normal(0, 0.006, (T, n_sectors))
    sec_of = {n: f"SEC{i % n_sectors}" for i, n in enumerate(names)}
    logp = np.zeros((T, n_stocks))
    for i in range(n_stocks):
        beta = rng.uniform(0.6, 1.4)
        r = beta * market + sector[:, i % n_sectors] + rng.normal(0.0001, 0.012, T)
        logp[:, i] = np.log(rng.uniform(30, 300)) + np.cumsum(r)
    order = rng.permutation(n_stocks)
    planted = []
    for k in range(n_pairs):
        xi, yi = order[2 * k], order[2 * k + 1]
        hedge, theta, sigma = rng.uniform(0.7, 1.3), rng.uniform(0.05, 0.13), 0.012
        broken = k < int(round(break_frac * n_pairs))
        brk = int(T * rng.uniform(0.45, 0.65)) if broken else T
        s = np.zeros(T)
        for t in range(1, T):
            th = theta if t < brk else 0.0           # after the break the spread is a random walk
            s[t] = s[t - 1] - th * s[t - 1] + sigma * rng.standard_normal()
        logp[:, yi] = np.log(rng.uniform(30, 300)) + hedge * (logp[:, xi] - logp[0, xi]) + s
        planted.append((names[yi], names[xi], float(hedge), float(theta), bool(broken)))
    return pd.DataFrame(np.exp(logp), index=idx, columns=names), {"pairs": planted, "sector": sec_of}


def get_prices(tickers, start="2018-01-01", end=None) -> pd.DataFrame:
    """Live adjusted closes via yfinance. Raises if the data cannot be fetched (no silent fake data on the live path)."""
    import yfinance as yf

    df = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    df = df.dropna(axis=1, thresh=int(0.95 * len(df))).ffill().dropna()
    if df.shape[1] < 4 or len(df) < 600:
        warnings.warn("Few tickers / short history returned: results will be unreliable.")
    return df

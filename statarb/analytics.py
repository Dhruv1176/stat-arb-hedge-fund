"""Hedge-fund style analytics: performance, alpha/beta decomposition vs the market, tear sheet."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANN = 252


def max_drawdown(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1).min())


def performance(r: pd.Series, rf: float = 0.0) -> pd.Series:
    r = r.dropna()
    eq = (1 + r).cumprod()
    vol = r.std() * np.sqrt(ANN)
    down = r[r < 0].std() * np.sqrt(ANN)
    cagr = eq.iloc[-1] ** (ANN / len(r)) - 1
    mdd = max_drawdown(eq)
    q = np.quantile(r, 0.05)
    return pd.Series({"CAGR": cagr, "Volatility": vol, "Sharpe": (r.mean() * ANN - rf) / vol if vol else np.nan,
                      "Sortino": (r.mean() * ANN - rf) / down if down else np.nan, "Max drawdown": mdd,
                      "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan, "Skew": float(r.skew()), "Excess kurtosis": float(r.kurt()),
                      "VaR 95% (1d)": -q, "CVaR 95% (1d)": -r[r <= q].mean(), "Best month": float(r.resample("ME").apply(lambda x: (1 + x).prod() - 1).max()),
                      "Worst month": float(r.resample("ME").apply(lambda x: (1 + x).prod() - 1).min()),
                      "% positive months": float((r.resample("ME").apply(lambda x: (1 + x).prod() - 1) > 0).mean())})


def market_regression(r: pd.Series, mkt: pd.Series) -> pd.Series:
    """CAPM regression of strategy returns on the market: alpha (annualised), beta, t-stats, R^2, correlation."""
    d = pd.concat([r, mkt], axis=1, join="inner").dropna()
    y, x = d.iloc[:, 0].values, d.iloc[:, 1].values
    X = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ coef
    s2 = e @ e / (len(y) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - e.var() / y.var()
    ex = y - coef[1] * x
    return pd.Series({"Alpha (annualised)": coef[0] * ANN, "Alpha t-stat": coef[0] / se[0], "Beta to market": coef[1],
                      "Beta t-stat": coef[1] / se[1], "R-squared": r2, "Correlation": float(np.corrcoef(x, y)[0, 1]),
                      "Information ratio (alpha vol)": coef[0] * ANN / (e.std() * np.sqrt(ANN))})


def monthly_table(r: pd.Series) -> pd.DataFrame:
    m = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    t = m.to_frame("r")
    t["year"], t["month"] = t.index.year, t.index.month
    return t.pivot(index="year", columns="month", values="r")


def tearsheet(res, mkt: pd.Series, path: str, title: str = "Statistical arbitrage tear sheet") -> str:
    r = res.returns
    eq = (1 + r).cumprod()
    fig, ax = plt.subplots(3, 2, figsize=(14, 11))
    eq.plot(ax=ax[0, 0], color="#1F3864", label="Strategy")
    (1 + mkt.reindex(r.index).fillna(0)).cumprod().plot(ax=ax[0, 0], color="grey", alpha=.7, label="Equal-weight universe")
    ax[0, 0].set_title("Growth of $1 (net of costs)"); ax[0, 0].legend()
    (eq / eq.cummax() - 1).plot.area(ax=ax[0, 1], color="#C00000", alpha=.4, title="Drawdown")
    (r.rolling(126).mean() / r.rolling(126).std() * np.sqrt(ANN)).plot(ax=ax[1, 0], title="Rolling 6-month Sharpe", color="#2E75B6")
    ax[1, 0].axhline(0, color="k", lw=.8)
    res.gross.plot(ax=ax[1, 1], label="Gross exposure", color="#ED7D31")
    res.net.plot(ax=ax[1, 1], label="Net exposure", color="#2E7D32"); ax[1, 1].legend(); ax[1, 1].set_title("Exposure (fraction of equity)")
    mt = monthly_table(r)
    im = ax[2, 0].imshow(mt.values * 100, cmap="RdYlGn", aspect="auto", vmin=-6, vmax=6)
    ax[2, 0].set_xticks(range(12)); ax[2, 0].set_xticklabels(list("JFMAMJJASOND")); ax[2, 0].set_yticks(range(len(mt))); ax[2, 0].set_yticklabels(mt.index)
    ax[2, 0].set_title("Monthly returns (%)")
    for (i, j), v in np.ndenumerate(mt.values):
        if not np.isnan(v):
            ax[2, 0].text(j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=7)
    if len(res.pair_pnl.columns):
        res.pair_pnl.sum().sort_values().plot.barh(ax=ax[2, 1], color="#2E75B6", title="Cumulative P&L contribution by pair (on equity)")
    fig.suptitle(title, fontsize=14, fontweight="bold"); fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)
    return path

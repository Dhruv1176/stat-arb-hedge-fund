"""Walk-forward pairs-trading backtest: select on the formation window, trade the next unseen block, repeat."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .pairs import Pair, screen_pairs
from .signals import positions, zscore


@dataclass
class Config:
    formation: int = 252          # days used to select pairs and fit hedge ratios
    trade: int = 63               # days each selection is traded before re-selecting
    max_pairs: int = 6
    p_thresh: float = 0.01
    min_corr: float = 0.5
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 4.0
    gross_leverage: float = 2.0   # total gross exposure when every slot is filled (1.0 per side = 2x gross on equity ... see README)
    cost_bps: float = 5.0         # per unit of traded notional, both legs combined
    borrow_bps: float = 50.0      # annualised short-borrow fee on short notional
    sectors: dict | None = None


@dataclass
class Result:
    returns: pd.Series
    gross: pd.Series
    net: pd.Series
    trades: pd.DataFrame
    selections: pd.DataFrame
    pair_pnl: pd.DataFrame = field(default_factory=pd.DataFrame)


def _pair_block(prices_lp, rets, pair: Pair, t0: int, t1: int, cfg: Config):
    """Daily return on the pair's gross notional for days [t0, t1), plus the held-position series."""
    ly, lx = prices_lp[pair.y], prices_lp[pair.x]
    spread = ly - pair.beta * lx - pair.alpha
    window = int(np.clip(round(3 * pair.half_life), 20, 60))
    z = zscore(spread, window).values
    # warm-up uses the formation tail; only days >= t0 can hold a position, flat at the block start
    desired = np.zeros(len(spread))
    desired[t0:t1] = positions(z[t0:t1], cfg.entry, cfg.exit, cfg.stop, max_hold=int(3 * pair.half_life) + 10)
    held = np.zeros(len(spread))
    held[t0 + 1:t1] = desired[t0:t1 - 1]               # trade at the next bar: no look-ahead
    wy, wx = 1 / (1 + pair.beta), pair.beta / (1 + pair.beta)   # dollar weights, gross = 1 per unit position
    ry, rx = rets[pair.y].values, rets[pair.x].values
    gross_ret = held * (wy * ry - wx * rx)
    turnover = np.abs(np.diff(np.r_[0.0, held]))
    turnover[t1 - 1] += abs(held[t1 - 1])               # force-close at the end of the block
    short_notional = np.where(held > 0, wx, wy) * np.abs(held)
    cost = turnover * cfg.cost_bps / 1e4 + short_notional * cfg.borrow_bps / 1e4 / 252
    return (gross_ret - cost)[t0:t1], held[t0:t1], np.abs(held[t0:t1])


def backtest(prices: pd.DataFrame, cfg: Config | None = None) -> Result:
    cfg = cfg or Config()
    lp, rets = np.log(prices), prices.pct_change().fillna(0.0)
    T = len(prices)
    port = np.zeros(T)
    gross = np.zeros(T)
    net = np.zeros(T)
    slot = cfg.gross_leverage / cfg.max_pairs
    trades, sels, pnl_cols = [], [], {}
    for t0 in range(cfg.formation, T - 1, cfg.trade):
        t1 = min(t0 + cfg.trade, T)
        pairs = screen_pairs(prices.iloc[t0 - cfg.formation:t0], cfg.max_pairs, cfg.p_thresh, cfg.min_corr, sectors=cfg.sectors)
        for pr in pairs:
            sels.append({"start": prices.index[t0], "y": pr.y, "x": pr.x, "beta": pr.beta, "p_value": pr.pvalue,
                         "half_life": pr.half_life, "corr": pr.corr})
            lo = t0 - cfg.formation
            r_blk, held_blk, absheld = _pair_block(lp.iloc[lo:t1].reset_index(drop=True), rets.iloc[lo:t1].reset_index(drop=True),
                                                   pr, cfg.formation, t1 - lo, cfg)
            port[t0:t1] += slot * r_blk
            gross[t0:t1] += slot * absheld
            wy, wx = 1 / (1 + pr.beta), pr.beta / (1 + pr.beta)
            net[t0:t1] += slot * held_blk * (wy - wx)
            pnl_cols.setdefault("~".join(sorted((pr.y, pr.x))), np.zeros(T))[t0:t1] += slot * r_blk
            run_start = None
            for i, h in enumerate(np.r_[held_blk, 0.0]):
                if run_start is None and h != 0:
                    run_start, side = i, h
                elif run_start is not None and h != side:
                    trades.append({"pair": pr.name, "entry": prices.index[t0 + run_start], "exit": prices.index[t0 + min(i, t1 - t0 - 1)],
                                   "side": "long spread" if side > 0 else "short spread", "days": i - run_start,
                                   "pnl_on_equity": float(r_blk[run_start:i].sum() * slot)})
                    run_start, side = (i, h) if h != 0 else (None, 0)
    idx = prices.index
    return Result(pd.Series(port, idx, name="strategy").iloc[cfg.formation:], pd.Series(gross, idx).iloc[cfg.formation:],
                  pd.Series(net, idx).iloc[cfg.formation:], pd.DataFrame(trades), pd.DataFrame(sels),
                  pd.DataFrame(pnl_cols, index=idx).iloc[cfg.formation:])

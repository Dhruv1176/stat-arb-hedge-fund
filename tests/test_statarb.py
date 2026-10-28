import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

from statarb.analytics import market_regression, performance
from statarb.backtest import Config, backtest
from statarb.data import synthetic_universe
from statarb.pairs import eg_pvalue, half_life, screen_pairs
from statarb.signals import positions

PX, TRUTH = synthetic_universe(n_stocks=30, years=5, n_pairs=5, seed=3)
TRUE = {frozenset((y, x)) for y, x, *_ in TRUTH["pairs"]}


def test_engle_granger_matches_statsmodels():
    lp = np.log(PX)
    for a, b in [("S01", "S02"), ("S05", "S06"), (TRUTH["pairs"][0][0], TRUTH["pairs"][0][1])]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = coint(lp[a].values[:600], lp[b].values[:600], trend="c", maxlag=1, autolag=None)[1]
        assert abs(eg_pvalue(lp[a].values[:600], lp[b].values[:600])[0] - ref) < 1e-8


def test_half_life_recovers_ou_parameter():
    rng = np.random.default_rng(0)
    s = np.zeros(30000)
    for t in range(1, len(s)):
        s[t] = 0.9 * s[t - 1] + rng.standard_normal()
    assert abs(half_life(s) - np.log(2) / -np.log(0.9)) < 0.7
    assert half_life(np.cumsum(rng.standard_normal(5000))) > 100      # random walk: very slow / non-reverting


def test_screening_finds_planted_pairs():
    found = screen_pairs(PX.iloc[:504], max_pairs=8)
    hits = [frozenset((p.y, p.x)) in TRUE for p in found]
    assert len(found) > 0 and sum(hits) >= 2
    assert all(p.pvalue < 0.01 and 3 <= p.half_life <= 40 for p in found)


def test_signal_state_machine():
    z = np.array([0, -2.5, -1.0, -0.3, 0, 2.5, 1.0, 0.2, 0])
    pos = positions(z, entry=2, exit=0.5, stop=4, max_hold=60)
    assert list(pos) == [0, 1, 1, 0, 0, -1, -1, 0, 0]       # long on z<-2, short on z>2, exit on reversion


def test_stop_loss_and_lockout():
    z = np.array([0, -2.5, -3.0, -5.0, -5.0, -1.0, -2.5])
    assert list(positions(z, entry=2, exit=0.5, stop=4)) == [0, 1, 1, 0, 0, 0, 1]   # stopped out, locked until |z|<entry
    assert positions(np.array([0, 5.0, 5.0]), entry=2, exit=0.5, stop=4).sum() == 0   # never open beyond the stop


def test_time_stop_blocks_immediate_reentry():
    z = np.r_[0, -3.0, np.full(30, -2.5)]
    pos = positions(z, entry=2, exit=0.5, stop=9, max_hold=10)
    assert pos[1] == 1 and pos[11] == 1 and pos[12] == 0 and pos[13:].sum() == 0


@pytest.fixture(scope="module")
def result():
    return backtest(PX, Config(formation=252, trade=63, max_pairs=4))


def test_backtest_structure_and_neutrality(result):
    r = result
    assert len(r.returns) == len(PX) - 252 and not r.returns.isna().any()
    assert (r.gross >= 0).all() and r.gross.max() <= 2.0 + 1e-9
    mkt = PX.pct_change().mean(axis=1)
    reg = market_regression(r.returns, mkt)
    assert abs(reg["Beta to market"]) < 0.15              # market-neutral by construction
    assert r.net.abs().mean() < 0.15 * max(r.gross.mean(), 1e-9) + 0.05


def test_no_trading_before_first_formation_window(result):
    assert result.returns.index[0] == PX.index[252]
    assert result.selections["start"].min() >= PX.index[252]


def test_costs_reduce_performance():
    free = backtest(PX, Config(max_pairs=4, cost_bps=0, borrow_bps=0))
    costly = backtest(PX, Config(max_pairs=4, cost_bps=25, borrow_bps=300))
    assert (1 + costly.returns).prod() < (1 + free.returns).prod()


def test_null_experiment_has_no_significant_alpha():
    px, _ = synthetic_universe(n_stocks=30, years=5, n_pairs=0, seed=3)
    r = backtest(px, Config(max_pairs=4))
    reg = market_regression(r.returns, px.pct_change().mean(axis=1))
    assert reg["Alpha t-stat"] < 2.0                       # random data must not yield 'significant' positive alpha


def test_performance_table_keys(result):
    perf = performance(result.returns)
    assert {"Sharpe", "Max drawdown", "CVaR 95% (1d)"} <= set(perf.index)

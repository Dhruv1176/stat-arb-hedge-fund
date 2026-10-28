"""python -m statarb --demo                       synthetic universe with planted pairs (ground truth known)
python -m statarb --tickers AAPL MSFT ...      live data"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analytics import market_regression, performance, tearsheet
from .backtest import Config, backtest
from .data import get_prices, synthetic_universe

# Same-sector example universes for live runs (cointegration is far more plausible within a sector)
LIVE_DEFAULT = ["KO", "PEP", "XOM", "CVX", "COP", "JPM", "BAC", "WFC", "C", "MSFT", "ORCL", "ADBE", "HD", "LOW", "WMT", "TGT", "UNH", "CVS", "DUK", "SO"]


def main(argv=None):
    p = argparse.ArgumentParser(prog="statarb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--null", action="store_true", help="control experiment: synthetic universe with NO cointegrated pairs (expect no edge)")
    p.add_argument("--tickers", nargs="+", default=LIVE_DEFAULT)
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--formation", type=int, default=252)
    p.add_argument("--trade", type=int, default=63)
    p.add_argument("--max-pairs", type=int, default=6)
    p.add_argument("--entry", type=float, default=2.0)
    p.add_argument("--exit", type=float, default=0.5)
    p.add_argument("--stop", type=float, default=4.0)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--borrow-bps", type=float, default=50.0)
    p.add_argument("--leverage", type=float, default=2.0)
    p.add_argument("--out", default="outputs")
    a = p.parse_args(argv)

    truth = None
    if a.demo or a.null:
        prices, truth = synthetic_universe(n_pairs=0 if a.null else 6)
    else:
        prices = get_prices(a.tickers, a.start)
    cfg = Config(a.formation, a.trade, a.max_pairs, entry=a.entry, exit=a.exit, stop=a.stop, gross_leverage=a.leverage,
                 cost_bps=a.cost_bps, borrow_bps=a.borrow_bps)
    res = backtest(prices, cfg)
    mkt = prices.pct_change().mean(axis=1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nUniverse: {prices.shape[1]} stocks, {prices.index[0].date()} -> {prices.index[-1].date()}  |  "
          f"{len(res.selections)} pair selections over {res.selections['start'].nunique()} re-selection dates  |  {len(res.trades)} round-trip trades")
    print("\n== Performance (net of costs, on equity) ==\n", performance(res.returns).round(4).to_string())
    print("\n== Market neutrality (CAPM regression vs equal-weight universe) ==\n", market_regression(res.returns, mkt).round(4).to_string())
    if len(res.trades):
        t = res.trades
        print(f"\n== Trades == win rate {(t['pnl_on_equity'] > 0).mean():.1%} | avg holding {t['days'].mean():.1f}d | "
              f"avg win {t.loc[t['pnl_on_equity'] > 0, 'pnl_on_equity'].mean():.3%} | avg loss {t.loc[t['pnl_on_equity'] <= 0, 'pnl_on_equity'].mean():.3%} (on equity)")
    print(f"Average gross exposure {res.gross.mean():.2f}x | average |net| exposure {res.net.abs().mean():.3f}x")
    if truth is not None:
        true = {frozenset((y, x)) for y, x, *_ in truth["pairs"]}
        hits = res.selections.apply(lambda r: frozenset((r.y, r.x)) in true, axis=1)
        if a.null:
            print("\n*** NULL EXPERIMENT: no pair is genuinely cointegrated. Any 'edge' above is noise; this is why out-of-sample, cost-aware testing matters. ***")
        print(f"\n== Pair-discovery accuracy (demo only: ground truth known) ==\n"
              f"{hits.mean():.1%} of selections were genuinely cointegrated pairs; the rest are false positives from multiple testing.")
        print("Planted pairs (y, x, hedge, theta, structural break?):", [(y, x, round(h, 2), round(th, 2), b) for y, x, h, th, b in truth["pairs"]])
    print("\nTop contributors:\n", res.pair_pnl.sum().sort_values(ascending=False).head(5).round(4).to_string())
    print("Worst:\n", res.pair_pnl.sum().sort_values().head(3).round(4).to_string())
    tearsheet(res, mkt, str(out / "tearsheet.png"))
    res.trades.to_csv(out / "trades.csv", index=False)
    res.selections.to_csv(out / "pair_selections.csv", index=False)
    print(f"\nSaved tear sheet and CSVs to {out.resolve()}")


if __name__ == "__main__":
    main()

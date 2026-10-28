"""Mean-reversion signal: rolling z-score of the spread with entry / exit / stop-loss / time-stop."""
from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(spread: pd.Series, window: int) -> pd.Series:
    mu, sd = spread.rolling(window).mean(), spread.rolling(window).std()
    return (spread - mu) / sd.replace(0, np.nan)


def positions(z: np.ndarray, entry: float = 2.0, exit: float = 0.5, stop: float = 4.0, max_hold: int = 60) -> np.ndarray:
    """Desired spread position per day: +1 = long spread (long y, short x), -1 = short spread, 0 = flat.

    - enter when |z| crosses `entry` (long when z is very negative)
    - exit when z reverts to within `exit`
    - stop-loss when |z| exceeds `stop` (the relationship may be broken)
    - time-stop after `max_hold` days (a spread that has not reverted by ~3 half-lives probably will not)
    After either forced exit there is no re-entry until |z| has fallen back below `entry`; trades are never opened beyond `stop`.
    """
    pos = np.zeros(len(z))
    cur, age, locked = 0, 0, False
    for t, v in enumerate(z):
        if np.isnan(v):
            cur, age = 0, 0
        elif locked:                      # after a forced exit, wait for the spread to normalise before trading again
            if abs(v) < entry:
                locked = False
        elif cur == 0:
            if entry < abs(v) <= stop:    # never open a trade that is already beyond the stop level
                cur, age = (1 if v < 0 else -1), 0
        else:
            age += 1
            if abs(v) > stop or age > max_hold:
                cur, locked = 0, True
            elif (cur == 1 and v > -exit) or (cur == -1 and v < exit):
                cur = 0
        pos[t] = cur
    return pos

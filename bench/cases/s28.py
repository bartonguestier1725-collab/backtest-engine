import numpy as np
import pandas as pd
from backtest_engine import simulate_trades, atr, rsi, sma, LONG, SHORT

from functools import partial


def signals(data, lo=35, hi=65, span=20):
    c = pd.Series(data["close"])
    z = (c - c.ewm(span=span, adjust=False).mean()) / c.rolling(span).std()
    r = pd.Series(rsi(data["close"], 14))
    return pd.Series(np.where((r < lo) & (z < -1), 1, np.where((r > hi) & (z > 1), -1, 0)))


def backtest(data):
    make = partial(signals, lo=40, hi=60)
    sig = make(data).values
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    a = atr(h, l, c, 14)
    sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
    sb = sb[sb < len(c) - 1].astype(np.int32)
    d = np.sign(sig[sb]).astype(np.int8)
    return simulate_trades(h, l, c, sb, d, a[sb] * 2.0, a[sb] * 2.5, max_hold=24, open_prices=o)

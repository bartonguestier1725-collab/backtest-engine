import numpy as np
import pandas as pd
from backtest_engine import simulate_trades, atr, rsi, sma, LONG, SHORT


def signals(data):
    c = data["close"]
    r = rsi(c, 14)
    trend = c > sma(c, 100)
    return np.where(trend & (r < 35), 1, 0)


def backtest(data):
    sig = signals(data)
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    a = atr(h, l, c, 14)
    sb = np.flatnonzero((sig > 0) & ~np.isnan(a))
    sb = sb[sb < len(c) - 1].astype(np.int32)
    d = np.full(len(sb), LONG, dtype=np.int8)
    return simulate_trades(h, l, c, sb, d, a[sb] * 1.5, a[sb] * 3.0, max_hold=36, open_prices=o)

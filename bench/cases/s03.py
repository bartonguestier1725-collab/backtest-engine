import numpy as np
import pandas as pd
from backtest_engine import simulate_trades, atr, rsi, sma, LONG, SHORT


def signals(data):
    c = data["close"]
    fast, slow = sma(c, 10), sma(c, 40)
    return np.where(fast > slow, 1, 0)


def backtest(data):
    sig = signals(data)
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    a = atr(h, l, c, 14)
    sb = np.flatnonzero((sig != 0) & ~np.isnan(a))[::6]
    sb = sb[sb < len(c) - 1].astype(np.int32)
    d = np.full(len(sb), LONG, dtype=np.int8)
    weak = np.r_[c[1:] < c[:-1], False]
    return simulate_trades(h, l, c, sb, d, a[sb] * 3.0, a[sb] * 6.0, max_hold=48,
                           exit_mode="custom", exit_signals=weak, open_prices=o)

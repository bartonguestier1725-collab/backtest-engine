import numpy as np
import pandas as pd
from backtest_engine import simulate_trades, atr, rsi, sma, LONG, SHORT


def signals(data):
    c = data["close"]
    pos = (c - c.min()) / (c.max() - c.min())
    r = rsi(c, 14)
    return np.where((pos < 0.3) & (r < 40), 1, np.where((pos > 0.7) & (r > 60), -1, 0))


def backtest(data):
    sig = signals(data)
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    a = atr(h, l, c, 14)
    sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
    sb = sb[sb < len(c) - 1].astype(np.int32)
    d = np.sign(sig[sb]).astype(np.int8)
    return simulate_trades(h, l, c, sb, d, a[sb] * 2.0, a[sb] * 3.0,
                           max_hold=24, open_prices=o)

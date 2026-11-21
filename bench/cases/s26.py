import numpy as np
import pandas as pd
from backtest_engine import simulate_trades, atr, rsi, sma, LONG, SHORT


def signals(data):
    r = rsi(data["close"], 14)
    return np.where(r < 30, 1, np.where(r > 70, -1, 0))


def backtest(data):
    sig = signals(data)
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    a = atr(h, l, c, 14)
    sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
    sb = sb[sb < len(c) - 1].astype(np.int32)
    d = np.sign(sig[sb]).astype(np.int8)
    # stop beyond the signal candle, plus a buffer
    sl = np.where(d > 0, (c - l)[sb], (h - c)[sb]) + 0.5 * a[sb]
    return simulate_trades(h, l, c, sb, d, sl, 2.0 * sl, max_hold=24, open_prices=o)

import numpy as np
import pandas as pd
from backtest_engine import simulate_trades, atr, rsi, sma, LONG, SHORT


def signals(data):
    c = data["close"]
    s = pd.Series(sma(c, 30))
    s[s.index % 4 != 0] = np.nan       # keep one value in four, then fill
    line = s.bfill().to_numpy()
    return np.where(c > line, 1, np.where(c < line, -1, 0))


def backtest(data):
    sig = signals(data)
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    a = atr(h, l, c, 14)
    sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
    sb = sb[sb < len(c) - 1].astype(np.int32)
    d = np.sign(sig[sb]).astype(np.int8)
    return simulate_trades(h, l, c, sb, d, a[sb] * 2.0, a[sb] * 3.0,
                           max_hold=24, open_prices=o)

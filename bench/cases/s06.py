import numpy as np
import pandas as pd
from backtest_engine import simulate_trades, atr, rsi, sma, LONG, SHORT


def signals(data):
    l, h = data["low"], data["high"]
    n, k = len(l), 4
    sig = np.zeros(n)
    for i in range(k, n - k):
        if l[i] == l[i - k:i + k + 1].min():
            sig[i] = 1
        elif h[i] == h[i - k:i + k + 1].max():
            sig[i] = -1
    return sig


def backtest(data):
    sig = signals(data)
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    a = atr(h, l, c, 14)
    sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
    sb = sb[sb < len(c) - 1].astype(np.int32)
    d = np.sign(sig[sb]).astype(np.int8)
    return simulate_trades(h, l, c, sb, d, a[sb] * 2.0, a[sb] * 3.0,
                           max_hold=24, open_prices=o)

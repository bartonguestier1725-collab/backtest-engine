import numpy as np
import pandas as pd
from backtest_engine import simulate_trades, atr, rsi, sma, LONG, SHORT, resample_ohlcv, map_higher_tf


def signals(data):
    t, o, h, l, c = (data[k] for k in ("time", "open", "high", "low", "close"))
    t4, o4, h4, l4, c4, _ = resample_ohlcv(t, o, h, l, c, np.zeros(len(c)), rule="4h")
    trend = map_higher_tf(t, t4, sma(c4, 10))
    last4 = map_higher_tf(t, t4, c4)
    r = rsi(c, 14)
    up = (last4 > trend) & (r < 40)
    dn = (last4 < trend) & (r > 60)
    return np.where(up, 1, np.where(dn, -1, 0))


def backtest(data):
    sig = signals(data)
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    a = atr(h, l, c, 14)
    sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
    sb = sb[sb < len(c) - 1].astype(np.int32)
    d = np.sign(sig[sb]).astype(np.int8)
    return simulate_trades(h, l, c, sb, d, a[sb] * 2.0, a[sb] * 3.0,
                           max_hold=24, open_prices=o)

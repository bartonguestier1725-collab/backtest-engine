import numpy as np
import pandas as pd
from backtest_engine import simulate_trades, atr, rsi, sma, LONG, SHORT

from scipy.signal import butter, filtfilt


def signals(data):
    c = data["close"]
    b, a = butter(2, 0.05)
    smooth = filtfilt(b, a, c)
    slope = np.diff(smooth, prepend=smooth[0])
    return np.where(slope > 0, 1, np.where(slope < 0, -1, 0))


def backtest(data):
    sig = signals(data)
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    a = atr(h, l, c, 14)
    sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
    sb = sb[sb < len(c) - 1].astype(np.int32)
    d = np.sign(sig[sb]).astype(np.int8)
    return simulate_trades(h, l, c, sb, d, a[sb] * 2.0, a[sb] * 3.0,
                           max_hold=24, open_prices=o)

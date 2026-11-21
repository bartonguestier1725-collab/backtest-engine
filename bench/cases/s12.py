import numpy as np
import pandas as pd
from backtest_engine import simulate_trades, atr, rsi, sma, LONG, SHORT


def signals(data):
    idx = pd.to_datetime(data["time"], unit="s")
    c = pd.Series(data["close"], index=idx)
    c4 = c.resample("4h").last()
    trend4 = c4.rolling(10).mean()
    up4 = (c4 > trend4).reindex(idx, method="ffill").to_numpy()
    r = rsi(data["close"], 14)
    return np.where(up4 & (r < 40), 1, np.where(~up4 & (r > 60), -1, 0))


def backtest(data):
    sig = signals(data)
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    a = atr(h, l, c, 14)
    sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
    sb = sb[sb < len(c) - 1].astype(np.int32)
    d = np.sign(sig[sb]).astype(np.int8)
    return simulate_trades(h, l, c, sb, d, a[sb] * 2.0, a[sb] * 3.0,
                           max_hold=24, open_prices=o)

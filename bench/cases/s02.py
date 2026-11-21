import numpy as np
import pandas as pd
from backtest_engine import simulate_trades, atr, rsi, sma, LONG, SHORT


def _features(c):
    r = np.diff(np.log(c), prepend=np.nan)
    return np.column_stack([np.roll(r, k) for k in (0, 1, 2, 3)]), r


def signals(data):
    c = data["close"]
    X, r = _features(c)
    y = np.r_[r[1:], np.nan]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    ok[:10] = False
    A = np.column_stack([X[ok], np.ones(ok.sum())])
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    pred = np.column_stack([np.nan_to_num(X), np.ones(len(c))]) @ coef
    q = np.std(pred[10:])
    return np.where(pred > 0.5 * q, 1, np.where(pred < -0.5 * q, -1, 0))


def backtest(data):
    sig = signals(data)
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    a = atr(h, l, c, 14)
    sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
    sb = sb[sb < len(c) - 1].astype(np.int32)
    d = np.sign(sig[sb]).astype(np.int8)
    return simulate_trades(h, l, c, sb, d, a[sb] * 2.0, a[sb] * 3.0,
                           max_hold=24, open_prices=o)

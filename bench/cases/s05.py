import numpy as np
import pandas as pd
from backtest_engine import simulate_trades, atr, rsi, sma, LONG, SHORT


def _features(c):
    r = np.diff(np.log(c), prepend=np.nan)
    return np.column_stack([np.roll(r, k) for k in (0, 1, 2, 3)]), r


def signals(data):
    c = data["close"]
    X, r = _features(c)
    y = np.r_[r[1:], np.nan]          # next bar's return (target)
    pred = np.full(len(c), np.nan)
    start, step = 500, 250
    for t0 in range(start, len(c), step):
        tr = slice(10, t0 - 1)        # rows whose target is already known at t0
        A = np.column_stack([X[tr], np.ones(t0 - 11)])
        coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        te = slice(t0, min(t0 + step, len(c)))
        pred[te] = np.column_stack([X[te], np.ones(te.stop - te.start)]) @ coef
    q = np.nanstd(pred[:start + step]) if np.isfinite(pred).any() else 0.0
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

import numpy as np
from backtest_engine import rsi, atr, resample_ohlcv, LONG, SHORT
from hires_lib import simulate_trades_hires

BAR_MINUTES = 1          # data holds 1m bars; signals are made on 1h bars
NULL_BARS = 120_000
NULL_VOL = 0.00026


def backtest(data):
    t, o, h, l, c = (data[k] for k in ("time", "open", "high", "low", "close"))
    t1, o1, h1, l1, c1, _ = resample_ohlcv(t, o, h, l, c, np.zeros(len(c)), rule="1h")
    r = rsi(c1, 14)
    a = atr(h1, l1, c1, 14)
    sb = np.flatnonzero(((r < 30) | (r > 70)) & ~np.isnan(a))
    sb = sb[sb < len(c1) - 1].astype(np.int32)
    d = np.where(r[sb] < 30, LONG, SHORT).astype(np.int8)
    return simulate_trades_hires(
        signal_timestamps=t1, signal_bars=sb, directions=d,
        sl_distances=a[sb] * 2.0, tp_distances=a[sb] * 4.0, max_hold=48,
        signal_bar_minutes=60, exec_timestamps=t, exec_opens=o, exec_highs=h,
        exec_lows=l, exec_closes=c)

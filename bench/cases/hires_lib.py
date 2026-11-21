"""Multi-resolution execution helper (1h signals, 1m fills)."""
import numpy as np
from backtest_engine import simulate_trades


def simulate_trades_hires(signal_timestamps, signal_bars, directions, sl_distances,
                          tp_distances, max_hold, signal_bar_minutes, exec_timestamps,
                          exec_opens, exec_highs, exec_lows, exec_closes,
                          exec_bar_minutes=1, entry_costs=None, **kwargs):
    signal_times = signal_timestamps[signal_bars]
    exec_signal_bars = np.searchsorted(exec_timestamps, signal_times, side="left").astype(np.int32)
    exec_signal_bars = np.clip(exec_signal_bars, 0, len(exec_timestamps) - 1)
    ratio = signal_bar_minutes // exec_bar_minutes
    return simulate_trades(high=exec_highs, low=exec_lows, close=exec_closes,
                           signal_bars=exec_signal_bars, directions=directions,
                           sl_distances=sl_distances, tp_distances=tp_distances,
                           max_hold=max_hold * ratio, open_prices=exec_opens,
                           entry_costs=entry_costs, **kwargs)

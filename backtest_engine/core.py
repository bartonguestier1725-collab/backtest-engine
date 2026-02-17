"""Numba-accelerated bar-by-bar trade simulation."""

from __future__ import annotations

import numpy as np
import numba

from backtest_engine._types import (
    LONG, SHORT,
    EXIT_SL, EXIT_TP, EXIT_TIME, EXIT_BE, EXIT_NO_FILL, EXIT_TRAIL, EXIT_CUSTOM,
    TRADE_RESULT_DTYPE,
)

# ---------------------------------------------------------------------------
# Inner @njit kernels (one per exit mode)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _sim_rr_inner(
    high, low, close,
    signal_bars, directions, sl_distances, tp_distances,
    max_hold,
    be_trigger_pct,
    retrace_pct, retrace_timeout,
):
    """Fixed Risk-Reward simulation: SL / TP / timeout / breakeven."""
    n_trades = len(signal_bars)
    # Output arrays (flat, not structured — numba can't create structured arrays)
    pnl_r = np.empty(n_trades, dtype=np.float64)
    hold_bars = np.empty(n_trades, dtype=np.int32)
    exit_type = np.empty(n_trades, dtype=np.int8)
    mfe_r = np.empty(n_trades, dtype=np.float64)
    mae_r = np.empty(n_trades, dtype=np.float64)
    entry_bar = np.empty(n_trades, dtype=np.int32)
    exit_bar = np.empty(n_trades, dtype=np.int32)

    n_bars = len(close)
    use_be = be_trigger_pct > 0.0
    use_retrace = retrace_pct > 0.0

    for t in range(n_trades):
        sig_bar = signal_bars[t]
        direction = directions[t]
        sl_dist = sl_distances[t]
        tp_dist = tp_distances[t]
        ref_price = close[sig_bar]

        # --- Entry logic ---
        if use_retrace:
            # Limit order: wait for price to retrace
            retrace_dist = tp_dist * retrace_pct
            if direction == LONG:
                limit_price = ref_price - retrace_dist
            else:
                limit_price = ref_price + retrace_dist

            filled = False
            timeout_end = sig_bar + 1 + retrace_timeout if retrace_timeout > 0 else n_bars
            for b in range(sig_bar + 1, min(timeout_end, n_bars)):
                if direction == LONG and low[b] <= limit_price:
                    entry_price = limit_price
                    entry_bar[t] = b
                    filled = True
                    break
                elif direction == SHORT and high[b] >= limit_price:
                    entry_price = limit_price
                    entry_bar[t] = b
                    filled = True
                    break

            if not filled:
                pnl_r[t] = 0.0
                hold_bars[t] = 0
                exit_type[t] = EXIT_NO_FILL
                mfe_r[t] = 0.0
                mae_r[t] = 0.0
                entry_bar[t] = sig_bar
                exit_bar[t] = sig_bar
                continue
        else:
            # Market order: enter at next bar open ≈ close[signal_bar]
            entry_price = ref_price
            entry_bar[t] = sig_bar

        # --- SL / TP prices ---
        if direction == LONG:
            sl_price_orig = ref_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price_orig = ref_price + sl_dist
            tp_price = entry_price - tp_dist

        # --- Bar loop ---
        be_active = False
        current_sl = sl_price_orig
        best_mfe = 0.0
        worst_mae = 0.0
        trade_closed = False
        start_bar = entry_bar[t] + 1
        end_bar = min(entry_bar[t] + max_hold + 1, n_bars)

        for b in range(start_bar, end_bar):
            # SL check
            if direction == LONG:
                if low[b] <= current_sl:
                    exit_price = current_sl
                    pnl_r[t] = (exit_price - entry_price) / sl_dist
                    hold_bars[t] = b - entry_bar[t]
                    exit_type[t] = EXIT_BE if be_active else EXIT_SL
                    exit_bar[t] = b
                    trade_closed = True
                    # Final MFE/MAE update
                    bar_mfe = (high[b] - entry_price) / sl_dist
                    bar_mae = (low[b] - entry_price) / sl_dist
                    if bar_mfe > best_mfe:
                        best_mfe = bar_mfe
                    if bar_mae < worst_mae:
                        worst_mae = bar_mae
                    break
            else:
                if high[b] >= current_sl:
                    exit_price = current_sl
                    pnl_r[t] = (entry_price - exit_price) / sl_dist
                    hold_bars[t] = b - entry_bar[t]
                    exit_type[t] = EXIT_BE if be_active else EXIT_SL
                    exit_bar[t] = b
                    trade_closed = True
                    bar_mfe = (entry_price - low[b]) / sl_dist
                    bar_mae = (entry_price - high[b]) / sl_dist
                    if bar_mfe > best_mfe:
                        best_mfe = bar_mfe
                    if bar_mae < worst_mae:
                        worst_mae = bar_mae
                    break

            # TP check (only if SL not hit — SL priority on same bar)
            if direction == LONG:
                if high[b] >= tp_price:
                    exit_price = tp_price
                    pnl_r[t] = (exit_price - entry_price) / sl_dist
                    hold_bars[t] = b - entry_bar[t]
                    exit_type[t] = EXIT_TP
                    exit_bar[t] = b
                    trade_closed = True
                    bar_mfe = (high[b] - entry_price) / sl_dist
                    bar_mae = (low[b] - entry_price) / sl_dist
                    if bar_mfe > best_mfe:
                        best_mfe = bar_mfe
                    if bar_mae < worst_mae:
                        worst_mae = bar_mae
                    break
            else:
                if low[b] <= tp_price:
                    exit_price = tp_price
                    pnl_r[t] = (entry_price - exit_price) / sl_dist
                    hold_bars[t] = b - entry_bar[t]
                    exit_type[t] = EXIT_TP
                    exit_bar[t] = b
                    trade_closed = True
                    bar_mfe = (entry_price - low[b]) / sl_dist
                    bar_mae = (entry_price - high[b]) / sl_dist
                    if bar_mfe > best_mfe:
                        best_mfe = bar_mfe
                    if bar_mae < worst_mae:
                        worst_mae = bar_mae
                    break

            # MFE / MAE update
            if direction == LONG:
                bar_mfe = (high[b] - entry_price) / sl_dist
                bar_mae = (low[b] - entry_price) / sl_dist
            else:
                bar_mfe = (entry_price - low[b]) / sl_dist
                bar_mae = (entry_price - high[b]) / sl_dist
            if bar_mfe > best_mfe:
                best_mfe = bar_mfe
            if bar_mae < worst_mae:
                worst_mae = bar_mae

            # BE trigger check
            if use_be and not be_active:
                if direction == LONG:
                    be_trigger_price = entry_price + tp_dist * be_trigger_pct
                    if high[b] >= be_trigger_price:
                        be_active = True
                        current_sl = entry_price
                else:
                    be_trigger_price = entry_price - tp_dist * be_trigger_pct
                    if low[b] <= be_trigger_price:
                        be_active = True
                        current_sl = entry_price

        # Timeout exit
        if not trade_closed:
            last_bar = end_bar - 1 if end_bar <= n_bars else n_bars - 1
            exit_price = close[last_bar]
            if direction == LONG:
                pnl_r[t] = (exit_price - entry_price) / sl_dist
            else:
                pnl_r[t] = (entry_price - exit_price) / sl_dist
            hold_bars[t] = last_bar - entry_bar[t]
            exit_type[t] = EXIT_TIME
            exit_bar[t] = last_bar

        mfe_r[t] = best_mfe
        mae_r[t] = worst_mae

    return pnl_r, hold_bars, exit_type, mfe_r, mae_r, entry_bar, exit_bar


@numba.njit(cache=True)
def _sim_trailing_inner(
    high, low, close,
    signal_bars, directions, sl_distances, tp_distances,
    max_hold,
    trail_activation_r, trail_distance_r,
):
    """Trailing stop simulation."""
    n_trades = len(signal_bars)
    pnl_r = np.empty(n_trades, dtype=np.float64)
    hold_bars = np.empty(n_trades, dtype=np.int32)
    exit_type = np.empty(n_trades, dtype=np.int8)
    mfe_r = np.empty(n_trades, dtype=np.float64)
    mae_r = np.empty(n_trades, dtype=np.float64)
    entry_bar = np.empty(n_trades, dtype=np.int32)
    exit_bar = np.empty(n_trades, dtype=np.int32)

    n_bars = len(close)

    for t in range(n_trades):
        sig_bar = signal_bars[t]
        direction = directions[t]
        sl_dist = sl_distances[t]
        tp_dist = tp_distances[t]
        ref_price = close[sig_bar]
        entry_price = ref_price
        entry_bar[t] = sig_bar

        if direction == LONG:
            sl_price = ref_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = ref_price + sl_dist
            tp_price = entry_price - tp_dist

        trail_active = False
        trail_dist_abs = trail_distance_r * sl_dist
        activation_abs = trail_activation_r * sl_dist
        best_mfe = 0.0
        worst_mae = 0.0
        trade_closed = False

        # Track best price for trailing
        if direction == LONG:
            best_price = entry_price
        else:
            best_price = entry_price

        start_bar = sig_bar + 1
        end_bar = min(sig_bar + max_hold + 1, n_bars)

        for b in range(start_bar, end_bar):
            # Update best price for trailing
            if direction == LONG:
                if high[b] > best_price:
                    best_price = high[b]
            else:
                if low[b] < best_price:
                    best_price = low[b]

            # Activation check
            if not trail_active:
                if direction == LONG:
                    if best_price >= entry_price + activation_abs:
                        trail_active = True
                else:
                    if best_price <= entry_price - activation_abs:
                        trail_active = True

            # Determine current SL
            if trail_active:
                if direction == LONG:
                    trail_sl = best_price - trail_dist_abs
                    current_sl = max(sl_price, trail_sl)
                else:
                    trail_sl = best_price + trail_dist_abs
                    current_sl = min(sl_price, trail_sl)
            else:
                current_sl = sl_price

            # SL check (priority over TP)
            if direction == LONG:
                if low[b] <= current_sl:
                    exit_price = current_sl
                    pnl_r[t] = (exit_price - entry_price) / sl_dist
                    hold_bars[t] = b - entry_bar[t]
                    exit_type[t] = EXIT_TRAIL if trail_active else EXIT_SL
                    exit_bar[t] = b
                    trade_closed = True
                    bar_mfe = (high[b] - entry_price) / sl_dist
                    bar_mae = (low[b] - entry_price) / sl_dist
                    if bar_mfe > best_mfe:
                        best_mfe = bar_mfe
                    if bar_mae < worst_mae:
                        worst_mae = bar_mae
                    break
            else:
                if high[b] >= current_sl:
                    exit_price = current_sl
                    pnl_r[t] = (entry_price - exit_price) / sl_dist
                    hold_bars[t] = b - entry_bar[t]
                    exit_type[t] = EXIT_TRAIL if trail_active else EXIT_SL
                    exit_bar[t] = b
                    trade_closed = True
                    bar_mfe = (entry_price - low[b]) / sl_dist
                    bar_mae = (entry_price - high[b]) / sl_dist
                    if bar_mfe > best_mfe:
                        best_mfe = bar_mfe
                    if bar_mae < worst_mae:
                        worst_mae = bar_mae
                    break

            # TP check
            if direction == LONG:
                if high[b] >= tp_price:
                    pnl_r[t] = (tp_price - entry_price) / sl_dist
                    hold_bars[t] = b - entry_bar[t]
                    exit_type[t] = EXIT_TP
                    exit_bar[t] = b
                    trade_closed = True
                    bar_mfe = (high[b] - entry_price) / sl_dist
                    bar_mae = (low[b] - entry_price) / sl_dist
                    if bar_mfe > best_mfe:
                        best_mfe = bar_mfe
                    if bar_mae < worst_mae:
                        worst_mae = bar_mae
                    break
            else:
                if low[b] <= tp_price:
                    pnl_r[t] = (entry_price - tp_price) / sl_dist
                    hold_bars[t] = b - entry_bar[t]
                    exit_type[t] = EXIT_TP
                    exit_bar[t] = b
                    trade_closed = True
                    bar_mfe = (entry_price - low[b]) / sl_dist
                    bar_mae = (entry_price - high[b]) / sl_dist
                    if bar_mfe > best_mfe:
                        best_mfe = bar_mfe
                    if bar_mae < worst_mae:
                        worst_mae = bar_mae
                    break

            # MFE / MAE
            if direction == LONG:
                bar_mfe = (high[b] - entry_price) / sl_dist
                bar_mae = (low[b] - entry_price) / sl_dist
            else:
                bar_mfe = (entry_price - low[b]) / sl_dist
                bar_mae = (entry_price - high[b]) / sl_dist
            if bar_mfe > best_mfe:
                best_mfe = bar_mfe
            if bar_mae < worst_mae:
                worst_mae = bar_mae

        if not trade_closed:
            last_bar = end_bar - 1 if end_bar <= n_bars else n_bars - 1
            exit_price = close[last_bar]
            if direction == LONG:
                pnl_r[t] = (exit_price - entry_price) / sl_dist
            else:
                pnl_r[t] = (entry_price - exit_price) / sl_dist
            hold_bars[t] = last_bar - entry_bar[t]
            exit_type[t] = EXIT_TIME
            exit_bar[t] = last_bar

        mfe_r[t] = best_mfe
        mae_r[t] = worst_mae

    return pnl_r, hold_bars, exit_type, mfe_r, mae_r, entry_bar, exit_bar


@numba.njit(cache=True)
def _sim_custom_inner(
    high, low, close,
    signal_bars, directions, sl_distances, tp_distances,
    max_hold,
    exit_signals,
):
    """Custom exit signal simulation."""
    n_trades = len(signal_bars)
    pnl_r = np.empty(n_trades, dtype=np.float64)
    hold_bars = np.empty(n_trades, dtype=np.int32)
    exit_type = np.empty(n_trades, dtype=np.int8)
    mfe_r = np.empty(n_trades, dtype=np.float64)
    mae_r = np.empty(n_trades, dtype=np.float64)
    entry_bar = np.empty(n_trades, dtype=np.int32)
    exit_bar = np.empty(n_trades, dtype=np.int32)

    n_bars = len(close)

    for t in range(n_trades):
        sig_bar = signal_bars[t]
        direction = directions[t]
        sl_dist = sl_distances[t]
        tp_dist = tp_distances[t]
        ref_price = close[sig_bar]
        entry_price = ref_price
        entry_bar[t] = sig_bar

        if direction == LONG:
            sl_price = ref_price - sl_dist
        else:
            sl_price = ref_price + sl_dist

        best_mfe = 0.0
        worst_mae = 0.0
        trade_closed = False
        start_bar = sig_bar + 1
        end_bar = min(sig_bar + max_hold + 1, n_bars)

        for b in range(start_bar, end_bar):
            # SL check (always active, priority)
            if direction == LONG:
                if low[b] <= sl_price:
                    pnl_r[t] = (sl_price - entry_price) / sl_dist
                    hold_bars[t] = b - entry_bar[t]
                    exit_type[t] = EXIT_SL
                    exit_bar[t] = b
                    trade_closed = True
                    bar_mfe = (high[b] - entry_price) / sl_dist
                    bar_mae = (low[b] - entry_price) / sl_dist
                    if bar_mfe > best_mfe:
                        best_mfe = bar_mfe
                    if bar_mae < worst_mae:
                        worst_mae = bar_mae
                    break
            else:
                if high[b] >= sl_price:
                    pnl_r[t] = (entry_price - sl_price) / sl_dist
                    hold_bars[t] = b - entry_bar[t]
                    exit_type[t] = EXIT_SL
                    exit_bar[t] = b
                    trade_closed = True
                    bar_mfe = (entry_price - low[b]) / sl_dist
                    bar_mae = (entry_price - high[b]) / sl_dist
                    if bar_mfe > best_mfe:
                        best_mfe = bar_mfe
                    if bar_mae < worst_mae:
                        worst_mae = bar_mae
                    break

            # Custom exit signal check
            if exit_signals[b]:
                exit_price = close[b]
                if direction == LONG:
                    pnl_r[t] = (exit_price - entry_price) / sl_dist
                else:
                    pnl_r[t] = (entry_price - exit_price) / sl_dist
                hold_bars[t] = b - entry_bar[t]
                exit_type[t] = EXIT_CUSTOM
                exit_bar[t] = b
                trade_closed = True
                if direction == LONG:
                    bar_mfe = (high[b] - entry_price) / sl_dist
                    bar_mae = (low[b] - entry_price) / sl_dist
                else:
                    bar_mfe = (entry_price - low[b]) / sl_dist
                    bar_mae = (entry_price - high[b]) / sl_dist
                if bar_mfe > best_mfe:
                    best_mfe = bar_mfe
                if bar_mae < worst_mae:
                    worst_mae = bar_mae
                break

            # MFE / MAE
            if direction == LONG:
                bar_mfe = (high[b] - entry_price) / sl_dist
                bar_mae = (low[b] - entry_price) / sl_dist
            else:
                bar_mfe = (entry_price - low[b]) / sl_dist
                bar_mae = (entry_price - high[b]) / sl_dist
            if bar_mfe > best_mfe:
                best_mfe = bar_mfe
            if bar_mae < worst_mae:
                worst_mae = bar_mae

        if not trade_closed:
            last_bar = end_bar - 1 if end_bar <= n_bars else n_bars - 1
            exit_price = close[last_bar]
            if direction == LONG:
                pnl_r[t] = (exit_price - entry_price) / sl_dist
            else:
                pnl_r[t] = (entry_price - exit_price) / sl_dist
            hold_bars[t] = last_bar - entry_bar[t]
            exit_type[t] = EXIT_TIME
            exit_bar[t] = last_bar

        mfe_r[t] = best_mfe
        mae_r[t] = worst_mae

    return pnl_r, hold_bars, exit_type, mfe_r, mae_r, entry_bar, exit_bar


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_trades(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    signal_bars: np.ndarray,
    directions: np.ndarray,
    sl_distances: np.ndarray,
    tp_distances: np.ndarray,
    max_hold: int,
    exit_mode: str = "rr",
    # RR mode options
    be_trigger_pct: float = 0.0,
    retrace_pct: float = 0.0,
    retrace_timeout: int = 0,
    # Trailing mode options
    trail_activation_r: float = 0.0,
    trail_distance_r: float = 0.0,
    # Custom mode options
    exit_signals: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate trades and return a structured array of results.

    Parameters
    ----------
    high, low, close : 1-D float64 arrays of bar prices.
    signal_bars : 1-D int array of bar indices where signals fire.
    directions : 1-D int8 array (1=LONG, -1=SHORT).
    sl_distances : 1-D float64 array of stop-loss distances from ref_price.
    tp_distances : 1-D float64 array of take-profit distances from entry.
    max_hold : Maximum bars to hold a trade before timeout.
    exit_mode : "rr" (fixed RR), "trailing", or "custom".
    be_trigger_pct : Fraction of TP distance that triggers breakeven (0=disabled).
    retrace_pct : Fraction of TP distance for limit entry (0=market order).
    retrace_timeout : Max bars to wait for limit fill (0=unlimited).
    trail_activation_r : R-multiple to activate trailing stop.
    trail_distance_r : R-multiple distance for trailing stop.
    exit_signals : Boolean array (same length as close) for custom exits.

    Returns
    -------
    Structured numpy array with TRADE_RESULT_DTYPE fields.
    """
    # --- Input validation and dtype coercion ---
    high = np.ascontiguousarray(high, dtype=np.float64)
    low = np.ascontiguousarray(low, dtype=np.float64)
    close = np.ascontiguousarray(close, dtype=np.float64)
    signal_bars = np.ascontiguousarray(signal_bars, dtype=np.int32)
    directions = np.ascontiguousarray(directions, dtype=np.int8)
    sl_distances = np.ascontiguousarray(sl_distances, dtype=np.float64)
    tp_distances = np.ascontiguousarray(tp_distances, dtype=np.float64)

    n_trades = len(signal_bars)
    if not (len(directions) == len(sl_distances) == len(tp_distances) == n_trades):
        raise ValueError("signal_bars, directions, sl_distances, tp_distances must have equal length")

    n_bars = len(close)
    if not (len(high) == len(low) == n_bars):
        raise ValueError("high, low, close must have equal length")

    if n_trades == 0:
        return np.empty(0, dtype=TRADE_RESULT_DTYPE)

    # --- Dispatch to appropriate kernel ---
    if exit_mode == "rr":
        result_tuple = _sim_rr_inner(
            high, low, close,
            signal_bars, directions, sl_distances, tp_distances,
            max_hold,
            be_trigger_pct,
            retrace_pct, retrace_timeout,
        )
    elif exit_mode == "trailing":
        result_tuple = _sim_trailing_inner(
            high, low, close,
            signal_bars, directions, sl_distances, tp_distances,
            max_hold,
            trail_activation_r, trail_distance_r,
        )
    elif exit_mode == "custom":
        if exit_signals is None:
            raise ValueError("exit_signals required for exit_mode='custom'")
        exit_signals = np.ascontiguousarray(exit_signals, dtype=np.bool_)
        if len(exit_signals) != n_bars:
            raise ValueError("exit_signals must have same length as price arrays")
        result_tuple = _sim_custom_inner(
            high, low, close,
            signal_bars, directions, sl_distances, tp_distances,
            max_hold,
            exit_signals,
        )
    else:
        raise ValueError(f"Unknown exit_mode: {exit_mode!r}. Use 'rr', 'trailing', or 'custom'.")

    # --- Pack into structured array ---
    pnl_r, hold_bars, exit_type_arr, mfe_r, mae_r, entry_bar, exit_bar_arr = result_tuple
    out = np.empty(n_trades, dtype=TRADE_RESULT_DTYPE)
    out["pnl_r"] = pnl_r
    out["hold_bars"] = hold_bars
    out["exit_type"] = exit_type_arr
    out["mfe_r"] = mfe_r
    out["mae_r"] = mae_r
    out["entry_bar"] = entry_bar
    out["exit_bar"] = exit_bar_arr
    return out

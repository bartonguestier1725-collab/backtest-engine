"""Numba-accelerated bar-by-bar trade simulation."""

from __future__ import annotations

import threading
import warnings

import numpy as np
import numba

from backtest_engine._types import (
    LONG, SHORT,
    EXIT_SL, EXIT_TP, EXIT_TIME, EXIT_BE, EXIT_NO_FILL, EXIT_TRAIL, EXIT_CUSTOM,
    TRADE_RESULT_DTYPE,
)
from backtest_engine.preflight import run_preflight, BacktestQualityWarning
from backtest_engine._results import TradeResults


# Internal hook for backtest_engine.audit's placebo, per thread. When
# ``_DIRECTION_FLIP.rng`` is a numpy Generator, simulate_trades (and so
# simulate_trades_hires) multiplies the directions by random +1/-1 drawn
# from it; SL/TP distances are left as they are. Other threads are never
# affected. Not part of the public API.
_DIRECTION_FLIP = threading.local()


class DataQualityWarning(UserWarning):
    """Issued when the price data looks broken: bar opens outside the bar's
    high/low (simulate_trades), or execution data that does not cover a
    signal (simulate_trades_hires: truncated or gapped fine-resolution data)."""


def _epoch_seconds(ts, name: str) -> np.ndarray:
    """int64 Unix seconds from an int array or a datetime64 array."""
    arr = np.asarray(ts)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[s]").astype(np.int64)
    if not np.issubdtype(arr.dtype, np.integer):
        if np.issubdtype(arr.dtype, np.floating) and np.all(np.isfinite(arr)) and np.all(arr == np.round(arr)):
            return arr.astype(np.int64)
        raise ValueError(f"{name} must be Unix seconds (int) or datetime64, got {arr.dtype}")
    return arr.astype(np.int64)


def _fmt_ts(ts) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(int(ts), tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")

# ---------------------------------------------------------------------------
# Inner @njit kernels (one per exit mode)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _stop_fill(direction, stop_price, bar_open, has_open):
    """Fill price of a stop order on a bar whose range reached it.

    If the bar opened beyond the stop (a gap), the order fills at the open,
    not at the stop price. Without open prices the gap is unknowable and the
    stop price is used.
    """
    if has_open:
        if direction == LONG and bar_open < stop_price:
            return bar_open
        if direction == SHORT and bar_open > stop_price:
            return bar_open
    return stop_price


@numba.njit(cache=True)
def _opens_beyond(direction, target, bar_open, has_open):
    """True if the bar opened at or beyond a profit target (a gap through it).

    The open is the first price of the bar, so the target filled before any
    stop the rest of the bar may touch.
    """
    if not has_open:
        return False
    if direction == LONG:
        return bar_open >= target
    return bar_open <= target


@numba.njit(cache=True)
def _sim_rr_inner(
    high, low, close,
    signal_bars, directions, sl_distances, tp_distances,
    max_hold,
    be_trigger_pct,
    retrace_pct, retrace_timeout,
    open_prices, use_open_entry,
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
            if use_open_entry:
                e_bar = sig_bar + 1
                if e_bar >= n_bars:
                    pnl_r[t] = 0.0
                    hold_bars[t] = 0
                    exit_type[t] = EXIT_NO_FILL
                    mfe_r[t] = 0.0
                    mae_r[t] = 0.0
                    entry_bar[t] = sig_bar
                    exit_bar[t] = sig_bar
                    continue
                entry_price = open_prices[e_bar]
                entry_bar[t] = e_bar
            else:
                # Market order: enter at current bar close (backward compat)
                entry_price = ref_price
                entry_bar[t] = sig_bar

        # --- SL / TP prices ---
        # Both are measured from the actual fill price, so -1R is always a
        # full stop-out (for a retrace fill too: the limit price, not the
        # signal-bar close).
        if direction == LONG:
            sl_price_orig = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price_orig = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        # --- Retrace fill bar ---
        # The limit fills somewhere inside bar entry_bar[t]. Whether the high
        # (TP) came before or after the fill is unknowable, so TP is not
        # checked on this bar. A stop beyond the limit can only be reached
        # after passing the limit, so an SL touch on the fill bar is a loss.
        if use_retrace:
            fb = entry_bar[t]
            if (direction == LONG and low[fb] <= sl_price_orig) or \
               (direction == SHORT and high[fb] >= sl_price_orig):
                pnl_r[t] = -1.0
                hold_bars[t] = 0
                exit_type[t] = EXIT_SL
                exit_bar[t] = fb
                mfe_r[t] = 0.0
                if direction == LONG:
                    mae_r[t] = (low[fb] - entry_price) / sl_dist
                else:
                    mae_r[t] = (entry_price - high[fb]) / sl_dist
                continue

        # --- Bar loop ---
        be_active = False
        current_sl = sl_price_orig
        best_mfe = 0.0
        worst_mae = 0.0
        trade_closed = False
        # When we enter on the next bar open, the remainder of that entry bar
        # is already "live" and can hit SL/TP. Close-entry starts on the bar
        # after the signal; retrace-fill starts after the fill bar (checked
        # above for SL only).
        open_entry = use_open_entry and not use_retrace
        start_bar = entry_bar[t] if open_entry else entry_bar[t] + 1
        # max_hold bars are held: bars entry..entry+max_hold-1 after an open
        # entry, bars entry+1..entry+max_hold after a close (or limit) fill.
        end_bar = min(entry_bar[t] + max_hold + (0 if open_entry else 1), n_bars)

        for b in range(start_bar, end_bar):
            # SL check
            if direction == LONG:
                if low[b] <= current_sl and not _opens_beyond(direction, tp_price, open_prices[b], use_open_entry):
                    exit_price = _stop_fill(direction, current_sl, open_prices[b], use_open_entry)
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
                if high[b] >= current_sl and not _opens_beyond(direction, tp_price, open_prices[b], use_open_entry):
                    exit_price = _stop_fill(direction, current_sl, open_prices[b], use_open_entry)
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
    open_prices, use_open_entry,
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

        # --- Entry logic ---
        if use_open_entry:
            e_bar = sig_bar + 1
            if e_bar >= n_bars:
                pnl_r[t] = 0.0
                hold_bars[t] = 0
                exit_type[t] = EXIT_NO_FILL
                mfe_r[t] = 0.0
                mae_r[t] = 0.0
                entry_bar[t] = sig_bar
                exit_bar[t] = sig_bar
                continue
            entry_price = open_prices[e_bar]
            entry_bar[t] = e_bar
        else:
            entry_price = ref_price
            entry_bar[t] = sig_bar

        if direction == LONG:
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
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

        start_bar = entry_bar[t] if use_open_entry else entry_bar[t] + 1
        end_bar = min(entry_bar[t] + max_hold + (0 if use_open_entry else 1), n_bars)

        for b in range(start_bar, end_bar):
            # Determine current SL using OLD best_price (before this bar's update)
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
                if low[b] <= current_sl and not _opens_beyond(direction, tp_price, open_prices[b], use_open_entry):
                    exit_price = _stop_fill(direction, current_sl, open_prices[b], use_open_entry)
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
                if high[b] >= current_sl and not _opens_beyond(direction, tp_price, open_prices[b], use_open_entry):
                    exit_price = _stop_fill(direction, current_sl, open_prices[b], use_open_entry)
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

            # Update best price for trailing (AFTER SL/TP checks)
            if direction == LONG:
                if high[b] > best_price:
                    best_price = high[b]
            else:
                if low[b] < best_price:
                    best_price = low[b]

            # Activation check (AFTER price update)
            if not trail_active:
                if direction == LONG:
                    if best_price >= entry_price + activation_abs:
                        trail_active = True
                else:
                    if best_price <= entry_price - activation_abs:
                        trail_active = True

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
    open_prices, use_open_entry,
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

        # --- Entry logic ---
        if use_open_entry:
            e_bar = sig_bar + 1
            if e_bar >= n_bars:
                pnl_r[t] = 0.0
                hold_bars[t] = 0
                exit_type[t] = EXIT_NO_FILL
                mfe_r[t] = 0.0
                mae_r[t] = 0.0
                entry_bar[t] = sig_bar
                exit_bar[t] = sig_bar
                continue
            entry_price = open_prices[e_bar]
            entry_bar[t] = e_bar
        else:
            entry_price = ref_price
            entry_bar[t] = sig_bar

        if direction == LONG:
            sl_price = entry_price - sl_dist
        else:
            sl_price = entry_price + sl_dist

        best_mfe = 0.0
        worst_mae = 0.0
        trade_closed = False
        start_bar = entry_bar[t] if use_open_entry else entry_bar[t] + 1
        end_bar = min(entry_bar[t] + max_hold + (0 if use_open_entry else 1), n_bars)

        for b in range(start_bar, end_bar):
            # SL check (always active, priority)
            if direction == LONG:
                if low[b] <= sl_price:
                    pnl_r[t] = (_stop_fill(direction, sl_price, open_prices[b], use_open_entry) - entry_price) / sl_dist
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
                    pnl_r[t] = (entry_price - _stop_fill(direction, sl_price, open_prices[b], use_open_entry)) / sl_dist
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

            # Custom exit signal check. The signal is known at bar b's close;
            # with open prices the exit fills at the next bar's open.
            if exit_signals[b]:
                xb = b
                exit_price = close[b]
                if use_open_entry and b + 1 < n_bars:
                    xb = b + 1
                    exit_price = open_prices[xb]
                if direction == LONG:
                    pnl_r[t] = (exit_price - entry_price) / sl_dist
                else:
                    pnl_r[t] = (entry_price - exit_price) / sl_dist
                hold_bars[t] = xb - entry_bar[t]
                exit_type[t] = EXIT_CUSTOM
                exit_bar[t] = xb
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


@numba.njit(cache=True)
def _sim_sar_trailing_inner(
    high, low, close, open_prices,
    signal_bars, directions, sl_distances, tp_distances,
    max_hold,
    sar_values,
    use_open_entry,
):
    """SAR-based trailing stop with direction-aware logic (B-plan fix).

    Compares SAR to close[b-1] each bar to determine its role:
      - SAR on the "correct" side of price -> trailing SL (tightens toward price).
      - SAR on the "wrong" side of price  -> TP target (price must reach SAR).
    Pessimistic: hard SL checked before SAR TP on every bar.

    Parameters
    ----------
    open_prices : bar-open array.  Used for entry when *use_open_entry* is True.
    use_open_entry : If True, enter at open_prices[sig_bar+1].
                     If False, enter at close[sig_bar] (backward compat).
    """
    n_trades = len(signal_bars)
    pnl_r     = np.empty(n_trades, dtype=np.float64)
    hold_bars = np.empty(n_trades, dtype=np.int32)
    exit_type = np.empty(n_trades, dtype=np.int8)
    mfe_r     = np.empty(n_trades, dtype=np.float64)
    mae_r     = np.empty(n_trades, dtype=np.float64)
    entry_bar = np.empty(n_trades, dtype=np.int32)
    exit_bar  = np.empty(n_trades, dtype=np.int32)

    n_bars = len(close)

    for t in range(n_trades):
        sig_bar   = signal_bars[t]
        direction = directions[t]
        sl_dist   = sl_distances[t]
        tp_dist   = tp_distances[t]

        # ── Entry ────────────────────────────────────────────
        if use_open_entry:
            e_bar = sig_bar + 1
            if e_bar >= n_bars:
                pnl_r[t]     = 0.0
                hold_bars[t] = 0
                exit_type[t] = EXIT_NO_FILL
                mfe_r[t]     = 0.0
                mae_r[t]     = 0.0
                entry_bar[t] = sig_bar
                exit_bar[t]  = sig_bar
                continue
            entry_price  = open_prices[e_bar]
            entry_bar[t] = e_bar
        else:
            entry_price  = close[sig_bar]
            entry_bar[t] = sig_bar

        # ── Hard SL / TP prices ──────────────────────────────
        if direction == LONG:
            hard_sl  = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            hard_sl  = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        best_mfe     = 0.0
        worst_mae    = 0.0
        trade_closed = False
        start_bar    = entry_bar[t] if use_open_entry else entry_bar[t] + 1
        end_bar      = min(entry_bar[t] + max_hold + (0 if use_open_entry else 1), n_bars)

        for b in range(start_bar, end_bar):
            sar_val = sar_values[b]
            ref     = close[b - 1]       # previous bar close

            if direction == LONG:
                # ── LONG ─────────────────────────────────────
                if sar_val <= ref:
                    # SAR below price -> normal trailing SL
                    current_sl    = max(hard_sl, sar_val)
                    sar_triggered = (sar_val > hard_sl)
                    if low[b] <= current_sl and not _opens_beyond(direction, tp_price, open_prices[b], use_open_entry):
                        pnl_r[t]     = (_stop_fill(direction, current_sl, open_prices[b], use_open_entry) - entry_price) / sl_dist
                        hold_bars[t] = b - entry_bar[t]
                        exit_type[t] = EXIT_TRAIL if sar_triggered else EXIT_SL
                        exit_bar[t]  = b
                        trade_closed = True
                        bar_mfe = (high[b] - entry_price) / sl_dist
                        bar_mae = (low[b] - entry_price) / sl_dist
                        if bar_mfe > best_mfe:  best_mfe  = bar_mfe
                        if bar_mae < worst_mae: worst_mae = bar_mae
                        break
                else:
                    # SAR above price -> TP target (counter-trend / SAR flip)
                    # Pessimistic: hard SL first
                    if low[b] <= hard_sl and not _opens_beyond(direction, tp_price, open_prices[b], use_open_entry):
                        pnl_r[t]     = (_stop_fill(direction, hard_sl, open_prices[b], use_open_entry) - entry_price) / sl_dist
                        hold_bars[t] = b - entry_bar[t]
                        exit_type[t] = EXIT_SL
                        exit_bar[t]  = b
                        trade_closed = True
                        bar_mfe = (high[b] - entry_price) / sl_dist
                        bar_mae = (low[b] - entry_price) / sl_dist
                        if bar_mfe > best_mfe:  best_mfe  = bar_mfe
                        if bar_mae < worst_mae: worst_mae = bar_mae
                        break
                    # Price must RISE to SAR
                    if high[b] >= sar_val:
                        pnl_r[t]     = (sar_val - entry_price) / sl_dist
                        hold_bars[t] = b - entry_bar[t]
                        exit_type[t] = EXIT_TRAIL
                        exit_bar[t]  = b
                        trade_closed = True
                        bar_mfe = (high[b] - entry_price) / sl_dist
                        bar_mae = (low[b] - entry_price) / sl_dist
                        if bar_mfe > best_mfe:  best_mfe  = bar_mfe
                        if bar_mae < worst_mae: worst_mae = bar_mae
                        break

                # Explicit TP (if SAR didn't trigger)
                if not trade_closed and high[b] >= tp_price:
                    pnl_r[t]     = (tp_price - entry_price) / sl_dist
                    hold_bars[t] = b - entry_bar[t]
                    exit_type[t] = EXIT_TP
                    exit_bar[t]  = b
                    trade_closed = True
                    bar_mfe = (high[b] - entry_price) / sl_dist
                    bar_mae = (low[b] - entry_price) / sl_dist
                    if bar_mfe > best_mfe:  best_mfe  = bar_mfe
                    if bar_mae < worst_mae: worst_mae = bar_mae
                    break

            else:
                # ── SHORT ────────────────────────────────────
                if sar_val >= ref:
                    # SAR above price -> normal trailing SL
                    current_sl    = min(hard_sl, sar_val)
                    sar_triggered = (sar_val < hard_sl)
                    if high[b] >= current_sl and not _opens_beyond(direction, tp_price, open_prices[b], use_open_entry):
                        pnl_r[t]     = (entry_price - _stop_fill(direction, current_sl, open_prices[b], use_open_entry)) / sl_dist
                        hold_bars[t] = b - entry_bar[t]
                        exit_type[t] = EXIT_TRAIL if sar_triggered else EXIT_SL
                        exit_bar[t]  = b
                        trade_closed = True
                        bar_mfe = (entry_price - low[b]) / sl_dist
                        bar_mae = (entry_price - high[b]) / sl_dist
                        if bar_mfe > best_mfe:  best_mfe  = bar_mfe
                        if bar_mae < worst_mae: worst_mae = bar_mae
                        break
                else:
                    # SAR below price -> TP target
                    # Pessimistic: hard SL first
                    if high[b] >= hard_sl and not _opens_beyond(direction, tp_price, open_prices[b], use_open_entry):
                        pnl_r[t]     = (entry_price - _stop_fill(direction, hard_sl, open_prices[b], use_open_entry)) / sl_dist
                        hold_bars[t] = b - entry_bar[t]
                        exit_type[t] = EXIT_SL
                        exit_bar[t]  = b
                        trade_closed = True
                        bar_mfe = (entry_price - low[b]) / sl_dist
                        bar_mae = (entry_price - high[b]) / sl_dist
                        if bar_mfe > best_mfe:  best_mfe  = bar_mfe
                        if bar_mae < worst_mae: worst_mae = bar_mae
                        break
                    # Price must FALL to SAR
                    if low[b] <= sar_val:
                        pnl_r[t]     = (entry_price - sar_val) / sl_dist
                        hold_bars[t] = b - entry_bar[t]
                        exit_type[t] = EXIT_TRAIL
                        exit_bar[t]  = b
                        trade_closed = True
                        bar_mfe = (entry_price - low[b]) / sl_dist
                        bar_mae = (entry_price - high[b]) / sl_dist
                        if bar_mfe > best_mfe:  best_mfe  = bar_mfe
                        if bar_mae < worst_mae: worst_mae = bar_mae
                        break

                # Explicit TP (if SAR didn't trigger)
                if not trade_closed and low[b] <= tp_price:
                    pnl_r[t]     = (entry_price - tp_price) / sl_dist
                    hold_bars[t] = b - entry_bar[t]
                    exit_type[t] = EXIT_TP
                    exit_bar[t]  = b
                    trade_closed = True
                    bar_mfe = (entry_price - low[b]) / sl_dist
                    bar_mae = (entry_price - high[b]) / sl_dist
                    if bar_mfe > best_mfe:  best_mfe  = bar_mfe
                    if bar_mae < worst_mae: worst_mae = bar_mae
                    break

            # ── MFE / MAE (no exit this bar) ─────────────────
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

        # ── Timeout ──────────────────────────────────────────
        if not trade_closed:
            last_bar = end_bar - 1 if end_bar <= n_bars else n_bars - 1
            exit_price = close[last_bar]
            if direction == LONG:
                pnl_r[t] = (exit_price - entry_price) / sl_dist
            else:
                pnl_r[t] = (entry_price - exit_price) / sl_dist
            hold_bars[t] = last_bar - entry_bar[t]
            exit_type[t] = EXIT_TIME
            exit_bar[t]  = last_bar

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
    # SAR trailing mode options
    sar_values: np.ndarray | None = None,
    open_prices: np.ndarray | None = None,
    entry_costs: np.ndarray | None = None,
    preflight: bool = True,
) -> TradeResults:
    """Simulate trades and return a structured array of results.

    Parameters
    ----------
    high, low, close : 1-D float64 arrays of bar prices.
    signal_bars : 1-D int array of bar indices where signals fire.
    directions : 1-D int8 array (1=LONG, -1=SHORT).
    sl_distances : 1-D float64 array of stop-loss distances from the entry
                   (fill) price. 1R = this distance.
    tp_distances : 1-D float64 array of take-profit distances from entry.
    max_hold : Bars a trade is held before a timeout exit at the close of the
               last one (>= 1). With next-bar-open entry the entry bar counts
               as the first; with close or limit entry the first is the next bar.
    exit_mode : "rr" (fixed RR), "trailing", "custom", or "sar_trailing".
    be_trigger_pct : Fraction of TP distance that triggers breakeven (0=disabled).
                     Each mode's options (``be_trigger_pct``, ``retrace_pct``,
                     ``retrace_timeout`` for "rr"; ``trail_*`` for "trailing";
                     ``exit_signals`` for "custom"; ``sar_values`` for
                     "sar_trailing") raise ValueError with another mode unless
                     left at 0 / None.
    retrace_pct : Fraction of TP distance for limit entry (0=market order).
                  The limit is placed this far from close[sig_bar]; SL/TP are
                  measured from the limit (fill) price. On the fill bar only
                  the SL is checked (TP order vs. fill is unknowable).
    retrace_timeout : Max bars to wait for limit fill (0=unlimited).
    trail_activation_r : R-multiple to activate trailing stop.
    trail_distance_r : R-multiple distance for trailing stop.
    exit_signals : Boolean array (same length as close) for custom exits.
    sar_values : Pre-computed Parabolic SAR array for sar_trailing mode.
    open_prices : Optional bar-open prices for next-bar-open entry.
                  Supported for all exit modes ("rr", "trailing", "custom", "sar_trailing").
                  If None, falls back to close[sig_bar] entry (backward compat).
                  When provided, a stop (SL, breakeven, trailing) on a bar
                  that opens beyond it fills at that bar's open (gap), not at
                  the stop price. TP and SAR targets still fill at their
                  price (conservative).
    entry_costs : Optional 1-D float64 array of per-trade costs in R-units.
                  Applied post-kernel: subtracted from pnl_r. NO_FILL trades get 0.
                  Use BrokerCost.per_trade_cost() to compute from spread arrays.
                  If None, cost_r=0.0 (backward compatible).
    preflight : If True (default), emit BacktestQualityWarning when the
                pre-flight grade is below A. Set False to suppress. The
                grade is attached to the result (``results.quality``) either way.

    Returns
    -------
    TradeResults (numpy structured array subclass with TRADE_RESULT_DTYPE).
    Access ``results.quality.grade`` for quality grade ("A"/"B"/"C").
    """
    # --- Pre-flight quality check ---
    # Always attach the quality report (GateKeeper reads it); only warn
    # when preflight is on.
    quality_report = run_preflight(open_prices, entry_costs)
    if preflight:
        if quality_report.grade != "A":
            warnings.warn(
                quality_report.format_message(),
                BacktestQualityWarning,
                stacklevel=2,
            )

    # --- Input validation and dtype coercion ---
    high = np.ascontiguousarray(high, dtype=np.float64)
    low = np.ascontiguousarray(low, dtype=np.float64)
    close = np.ascontiguousarray(close, dtype=np.float64)
    signal_bars = np.ascontiguousarray(signal_bars, dtype=np.int32)
    directions = np.ascontiguousarray(directions, dtype=np.int8)
    _flip_rng = getattr(_DIRECTION_FLIP, "rng", None)
    if _flip_rng is not None and len(directions):
        flips = _flip_rng.choice(np.array([-1, 1], dtype=np.int8), len(directions))
        directions = np.ascontiguousarray(directions * flips, dtype=np.int8)
    sl_distances = np.ascontiguousarray(sl_distances, dtype=np.float64)
    tp_distances = np.ascontiguousarray(tp_distances, dtype=np.float64)

    n_trades = len(signal_bars)
    if not (len(directions) == len(sl_distances) == len(tp_distances) == n_trades):
        raise ValueError("signal_bars, directions, sl_distances, tp_distances must have equal length")

    n_bars = len(close)
    if not (len(high) == len(low) == n_bars):
        raise ValueError("high, low, close must have equal length")

    # --- Value validation (prevents garbage output / kernel crashes) ---
    if np.any(np.isnan(close)):
        raise ValueError("close contains NaN — clean data before simulation")
    if n_trades > 0:
        if np.any(signal_bars < 0) or np.any(signal_bars >= n_bars):
            bad = signal_bars[(signal_bars < 0) | (signal_bars >= n_bars)]
            raise ValueError(
                f"signal_bars contains out-of-range indices {bad[:5].tolist()} "
                f"(valid range: 0..{n_bars - 1})"
            )
        if np.any(sl_distances <= 0):
            raise ValueError("sl_distances must be > 0 (used as divisor for R-multiples)")
    if max_hold < 1:
        raise ValueError("max_hold must be >= 1 (bars a trade may be held)")
    if exit_mode not in ("rr", "trailing", "custom", "sar_trailing"):
        raise ValueError(f"Unknown exit_mode: {exit_mode!r}. Use 'rr', 'trailing', 'sar_trailing', or 'custom'.")
    # Each kernel reads only its own mode's options; another mode would
    # ignore them silently (e.g. trailing params with the default "rr").
    for name, value, mode in (
        ("be_trigger_pct", be_trigger_pct, "rr"),
        ("retrace_pct", retrace_pct, "rr"),
        ("retrace_timeout", retrace_timeout, "rr"),
        ("trail_activation_r", trail_activation_r, "trailing"),
        ("trail_distance_r", trail_distance_r, "trailing"),
        ("exit_signals", exit_signals is not None, "custom"),
        ("sar_values", sar_values is not None, "sar_trailing"),
    ):
        if value and exit_mode != mode:
            raise ValueError(
                f"{name} only works with exit_mode={mode!r}; exit_mode={exit_mode!r} "
                f"would ignore it. Leave it at its default (0 or None) or use "
                f"exit_mode={mode!r}."
            )

    if open_prices is not None:
        o_chk = np.asarray(open_prices, dtype=np.float64)
        if len(o_chk) == n_bars:
            tol = 1e-9 * np.abs(close)
            n_bad = int(np.sum((o_chk > high + tol) | (o_chk < low - tol)))
            if n_bad:
                warnings.warn(
                    f"{n_bad}/{n_bars} bars open outside their own high/low. "
                    f"Fills at the open (next-bar entry, stop gaps) use these "
                    f"prices, so check the data (e.g. open must be inside "
                    f"[low, high]).",
                    DataQualityWarning,
                    stacklevel=2,
                )

    if n_trades == 0:
        return TradeResults(np.empty(0, dtype=TRADE_RESULT_DTYPE), quality=quality_report)

    # --- Dispatch to appropriate kernel ---
    if exit_mode == "rr":
        if open_prices is not None:
            open_arr = np.ascontiguousarray(open_prices, dtype=np.float64)
            if len(open_arr) != n_bars:
                raise ValueError("open_prices must have same length as price arrays")
            use_open_entry = True
        else:
            open_arr = close          # dummy — unused when use_open_entry=False
            use_open_entry = False
        result_tuple = _sim_rr_inner(
            high, low, close,
            signal_bars, directions, sl_distances, tp_distances,
            max_hold,
            be_trigger_pct,
            retrace_pct, retrace_timeout,
            open_arr, use_open_entry,
        )
    elif exit_mode == "trailing":
        if open_prices is not None:
            open_arr = np.ascontiguousarray(open_prices, dtype=np.float64)
            if len(open_arr) != n_bars:
                raise ValueError("open_prices must have same length as price arrays")
            use_open_entry = True
        else:
            open_arr = close          # dummy — unused when use_open_entry=False
            use_open_entry = False
        result_tuple = _sim_trailing_inner(
            high, low, close,
            signal_bars, directions, sl_distances, tp_distances,
            max_hold,
            trail_activation_r, trail_distance_r,
            open_arr, use_open_entry,
        )
    elif exit_mode == "sar_trailing":
        if sar_values is None:
            raise ValueError("sar_values required for exit_mode='sar_trailing'")
        sar_values = np.ascontiguousarray(sar_values, dtype=np.float64)
        if len(sar_values) != n_bars:
            raise ValueError("sar_values must have same length as price arrays")
        if open_prices is not None:
            open_arr = np.ascontiguousarray(open_prices, dtype=np.float64)
            if len(open_arr) != n_bars:
                raise ValueError("open_prices must have same length as price arrays")
            use_open_entry = True
        else:
            open_arr = close          # dummy — unused when use_open_entry=False
            use_open_entry = False
        result_tuple = _sim_sar_trailing_inner(
            high, low, close, open_arr,
            signal_bars, directions, sl_distances, tp_distances,
            max_hold,
            sar_values,
            use_open_entry,
        )
    elif exit_mode == "custom":
        if exit_signals is None:
            raise ValueError("exit_signals required for exit_mode='custom'")
        exit_signals = np.ascontiguousarray(exit_signals, dtype=np.bool_)
        if len(exit_signals) != n_bars:
            raise ValueError("exit_signals must have same length as price arrays")
        if open_prices is not None:
            open_arr = np.ascontiguousarray(open_prices, dtype=np.float64)
            if len(open_arr) != n_bars:
                raise ValueError("open_prices must have same length as price arrays")
            use_open_entry = True
        else:
            open_arr = close          # dummy — unused when use_open_entry=False
            use_open_entry = False
        result_tuple = _sim_custom_inner(
            high, low, close,
            signal_bars, directions, sl_distances, tp_distances,
            max_hold,
            exit_signals,
            open_arr, use_open_entry,
        )

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
    out["signal_bar"] = signal_bars

    # --- Apply per-trade costs (post-kernel) ---
    if entry_costs is not None:
        costs = np.ascontiguousarray(entry_costs, dtype=np.float64)
        if len(costs) != n_trades:
            raise ValueError("entry_costs must have same length as signal_bars")
        # Zero out cost for NO_FILL trades
        no_fill_mask = exit_type_arr == EXIT_NO_FILL
        costs_applied = np.where(no_fill_mask, 0.0, costs)
        out["pnl_r"] -= costs_applied
        out["cost_r"] = costs_applied
    else:
        out["cost_r"] = 0.0

    return TradeResults(out, quality=quality_report)


def simulate_trades_hires(
    # Signal-timeframe data (e.g. 1h — used only for index mapping)
    signal_timestamps: np.ndarray,
    signal_bars: np.ndarray,
    directions: np.ndarray,
    sl_distances: np.ndarray,
    tp_distances: np.ndarray,
    max_hold: int,
    signal_bar_minutes: int,
    # Execution-timeframe data (e.g. 1m — used for actual simulation)
    exec_timestamps: np.ndarray,
    exec_opens: np.ndarray,
    exec_highs: np.ndarray,
    exec_lows: np.ndarray,
    exec_closes: np.ndarray,
    exec_bar_minutes: int = 1,
    # Pass-through to simulate_trades
    entry_costs: np.ndarray | None = None,
    **kwargs,
) -> TradeResults:
    """Simulate trades at higher resolution than signal generation.

    Signals are generated on lower-resolution bars (e.g. 1h) but trade
    execution (SL/TP hit detection) runs on higher-resolution bars (e.g. 1m).
    This eliminates the intra-bar SL/TP ordering problem (BG-05).

    Timing: timestamps are bar OPEN times (as from fetch_aggvault). A signal
    on signal bar k is known when that bar closes, so the trade enters at the
    open of the first execution bar starting at or after
    ``signal_timestamps[k] + signal_bar_minutes * 60``. The returned
    ``signal_bar`` field is the execution bar just before the entry bar
    (normally the last one inside signal bar k), so ``entry_bar > signal_bar``
    for every filled trade.

    Signals that close outside the execution data are returned as NO_FILL
    (pnl_r=0) with a DataQualityWarning. Entries delayed by a gap in the
    execution data still trade, later, and also emit a DataQualityWarning.
    Timestamps must be Unix seconds (datetime64 arrays are converted); their
    usual spacing must match ``signal_bar_minutes`` / ``exec_bar_minutes``.

    Parameters
    ----------
    signal_timestamps : int64 array of Unix timestamps (bar open) for the signal timeframe.
    signal_bars : int32 array of bar indices in the signal timeframe.
    directions, sl_distances, tp_distances : Same as simulate_trades.
    max_hold : Max hold in signal-timeframe bars (auto-converted to exec bars).
               ``retrace_timeout`` in kwargs is converted the same way.
    signal_bar_minutes : Bar size of signal timeframe (e.g. 60 for 1h).
    exec_timestamps : int64 array of Unix timestamps for execution timeframe.
    exec_opens, exec_highs, exec_lows, exec_closes : Execution-timeframe OHLC.
    exec_bar_minutes : Bar size of execution timeframe (default 1 for 1m).
    entry_costs : Per-trade costs (same as simulate_trades).
    **kwargs : Passed to simulate_trades (exit_mode, be_trigger_pct, etc.).

    Returns
    -------
    TradeResults with bar indices in execution timeframe.

    Example
    -------
    ::

        # Signal generation on 1h
        ts_1h, opens_1h, highs_1h, lows_1h, closes_1h, _ = fetch_aggvault(
            "EURUSD", "1h", "2021-04-01", "2026-03-31",
        )
        rsi_vals = rsi(closes_1h, 14)
        # ... generate signal_bars, directions, sl_distances, tp_distances

        # Execution simulation on 1m
        ts_1m, opens_1m, highs_1m, lows_1m, closes_1m, _ = fetch_aggvault(
            "EURUSD", "1m", "2021-04-01", "2026-03-31",
        )

        results = simulate_trades_hires(
            signal_timestamps=ts_1h,
            signal_bars=signal_bars,
            directions=directions,
            sl_distances=sl_distances,
            tp_distances=tp_distances,
            max_hold=48,
            signal_bar_minutes=60,
            exec_timestamps=ts_1m,
            exec_opens=opens_1m,
            exec_highs=highs_1m,
            exec_lows=lows_1m,
            exec_closes=closes_1m,
        )
    """
    if signal_bar_minutes <= exec_bar_minutes:
        raise ValueError(
            f"signal_bar_minutes ({signal_bar_minutes}) must be greater than "
            f"exec_bar_minutes ({exec_bar_minutes}). "
            f"Use simulate_trades() directly if resolutions are equal."
        )

    signal_timestamps = _epoch_seconds(signal_timestamps, "signal_timestamps")
    exec_timestamps = _epoch_seconds(exec_timestamps, "exec_timestamps")
    signal_bars = np.ascontiguousarray(signal_bars, dtype=np.int32)
    n_exec = len(exec_timestamps)
    if n_exec < 2:
        raise ValueError("exec_timestamps must contain at least 2 bars")
    if np.any(np.diff(exec_timestamps) <= 0):
        raise ValueError("exec_timestamps must be strictly increasing")
    # The usual spacing must match the declared bar sizes. This catches
    # milliseconds passed as seconds (which would bring the 59-minute
    # look-ahead back) and 5m data passed without exec_bar_minutes=5.
    for name, ts, minutes in (("exec_timestamps", exec_timestamps, exec_bar_minutes),
                              ("signal_timestamps", signal_timestamps, signal_bar_minutes)):
        if len(ts) >= 3:
            step = int(np.median(np.diff(ts)))
            if step != int(minutes) * 60:
                raise ValueError(
                    f"{name}: bars are usually {step} s apart, but the bar size is "
                    f"{minutes} min ({int(minutes) * 60} s). Timestamps must be Unix "
                    f"seconds of the bar open, and the *_bar_minutes arguments must "
                    f"match the data.")
    if len(signal_bars) > 0 and (
        signal_bars.min() < 0 or signal_bars.max() >= len(signal_timestamps)
    ):
        raise ValueError(
            f"signal_bars must index signal_timestamps "
            f"(valid range: 0..{len(signal_timestamps) - 1})"
        )

    # Map signal bars from signal TF -> exec TF.
    #
    # Timestamps are bar OPEN times (as returned by fetch_aggvault and
    # load_ohlcv). A signal computed from signal bar k is known only when that
    # bar closes, at signal_timestamps[k] + signal_bar_minutes * 60. The trade
    # enters at the open of the first exec bar starting at or after that time.
    # The exec bar just before it (the last exec bar inside the signal bar) is
    # passed to simulate_trades as the signal bar, so simulate_trades enters on
    # the next exec bar's open, exactly like next-bar-open entry on 1h bars.
    #
    # Before v0.6.0 the signal bar was mapped to the FIRST exec bar of the
    # signal bar, which entered 1 exec bar after the signal bar opened, i.e.
    # (signal_bar_minutes - exec_bar_minutes) minutes before the signal was
    # known (look-ahead).
    bar_seconds = int(signal_bar_minutes) * 60
    close_times = signal_timestamps[signal_bars] + bar_seconds
    entry_idx = np.searchsorted(exec_timestamps, close_times, side="left")
    covered = (entry_idx >= 1) & (entry_idx < n_exec)
    exec_signal_bars = np.clip(entry_idx - 1, 0, n_exec - 1).astype(np.int32)

    # A signal on the last bar of data that ends with the exec data has no
    # bar to enter on: NO_FILL, as in simulate_trades, without a warning.
    # Warn only when the exec data starts after, or ends before, the signal.
    exec_end = exec_timestamps[-1] + int(exec_bar_minutes) * 60
    outside = (entry_idx == 0) | (close_times > exec_end)
    n_outside = int(np.sum(outside))
    n_uncovered = int(np.sum(~covered))
    if n_outside:
        warnings.warn(
            f"{n_outside}/{len(signal_bars)} signals close outside the execution "
            f"data ({_fmt_ts(exec_timestamps[0])} .. {_fmt_ts(exec_end)}). "
            f"They are returned as NO_FILL with pnl_r=0. If the exec data was "
            f"fetched in one request, it may have been truncated.",
            DataQualityWarning,
            stacklevel=2,
        )

    # Exec data missing for a whole signal bar that exists in the signal data
    # (the entry would happen later than on the signal timeframe).
    has_next = signal_bars + 1 < len(signal_timestamps)
    next_close = np.where(
        has_next,
        signal_timestamps[np.minimum(signal_bars + 1, len(signal_timestamps) - 1)] + bar_seconds,
        np.iinfo(np.int64).max,
    )
    late = covered & (exec_timestamps[np.minimum(entry_idx, n_exec - 1)] >= next_close)
    n_late = int(np.sum(late))
    if n_late:
        warnings.warn(
            f"{n_late}/{len(signal_bars)} signals enter later than the next "
            f"signal bar because the execution data has a gap there. "
            f"Check the execution data for missing bars.",
            DataQualityWarning,
            stacklevel=2,
        )

    # Convert max_hold (and retrace_timeout) from signal bars to exec bars
    ratio = signal_bar_minutes // exec_bar_minutes
    max_hold_exec = max_hold * ratio
    if kwargs.get("retrace_timeout", 0) > 0:
        kwargs["retrace_timeout"] = int(kwargs["retrace_timeout"]) * ratio

    results = simulate_trades(
        high=exec_highs,
        low=exec_lows,
        close=exec_closes,
        signal_bars=exec_signal_bars,
        directions=directions,
        sl_distances=sl_distances,
        tp_distances=tp_distances,
        max_hold=max_hold_exec,
        open_prices=exec_opens,
        entry_costs=entry_costs,
        **kwargs,
    )

    if n_uncovered and len(results):
        out = np.asarray(results)
        miss = ~covered
        out["pnl_r"][miss] = 0.0
        out["cost_r"][miss] = 0.0
        out["hold_bars"][miss] = 0
        out["exit_type"][miss] = EXIT_NO_FILL
        out["mfe_r"][miss] = 0.0
        out["mae_r"][miss] = 0.0
        out["entry_bar"][miss] = exec_signal_bars[miss]
        out["exit_bar"][miss] = exec_signal_bars[miss]
    return results

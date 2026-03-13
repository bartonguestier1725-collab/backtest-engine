# backtest_engine

Numba JIT-accelerated backtesting engine for trading strategies. Turns Python for-loop backtests into **10–50x faster** simulations.

## Features

- **`simulate_trades()`** — SL/TP/trailing/SAR trailing/limit entry/custom exit, all running inside `@njit` kernels
- **BugGuard** — 11 automated checks (BG-01–BG-11) that catch look-ahead bias, cost underestimation, overfitting on short data, and more — before you even see the results
- **GateKeeper** — fast-kill pipeline (Gate 0→1→2→3→4) that eliminates hopeless parameter combos within minutes
- **Indicators** — SMA, ATR, Bollinger Bands, RCI, Parabolic SAR, expanding quantile, multi-timeframe mapping (all `@njit`)
- **Monte Carlo DD** — 10,000-shuffle drawdown distribution, Kelly criterion, optimal risk sizing
- **Walk-Forward / CSCV** — out-of-sample validation and Probability of Backtest Overfitting (PBO, Bailey 2015)
- **Broker cost models** — measured spread + commission presets for ECN brokers (27 FX pairs + XAUUSD)

## Install

```bash
git clone https://github.com/bartonguestier1725-collab/backtest-engine.git
cd backtest-engine
uv venv .venv && source .venv/bin/activate
uv pip install -e '.[dev]'
```

> **Note:** numba 0.63 requires NumPy < 2.4. `numpy>=2.2,<2.4` is pinned.

## Quick start

```python
import numpy as np
from backtest_engine import simulate_trades, LONG, MonteCarloDD, atr

# OHLC data (numpy arrays)
high, low, close = ...

# ATR-based SL/TP
atr_vals = atr(high, low, close, 14)

# Define signals
signal_bars = np.array([100, 250, 400], dtype=np.int32)
directions = np.array([LONG, LONG, LONG], dtype=np.int8)
sl_distances = atr_vals[signal_bars] * 1.5
tp_distances = atr_vals[signal_bars] * 3.0

# Run simulation
results = simulate_trades(
    high, low, close,
    signal_bars, directions, sl_distances, tp_distances,
    max_hold=100,
    be_trigger_pct=0.5,  # Move SL to breakeven at 50% of TP
)

# Results
print(f"Win rate: {np.mean(results['pnl_r'] > 0) * 100:.1f}%")
print(f"Avg PnL: {np.mean(results['pnl_r']):.3f}R")

# Monte Carlo DD analysis
mc = MonteCarloDD(results['pnl_r'], risk_pct=0.01)
mc.run()
print(f"95th DD: {mc.dd_percentile(95) * 100:.1f}%")
```

## simulate_trades options

| Parameter | Description |
|-----------|-------------|
| `exit_mode="rr"` | Fixed risk-reward: SL/TP/timeout/breakeven |
| `exit_mode="trailing"` | Trailing stop |
| `exit_mode="sar_trailing"` | Parabolic SAR-based trailing (with direction-aware Plan B) |
| `exit_mode="custom"` | Exit on external signal |
| `be_trigger_pct` | Move SL to entry price when profit reaches this fraction of TP (0 = disabled) |
| `retrace_pct` | Limit entry: wait for price to retrace this fraction of TP before filling (0 = market entry) |
| `retrace_timeout` | Max bars to wait for limit fill before cancelling |
| `trail_activation_r` | R-multiple required to activate trailing stop |
| `trail_distance_r` | Trailing stop distance in R-multiples |
| `open_prices` | Enter at next-bar open instead of signal-bar close |

## BugGuard

Automatically checks for 11 known backtesting bugs before you trust any result:

```python
from backtest_engine import bug_guard

report = bug_guard.run_all_checks(
    source_path="my_strategy.py",
    signal_bars=signal_bars,
    entry_bars=results["entry_bar"],
    exit_bars=results["exit_bar"],
    spreads_used={"USDJPY": 0.017, "EURUSD": 0.000075},
    resolution_minutes=5,
    n_bars=len(close),
    n_trades=len(signal_bars),
)
# BG-01: Look-ahead bias      BG-02: Cost underestimation  BG-04: bfill data leak
# BG-05: Coarse-bar SL/TP     BG-06: Short-period overfit  BG-07: Full-period quantile
# BG-08: Same-bar re-entry     BG-09: Close-price entry     ... and more
```

## GateKeeper

Stage-gate pipeline that kills hopeless strategies early, before you waste hours on full parameter sweeps:

```python
from backtest_engine import GateKeeper

gk = GateKeeper(data, param_grid, strategy_fn)
gk.gate0_validate("my_strategy.py")  # BugGuard checks
gk.gate1_quick(n_combos=20)          # 20-param quick kill
gk.gate2_screen(n_combos=100)        # 100-param screening
```

## Walk-Forward & CSCV

```python
from backtest_engine import WalkForward, CSCV

# Walk-Forward analysis
wf = WalkForward(n_bars=len(close), is_ratio=0.7, n_splits=5, anchored=False)
wf_result = wf.run(param_grid, evaluate_fn)
print(f"OOS mean: {wf_result['oos_mean']:.3f}R")

# CSCV — Probability of Backtest Overfitting (Bailey 2015)
cscv = CSCV(n_splits=10)
cscv_result = cscv.run(param_grid, evaluate_fn, n_bars=len(close))
print(f"PBO: {cscv_result['pbo']:.1%}")  # < 0.5 = probably not overfit
```

## Utilities

```python
from backtest_engine import load_ohlcv, resample_ohlcv, map_higher_tf

# Load CSV → numpy arrays
timestamps, open_, high, low, close, volume = load_ohlcv("data.csv")

# Resample 5-min to 1-hour
ts_1h, o_1h, h_1h, l_1h, c_1h, v_1h = resample_ohlcv(
    timestamps, open_, high, low, close, volume, rule="1h"
)

# Map higher-TF indicator to lower-TF bars (uses last completed bar)
atr_1h = atr(h_1h, l_1h, c_1h, 14)
atr_mapped = map_higher_tf(timestamps, ts_1h, atr_1h)
```

## Broker cost presets

```python
from backtest_engine import BrokerCost

# ECN broker (tight spread + commission)
cost = BrokerCost.tradeview_ilc()

# Prop firm (wider spread, no commission)
cost = BrokerCost.fundora()

# Cost per trade in R-units
cost_r = cost.as_r("EURUSD", sl_distance=0.00050)
pnl_after_costs = results['pnl_r'] - cost_r
```

## Tests

```bash
python -m pytest tests/ -v
```

98 tests, all passing.

## License

MIT

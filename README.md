# backtest_engine

Numba JIT-accelerated trade simulator and validation toolkit. Turns Python for-loop backtests into **10-50x faster** simulations.

**What this is:** A fast bar-based trade simulator that takes OHLC data + pre-computed signals and evaluates each trade independently. Includes automated bug detection and stage-gate validation.

**What this is not:** A portfolio backtester. There is no account balance tracking, multi-instrument position management, or order book simulation. If you need those, look at backtrader or vectorbt.

## Features

### Core (instrument-agnostic)

- **`simulate_trades()`** — SL/TP/trailing/SAR trailing/limit entry/custom exit, all `@njit` kernels. Supports next-bar-open entry via `open_prices` in all modes
- **Indicators** — SMA, ATR, Bollinger Bands, RCI, Parabolic SAR, expanding quantile, multi-timeframe mapping (all `@njit`)
- **Monte Carlo DD** — 10,000-shuffle drawdown distribution, Kelly criterion, optimal risk sizing, prop firm DD check
- **Walk-Forward / CSCV** — out-of-sample validation with IS/OOS ratio and Probability of Backtest Overfitting (PBO, Bailey 2015)
- **Utilities** — CSV loading, OHLC resampling, higher-TF indicator mapping

### Opinionated modules (FX defaults, fully customizable)

- **BrokerCost** — measured spread + commission model. Ships with FX presets (28 pairs + XAUUSD), but accepts any instrument via constructor
- **BugGuard** — 11 automated checks (BG-01–BG-11) that catch look-ahead bias, cost underestimation, overfitting, and more. Data period estimation defaults to FX market hours but accepts explicit timestamps for any market
- **GateKeeper** — fast-kill pipeline (Gate 0–2) that eliminates hopeless parameter combos early. Thresholds (PF, RF, trades) are class variables — override them for your asset class

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
| `open_prices` | Enter at next-bar open instead of signal-bar close (supported in all modes) |

## Broker cost models

BrokerCost works with any instrument. Pass your own spreads, commissions, and pip definitions.

```python
from backtest_engine import BrokerCost

# Custom broker — any instrument
cost = BrokerCost(
    spreads={"EURUSD": 0.00008, "USDJPY": 0.008, "BTCUSD": 15.0},
    commission_per_lot=4.0,
    pip_values={"EURUSD": 10.0, "USDJPY": 6.67, "BTCUSD": 1.0},
    pip_sizes={"EURUSD": 0.0001, "USDJPY": 0.01, "BTCUSD": 1.0},
)

# FX presets (ships with 28 pairs + XAUUSD, measured spreads)
cost = BrokerCost.tradeview_ilc()   # ECN (tight spread + $5 RT commission)
cost = BrokerCost.fundora()          # Prop firm (wider spread, no commission)

# Cost per trade in R-units
cost_r = cost.as_r("EURUSD", risk_price=0.00050)
pnl_after_costs = results['pnl_r'] - cost_r

# Get all costs as a dict (useful for BugGuard)
expected = cost.cost_prices()  # {"EURUSD": 0.00007, "USDJPY": 0.014, ...}
```

## BugGuard

Automatically checks for 11 known backtesting bugs before you trust any result.

The checks themselves are instrument-agnostic (look-ahead bias, bfill leak, same-bar reentry, etc.). The only FX-specific part is data period estimation — pass `start_ts`/`end_ts` to use calendar dates instead.

```python
from backtest_engine import bug_guard, BrokerCost

report = bug_guard(
    source_path="my_strategy.py",
    signal_bars=signal_bars,
    entry_bars=results["entry_bar"],
    exit_bars=results["exit_bar"],
    spreads_used={"USDJPY": 0.017, "EURUSD": 0.000075},
    expected_costs=BrokerCost.tradeview_ilc().cost_prices(),
    resolution_minutes=5,
    n_bars=len(close),
    n_trades=len(signal_bars),
)
# BG-01: Look-ahead bias      BG-02: Cost underestimation  BG-04: bfill data leak
# BG-05: Coarse-bar SL/TP     BG-06: Short-period overfit  BG-07: Full-period quantile
# BG-08: Same-bar re-entry     BG-09: Close-price entry     ... and more
```

Individual checks can also be called standalone:

```python
from backtest_engine import check_look_ahead, check_cost_registry

result = check_look_ahead(signal_bars, entry_bars)
print(f"{result.check_id}: {result.message}")
```

## GateKeeper

Stage-gate pipeline that kills hopeless strategies early. Default thresholds are tuned for FX — override them for your asset class.

```python
from backtest_engine import GateKeeper, BrokerCost

gk = GateKeeper(
    strategy_name="SMA Crossover v3",
    n_bars=7022,
    bar_minutes=60,
    resolution_minutes=1,
    spreads_used=spreads,
    source_path=__file__,
    expected_costs=BrokerCost.tradeview_ilc().cost_prices(),
)

gk.gate0_validate()                    # BugGuard checks
gk.gate1_quick(run_func, quick_params)  # ~20 combos → PF >= 1.05?
gk.gate2_screen(run_func, full_params)  # ~100 combos → PF >= 1.10, RF >= 1.5?
gk.summary()

# Override thresholds for a different asset class
gk.GATE1_MIN_PF = 1.10
gk.GATE2_MIN_RF = 2.0
```

`run_func` takes a parameter dict and returns a metric dict:

```python
def run_func(params: dict) -> dict | None:
    """Run a single backtest with the given params.
    Must return {'pf': ..., 'total_r': ..., 'n_trades': ..., 'max_dd_r': ...}
    or None if no trades.
    """
```

## Monte Carlo & prop firm check

```python
from backtest_engine import MonteCarloDD

mc = MonteCarloDD(pnl_after_costs, n_sims=10_000, risk_pct=0.01, seed=42)
mc.run()

print(f"DD 95th: {mc.dd_percentile(95)*100:.1f}%")
print(f"Kelly: {mc.kelly_fraction()*100:.1f}%")
print(f"Optimal risk: {mc.optimal_risk_pct(max_dd=0.15)*100:.2f}%")

# Prop firm drawdown check (any firm, any limits)
result = mc.prop_firm_check(daily_dd_limit=0.04, total_dd_limit=0.08, confidence=95.0)
print(f"Pass: {result['pass']}")
```

## Walk-Forward & CSCV

```python
from backtest_engine import WalkForward, CSCV

# Walk-Forward analysis
wf = WalkForward(n_bars=len(close), is_ratio=0.7, n_splits=5, anchored=False)
wf_result = wf.run(param_grid, evaluate_fn)
print(f"OOS mean: {wf_result['oos_mean']:.3f}R")
print(f"IS/OOS ratio: {wf_result['is_oos_ratio']:.2f}")

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

## FX defaults reference

The following FX-specific defaults ship with the library. They are used by BugGuard and GateKeeper when no custom values are provided.

| Module | Default | How to override |
|--------|---------|-----------------|
| `BugGuard` | Data period estimation uses FX hours (22 days/mo, 17h/day) | Pass `start_ts`/`end_ts` to `check_data_period()` |
| `BugGuard` | Fundora cost registry (28 FX pairs + XAUUSD) | Pass your own `expected_costs` dict |
| `GateKeeper` | PF >= 1.05/1.10, RF >= 1.5, trades >= 30/50, PBO <= 0.40 | Override class variables: `gk.GATE1_MIN_PF = 1.10` |
| `BrokerCost` | `tradeview_ilc()`, `fundora()` presets | Use the constructor directly with your own spreads |

## Tests

```bash
python -m pytest tests/ -v
```

123 tests, all passing.

## License

MIT

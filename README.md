# backtest_engine

Numba JIT-accelerated trade simulator and validation toolkit. Turns Python for-loop backtests into **10-50x faster** simulations.

**What this is:** A fast bar-based trade simulator that takes OHLC data + pre-computed signals and evaluates each trade independently. Includes automated bug detection and stage-gate validation.

**What this is not:** A portfolio backtester. There is no account balance tracking, multi-instrument position management, or order book simulation. If you need those, look at backtrader or vectorbt.

## Features

### Core (instrument-agnostic)

- **`simulate_trades()`** — SL/TP/trailing/SAR trailing/limit entry/custom exit, all `@njit` kernels. Supports next-bar-open entry via `open_prices` in all modes
- **Indicators** — SMA, ATR, RSI, Bollinger Bands, RCI, Parabolic SAR, expanding quantile, multi-timeframe mapping (all `@njit`)
- **Monte Carlo DD** — 10,000-shuffle drawdown distribution, Kelly criterion, optimal risk sizing, prop firm DD check
- **StressTest** — block bootstrap (preserves losing-streak autocorrelation), parameter degradation scenarios (win rate / RR / cost what-if)
- **Walk-Forward / CSCV** — out-of-sample validation with IS/OOS ratio and Probability of Backtest Overfitting (PBO, Bailey 2015)
- **TradeResults** — structured array with convenience metrics: profit factor, win rate, expectancy, Sharpe/Sortino, max drawdown, recovery factor
- **Utilities** — CSV loading, AggVault API fetcher, OHLC resampling, higher-TF indicator mapping
- **Strategy Builder** — time-based entry/exit signal generation with timezone conversion, market-hours filtering, and match reporting

### Opinionated modules (FX defaults, fully customizable)

- **BrokerCost** — measured spread + commission model. Ships with FX presets (29 pairs (28 FX + XAUUSD)), but accepts any instrument via constructor. Supports per-trade cost arrays for variable-spread modeling
- **Strategy audit** (`python -m backtest_engine.audit my_strategy.py --json`) — finds look-ahead by behavior, not by reading code: rewrites the future and checks that past signals do not change, and runs the whole backtest on random walks where no edge can exist. JSON output for AI assistants. Measured on a labeled benchmark of 28 strategies ([bench/](bench/README.md))
- **BugGuard** — 14 automated checks (BG-01–BG-14) for cost underestimation, fixed-cost misuse, spread filter violations, short data periods, bfill and full-period quantiles in the source, and more. Data period estimation defaults to FX market hours but accepts explicit timestamps for any market. These are pattern and index checks: use the audit for look-ahead in signal code
- **GateKeeper** — fast-kill pipeline (Gate 0–4) that eliminates hopeless strategies early. Gate 0: BugGuard (pass your simulation result so the trade-level checks run), Gate 1–2: PF/RF screening, Gate 3: WFA+CSCV overfitting, Gate 4: Monte Carlo DD. All thresholds are class variables — override them for your asset class

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
from backtest_engine import fetch_aggvault, simulate_trades, LONG, atr

# Fetch EURUSD 1h data from AggVault API (export AGGVAULT_KEY=tk_your_key)
timestamps, open_, high, low, close, _ = fetch_aggvault(
    "EURUSD", "1h", "2024-01-01", "2025-12-31",
)

# ATR-based SL/TP
atr_vals = atr(high, low, close, 14)

# Define signals
signal_bars = np.array([100, 250, 400], dtype=np.int32)
directions = np.array([LONG, LONG, LONG], dtype=np.int8)
sl_distances = atr_vals[signal_bars] * 1.5
tp_distances = atr_vals[signal_bars] * 3.0

# Run simulation (GROSS — no execution costs)
# This tests whether the technical edge EXISTS, not whether it's profitable
# on a specific broker. Add costs later with BrokerCost for broker-specific testing.
results = simulate_trades(
    high, low, close,
    signal_bars, directions, sl_distances, tp_distances,
    max_hold=100,
    open_prices=open_,
)

# Results (GROSS: before broker-specific costs)
print(f"Win rate: {np.mean(results['pnl_r'] > 0) * 100:.1f}%")
print(f"Avg PnL: {np.mean(results['pnl_r']):.3f}R")
```

## Multi-resolution execution

`simulate_trades_hires()` generates signals on coarse bars (e.g. 1h) but runs trade simulation on fine bars (e.g. 1m). SL/TP hits are then resolved in the order they happened inside the coarse bar, instead of assuming SL first.

```python
import numpy as np
from backtest_engine import fetch_aggvault, simulate_trades_hires, rsi, atr, LONG, SHORT

# Signals on 1h
ts_1h, o_1h, h_1h, l_1h, c_1h, _ = fetch_aggvault("EURUSD", "1h", "2024-01-01", "2025-01-01")
rsi_vals = rsi(c_1h, 14)
atr_vals = atr(h_1h, l_1h, c_1h, 14)
# ... generate signal_bars, directions, sl_distances, tp_distances

# Execution on 1m (one year = 3 requests, ~15 s)
ts_1m, o_1m, h_1m, l_1m, c_1m, _ = fetch_aggvault("EURUSD", "1m", "2024-01-01", "2025-01-01")

results = simulate_trades_hires(
    signal_timestamps=ts_1h, signal_bars=signal_bars,
    directions=directions, sl_distances=sl_distances, tp_distances=tp_distances,
    max_hold=48, signal_bar_minutes=60,
    exec_timestamps=ts_1m, exec_opens=o_1m, exec_highs=h_1m,
    exec_lows=l_1m, exec_closes=c_1m,
)
```

**Timing.** Timestamps are bar open times (as returned by `fetch_aggvault`). A signal on the 10:00 1h bar is known when that bar closes at 11:00, so the trade enters at the open of the 11:00 1m bar. `results["signal_bar"]` is the 1m bar just before the entry bar (normally the last one inside the signal bar, 10:59), so `entry_bar > signal_bar` for every filled trade. Signals that close outside the 1m data come back as `NO_FILL` with a `DataQualityWarning`. Timestamps must be Unix seconds (`datetime64` arrays are converted); if the usual bar spacing does not match `signal_bar_minutes` / `exec_bar_minutes` (milliseconds, or 5m data without `exec_bar_minutes=5`), it raises. `max_hold` and `retrace_timeout` are given in signal bars.

**How much it changes results.** On EURUSD RSI(14) 30/70, 2024, SL 2×ATR / TP 4×ATR (696 trades, [`examples/rsi_reversal/compare_1h_vs_hires.py`](examples/rsi_reversal/compare_1h_vs_hires.py)):

| | 1h fills | 1m fills |
|---|---|---|
| GROSS | PF 0.959 | PF 0.958 |
| NET (Tradeview ILC) | PF 0.909 | PF 0.908 |

For this strategy the two are close. The difference can be larger when SL/TP are small relative to the bar range, because more bars then touch both.

**Correction (v0.6.0):** earlier versions of this README said this example drops from PF 0.90 on 1h to 0.53 on 1m, "a 41% overestimate". That 0.53 came from a bug: before v0.6.0, `simulate_trades_hires` entered one 1m bar after the signal bar *opened*, 59 minutes before the signal was known. See the [CHANGELOG](CHANGELOG.md).

## simulate_trades options

| Parameter | Description |
|-----------|-------------|
| `exit_mode="rr"` | Fixed risk-reward: SL/TP/timeout/breakeven |
| `exit_mode="trailing"` | Trailing stop |
| `exit_mode="sar_trailing"` | Parabolic SAR-based trailing (with direction-aware Plan B) |
| `exit_mode="custom"` | Exit on external signal |
| `be_trigger_pct` | Move SL to entry price when profit reaches this fraction of TP (0 = disabled) |
| `retrace_pct` | Limit entry: wait for price to retrace this fraction of TP from the signal close before filling (0 = market entry). SL/TP are measured from the fill price |
| `retrace_timeout` | Max bars to wait for limit fill before cancelling |
| `trail_activation_r` | R-multiple required to activate trailing stop |
| `trail_distance_r` | Trailing stop distance in R-multiples |
| `open_prices` | Enter at next-bar open instead of signal-bar close (supported in all modes). Also lets stops fill at the open on a gap |
| `entry_costs` | Per-trade cost array in R-units (e.g. from `BrokerCost.per_trade_cost()`). Subtracted from `pnl_r` post-simulation. `NO_FILL` trades automatically get zero cost |

### Fill rules

- **Entry**: next bar's open (`open_prices` given), signal bar's close (not given), or the limit price (`retrace_pct`).
- **Same bar touches SL and TP**: SL (conservative), unless the bar opened beyond the TP (then TP: the open came first; SAR targets keep SL priority). Use `simulate_trades_hires()` to resolve the order.
- **Stops on a gap**: a stop (SL, breakeven, trailing) on a bar that opens beyond it fills at that open, not at the stop price, so a loss can exceed 1R. Needs `open_prices` (`simulate_trades_hires` always passes them); without them the stop price is used.
- **TP and SAR targets**: fill at their price even on a gap (conservative).
- **Limit (retrace) fill bar**: only the SL is checked on the bar where the limit fills; whether the high/low that would hit TP came before or after the fill is unknowable.
- **Custom exits** (`exit_mode="custom"`): the exit signal is known at the close of its bar, so with `open_prices` the exit fills at the next bar's open; without them, at that close.
- **Timeout**: a trade is held `max_hold` bars and exits at the close of the last one. With next-bar-open entry the entry bar is the first of them (so `hold_bars = exit_bar - entry_bar = max_hold - 1`); with close or limit entry, the bar after (`hold_bars = max_hold`).

All prices must be consistent: if any bar opens outside its own high/low, `simulate_trades()` emits a `DataQualityWarning`.

## Pre-flight quality check

`simulate_trades()` inspects its inputs before running and assigns a quality grade:

| Grade | Condition | Meaning |
|-------|-----------|---------|
| A | `open_prices` provided | Next-bar-open entry (realistic) |
| B | `open_prices` not provided | Signal-bar close entry (optimistic bias) |

`entry_costs` is not part of the grade — GROSS (no costs) is valid when testing edge existence on aggregated data.

Grade indicates whether inputs were provided, not whether they are correct. Use BugGuard (BG-02, BG-12) to validate cost accuracy.

Grade B emits a `BacktestQualityWarning` with details on what's missing. The simulation still runs — nothing is blocked.

```python
# Grade B — warns you
results = simulate_trades(high, low, close, ...)
# BacktestQualityWarning: Backtest Quality: B
#   open_prices:  NOT PROVIDED — entry at signal-bar close (optimistic bias)

# Grade A — no warning
results = simulate_trades(..., open_prices=open_arr)

# Access quality info
print(results.quality.grade)  # "A"

# Suppress warnings
import warnings
from backtest_engine import BacktestQualityWarning
warnings.filterwarnings("ignore", category=BacktestQualityWarning)

# Or disable per call (results.quality is still set)
results = simulate_trades(..., preflight=False)
```

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

# FX presets (ships with 29 pairs (28 FX + XAUUSD), measured spreads)
cost = BrokerCost.tradeview_ilc()   # ECN (tight spread + $5 RT commission)
cost = BrokerCost.fundora()          # Prop firm (wider spread, no commission)

# Per-trade cost — pass to simulate_trades() for accurate cost modeling
instruments = ["EURUSD"] * len(sl_distances)
cost_array = cost.per_trade_cost(instruments, sl_distances)
results = simulate_trades(..., entry_costs=cost_array)
# results['pnl_r'] already has costs subtracted; results['cost_r'] shows each trade's cost

# Get all costs as a dict (useful for BugGuard)
expected = cost.cost_prices()  # {"EURUSD": 0.00007, "USDJPY": 0.014, ...}
```

## BugGuard

Checks for 14 known backtesting bugs before you trust any result. Each check runs only when you pass the inputs it needs (source path, trade arrays, costs, timestamps); a check whose inputs are missing is skipped, and `GateKeeper` reports the skipped ones as `GK-00`.

The checks themselves are instrument-agnostic (look-ahead in the trade arrays, bfill leak, same-bar reentry, etc.). The only FX-specific part is data period estimation — pass `start_ts`/`end_ts` to use calendar dates instead.

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
    timestamps=timestamps,
)
# BG-01: signal_bar >= entry_bar  BG-02: Cost underestimation  BG-04: bfill/backfill leak
# BG-05: Coarse-bar SL/TP         BG-06: Short period          BG-06b: Few trades
# BG-07: Full-period quantile     BG-08: Same-bar re-entry     BG-09: Close-price entry
# BG-09b: Missing open_prices     BG-10: Unsorted / still-forming bars
# BG-12: Fixed-cost misuse        BG-13: Spread filter gap     BG-14: Effectively no SL
```

Individual checks can also be called standalone:

```python
from backtest_engine import check_look_ahead, check_cost_registry, check_spread_filter

result = check_look_ahead(signal_bars, entry_bars)
print(f"{result.check_id}: {result.message}")

# Check if backtest signals respect a max spread constraint
spread_result = check_spread_filter(spreads_at_entry, max_spread=0.00020)
print(f"{spread_result.check_id}: {spread_result.message}")
```

## Strategy audit

BugGuard reads patterns; the audit tests behavior, so it works on code it cannot read (imported helpers, libraries). Put the strategy in one file:

```python
# my_strategy.py
import numpy as np
from backtest_engine import simulate_trades, rsi, atr

def signals(data):            # data: dict of numpy arrays time/open/high/low/close
    r = rsi(data["close"], 14)
    return np.where(r < 30, 1, np.where(r > 70, -1, 0))   # one value per bar

def backtest(data):           # the whole pipeline on the same data
    sig = signals(data)
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    a = atr(h, l, c, 14)
    sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
    sb = sb[sb < len(c) - 1].astype(np.int32)
    return simulate_trades(h, l, c, sb, np.sign(sig[sb]).astype(np.int8),
                           a[sb] * 2, a[sb] * 3, max_hold=24, open_prices=o)
```

```bash
uv run python -m backtest_engine.audit my_strategy.py          # or --json; --csv data.csv to test on your data
```

| Check | How | Catches |
|---|---|---|
| `lookahead` | Rewrites every bar after a cut point (mirror, 2× moves, ±20–25% jumps; volume is scaled too) and re-runs `signals`. A signal at or before the cut must not change. 32 cuts, half of them on or just before bars with a signal | Leaks that change signals: `shift(-1)`, centered windows, full-sample mean/std/min/max/quantiles, `bfill`, zero-phase filters, FFT over the sample, pivots confirmed by later bars, the current (unfinished) higher-timeframe bar, models fitted on the whole sample. Probabilistic: a leak that rarely flips a signal can pass |
| `null_edge` | Runs `backtest` on 30 driftless random walks and tests the mean R per trade before costs (paths as clusters). Each bar's high/low come from a continuous (Brownian bridge) path, so stops and targets fill at their level fairly. The same backtest is also run with every trade's direction replaced by a coin flip inside `simulate_trades` (same entries, SL/TP distances and exits) as a placebo | Entries placed before their signal, exits or targets that use later bars, timing bugs inside libraries; and a backtest made pessimistic by the SL-first rule (`warn`) |
| `source:BG-04/07/09` | BugGuard source patterns | Some backward-fill and full-period quantile patterns, entry at close (pattern matching only) |

Exit code 1 on `fail`, 2 on `error`. In Python: `from backtest_engine.audit import check_lookahead, check_null_edge`. `null_edge` flags |t| > 3.5; on the benchmark it flagged none of 360 runs of clean strategies (12 strategies × 30 seeds). Different `--seed` values share no random path, so a real leak fails again with another seed and a chance failure does not.

On the [benchmark](bench/README.md) (16 strategies with a planted leak, 12 without), the two checks together caught all 16 with no false alarm. An LLM reading each strategy file caught 15, with 1 false alarm. It missed the one leak inside an imported helper whose source it was not shown (the pre-v0.6.0 `simulate_trades_hires`, described in the prompt as documented); `null_edge` found it without reading code, and the LLM found it once given the helper's source. Passing does not prove the absence of bias: a leak that never flips a signal, survivorship, and bad data are not covered.

## GateKeeper

Stage-gate pipeline that kills hopeless strategies early. Default thresholds are tuned for FX — override them for your asset class.

```python
from backtest_engine import GateKeeper, BrokerCost

gk = GateKeeper(
    strategy_name="SMA Crossover v3",
    n_bars=7022,
    bar_minutes=60,
    resolution_minutes=1,          # fills simulated on 1m bars (simulate_trades_hires)
    spreads_used=spreads,
    source_path=__file__,
    expected_costs=BrokerCost.tradeview_ilc().cost_prices(),
)

# BugGuard + GK-00 input completeness. Pass a simulation result so the
# trade-level checks (BG-06b/08/09b/12/14) run.
gk.gate0_validate(trades=results, sl_distances=sl_distances,
                  avg_price=float(np.median(close)), timestamps=timestamps)
gk.gate1_quick(run_func, quick_params)  # ~20 combos → PF >= 1.05?
gk.gate2_screen(run_func, full_params)  # ~100 combos → one combo with PF >= 1.10 and RF >= 1.5?
gk.gate3_validate(wfa_result, cscv_result)  # WFA OOS win rate + CSCV PBO
gk.gate4_montecarlo(mc)                     # MC drawdown pass rate >= 70%
gk.summary()

# Override thresholds for a different asset class
gk.GATE1_MIN_PF = 1.10
gk.GATE2_MIN_RF = 2.0
gk.GATE3_MAX_PBO = 0.30
gk.GATE4_MIN_MC_PASS = 0.80
```

### Gate details

| Gate | Time | What it checks | Kill condition |
|------|------|---------------|----------------|
| Gate 0 | 1 min | BugGuard (the checks whose inputs you passed) + input completeness | Any BG ERROR → RuntimeError |
| Gate 1 | 5 min | ~20 param combos on full data | Best PF < 1.05 |
| Gate 2 | 20 min | ~100 param combos, PF + Recovery Factor | No single combo with PF >= 1.10 and RF >= 1.5 |
| Gate 3 | 30 min | WFA out-of-sample win rate + CSCV PBO | OOS win rate < 0.55 or PBO > 0.40 |
| Gate 4 | 5 min | Monte Carlo DD pass rate + confidence percentile | DD@confidence > dd_limit or pass rate < 0.70 |

Gate 0 also emits a `GK-00 WARN` listing every missing input (`source_path`, `spreads_used`, `n_bars`, `resolution_minutes`, `trades`, `sl_distances` + `avg_price`, `timestamps`) and the checks it skips. If `resolution_minutes` is not given, BG-05 assumes the fills were simulated on `bar_minutes` bars. With `trades`, BG-08 reports overlapping trades as a WARN (trades are simulated independently, so overlap is not a bug by itself). Gate 0 does not look for look-ahead in your signal code; run the [strategy audit](#strategy-audit) for that.

### run_func

`run_func` takes a parameter dict and returns a metric dict:

```python
def run_func(params: dict) -> dict | None:
    """Run a single backtest with the given params.
    Must return {'pf': ..., 'total_r': ..., 'n_trades': ..., 'max_dd_r': ...}
    or None if no trades.
    """
```

### Gate 3: WFA + CSCV

Gate 3 uses results from `WalkForward` and `CSCV` to detect overfitting. CSCV is optional — pass `None` to skip the PBO check.

```python
from backtest_engine import WalkForward, CSCV

# Run WFA
wf = WalkForward(n_bars=len(close), is_ratio=0.7, n_splits=5)
wfa_result = wf.run(param_grid, evaluate_fn)

# Run CSCV
cscv = CSCV(n_splits=10)
cscv_result = cscv.run(param_grid, evaluate_fn, n_bars=len(close))

# Feed into GateKeeper
gk.gate3_validate(wfa_result, cscv_result)
# → Checks: OOS win rate >= 0.55, PBO <= 0.40
```

### Gate 4: Monte Carlo DD

Gate 4 validates that the strategy survives randomized trade-order scenarios.

```python
from backtest_engine import MonteCarloDD

mc = MonteCarloDD(results['pnl_r'], n_sims=10_000, risk_pct=0.01, seed=42)
mc.run()

gk.gate4_montecarlo(mc, dd_limit=0.20, confidence=95.0)
# → Checks: DD@95% <= 20% AND fraction of sims with max DD < 20% >= 0.70
```

## TradeResults metrics

`simulate_trades()` returns a `TradeResults` object (numpy structured array) with convenience properties:

```python
results = simulate_trades(...)

print(f"PF: {results.profit_factor:.2f}")       # Gross profit / gross loss
print(f"Win rate: {results.win_rate:.1%}")       # Fraction of winning trades
print(f"Expectancy: {results.expectancy_r:.3f}R") # Mean PnL per trade
print(f"Geo mean: {results.geometric_mean_r:.4f}") # Geometric growth rate
print(f"Sharpe: {results.sharpe_r:.2f}")         # mean / std of pnl_r
print(f"Sortino: {results.sortino_r:.2f}")       # mean / downside_std
print(f"Max DD: {results.max_drawdown_r:.1f}R")  # Peak-to-trough in R
print(f"RF: {results.recovery_factor:.2f}")      # total_r / max_dd_r
```

All properties handle edge cases (empty results, no losses, no wins) and return plain floats.

## Monte Carlo & prop firm check

```python
from backtest_engine import MonteCarloDD

mc = MonteCarloDD(pnl_after_costs, n_sims=10_000, risk_pct=0.01, seed=42)
mc.run()

print(f"DD 95th: {mc.dd_percentile(95)*100:.1f}%")
print(f"Kelly: {mc.kelly_fraction()*100:.1f}%")
print(f"Optimal risk: {mc.optimal_risk_pct(max_dd=0.15)*100:.2f}%")

# Prop firm drawdown check (any firm, any limits)
result = mc.prop_firm_check(max_dd_limit=0.04, total_dd_limit=0.08, confidence=95.0)
print(f"Pass: {result['pass']}, Max DD OK: {result['max_dd_ok']}")
```

## StressTest

Stress-test a strategy's robustness beyond simple Monte Carlo shuffling.

```python
from backtest_engine import StressTest

st = StressTest(results['pnl_r'], n_sims=1000, seed=42)

# Block bootstrap — preserves losing-streak autocorrelation
bb = st.block_bootstrap(block_size=10)
print(f"Block bootstrap DD@95%: {bb['dd_95']*100:.1f}%")

# Parameter degradation — what-if scenarios
degraded = st.degrade(win_rate_delta=-0.05, rr_scale=0.90, cost_add_r=0.02)
# degraded is a modified pnl array you can feed into MonteCarloDD

# Run all scenarios at once
report = st.run_all(block_size=10)
# report["baseline"]           — standard MC shuffle
# report["block_bootstrap"]    — block bootstrap with autocorrelation
# report["degraded"]["wr_minus5"]     — win rate -5%
# report["degraded"]["rr_80pct"]      — reward:risk scaled to 80%
# report["degraded"]["cost_plus_01r"] — extra 0.1R cost per trade
# report["degraded"]["combined"]      — milder mix: win rate -3%, RR x0.90, +0.05R cost
```

Block bootstrap resamples consecutive blocks of trades instead of individual shuffling. This preserves the natural clustering of losing streaks, giving more realistic worst-case DD estimates.

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

## Data loading

Two ways to get OHLC data — both return the same 6-tuple `(timestamps, opens, highs, lows, closes, volume)`:

```python
from backtest_engine import load_ohlcv, fetch_aggvault

# Option 1: From CSV file (time column: date strings, or epoch seconds/milliseconds)
timestamps, opens, highs, lows, closes, volume = load_ohlcv("data.csv")

# Option 2: From AggVault API (ISO dates, no epoch math)
#   export AGGVAULT_KEY=tk_your_key
timestamps, opens, highs, lows, closes, volume = fetch_aggvault(
    "EURUSD", "1h", "2021-04-01", "2026-03-31",
)
```

`fetch_aggvault` supports `1m`, `5m`, `15m`, `1h`, `4h`, `1d` timeframes and 29 symbols (major/minor/cross FX pairs + XAUUSD). Case-insensitive, slashes optional (`"eur/usd"` works). API key via `api_key=` parameter or `AGGVAULT_KEY` environment variable.

The API returns at most 500,000 bars per request. Long 1m/5m/15m ranges are fetched in several requests (1m: 180 days each, paced to the 10 requests/minute limit), so 5 years of 1m data takes about a minute. Before v0.6.0 a single request silently returned only the first 500,000 bars (about 1.3 years of 1m data). A request window with no data inside the range emits a `DataQualityWarning`.

All timestamps are bar open times in Unix seconds (UTC).

## HTML Report Generation

Generate publication-ready HTML reports from `TradeResults`. Auto-generates PnL distribution, monthly/yearly breakdown, exit analysis, and GateKeeper results.

```python
from backtest_engine import generate_report, ReportConfig, SummaryCell, GateRow

cfg = ReportConfig(
    title="RSI Reversal — EURUSD",
    subtitle="RSI(14) 30/70 | 1H | 2021-2026 | GROSS",
    verdict="PASS",
    equity_curve=np.cumsum(results["pnl_r"]),
    equity_timestamps=timestamps[results["entry_bar"]],
    summary=[
        SummaryCell("Trades", str(len(results)), "総トレード数。"),
        SummaryCell("PF", f"{results.profit_factor:.2f}", "プロフィットファクター。"),
    ],
    strategy_html="<b>Entry:</b> RSI < 30 → Long ...",
    trades=results,              # TradeResults → auto sections
    trade_timestamps=timestamps,
    gates=[GateRow("BugGuard", "PASS", "14/14 checks")],
)
generate_report(cfg, "report.html")
```

Auto-generated sections (when `trades` is provided): PnL Distribution histogram, entry-month returns, exit breakdown with hold duration, entry-year P&L. Optional: WFA table (`wfa=`), CSCV overfitting detection (`cscv=`), free-form sections (`sections=`).

## Utilities

```python
from backtest_engine import resample_ohlcv, map_higher_tf

# Resample 5-min to 1-hour
ts_1h, o_1h, h_1h, l_1h, c_1h, v_1h = resample_ohlcv(
    timestamps, opens, highs, lows, closes, volume, rule="1h"
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
| `BugGuard` | Fundora cost registry (29 pairs: 28 FX + XAUUSD) | Pass your own `expected_costs` dict |
| `GateKeeper` | PF >= 1.05/1.10, RF >= 1.5, trades >= 30/50, PBO <= 0.40, MC pass >= 0.70 | Override class variables: `gk.GATE1_MIN_PF = 1.10` |
| `BrokerCost` | `tradeview_ilc()`, `fundora()` presets | Use the constructor directly with your own spreads |

## Known limitations

Things this engine handles, and things it doesn't. Read before trusting any result.

**Handled automatically:**
- **Entry timing** — Use `open_prices` for next-bar-open entry. The strategy audit's `null_edge` check flags entries placed before their signal.
- **Intra-bar SL/TP ordering** — Use `simulate_trades_hires()` to run signals on coarse bars but simulate execution on 1m bars. Without it, a bar that touches both SL and TP counts as SL.
- **Stop gaps** — With `open_prices`, a stop that the market gaps through fills at the open (see [Fill rules](#fill-rules)).
- **Cost underestimation** — BugGuard BG-02 compares your costs against measured broker spreads. Use `BrokerCost.per_trade_cost()` for per-trade costs, not fixed scalars.
- **One-way vs round-trip clarity** — `BrokerCost.spreads` is **one-way** bid-ask spread (pay once on entry). `commission_per_lot` is **round-trip** (e.g. $5 RT, not $2.50 per side). Getting these backwards doubles or halves your cost estimate. The presets (`tradeview_ilc`, `fundora`) are already correct.
- **Short-period overfitting** — BugGuard BG-06 warns on <12 months of data. CSCV gives a formal PBO estimate.
- **Timezone mixing** — All timestamps are Unix epoch (UTC). Strategy Builder handles local→UTC conversion.

**NOT handled — your responsibility:**
- **Look-ahead the audit cannot see** — The [strategy audit](#strategy-audit) finds look-ahead that changes signals or creates an edge on random data. BugGuard only checks a few source patterns. Neither proves the absence of bias: a leak that never flips a signal, survivorship in the instrument list, and data errors are not covered. A strategy that looks too good usually has a leak.
- **Portfolio effects** — Trades are simulated independently: no position overlap limits, margin, or account balance.
- **Backtest-to-live gap** — Backtesting on aggregated multi-source data (e.g. AggVault median) gives cleaner signals than any single broker. A strategy that works on aggregated data may underperform on your broker's feed due to spread differences, requotes, and slippage. Always forward-test on your actual broker before going live.
- **Slippage** — Not included in `BrokerCost` presets because it varies per user (colocation ≈ 0, retail ≈ 0.5-1+ pips). Estimate your own slippage and either add it to `spreads` or use `StressTest.run_all()` which includes a `cost_plus_01r` scenario (extra 0.1R per trade) to check if your edge survives higher costs.
- **Variable spreads** — `BrokerCost` uses static measured spreads. Real spreads widen during news, low liquidity, and session boundaries. If your strategy trades during these times, backtest results will be more optimistic than live.
- **Swap/rollover costs** — Not modeled. For intraday strategies this is negligible. For multi-day holds, swap can erase thin edges.
- **Execution latency** — The engine assumes instant fills at bar prices. In live trading, latency causes slippage, especially on fast moves.

## Tests

```bash
uv run pytest
```

469 tests, all passing.

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.

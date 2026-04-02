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
- **BugGuard** — 14 automated checks (BG-01–BG-13) that catch look-ahead bias, cost underestimation, fixed-cost misuse, spread filter violations, overfitting, and more. Data period estimation defaults to FX market hours but accepts explicit timestamps for any market
- **GateKeeper** — fast-kill pipeline (Gate 0–4) that eliminates hopeless strategies early. Gate 0: BugGuard, Gate 1–2: PF/RF screening, Gate 3: WFA+CSCV overfitting, Gate 4: Monte Carlo DD. All thresholds are class variables — override them for your asset class

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

`simulate_trades_hires()` generates signals on coarse bars (e.g. 1h) but runs trade simulation on fine bars (e.g. 1m). This eliminates the intra-bar SL/TP ordering problem that inflates results on coarse timeframes.

```python
from backtest_engine import fetch_aggvault, simulate_trades_hires, rsi, atr, BrokerCost, LONG, SHORT

# Signals on 1h
ts_1h, o_1h, h_1h, l_1h, c_1h, _ = fetch_aggvault("EURUSD", "1h", "2024-01-01", "2025-01-01")
rsi_vals = rsi(c_1h, 14)
atr_vals = atr(h_1h, l_1h, c_1h, 14)
# ... generate signal_bars, directions, sl_distances, tp_distances

# Execution on 1m (7s fetch, 0.01s simulation)
ts_1m, o_1m, h_1m, l_1m, c_1m, _ = fetch_aggvault("EURUSD", "1m", "2024-01-01", "2025-01-01")

results = simulate_trades_hires(
    signal_timestamps=ts_1h, signal_bars=signal_bars,
    directions=directions, sl_distances=sl_distances, tp_distances=tp_distances,
    max_hold=48, signal_bar_minutes=60,
    exec_timestamps=ts_1m, exec_opens=o_1m, exec_highs=h_1m,
    exec_lows=l_1m, exec_closes=c_1m, entry_costs=entry_costs,
)
```

**Why this matters:** On EURUSD RSI 30/70 (2024), 1h-only simulation reports PF=0.90 while hires reports PF=0.53. The 1h version overestimates performance by ~41% because it can't determine SL/TP hit order within a bar.

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
| `entry_costs` | Per-trade cost array in R-units (e.g. from `BrokerCost.per_trade_cost()`). Subtracted from `pnl_r` post-simulation. `NO_FILL` trades automatically get zero cost |

## Pre-flight quality check

`simulate_trades()` inspects its inputs before running and assigns a quality grade:

| Grade | Condition | Meaning |
|-------|-----------|---------|
| A | `open_prices` provided | Next-bar-open entry (realistic) |
| B | `open_prices` not provided | Signal-bar close entry (optimistic bias) |

`entry_costs` is not part of the grade — GROSS (no costs) is valid when testing edge existence on aggregated data.

Grade indicates whether inputs were provided, not whether they are correct. Use BugGuard (BG-02, BG-12) to validate cost accuracy.

Grade B/C emit a `BacktestQualityWarning` with details on what's missing. The simulation still runs — nothing is blocked.

```python
# Grade C — warns you
results = simulate_trades(high, low, close, ...)
# BacktestQualityWarning: Backtest Quality: C
#   entry_costs:  NOT PROVIDED — costs will be 0 (use BrokerCost.per_trade_cost())
#   open_prices:  NOT PROVIDED — entry at signal-bar close (optimistic bias)

# Grade A — no warning
results = simulate_trades(..., open_prices=open_arr, entry_costs=cost_arr)

# Access quality info
print(results.quality.grade)  # "A"

# Suppress warnings
import warnings
from backtest_engine import BacktestQualityWarning
warnings.filterwarnings("ignore", category=BacktestQualityWarning)

# Or disable per call
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

Automatically checks for 14 known backtesting bugs before you trust any result.

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
# BG-08: Same-bar re-entry     BG-09: Close-price entry     BG-09b: Missing open_prices
# BG-12: Fixed-cost misuse    BG-13: Spread filter gap     ... and more
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

gk.gate0_validate()                    # BugGuard 14 checks + GK-00 input completeness
gk.gate1_quick(run_func, quick_params)  # ~20 combos → PF >= 1.05?
gk.gate2_screen(run_func, full_params)  # ~100 combos → PF >= 1.10, RF >= 1.5?
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
| Gate 0 | 1 min | BugGuard (14 checks) + input completeness | Any BG ERROR → RuntimeError |
| Gate 1 | 5 min | ~20 param combos on full data | Best PF < 1.05 |
| Gate 2 | 20 min | ~100 param combos, PF + Recovery Factor | Best PF < 1.10 or RF < 1.5 |
| Gate 3 | 30 min | WFA out-of-sample win rate + CSCV PBO | OOS win rate < 0.55 or PBO > 0.40 |
| Gate 4 | 5 min | Monte Carlo DD pass rate + confidence percentile | DD@confidence > dd_limit or pass rate < 0.70 |

Gate 0 also emits a `GK-00 WARN` when key inputs are missing (`source_path`, `spreads_used`, `n_bars`), which causes important BugGuard checks to be silently skipped.

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
# report["degraded"]["combined"]      — all degradations at once
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

# Option 1: From CSV file
timestamps, opens, highs, lows, closes, volume = load_ohlcv("data.csv")

# Option 2: From AggVault API (ISO dates, no epoch math)
#   export AGGVAULT_KEY=tk_your_key
timestamps, opens, highs, lows, closes, volume = fetch_aggvault(
    "EURUSD", "1h", "2021-04-01", "2026-03-31",
)
```

`fetch_aggvault` supports `1m`, `5m`, `15m`, `1h` timeframes and 14 symbols (major FX pairs + XAUUSD). API key via `api_key=` parameter or `AGGVAULT_KEY` environment variable.

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
- **Look-ahead bias** — BugGuard BG-01 detects signal/entry on same bar. Use `open_prices` for next-bar-open entry.
- **Intra-bar SL/TP ordering** — Use `simulate_trades_hires()` to run signals on coarse bars but simulate execution on 1m bars. Without this, 1h-bar simulation overestimates PF by ~40%.
- **Cost underestimation** — BugGuard BG-02 compares your costs against measured broker spreads. Use `BrokerCost.per_trade_cost()` for per-trade costs, not fixed scalars.
- **One-way vs round-trip clarity** — `BrokerCost.spreads` is **one-way** bid-ask spread (pay once on entry). `commission_per_lot` is **round-trip** (e.g. $5 RT, not $2.50 per side). Getting these backwards doubles or halves your cost estimate. The presets (`tradeview_ilc`, `fundora`) are already correct.
- **Short-period overfitting** — BugGuard BG-06 warns on <12 months of data. CSCV gives a formal PBO estimate.
- **Timezone mixing** — All timestamps are Unix epoch (UTC). Strategy Builder handles local→UTC conversion.

**NOT handled — your responsibility:**
- **Backtest-to-live gap** — Backtesting on aggregated multi-source data (e.g. AggVault median) gives cleaner signals than any single broker. A strategy that works on aggregated data may underperform on your broker's feed due to spread differences, requotes, and slippage. Always forward-test on your actual broker before going live.
- **Slippage** — Not included in `BrokerCost` presets because it varies per user (colocation ≈ 0, retail ≈ 0.5-1+ pips). Estimate your own slippage and either add it to `spreads` or use `StressTest.run_all()` which includes a `cost_plus_01r` scenario (extra 0.1R per trade) to check if your edge survives higher costs.
- **Variable spreads** — `BrokerCost` uses static measured spreads. Real spreads widen during news, low liquidity, and session boundaries. If your strategy trades during these times, backtest results will be more optimistic than live.
- **Swap/rollover costs** — Not modeled. For intraday strategies this is negligible. For multi-day holds, swap can erase thin edges.
- **Execution latency** — The engine assumes instant fills at bar prices. In live trading, latency causes slippage, especially on fast moves.

## Tests

```bash
python -m pytest tests/ -v
```

326 tests, all passing.

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.

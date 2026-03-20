# Changelog

## v0.5.0

### New features

- **Gate 3: WFA + CSCV overfitting check** — `GateKeeper.gate3_validate(wfa_result, cscv_result)` checks WFA out-of-sample win rate (>= 0.55) and CSCV Probability of Backtest Overfitting (<= 0.40). CSCV is optional — pass `None` to skip the PBO check.
- **Gate 4: Monte Carlo DD check** — `GateKeeper.gate4_montecarlo(mc, dd_limit, confidence)` validates that >= 70% of MC simulations survive within the DD limit. Auto-runs MC if not already run.
- **GK-00 input completeness warning** — `gate0_validate()` now emits a `WARN` when `source_path`, `spreads_used`, or `n_bars` are missing, since this causes important BugGuard checks to be silently skipped.
- **`StressTest` class** — block bootstrap (preserves losing-streak autocorrelation via `@njit` kernel) and parameter degradation scenarios (win rate delta, RR scaling, cost addition). `run_all()` executes baseline + block bootstrap + 4 degradation scenarios in one call.
- **`_block_bootstrap_dd()` kernel** — new `@numba.njit` function that resamples consecutive blocks of trades, using the same LCG PRNG pattern as the existing Monte Carlo shuffle.
- **TradeResults convenience metrics** — 8 new `@property` methods: `profit_factor`, `win_rate`, `expectancy_r`, `geometric_mean_r`, `sharpe_r`, `sortino_r`, `max_drawdown_r`, `recovery_factor`. All handle edge cases (empty results, no losses, no wins) and return plain floats.

### Bug fixes

- **`max_drawdown_r` missed initial drawdown** — cumulative PnL curve started from the first trade, not from 0R. If the first trades were losers, the drawdown from starting equity was invisible. Fixed by prepending 0.0 to the cumulative curve. `recovery_factor` (which depends on `max_drawdown_r`) is also corrected.
- **`sortino_r` returned `nan` with exactly 1 loss** — `np.std(downside, ddof=1)` on a single-element array produces `nan` (division by zero). Now returns `inf` (positive expectancy) or `0.0` when fewer than 2 losing trades.
- **`summary()` claimed "Strategy is validated" with partial gates** — running only Gate 0–1 and calling `summary()` printed "All gates passed. Strategy is validated", giving false confidence. Now checks that all 5 gates (0–4) ran before printing "validated". Partial runs show "INCOMPLETE (Gate X, Y not run)".
- **GK-00 warning not counted in `n_warnings`** — the GK-00 `CheckResult` was created with `passed=True`, but `GuardReport.n_warnings` counts `not passed and severity == "WARN"`. Changed to `passed=False` so the warning is properly counted without affecting gate pass/fail (which only checks ERROR severity).
- **Gate 4 `passed` ignored `confidence` percentile** — the pass condition only checked the pass rate (fraction of sims with DD < dd_limit), ignoring the DD at the `confidence` percentile entirely. A strategy with DD@95%=30% could pass with dd_limit=20% if 70% of sims were below. Now requires BOTH `dd_conf <= dd_limit` and `pass_rate >= GATE4_MIN_MC_PASS`.
- **Gate 3 silently skipped PBO when CSCV=None** — passing `cscv_result=None` caused Gate 3 to omit any mention of PBO in its message. Now explicitly reports "CSCV PBO=skipped (not provided)" so the omission is visible in logs and summary output. Additionally, `summary()` now distinguishes "Strategy is validated" from "Strategy is validated (CSCV/PBO not checked)" to prevent false confidence when PBO was not actually tested.
- **Gate 2 killed perfect strategies (max_dd_r=0)** — Recovery Factor calculation returned 0 when `max_dd_r=0` (zero drawdown), causing `rf < GATE2_MIN_RF` to kill the strategy. Now correctly returns `inf` when `max_dd_r=0` and `total_r > 0`.
- **Gate 0 log contradiction** — `run_all_checks()` printed "ALL CLEAR (0 warnings)" before GK-00 was appended, producing a numerical mismatch with the Gate 0 footer. GK-00 is now appended to the report *before* `print_report()`, so BugGuard's own output includes the correct warning count.
- **`StressTest.block_bootstrap(block_size=0)` raised raw `ZeroDivisionError`** — now raises `ValueError("block_size must be >= 1")`.
- **`summary()` could be fooled by duplicate gate runs** — `len(self.gates) >= 5` could be satisfied by running Gate 0 and Gate 1 three times each. Now uses set-based gate name validation instead of count.

### Other

- GateKeeper `summary()` uses `TOTAL_GATES = 5` class variable for completeness check.
- `examples/gatekeeper_pipeline.py` — new example demonstrating the full Gate 0–4 pipeline with relaxed thresholds for demo purposes.
- `examples/example_strategy.py` — updated to use TradeResults metrics and StressTest.
- README updated with Gate 3/4 documentation, StressTest section, TradeResults metrics section, and gate details table.
- Test count: 174 → 249.

## v0.4.0

### New features

- **Pre-flight quality check** — `simulate_trades()` now inspects inputs before running and assigns a quality grade (A/B/C). Grade B/C emit `BacktestQualityWarning` explaining what's missing. Disable with `preflight=False` or `warnings.filterwarnings("ignore", category=BacktestQualityWarning)`.
- **`TradeResults`** — return type of `simulate_trades()` is now a numpy structured array subclass with a `.quality` attribute. All existing code (`results["pnl_r"]`, `isinstance(results, np.ndarray)`, slicing, pickle) continues to work unchanged.
- **BG-12/BG-13 integrated into `run_all_checks()`** — pass `sl_distances`/`cost_is_fixed` for BG-12 and `signal_spreads`/`max_spread` for BG-13. Previously these checks existed but had to be called individually.

### Bug fixes

- **FX pair count display** — README/CHANGELOG corrected from "29 pairs + XAUUSD" to "29 pairs (28 FX + XAUUSD)".
- **BG-13 false PASS** — `check_spread_filter(max_spread=0.0002)` without `signal_spreads` returned PASS ("No constraint to check"). Now returns WARN ("signal_spreads not provided").
- **Input validation gaps** — `simulate_trades()` now rejects `signal_bars` out of range, `sl_distances <= 0`, and `NaN` in `close` with clear `ValueError` messages. Previously these caused garbage output, `ZeroDivisionError`, or silent NaN propagation.
- **Fundora cost registry mismatch** — `_fundora_expected_costs()` (used by deprecated `broker="fundora"` path) produced different values from `BrokerCost.fundora().cost_prices()` for 10/29 instruments. Now delegates to `BrokerCost.fundora().cost_prices()` so only one cost registry exists.

### Tests

- Trailing phantom profit regression tests tightened with exact expected values.
- `is_oos_ratio` test updated from `> 0` to bounded range assertion.
- 6 new tests for BG-12/BG-13 `run_all_checks()` integration (incl. max_spread-only case).
- 15 new tests for pre-flight quality check and TradeResults.
- 5 new input validation tests (signal_bars range, sl_distances, NaN).
- Test count: 140 → 174.

## v0.3.0

### Breaking changes

None. All changes are backward-compatible with v0.2.0 code.

### Bug fixes

- **Trailing stop phantom profit** — `_sim_trailing_inner` checked SL against the *updated* `best_price` (which included the current bar's high/low), creating optimistic bias. Now checks SL against the *previous* `best_price` before updating. Affects `exit_mode="trailing"` only; `sar_trailing` was already correct.
- **BG-02 silent pass on missing costs** — `check_cost_registry()` returned `PASS` when `expected_costs=None`, silently skipping cost validation. Now returns `WARN` to surface the omission.

### New features

- **`entry_costs` parameter in `simulate_trades()`** — per-trade cost array (in R-units) subtracted from `pnl_r` post-kernel. Enables accurate per-trade spread modeling instead of a single fixed cost. `NO_FILL` trades automatically get zero cost.
- **`cost_r` field in trade results** — `TRADE_RESULT_DTYPE` now includes `cost_r` for full cost auditability.
- **`BrokerCost.per_trade_cost()`** — convenience method that returns per-trade cost arrays for use with `entry_costs`. Delegates to `as_r_array()`.
- **BG-09b: `check_open_prices_provided()`** — warns when `open_prices` is not passed to `simulate_trades()`, flagging potential close-price entry bias.
- **BG-12: `check_fixed_cost_usage()`** — warns when a single fixed cost is used with varying SL distances (CV > 10%), indicating per-trade costs should be used instead.
- **BG-13: `check_spread_filter()`** — warns when backtest signals include spreads exceeding a max constraint, catching BT/LIVE cost parity violations.
- **EURCHF** added to `BrokerCost.fundora()` preset (28 FX → 29 pairs incl. XAUUSD).
- **`is_oos_ratio_raw`** — new key in `WalkForward.run()` output with docstring noting upward bias from asymmetric split lengths. `is_oos_ratio` retained as backward-compat alias.

### Deprecations (with backward-compatible wrappers)

- `prop_firm_check(daily_dd_limit=...)` → use `max_dd_limit=...` (the old parameter name implied per-day DD, but it controls max drawdown limit). Return dict: `max_dd_ok` is the primary key, `daily_dd_ok` retained as alias.

All deprecated APIs emit `DeprecationWarning` and continue to work.

### Other

- Example strategy updated to use `per_trade_cost()` instead of scalar `as_r()`.
- Test count: 123 → 140.

## v0.2.0

### Breaking changes

- `FUNDORA_COST_PIPS` and `cost_pips_to_price` removed from public API (`__all__`).
  Access via `bug_guard._FUNDORA_COST_PIPS` / `bug_guard._cost_pips_to_price` if needed.

### New features

- **`BrokerCost.cost_prices()`** — returns `{instrument: total_RT_cost}` dict for all instruments.
- **`MonteCarloDD.prop_firm_check()`** — generic prop firm DD check with configurable daily/total DD limits and confidence level.
- **`check_cost_registry(expected_costs=...)`** — accepts any cost dict instead of hardcoded broker name.
- **`GateKeeper(expected_costs=...)`** — same generalization for the gate pipeline.
- **`open_prices`** support in all exit modes (rr, trailing, sar_trailing, custom).
- **`WalkForward.run()` returns `is_oos_ratio`** — ratio of OOS to IS performance.
- **GitHub Actions CI** — runs `pytest tests/ -v` on push/PR.

### Deprecations (with backward-compatible wrappers)

- `MonteCarloDD.fundora_check()` — use `prop_firm_check()` instead.
- `run_all_checks(broker="fundora")` — use `expected_costs=BrokerCost.fundora().cost_prices()`.
- `GateKeeper(broker="fundora")` — use `expected_costs=BrokerCost.fundora().cost_prices()`.

All deprecated APIs emit `DeprecationWarning` and continue to work.

### Bug fixes

- Fixed duplicate `CADCHF` key in internal cost registry.

### Other

- `pyproject.toml`: added `license`, `readme`, `authors`, `classifiers`, `[project.urls]`.
- `.gitignore`: added Numba cache files (`*.nbi`, `*.nbc`).
- README rewritten with correct API examples.
- Test count: 98 → 123.

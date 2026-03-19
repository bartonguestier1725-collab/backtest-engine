# Changelog

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

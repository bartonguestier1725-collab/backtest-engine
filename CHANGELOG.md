# Changelog

## v0.6.2

Documentation, names and instructions for AI assistants. Simulation results
do not change.

### Changed

- **The `BrokerCost` presets have new names:** `BrokerCost.sample_ecn()` and
  `BrokerCost.sample_wide_spread()`, documented as sample values to replace
  with your own broker's numbers. The old names still work and return the
  same values, with a `DeprecationWarning` naming the new one.
- The deprecated `broker=` argument of `run_all_checks` / `GateKeeper` and
  the deprecated prop-firm check method of `MonteCarloDD` still work; their
  warnings point to `BrokerCost(...).cost_prices()` and `prop_firm_check()`.
- Removed an unused internal cost table from `bug_guard` (not used for
  validation since v0.4.0).
- The missing-key error of `fetch_aggvault` names the environment variable
  `AGGVAULT_KEY` first and says not to write the key into files.

### Docs

- CLAUDE.md, AGENTS.md: rules for AI assistants on costs (GROSS first; NET
  on the user's own numbers, not refused), data (the user's own broker data
  may be used, aligned to UTC bar opens), the next step after Gate 3 (check
  the result at the user's broker), the API key (environment or `.env` only),
  fetching (once, saved; not at the top level of a strategy file) and not
  editing the engine. AGENTS.md now carries these rules too.
- README: pointer to AGENTS.md / CLAUDE.md; install with `uv sync` and update
  with `git pull` + `uv sync`; cost examples with your own `BrokerCost(...)`;
  `load_ohlcv` column names; the PBO comment follows Gate 3 (fails above
  0.40); fetch times without unmeasured figures.
- Examples: `gatekeeper_pipeline.py` runs GROSS with the default thresholds
  (it passes Gates 0-3 and stops at Gate 4) and drops the unmeasured per-gate
  times; `example_strategy.py` passes example costs through the constructor;
  `level1_from_api.py` follows the README install steps.

## v0.6.1

Fixes found in a clean-clone check of v0.6.0. Fills and `pnl_r` do not
change. Some metrics and checks do (first section).

### Changed

- **Win rate, expectancy, geometric mean, Sharpe and Sortino count filled
  trades only.** `NO_FILL` rows (pnl_r = 0: a limit order that never filled,
  a signal on the last bar or outside the 1m data) were counted as trades and
  diluted them. PF, total R, max DD and recovery factor are unchanged. New
  `results.n_filled` gives the trade count; use it as `n_trades` in a
  GateKeeper `run_func` (`len(results)` counts the `NO_FILL` rows too, and
  Gates 1–2 compare it with their minimum trade counts). New
  `results.filled` drops the `NO_FILL` rows; the README and examples pass
  `results.filled["pnl_r"]` to `MonteCarloDD` and `StressTest` (the
  shuffled max DD has the same distribution, but numbers at a fixed seed
  change; Kelly's win rate, the block bootstrap, `degrade(win_rate_delta=)`
  and `cost_add_r` no longer count orders that never filled). The HTML report
  leaves `NO_FILL` rows out of the histogram, the monthly and yearly tables
  and the exit shares, and says how many it left out.
- **BG-06 uses the bar times.** `bug_guard(timestamps=)` and
  `gk.gate0_validate(timestamps=)` ignored them for the data period and
  estimated it from the bar count. They now measure from the first bar's open
  to the last bar's close; timestamps in milliseconds or nanoseconds are a BG-06 ERROR (and
  BG-10 reports times that are not numbers instead of raising). With
  fewer `n_bars` than timestamps (a sliced `close` with the full `ts`), the
  span is scaled down by `n_bars`/timestamps. Without
  timestamps the estimate is 24 h × 22 days a month (528 1h bars), not
  17 h × 22 days (374): FX trades about 21.7 days of 24 h, so the old
  estimate read 8.7 months of AggVault EURUSD 1h as 12.1 months and passed
  the 12-month minimum. The new one errs short (5 years read as 59 months),
  except for markets open every day (crypto reads about 1.4× long: pass
  timestamps). Data that passed before can now fail, e.g. 300,000 1m bars
  (9.5 months). BG-06 passes at 12 months less 1 week (11.75): a calendar
  year of FX data starts after the New Year close and ends at the last
  close of the year, so it measures 11.86–11.99 months (EURUSD and XAUUSD 1h,
  2022–2024) and would fail a strict 12. Periods are shown with two
  decimals, and a failure on the estimate says to pass the bar times.
- **`simulate_trades` raises on options of another exit mode.** Each mode
  ignored the others' options: a non-zero `be_trigger_pct`, `retrace_pct` or
  `retrace_timeout` outside `"rr"` entered at market without breakeven, a
  non-zero `trail_activation_r` / `trail_distance_r` outside `"trailing"`
  (e.g. with the default `"rr"`) gave fixed SL/TP exits, and `exit_signals`
  outside `"custom"` or `sar_values` outside `"sar_trailing"` were unused.
  These now raise `ValueError`, also through `simulate_trades_hires`. An
  unknown `exit_mode` now raises also when there are no signals.
- **The audit no longer passes a file it tested only half of.** A file with
  only `signals()` or only `backtest()` gets a `warn` for the missing test
  (`details["not_run"]`, and a summary of what was not tested) instead of an
  overall `pass`. The exit code stays 0. `bench/run_bench.py` does not count
  these as catches, so the benchmark numbers do not change.

### Examples

- **`rsi_reversal/level4_filters.py` reached the wrong conclusion** — its
  `evaluate_fn` returned -999 for a window with fewer than 10 trades. WFA
  averaged that into the OOS mean (-209.7R instead of -9.6R per window), and
  CSCV summed it per window, so the ranking followed how often a parameter
  set traded, not how well: PBO came out 2% instead of 96%, and the script
  printed "Filters reduced overfitting!". A window now scores its real total
  R (0 without trades), and the verdict uses the Gate 3 rule (WFA and CSCV
  must both pass). The filtered strategy fails it. `level3_gatekeeper.py`
  had the same placeholder (PBO 87.7% → 88.1%), and judged the OOS/IS
  ratio of window totals, which is about 0.43 even with a perfect transfer
  (the OOS windows are 30/70 as long); it now compares R per bar.
- `level2`/`level3`/`level4`: the Level 1–3 numbers printed from constants
  were from v0.5.x (PF 1.05 / 1.33, PBO 80%); level2 now reads the Level 1
  row from its own grid. The WFA mean is labelled per OOS window (it is total
  R, not R per trade), and the CSCV question describes PBO correctly.
- `gold_rollover/with_strategy_builder.py` stopped with `NameError: df` since
  the move to `fetch_aggvault`; it notes that BrokerCost's XAUUSD pip
  ($0.01) makes its pips 10× those of the `stage1_*` scripts ($0.10).

### Fixes

- `uv sync` now installs pytest (a `dev` dependency group), so `uv run
  pytest` works in a fresh clone. The `dev` extra stays for `pip install -e
  '.[dev]'`.
- `fetch_aggvault` closes the HTTP 429/503 reply before retrying (it left a
  `ResourceWarning`).
- BG-05 no longer quotes "SL=2.0×ATR≈20pip" built from default arguments; it
  quotes the SL only when `check_resolution` is given `sl_atr_mult` and
  `avg_atr_pips`.
- `check_data_period(start_ts=, end_ts=)` reads numbers as Unix seconds (it
  read them as nanoseconds and reported 0.0 months) and raises on numbers
  that look like milliseconds or nanoseconds, and on NaT (which passed as
  "nan months"). Timestamps with and without a time zone can be mixed.
- Gate 3's message says "WFA OOS positive split share" (the share of WFA
  splits with a positive OOS metric), not "win rate".
- BG-12 said "Per-trade cost used (correct)" also when no cost was applied.
  It now says "No fixed cost reported (per-trade costs, or none applied)".

### Docs

- README, CLAUDE.md, AGENTS.md: BG-06 is an ERROR, not a warning; BugGuard has no
  default cost registry; `generate_report` builds the equity curve and the
  Gate/WFA/CSCV tables only from values you pass (`WfaRow`, `CscvResult`),
  and DD% needs a balance curve, not cumulative R; `custom` exits ignore TP;
  Gate 3 uses the share of WFA splits with a
  positive OOS metric, not a trade win rate; Gates 1–4 do not stop on a
  failure; BG-08 reports only entries on the previous exit bar; the audit's
  third `null_edge` failure (LOSS ON RANDOM DATA) and its exit code 0 for
  `warn`; 5 years of 1m data take 2–3 minutes (146 s measured), not one.
- `gatekeeper.py` docstring stated the kill conditions backwards.
- CLAUDE.md, AGENTS.md, README and the `WalkForward`/`CSCV` docstrings:
  `evaluate_fn` returns the window's real metric, 0 without trades, never a
  placeholder such as -999.
- README: removed "10-50x faster" and the Time column of the Gate table (no
  measurement behind either), and the per-gate times in `gatekeeper.py`.

## v0.6.0

This release fixes several bugs that made results look different from what the
code would do in a live market, and adds a strategy audit that finds
look-ahead by behavior. **Results change** for `simulate_trades_hires`, for
long `fetch_aggvault` 1m/5m/15m requests, for `retrace_pct` entries, for
`exit_mode="custom"` with `open_prices`, for timeouts with `open_prices` (one
bar shorter), and slightly for gapped stops and targets. Re-run backtests made
with v0.5.x before relying on them.

### Bug fixes (results change)

- **`simulate_trades_hires` look-ahead** — the signal bar was mapped to the
  first execution bar of the signal bar, so trades entered one execution bar
  after the signal bar *opened* (with 1h signals and 1m fills: 59 minutes
  before the 1h close that produced the signal was known). Trades now enter at
  the open of the first execution bar at or after the signal bar's close.
  Measured on EURUSD 2024, RSI(14) 30/70, SL 2×ATR / TP 4×ATR, 696 trades,
  NET with the sample ECN cost preset: PF 0.525 before, 0.908 after (1h fills: 0.909).
  The README claim "1h-only PF 0.90 vs hires 0.53, a 41% overestimate" was an
  artifact of this bug and has been removed. The old test that fixed the old
  mapping (`test_signal_bar_mapping_correct`) now asserts the corrected one.
- **`fetch_aggvault` silently truncated long ranges** — the API returns at most
  500,000 bars per request (oldest first, HTTP 200, no flag). For the RSI
  tutorial (`examples/rsi_reversal/level1_from_api.py`, EURUSD 1m,
  2021-04-01 – 2026-03-31) the single request ends on 2022-08-02, and
  replaying those first 500,000 bars, 2,608 of 3,573 signals (73%) come back
  as `NO_FILL` with zero PnL, without any warning. Long 1m/5m/15m ranges are now fetched in windows (1m: 180 days),
  paced to the 10 requests/minute limit, with up to 2 retries on HTTP 429. The
  tutorial now fills all 3,573 signals (1,861,388 1m bars). A windowed
  fetch of 2024 returned exactly the same bars as the single request.
- **Stops filled at the stop price through gaps** — with `open_prices`, a stop
  (SL, breakeven, trailing, SAR hard stop) on a bar that opens beyond it now
  fills at that open. A gapped stop can lose more than 1R. TP and SAR targets
  still fill at their price. Without `open_prices` the gap is unknowable and
  the stop price is used as before. EURUSD 2024 example above, 1h fills,
  GROSS: PF 0.953 → 0.948 from this rule alone (0.959 with all v0.6.0
  changes, mostly the hold-window fix below).
- **`retrace_pct` SL measured from the signal close** — the limit fills at
  `close ∓ retrace_pct × TP`, but the SL was placed `sl_distance` from the
  signal close. With a retrace deeper than the SL distance, a stop-out showed a
  profit (e.g. +1.5R). SL and TP are now measured from the fill price, so a
  stop-out is −1R. The fill bar is no longer skipped: an SL touch on the fill
  bar is a loss (the price must pass the limit before reaching the stop); TP
  is not checked on the fill bar, since the high may have come first.
- **`load_ohlcv` / `resample_ohlcv` returned wrong timestamps on pandas 3**
  (GitHub issue #1) — `astype(int64) // 10**9` assumed nanosecond datetimes.
  On pandas 3.0.1, `load_ohlcv` returned 1704067 instead of 1704067200 and
  `resample_ohlcv` returned 1 for every bar of 2001–2033. Both now convert any datetime resolution,
  and `load_ohlcv` also accepts numeric epoch seconds or milliseconds and
  date strings with UTC offsets.
- **Gate 2 passed when PF and RF came from different combos** — the best PF
  and the best RF were checked separately. A grid with one combo at PF 1.2 /
  RF 0.1 and another at PF 1.01 / RF 3 passed. Now at least one combo must meet
  both thresholds; `best_metric` is the best-PF combo among those.

- **Open entry held one bar more than `max_hold`** — with `open_prices` the
  entry bar counted as held but the timeout came `max_hold` bars after it, so
  a trade was held `max_hold + 1` bars (close entry: `max_hold`). Both now hold
  `max_hold` bars, so an open-entry timeout has `hold_bars = max_hold - 1`
  (it was `max_hold`). `max_hold < 1` now raises.
- **Custom exits filled at the close that produced the exit signal** — the
  same optimism as entering at the signal bar's close. With `open_prices` the
  exit now fills at the next bar's open.
- **A bar that opened beyond the TP still counted as SL** when its low later
  touched the SL. The open proves the TP filled first; it is now a TP (at the
  TP price). Applies with `open_prices`, to the TP in the rr, trailing and
  SAR modes (SAR targets keep SL priority).
- **`simulate_trades_hires` accepted timestamps in other units** — with
  milliseconds or `datetime64[ns]` the 59-minute look-ahead came back silently
  (found in review). `datetime64` is now converted; a bar spacing that does
  not match `signal_bar_minutes` / `exec_bar_minutes` raises (this also
  catches 5m data passed without `exec_bar_minutes=5`, which made `max_hold`
  five times too long). `retrace_timeout` is now converted from signal bars to
  execution bars like `max_hold`.

### Bug fixes (validation coverage)

- **Gate 0 never ran the trade-level BugGuard checks** — `gate0_validate()`
  took no trade data, so BG-06b, BG-08, BG-09b, BG-12 and BG-14 never ran
  inside the pipeline (only 6 of the 14 checks could run). It now takes
  `trades=` (a `TradeResults`), `sl_distances=`, `avg_price=` and
  `timestamps=`. NO_FILL trades are ignored; BG-12 (fixed cost) is inferred
  from `cost_r`; the entry mode for BG-09b is inferred from the trades when
  the metadata is lost (e.g. after `np.concatenate`). BG-08 is a WARN here:
  trades are simulated independently, so overlapping trades are not a
  same-bar re-entry bug. BG-01 is not run in Gate 0, since `simulate_trades`
  sets the entry bar itself and it cannot fail; BG-13 is not run either (no
  input for it). GK-00 lists every missing
  input.
- **`resolution_minutes` defaulted to 1** — a GateKeeper created without it
  passed BG-05 even for 1h fills. The default is now `None`, which means
  "same as `bar_minutes`", and GK-00 notes the assumption.
- **BG-10 always passed** — it was an empty stub. It now reports unsorted or
  duplicate timestamps (ERROR) and a last bar that is still forming (WARN),
  and runs when `timestamps=` is passed.
- **BG-04 missed other backward fills** — now also catches `.backfill(`,
  `fillna(method="backfill")` and `interpolate(limit_direction="backward"|"both")`.
- **BG-09 only matched `entry_price = ...Close...values[`** — lowercase and
  other spellings (`close[i]`, `closes[i]`, `df["close"].iloc[i]`) now give a
  WARN; next-bar indexing (`[i + 1]`) is not flagged. The original pattern is
  still an ERROR.

### New

- **Strategy audit** (`backtest_engine/audit.py`, `python -m backtest_engine.audit
  file.py [--json]`). `check_lookahead` rewrites the bars after a cut point
  four ways and re-runs the signal function; any change at or before the cut
  is look-ahead (half the cuts sit on or just before signal bars, so rare
  signals are tested; float noise is tolerated with `rtol`). `check_null_edge`
  runs the whole backtest on 120 driftless random walks whose bar extremes
  come from a Brownian bridge (fair stop/target fills), tests the mean R
  before costs with paths as clusters, and compares with the same entries
  under coin-flip directions (flipped inside `simulate_trades` in the
  auditing thread, SL/TP distances unchanged). On a labeled benchmark of 28
  strategies (`bench/`), the two together caught all 16 leaks (15 planted,
  1 found in a strategy written as a clean control) with no false alarm on
  12 clean ones; on real EURUSD 1h data the look-ahead
  verdicts were the same; `null_edge` flagged 3 of 360 runs of the clean
  strategies (all at one seed) and caught a leak affecting 20% of trades in
  18 of 20 runs. An LLM
  (Claude Sonnet) reading each strategy file caught 15 with 1 false alarm;
  it missed a leak inside an imported helper whose source it was not shown,
  which `check_null_edge` found. `audit_strategy` runs each audit in a fresh
  Python process (as the CLI does), so a strategy's helpers never meet
  another audit's modules or the caller's.
- `AGENTS.md`: how an AI assistant should backtest with this library (write
  `signals`/`backtest`, run the audit, do not report a failing result).
- `TradeResults` has a `signal_bar` field (same timeframe as `entry_bar`).
  For `simulate_trades_hires` it is the execution bar just before the entry
  bar (normally the last one inside the signal bar), so `entry_bar >
  signal_bar` for every filled trade.
- `DataQualityWarning`: emitted when bars open outside their own high/low
  (`simulate_trades` with `open_prices`), when signals close outside the
  execution data, or when a gap in the execution data delays an entry
  (`simulate_trades_hires`), and when a `fetch_aggvault` request window has
  no data. `simulate_trades_hires` now also rejects
  unsorted execution timestamps.
- `check_incomplete_bars` is exported.
- `results.quality` is set even with `preflight=False` (only the warning is
  suppressed). Previously it was `None`.
- `examples/rsi_reversal/compare_1h_vs_hires.py` reproduces the README table.
- `fetch_aggvault`: sends `limit=500000` explicitly (a server that rejects
  it fails loudly; one that silently caps a reply below a window's size can
  still cut the last window short), keeps requesting inside the range when a reply stops more
  than 5 days before its window end, warns about request windows with no
  data, paces requests across calls, and retries HTTP 503 as well as 429.

### Docs

- README: fill rules, hires timing, corrected hires numbers, Gate 0/2
  behavior, BugGuard check list (BG-01–BG-14), what BugGuard cannot detect
  (look-ahead inside signal code), fetch windowing, StressTest "combined"
  scenario values, test count. Removed the Grade C description (grades are A/B
  since v0.5.1) and the undefined `entry_costs`/unused `BrokerCost` in the
  hires example.
- `examples/gatekeeper_pipeline.py`: synthetic bars now open near the previous
  close and inside their own range (the old generator put opens outside
  high/low, which the gap rule turns into large losses); Gate 0 gets a
  simulation result.

### Tests

- 343 → 492. New: `tests/test_audit.py`, `tests/test_fills.py` (gap, retrace, TP-gap, hold window, custom exit fills),
  `tests/test_utils.py` (`load_ohlcv`, `resample_ohlcv`, `find_signal_bar`),
  hires timing, coverage and timestamp units, Gate 0 trade checks (including
  overlapping 1h trades, which must not block), Gate 2 same-combo rule,
  BG-04/09/10 patterns, windowed fetch against a fake server (holes, lower
  limits, 429/503).

## v0.5.1

Written in v0.6.0 from the commit history (2026-04-01 – 2026-04-27); this
version was not recorded at the time.

- `simulate_trades_hires()` for 1h signals with 1m fills (its entry timing was
  wrong until v0.6.0, see above).
- `fetch_aggvault()` (AggVault API), with 4h/1d timeframes, inclusive end
  date and symbol normalization. Examples moved from CSV paths to it.
- BG-14 (effectively no SL), `BrokerCost.pip_size` / `price_to_pips`.
- `strategy_builder` for time-based entry/exit signals; `rsi` indicator;
  RSI reversal tutorial (levels 1–4) and gold rollover examples.
- BG-05 downgraded from ERROR to WARN for 1h resolution.
- GROSS (no costs) is the default in examples; the pre-flight grade depends
  only on `open_prices` (A/B; Grade C removed).
- HTML report generator (`generate_report`), with escaping fixes and the
  equity curve sorted by exit bar.
- `matplotlib` added to dependencies (CI failure).

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
- **Cost registry mismatch** — the expected costs of the deprecated `broker=` path differed from the no-commission sample preset's `cost_prices()` for 10/29 instruments. Now delegates to the preset (renamed `BrokerCost.sample_wide_spread()` in v0.6.2) so only one cost registry exists.

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
- **EURCHF** added to the no-commission sample preset (renamed `BrokerCost.sample_wide_spread()` in v0.6.2; 28 FX → 29 pairs incl. XAUUSD).
- **`is_oos_ratio_raw`** — new key in `WalkForward.run()` output with docstring noting upward bias from asymmetric split lengths. `is_oos_ratio` retained as backward-compat alias.

### Deprecations (with backward-compatible wrappers)

- `prop_firm_check(daily_dd_limit=...)` → use `max_dd_limit=...` (the old parameter name implied per-day DD, but it controls max drawdown limit). Return dict: `max_dd_ok` is the primary key, `daily_dd_ok` retained as alias.

All deprecated APIs emit `DeprecationWarning` and continue to work.

### Other

- Example strategy updated to use `per_trade_cost()` instead of scalar `as_r()`.
- Test count: 123 → 140.

## v0.2.0

### Breaking changes

- The internal cost table and `cost_pips_to_price` removed from public API (`__all__`).
  Both were removed from `bug_guard` in v0.6.2 (unused since v0.4.0).

### New features

- **`BrokerCost.cost_prices()`** — returns `{instrument: total_RT_cost}` dict for all instruments.
- **`MonteCarloDD.prop_firm_check()`** — generic prop firm DD check with configurable daily/total DD limits and confidence level.
- **`check_cost_registry(expected_costs=...)`** — accepts any cost dict instead of hardcoded broker name.
- **`GateKeeper(expected_costs=...)`** — same generalization for the gate pipeline.
- **`open_prices`** support in all exit modes (rr, trailing, sar_trailing, custom).
- **`WalkForward.run()` returns `is_oos_ratio`** — ratio of OOS to IS performance.
- **GitHub Actions CI** — runs `pytest tests/ -v` on push/PR.

### Deprecations (with backward-compatible wrappers)

- The old prop-firm check method of `MonteCarloDD` — use `prop_firm_check()` instead.
- `run_all_checks(broker=...)` — use `expected_costs=BrokerCost(...).cost_prices()`.
- `GateKeeper(broker=...)` — use `expected_costs=BrokerCost(...).cost_prices()`.

All deprecated APIs emit `DeprecationWarning` and continue to work.

### Bug fixes

- Fixed duplicate `CADCHF` key in internal cost registry.

### Other

- `pyproject.toml`: added `license`, `readme`, `authors`, `classifiers`, `[project.urls]`.
- `.gitignore`: added Numba cache files (`*.nbi`, `*.nbc`).
- README rewritten with correct API examples.
- Test count: 98 → 123.

# AGENTS.md

Instructions for AI coding assistants (Claude, ChatGPT, Codex, Cursor, etc.) that use this repository to backtest a trading idea for a user. `CLAUDE.md` has the same rules in Japanese plus Claude-specific notes.

## Before you report any backtest result

1. Write the strategy as one Python file with two functions:

   ```python
   def signals(data):   # data: dict of numpy arrays time/open/high/low/close
       ...              # one value per bar: +1 long, -1 short, 0 none,
       return sig       # computed from that bar's close and earlier bars

   def backtest(data):  # the full pipeline on the same data
       ...              # signals -> simulate_trades(..., open_prices=data["open"])
       return results   # TradeResults
   ```

   Keep `signals` and `backtest` free of global state that depends on the data you fetched: the audit calls them on other data. Use the engine's `simulate_trades` / `simulate_trades_hires` inside `backtest` (the placebo flips directions there). Set `BAR_MINUTES` (and, for 1m data resampled inside the strategy, `NULL_BARS = 120_000` and `NULL_VOL = 0.00026`) as module constants when the defaults (60-minute bars, 5,000 bars) do not fit.

2. Run the audit:

   ```bash
   uv run python -m backtest_engine.audit my_strategy.py --json
   ```

   It runs three groups of tests and prints JSON (`overall`: `pass` / `warn` / `fail` / `error`; exit code 1 on `fail`, 2 on `error`):

   | Check | What it does | Catches |
   |---|---|---|
   | `lookahead` | Rewrites every bar after a cut point (mirror, 2× moves, ±20–25% jump) and re-runs `signals`. A signal at or before the cut must not change. | Leaks that change signals: `shift(-1)`, centered windows, full-sample mean/std/min/max/quantile, `bfill`, zero-phase filters (`filtfilt`), FFT over the sample, pivots confirmed by later bars, the current (unfinished) higher-timeframe bar, models fitted on the whole sample. Probabilistic |
   | `null_edge` | Runs `backtest` on 120 driftless random walks, where no strategy can make money, and compares with the same entries under coin-flip directions. | Entries before their signal, exits or targets computed from later bars, look-ahead in library code; a backtest biased by the SL-first rule (`warn`) |
   | `source:BG-04/07/09` | BugGuard source patterns | Some backward-fill and full-period quantile patterns, entry at close (pattern matching only) |

3. Report the audit with the result:
   - `fail` on `lookahead`: the backtest is invalid. Do not report its PF or win rate as a result. Show the user the `summary` (it names the likely cause from which bars changed), fix it, and run the audit again.
   - `fail` on `null_edge`: treat it as a problem and look for it: the summary says whether the strategy makes money on random data (look-ahead) or differs from coin-flip directions (signals misaligned in time). A clean strategy fails in under 1% of runs (3 of 360 on the benchmark). Running once more with `--seed 1` (no shared random paths) helps: a second fail is conclusive; a pass after a fail is usually chance, but a weak leak (affecting a few trades) can also pass one run, so check the code at the place the summary points to before reporting.
   - `warn` on `null_edge`: the SL-first rule (a bar touching both SL and TP counts as SL) makes this backtest pessimistic. Re-run with `simulate_trades_hires` (1m fills) before reporting. (Only costs passed as `entry_costs` are added back before the test; costs subtracted from `pnl_r` by hand also show up as this loss.)
   - `error`: the test could not run (too few signals or trades, a load error). Fix the input; do not treat it as a pass.
   - `pass`: say what was tested. Passing does not prove the absence of bias (survivorship, a leak that never changes a signal, or data errors are not covered).

## Other rules

- Use `simulate_trades_hires()` (1h signals, 1m fills) when the strategy is on 1h or coarser bars. Fetch 1m data with `fetch_aggvault`; long ranges are split into requests automatically (5 years of 1m ≈ 1 minute).
- Pass `open_prices` to `simulate_trades` (next-bar-open entry). Without it, entry is at the signal bar's close (optimistic).
- Run `GateKeeper` (with `bar_minutes` and `resolution_minutes`; `resolution_minutes=1` for `simulate_trades_hires`) and pass the simulation result to Gate 0: `gk.gate0_validate(trades=results, sl_distances=sl, avg_price=float(np.median(close)))`. Gate 0 checks trade count, overlap, missing SL, open-price entry and fixed costs. It does not look for look-ahead in signal code; that is what the audit is for.
- State whether results are GROSS (no costs) or NET (`entry_costs` from `BrokerCost`).
- Interpret PF: < 1.0 losing; 1.0–1.1 weak, likely gone after costs; 1.1–1.5 a possible edge, validate with GateKeeper; above 1.5 suspect look-ahead or overfitting first.
- Timestamps are bar open times in UTC (Unix seconds).

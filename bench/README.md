# Audit benchmark

How well do the checks in `backtest_engine.audit` find look-ahead bias? 28 small strategy files with known answers: 16 with a leak (15 planted, and s05, written as a clean control, which turned out to leak; see Limits), 12 without. File names (`s01.py` …) carry no hint, but a few comments say what a line does (s08: `# enter at the open of the signal bar itself`; s11: `# keep one value in four, then fill`), which may have helped the LLM. The answers are in `labels.json`.

```bash
uv run python bench/run_bench.py      # all checks on all cases (~1 min), writes results.json
uv run python bench/robustness.py     # other seeds, and real EURUSD 1h if AGGVAULT_KEY is set (without it, robustness.json is rewritten without the real-data verdicts)
uv run python bench/false_alarms.py   # null_edge on the clean cases, 30 seeds each (~7 min)
uv run python bench/weak_leaks.py     # null_edge against a leak in only some trades
uv run python bench/score_llm.py      # scores the saved LLM answers in llm/
```

## Results (2026-11-22, v0.6.0)

A check "catches" a case when its verdict is `fail` or `warn`.

| Check | Leaky cases caught (of 16) | False alarms (of 12 clean) |
|---|---|---|
| BugGuard source patterns (BG-04/07/09) | 2 | 0 |
| `lookahead` (rewrite the future, re-run `signals`) | 12 | 0 |
| `null_edge` (backtest on random walks, vs. coin-flip directions) | 12 | 0 |
| Either audit check | **16** | **0** |
| LLM reading the strategy file (Claude Code `--model sonnet`, `run_llm.sh`; the model version was not recorded) | 15 | 1 |

s24 and s25 define only `backtest()`, so `lookahead` does not run on them: it caught 12 of the 15 leaky cases it ran on, with no false alarm on 11 clean ones.

What each check missed, and why:

- `lookahead` cannot see leaks in `backtest()` (an entry one bar early, an exit that uses the next bar, a target computed from later highs) or in library code, since it only re-runs `signals()`: s03, s07, s08, s24.
- `null_edge` misses leaks that do not create an edge on random data or tilt the directions: a full-sample z-score, a full-sample volatility quantile, a backward fill, and a threshold computed from a window that includes later bars (s22, s20, s11, s05). These still distort a real backtest.
- The LLM caught every leak that is visible in the strategy file, including the one in s05. It missed s24, whose strategy file is correct and whose leak is inside an imported helper (`cases/hires_lib.py`, a copy of `simulate_trades_hires` before v0.6.0, which entered 59 minutes early). `null_edge` finds it without reading any code: −0.21R per trade before costs on random walks (t = −12.2), against +0.009R for the same entries with coin-flip directions; the fixed engine (s25) gives −0.002R (t = −0.1).
- The LLM's false alarm is s28 (clean): its JSON said `true` while its own reason said no look-ahead was found.

Conditions of the LLM comparison, which matter for reading it:

- One answer per case, from `bench/llm/prompt.txt` plus the strategy file, no tools.
- The prompt describes `simulate_trades_hires` as the library documented it ("entering after the signal bar closes"). s24 imports a local function with the same name and its source was not shown, as with an installed library an assistant does not read. Given the helper's source too, the LLM found the leak (`llm/s24_with_lib.sonnet.json`).
- The prompt says custom exits fill at the close of the exit-signal bar; that was the engine's behavior when the first 25 answers were collected (it fills at the next open since v0.6.0). Only s03 uses custom exits; the LLM flagged it. s26–s28 were added later and asked with the same prompt.
- For s21 the reply first gave a false alarm, then corrected itself; the last JSON object in each reply is scored (`score_llm.py`).

False alarms of `null_edge` (`false_alarms.json`), with the defaults (120 paths, |t| > 3.5): 3 of 360 runs (each of the 12 clean cases at 30 seeds) failed, all three at the same seed (1004). At a given seed every case with the default `NULL_BARS` runs on the same 120 random walks; those happened to revert slightly more than usual, which favoured the mean-reversion case (s28: t = +3.65 on random data, +3.87 against its placebo) and penalized two trend-following ones (s09, s18). No single path drove it (the largest path carried 6% of the variance of the comparison statistics, and 11% of s09's raw statistic). Different seeds share no path, so a clean strategy almost never fails twice.

Weak leaks (`weak_leaks.json`): an RSI strategy where a fraction of the trades enters one bar early (at the open of its signal bar), 20 seeds each:

| Trades entering early | 30 paths | 120 paths (default) |
|---|---|---|
| 0% (clean) | 0/20 fail | 0/20 fail |
| 20% | 4/20 fail | 18/20 fail |
| 50% | 20/20 fail | 20/20 fail |

Robustness (`robustness.json`): on real EURUSD 1h data (2024) instead of a random walk, the `lookahead` verdicts are the same for every case (no false alarm on the 11 clean cases that have `signals`). Over 5 other seeds, every signal-level leak failed 5/5 times except s05, whose leak only affects bars 500–749 (4/5). Each of the 12 leaky cases that `null_edge` catches was also flagged (fail or warn; the file does not tell them apart) at 3 more seeds.

## Limits

- The cases are small and each has one leak in plain sight; real strategies are longer and leak in less obvious ways. The LLM's 15/16 here says little about a 2,000-line strategy.
- The labels were written by the same author as the checks (with Claude). One case written as a clean control, s05, turned out to have a real look-ahead, which the `lookahead` check found; it is kept as a leaky case and a fixed version was added (s23). s26–s28 (clean) were added after a review found that an earlier placebo, which flipped the signals before `backtest()`, failed strategies whose stops depend on the direction.
- Passing both checks does not prove a strategy has no bias. Leaks that never flip a signal, survivorship in the instrument list, and bad data are not covered.
- The random-walk null has no sessions, volatility regimes or real volume, so strategies that only trade under real market structure may produce too few trades to test (`error`).

## Files

- `cases/sNN.py` — strategies (`signals(data)` and `backtest(data)`); `cases/hires_lib.py` is the helper imported by s24.
- `labels.json` — answers: leak or not, the kind of leak, and where it sits (signal, execution, library).
- `results.json`, `robustness.json`, `false_alarms.json`, `weak_leaks.json` — verdicts from the runs above.
- `llm/prompt.txt`, `llm/*.sonnet.json`, `llm/summary.json` — the LLM prompt, raw answers and scores.

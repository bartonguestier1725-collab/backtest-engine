# Audit benchmark

How well do the checks in `backtest_engine.audit` find look-ahead bias? 25 small strategy files with known answers: 16 with a planted leak, 9 without. File names (`s01.py` …) and code carry no hint; the answers are in `labels.json`.

```bash
uv run python bench/run_bench.py      # all checks on all cases (~20 s), writes results.json
uv run python bench/robustness.py     # other seeds, and real EURUSD 1h if AGGVAULT_KEY is set
uv run python bench/false_alarms.py   # null_edge on the clean cases, 10 seeds each
uv run python bench/score_llm.py      # scores the saved LLM answers in llm/
```

## Results (2026-11-22, v0.6.0)

A check "catches" a case when its verdict is `fail` or `warn`.

| Check | Leaky cases caught (of 16) | False alarms (of 9 clean) |
|---|---|---|
| BugGuard source patterns (BG-04/07/09) | 2 | 0 |
| `lookahead` (rewrite the future, re-run `signals`) | 12 | 0 |
| `null_edge` (backtest on random walks, vs. coin-flip directions) | 12 | 0 |
| Either audit check | **16** | **0** |
| LLM reading the strategy file (Claude Sonnet, `run_llm.sh`) | 15 | 0 |

What each check missed, and why:

- `lookahead` cannot see leaks in `backtest()` (an entry one bar early, an exit that uses the next bar, a target computed from later highs) or in library code, since it only re-runs `signals()`: s03, s07, s08, s24.
- `null_edge` misses leaks that do not create an edge on random data or tilt the directions: a full-sample z-score, a full-sample volatility quantile, a backward fill, and a threshold computed from a window that includes later bars (s05, s11, s20, s22). These still distort a real backtest.
- The LLM caught every leak that is visible in the strategy file, including the one in s05. It missed s24, whose strategy file is correct and whose leak is inside an imported helper (`cases/hires_lib.py`, a copy of `simulate_trades_hires` before v0.6.0, which entered 59 minutes early). `null_edge` finds it without reading any code: −0.16R per trade before costs on random walks (t = −4.5), against +0.08R (t = 1.9) for the fixed engine (s25).

Conditions of the LLM comparison, which matter for reading it:

- One answer per case, from `bench/llm/prompt.txt` plus the strategy file, no tools.
- The prompt describes `simulate_trades_hires` as the library documented it ("entering after the signal bar closes"). s24 imports a local function with the same name and its source was not shown, as with an installed library an assistant does not read. Given the helper's source too, the LLM found the leak (`llm/s24_with_lib.sonnet.json`).
- The prompt says custom exits fill at the close of the exit-signal bar; that was the engine's behavior when the answers were collected (it fills at the next open since v0.6.0). Only s03 uses custom exits; the LLM flagged it.
- For s21 the reply first gave a false alarm, then corrected itself; the last JSON object in each reply is scored (`score_llm.py`).

False alarms of `null_edge` (`false_alarms.json`): it uses |t| > 3 on two statistics, so a clean strategy is flagged now and then. Over 10 seeds for each of the 9 clean cases, it flagged 1 of 90 runs (s09 at one seed). A real leak fails again with another seed: each of the 12 leaky cases that `null_edge` catches failed at all 3 extra seeds (`robustness.json`).

Robustness (`robustness.json`): on real EURUSD 1h data (2024) instead of a random walk, the `lookahead` verdicts are the same for every case (no false alarm on the 8 clean cases that have `signals`). Over 5 other seeds, every signal-level leak failed 5/5 times except s05, whose leak only affects bars 500–749 (4/5).

## Limits

- The cases are small and each has one leak in plain sight; real strategies are longer and leak in less obvious ways. The LLM's 15/16 here says little about a 2,000-line strategy.
- The labels were written by the same author as the checks (with Claude). One case written as a clean control, s05, turned out to have a real look-ahead, which the `lookahead` check found; it is kept as a leaky case and a fixed version was added (s23).
- Passing both checks does not prove a strategy has no bias. Leaks that never flip a signal, survivorship in the instrument list, and bad data are not covered.
- The random-walk null has no sessions, volatility regimes or real volume, so strategies that only trade under real market structure may produce too few trades to test (`error`).

## Files

- `cases/sNN.py` — strategies (`signals(data)` and `backtest(data)`); `cases/hires_lib.py` is the helper imported by s24.
- `labels.json` — answers: leak or not, the kind of leak, and where it sits (signal, execution, library).
- `results.json`, `robustness.json`, `false_alarms.json` — per-case verdicts from the runs above.
- `llm/prompt.txt`, `llm/*.sonnet.json`, `llm/summary.json` — the LLM prompt, raw answers and scores.

"""RSI Reversal Strategy — Level 3: Overfitting Detection.

Level 2 found RSI(21) 20/70 with PF=1.37 (in-sample, 1h fills, GROSS) via
parameter optimization.
Is this real edge or just curve-fitting to 2023's range market?

This level uses:
  1. BugGuard — catches implementation bugs (1h resolution warning, cost check)
  2. Walk-Forward Analysis (WFA) — does the best IS param work on unseen OOS data?
  3. CSCV (Combinatorially Symmetric Cross-Validation) — what's the probability
     that the "best" params are actually overfitting? (Bailey 2015)
  4. Monte Carlo — stress test the best params
"""

import itertools
import numpy as np
import pandas as pd
from backtest_engine import (
    fetch_aggvault, simulate_trades,
    LONG, SHORT, rsi, atr,
    WalkForward, CSCV, MonteCarloDD, StressTest,
    bug_guard,
)

# ── Step 1: データ取得 ────────────────────────────────────────────────
timestamps, opens, highs, lows, closes, _ = fetch_aggvault(
    "EURUSD", "1h", "2021-04-01", "2026-03-31",
)
dt_index = pd.to_datetime(timestamps, unit="s", utc=True)
n_bars = len(timestamps)
print(f"Loaded: {n_bars} bars, {dt_index[0]} → {dt_index[-1]}")

# ── Pre-compute ──────────────────────────────────────────────────────
atr_vals = atr(highs, lows, closes, 14)


# ── evaluate_fn: WFA / CSCV が呼ぶ関数 ───────────────────────────────
# シグネチャ: evaluate_fn(params, start_bar, end_bar) → float
def evaluate_fn(params: dict, start_idx: int = 0, end_idx: int = 0) -> float:
    """Run a single RSI backtest. Returns total_r (float) for WFA/CSCV."""
    if end_idx == 0:
        end_idx = n_bars

    rsi_period = params.get("rsi_period", 14)
    oversold = params.get("oversold", 30)
    overbought = params.get("overbought", 70)
    sl_mult = params.get("sl_atr_mult", 2.0)
    rr_ratio = params.get("rr_ratio", 2.0)
    max_hold = params.get("max_hold", 48)

    rsi_vals = rsi(closes, rsi_period)
    warmup = rsi_period + 1

    signal_bars = []
    directions = []
    for j in range(max(warmup, start_idx), min(end_idx, len(closes) - 1)):
        if np.isnan(rsi_vals[j]) or np.isnan(atr_vals[j]):
            continue
        if rsi_vals[j] < oversold:
            signal_bars.append(j)
            directions.append(LONG)
        elif rsi_vals[j] > overbought:
            signal_bars.append(j)
            directions.append(SHORT)

    # A window without trades scores 0R. Do not return a placeholder such as
    # -999 for "too few trades": WFA averages it into the OOS mean, and CSCV
    # sums it per window, which ranks parameter sets by how often they trade
    # instead of by how well.
    if len(signal_bars) == 0:
        return 0.0

    signal_bars = np.array(signal_bars, dtype=np.int32)
    directions = np.array(directions, dtype=np.int8)
    sl_distances = atr_vals[signal_bars] * sl_mult
    tp_distances = sl_distances * rr_ratio
    res = simulate_trades(
        highs, lows, closes,
        signal_bars, directions, sl_distances, tp_distances,
        max_hold=max_hold, exit_mode="rr",
        open_prices=opens, preflight=False,
    )

    return float(np.sum(res["pnl_r"]))


def run_full(params: dict) -> dict | None:
    """Run backtest on full data — for Monte Carlo / StressTest."""
    rsi_period = params.get("rsi_period", 14)
    rsi_vals = rsi(closes, rsi_period)
    warmup = rsi_period + 1

    signal_bars = []
    directions = []
    for j in range(warmup, len(closes) - 1):
        if np.isnan(rsi_vals[j]) or np.isnan(atr_vals[j]):
            continue
        if rsi_vals[j] < params.get("oversold", 30):
            signal_bars.append(j)
            directions.append(LONG)
        elif rsi_vals[j] > params.get("overbought", 70):
            signal_bars.append(j)
            directions.append(SHORT)

    signal_bars = np.array(signal_bars, dtype=np.int32)
    directions = np.array(directions, dtype=np.int8)
    sl_distances = atr_vals[signal_bars] * params.get("sl_atr_mult", 2.0)
    tp_distances = sl_distances * params.get("rr_ratio", 2.0)

    res = simulate_trades(
        highs, lows, closes,
        signal_bars, directions, sl_distances, tp_distances,
        max_hold=params.get("max_hold", 48), exit_mode="rr",
        open_prices=opens, preflight=False,
    )
    if len(res["pnl_r"]) < 10:
        return None
    return {"pnl_r": res["pnl_r"], "pf": res.profit_factor}


def per_bar_ratio(wfa: dict) -> float:
    """OOS/IS of R per bar. The IS and OOS windows differ in length (70/30),
    so the raw 'is_oos_ratio' of window totals is about 0.43 even when the
    edge carries over perfectly."""
    is_len = np.mean([r["is_range"][1] - r["is_range"][0] for r in wfa["splits"]])
    oos_len = np.mean([r["oos_range"][1] - r["oos_range"][0] for r in wfa["splits"]])
    if wfa["is_mean"] <= 0:
        return float("nan")  # no in-sample edge to carry over
    return (wfa["oos_mean"] / oos_len) / (wfa["is_mean"] / is_len)

# ── Step 2: BugGuard (教育目的で実行、ブロックはしない) ───────────────
print("\n" + "=" * 70)
print("Step 2: BugGuard — Implementation Checks")
print("=" * 70)
print("Note: BugGuard catches real issues. These are limitations we")
print("acknowledge for this tutorial (1h bars, not 1m).\n")

bug_guard(
    resolution_minutes=60,
    n_bars=n_bars,
    bar_minutes=60,
    n_trades=943,  # From Level 2 best
    # No costs — GROSS edge testing on aggregated data
    source_path=__file__,
    strict=False,
)

print("\n  → BG-05 (1h resolution) is a real limitation.")
print("    For production: re-run on 1m bars. For this tutorial: proceed.")

# ── Step 3: パラメータグリッド (WFA / CSCV 用) ────────────────────────
param_grid = [
    {"rsi_period": p, "oversold": os_val, "overbought": ob_val,
     "sl_atr_mult": sl, "rr_ratio": rr, "max_hold": 48}
    for p in [7, 14, 21]
    for os_val in [20, 25, 30]
    for ob_val in [65, 70, 75]
    for sl in [2.0, 2.5]
    for rr in [2.0, 3.0]
]
print(f"\nParameter grid: {len(param_grid)} combinations")

# ── Step 4: Walk-Forward Analysis ────────────────────────────────────
print("\n" + "=" * 70)
print("Step 4: Walk-Forward Analysis (5 splits, 70/30 IS/OOS)")
print("=" * 70)
print("  Question: Does the best in-sample param work on unseen data?\n")

wf = WalkForward(n_bars=n_bars, is_ratio=0.7, n_splits=5, anchored=False)
wfa_result = wf.run(param_grid, evaluate_fn)

print(f"  OOS mean PnL:     {wfa_result['oos_mean']:.3f}R per OOS window (total R)")
print(f"  OOS positive frac: {wfa_result['oos_positive_frac']:.1%} of splits profitable")
ratio = per_bar_ratio(wfa_result)  # 1.0 = the IS edge per bar carries over
print(f"  OOS/IS ratio:      {ratio:.2f} (R per bar)")

if np.isnan(ratio):
    print("  ✗ The best in-sample params do not make money in-sample: no ratio to judge")
elif ratio < 0.3:
    print(f"  ✗ OOS/IS ratio {ratio:.2f} — OOS is {ratio:.0%} of IS. Severe overfitting.")
elif ratio < 0.7:
    print(f"  ⚠ OOS/IS ratio {ratio:.2f} — OOS degrades significantly vs IS")
else:
    print(f"  ✓ OOS/IS ratio {ratio:.2f} — performance transfers to OOS")

# ── Step 5: CSCV — Probability of Backtest Overfitting ────────────────
print("\n" + "=" * 70)
print("Step 5: CSCV — Probability of Backtest Overfitting (Bailey 2015)")
print("=" * 70)
print("  Question: In what fraction of IS/OOS splits does the best IS param")
print("  rank in the bottom half on the unseen half?\n")

cscv = CSCV(n_splits=10)
cscv_result = cscv.run(param_grid, evaluate_fn, n_bars=n_bars)

pbo = cscv_result['pbo']
print(f"  PBO:           {pbo:.1%}")
print(f"  Logit mean:    {cscv_result['logit_mean']:.2f}")
print(f"  Combinations:  {cscv_result['n_combinations']}")

if pbo > 0.40:
    print(f"\n  ✗ PBO = {pbo:.0%} > 40% — HIGH probability of overfitting")
    print(f"    The 'best' params from Level 2 are likely curve-fit to historical data.")
elif pbo > 0.20:
    print(f"\n  ⚠ PBO = {pbo:.0%} — moderate overfitting risk")
else:
    print(f"\n  ✓ PBO = {pbo:.0%} — low overfitting probability")

# ── Step 6: Monte Carlo + StressTest ─────────────────────────────────
print("\n" + "=" * 70)
print("Step 6: Monte Carlo & Stress Test (Level 2 best params)")
print("=" * 70)

best_result = run_full({
    "rsi_period": 21, "oversold": 20, "overbought": 70,
    "sl_atr_mult": 2.5, "rr_ratio": 3.0, "max_hold": 48,
})
best_pf = best_result["pf"] if best_result is not None else float("nan")

if best_result is not None:
    pnl = best_result["pnl_r"]

    # Monte Carlo
    mc = MonteCarloDD(pnl, n_sims=10_000, risk_pct=0.01, seed=42)
    mc.run()
    print(f"\n  Monte Carlo ({len(pnl)} trades, 10K shuffles):")
    print(f"    DD 50th: {mc.dd_percentile(50)*100:.1f}%")
    print(f"    DD 95th: {mc.dd_percentile(95)*100:.1f}%")
    print(f"    DD 99th: {mc.dd_percentile(99)*100:.1f}%")
    print(f"    Kelly:   {mc.kelly_fraction()*100:.1f}%")

    # Prop firm check (Fintokei: 4% daily, 8% total)
    prop = mc.prop_firm_check(max_dd_limit=0.04, total_dd_limit=0.08, confidence=95.0)
    print(f"\n  Prop firm check (4%/8% DD limits):")
    print(f"    Pass: {prop['pass']}")
    print(f"    Max DD OK: {prop['max_dd_ok']}, Total DD OK: {prop['total_dd_ok']}")

    # StressTest
    st = StressTest(pnl, n_sims=1000, seed=42)
    report = st.run_all(block_size=10)
    print(f"\n  Stress Test:")
    print(f"    Baseline DD@95%:       {report['baseline']['dd_95']*100:.1f}%")
    print(f"    Block bootstrap DD@95%: {report['block_bootstrap']['dd_95']*100:.1f}%")
    print(f"    WR -5% DD@95%:         {report['degraded']['wr_minus5']['dd_95']*100:.1f}%")
    print(f"    RR 80% DD@95%:         {report['degraded']['rr_80pct']['dd_95']*100:.1f}%")
    print(f"    Combined DD@95%:       {report['degraded']['combined']['dd_95']*100:.1f}%")

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("Level 3 Summary")
print(f"{'='*70}")
print(f"""
  Level 1: RSI(14) 30/70 → PF≈1.10 GROSS (1m fills), carried by 2023 alone
  Level 2: Optimized to RSI(21) 20/70 → PF={best_pf:.2f} in-sample (looks good!)
  Level 3: WFA + CSCV reveal the truth:
    - PBO: {pbo:.0%} — {'HIGH overfitting probability' if pbo > 0.40 else 'moderate risk' if pbo > 0.20 else 'low risk'}
    - OOS/IS ratio (R per bar): {ratio:.2f} — {'no in-sample edge' if np.isnan(ratio) else 'severe overfitting' if ratio < 0.3 else 'significant degradation' if ratio < 0.7 else 'transfers well'}

  Conclusion: RSI reversal alone does not produce a robust edge on EURUSD.
  The Level 2 "improvement" was mostly fitting to 2023's range market.

  → Level 4: Can structural filters (trend, time-of-day) create genuine edge?
""")

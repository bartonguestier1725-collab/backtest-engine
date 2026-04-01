"""RSI Reversal Strategy — Level 2: Parameter Optimization.

Level 1 showed RSI(14) 30/70 has almost no edge (PF=1.05).
Can we find better parameters?

Grid search over:
  - RSI period:      7, 10, 14, 21
  - Oversold (buy):  20, 25, 30, 35
  - Overbought (sell): 65, 70, 75, 80
  - SL (ATR mult):   1.5, 2.0, 2.5
  - RR ratio:        1.5, 2.0, 3.0
  - Max hold (bars): 24, 48

Total: 4 × 4 × 4 × 3 × 3 × 2 = 1,152 combinations.

Spoiler: We'll find params that look great in-sample.
         Level 3 will test whether it's real or overfit.
"""

import itertools
import numpy as np
import pandas as pd
from backtest_engine import (
    fetch_aggvault, simulate_trades,
    LONG, SHORT, rsi, atr,
)

# ── Step 1: データ取得 ────────────────────────────────────────────────
timestamps, opens, highs, lows, closes, _ = fetch_aggvault(
    "EURUSD", "1h", "2021-04-01", "2026-03-31",
)
dt_index = pd.to_datetime(timestamps, unit="s", utc=True)
print(f"Loaded: {len(timestamps)} bars, {dt_index[0]} → {dt_index[-1]}")

# ── Step 2: パラメータグリッド ────────────────────────────────────────
param_grid = {
    "rsi_period":    [7, 10, 14, 21],
    "oversold":      [20, 25, 30, 35],
    "overbought":    [65, 70, 75, 80],
    "sl_atr_mult":   [1.5, 2.0, 2.5],
    "rr_ratio":      [1.5, 2.0, 3.0],
    "max_hold":      [24, 48],
}

keys = list(param_grid.keys())
combos = list(itertools.product(*param_grid.values()))
print(f"\nGrid: {len(combos)} combinations")

# ── Pre-compute ATR (period 14 for all — SL scaling only) ────────────
atr_vals = atr(highs, lows, closes, 14)

# ── Step 3: Grid search (GROSS — no execution costs) ─────────────────
results_list = []

for i, combo in enumerate(combos):
    params = dict(zip(keys, combo))

    # Compute RSI with this period
    rsi_vals = rsi(closes, params["rsi_period"])

    # Generate signals
    warmup = params["rsi_period"] + 1
    signal_bars = []
    directions = []

    for j in range(warmup, len(closes) - 1):
        if np.isnan(rsi_vals[j]) or np.isnan(atr_vals[j]):
            continue
        if rsi_vals[j] < params["oversold"]:
            signal_bars.append(j)
            directions.append(LONG)
        elif rsi_vals[j] > params["overbought"]:
            signal_bars.append(j)
            directions.append(SHORT)

    if len(signal_bars) < 30:
        continue

    signal_bars = np.array(signal_bars, dtype=np.int32)
    directions = np.array(directions, dtype=np.int8)

    # SL/TP
    sl_distances = atr_vals[signal_bars] * params["sl_atr_mult"]
    tp_distances = sl_distances * params["rr_ratio"]

    # Simulate (GROSS — no costs)
    res = simulate_trades(
        highs, lows, closes,
        signal_bars, directions, sl_distances, tp_distances,
        max_hold=params["max_hold"],
        exit_mode="rr",
        open_prices=opens,
        preflight=False,
    )

    pnl = res["pnl_r"]
    if len(pnl) < 30:
        continue

    pf = res.profit_factor
    wr = res.win_rate
    exp = res.expectancy_r
    dd = res.max_drawdown_r
    total_r = float(np.sum(pnl))

    results_list.append({
        **params,
        "n_trades": len(pnl),
        "pf": pf,
        "win_rate": wr,
        "expectancy_r": exp,
        "total_r": total_r,
        "max_dd_r": dd,
        "sharpe_r": res.sharpe_r,
        "recovery": res.recovery_factor,
    })

    if (i + 1) % 200 == 0:
        print(f"  {i + 1}/{len(combos)} done...")

print(f"\nCompleted: {len(results_list)} valid combinations (≥30 trades)")

# ── Step 4: Top 10 by Profit Factor ──────────────────────────────────
df_results = pd.DataFrame(results_list)
top10 = df_results.nlargest(10, "pf")

print(f"\n{'='*90}")
print(f"Top 10 by Profit Factor")
print(f"{'='*90}")
print(f"{'RSI':>5} {'OS':>4} {'OB':>4} {'SL':>5} {'RR':>4} {'Hold':>4} | "
      f"{'PF':>5} {'WR':>5} {'Exp':>6} {'TotalR':>8} {'MaxDD':>7} {'n':>5}")
print(f"{'-'*90}")

for _, row in top10.iterrows():
    print(f"{row['rsi_period']:>5} {row['oversold']:>4} {row['overbought']:>4} "
          f"{row['sl_atr_mult']:>5.1f} {row['rr_ratio']:>4.1f} {row['max_hold']:>4} | "
          f"{row['pf']:>5.2f} {row['win_rate']:>4.0%} {row['expectancy_r']:>+6.3f} "
          f"{row['total_r']:>+8.1f} {row['max_dd_r']:>7.1f} {row['n_trades']:>5.0f}")

# ── Step 5: Best params — detailed results ───────────────────────────
best = top10.iloc[0]
print(f"\n{'='*90}")
print(f"Best params: RSI({int(best['rsi_period'])}) {int(best['oversold'])}/{int(best['overbought'])}, "
      f"SL={best['sl_atr_mult']:.1f}×ATR, RR={best['rr_ratio']:.1f}, Hold={int(best['max_hold'])}h")
print(f"{'='*90}")

# Re-run best params for detailed output
best_rsi = rsi(closes, int(best["rsi_period"]))
signal_bars = []
directions = []
warmup = int(best["rsi_period"]) + 1

for j in range(warmup, len(closes) - 1):
    if np.isnan(best_rsi[j]) or np.isnan(atr_vals[j]):
        continue
    if best_rsi[j] < best["oversold"]:
        signal_bars.append(j)
        directions.append(LONG)
    elif best_rsi[j] > best["overbought"]:
        signal_bars.append(j)
        directions.append(SHORT)

signal_bars = np.array(signal_bars, dtype=np.int32)
directions = np.array(directions, dtype=np.int8)
sl_distances = atr_vals[signal_bars] * best["sl_atr_mult"]
tp_distances = sl_distances * best["rr_ratio"]

best_results = simulate_trades(
    highs, lows, closes,
    signal_bars, directions, sl_distances, tp_distances,
    max_hold=int(best["max_hold"]),
    exit_mode="rr",
    open_prices=opens,
)

pnl = best_results["pnl_r"]
print(f"Quality Grade: {best_results.quality.grade}")
print(f"Trades:        {len(pnl)}")
print(f"Profit Factor: {best_results.profit_factor:.2f}")
print(f"Win Rate:      {best_results.win_rate:.1%}")
print(f"Expectancy:    {best_results.expectancy_r:.3f}R")
print(f"Sharpe (R):    {best_results.sharpe_r:.2f}")
print(f"Sortino (R):   {best_results.sortino_r:.2f}")
print(f"Max DD:        {best_results.max_drawdown_r:.1f}R")
print(f"Recovery:      {best_results.recovery_factor:.2f}")
print(f"Cost:          {best_results.cost_label}")

# Yearly breakdown
entry_bars_arr = best_results["entry_bar"]
print(f"\n{'Year':>6} {'PnL(R)':>10} {'n':>5} {'WR':>6} {'PF':>6}")
for year in range(2021, 2027):
    year_mask = np.array([dt_index[b].year == year for b in entry_bars_arr])
    if not np.any(year_mask):
        continue
    y_pnl = pnl[year_mask]
    y_total = np.sum(y_pnl)
    y_wr = (y_pnl > 0).mean()
    y_wins = y_pnl[y_pnl > 0]
    y_losses = y_pnl[y_pnl < 0]
    y_pf = np.sum(y_wins) / abs(np.sum(y_losses)) if len(y_losses) > 0 and np.sum(y_losses) != 0 else float('inf')
    print(f"  {year}  {y_total:>+9.1f}R  {len(y_pnl):>4}  {y_wr:>5.0%}  {y_pf:>5.2f}")

# ── Distribution analysis ────────────────────────────────────────────
print(f"\n{'='*90}")
print("Parameter distribution in Top 10")
print(f"{'='*90}")
for key in keys:
    counts = top10[key].value_counts().sort_index()
    print(f"  {key:>15}: {dict(counts)}")

# ── Comparison: Level 1 vs Level 2 ───────────────────────────────────
print(f"\n{'='*90}")
print(f"Level 1 (RSI 14/30/70) vs Level 2 (optimized)")
print(f"{'='*90}")
print(f"  {'':>20} {'Level 1':>10} {'Level 2':>10}")
print(f"  {'PF':>20} {'1.05':>10} {best_results.profit_factor:>10.2f}")
print(f"  {'Expectancy':>20} {'0.028R':>10} {best_results.expectancy_r:>+10.3f}R")
print(f"  {'Sharpe':>20} {'0.02':>10} {best_results.sharpe_r:>10.2f}")
print(f"  {'Trades':>20} {'3550':>10} {len(pnl):>10}")

print(f"\n{'='*90}")
print("Looks promising? → Level 3 will test if this is real or overfit.")
print(f"{'='*90}")

"""RSI Reversal Strategy — Level 1 (API版).

新規ユーザーの初回体験を想定:
  1. AggVault APIからEURUSD 1hデータを取得
  2. RSI(14) 30/70 逆張りを5年バックテスト
  3. 結果: エッジはあるか？

前提:
  pip install git+ssh://git@github.com/bartonguestier1725-collab/backtest-engine.git
  export AGGVAULT_KEY=tk_あなたのキー
"""

import numpy as np
import pandas as pd
from backtest_engine import (
    fetch_aggvault, simulate_trades,
    LONG, SHORT, EXIT_SL, EXIT_TP, EXIT_TIME,
    rsi, atr, BrokerCost,
)

# ── Step 1: AggVault APIからデータ取得 ────────────────────────────────
# load_ohlcv() と同じ (timestamps, opens, highs, lows, closes, volume) を返す
timestamps, opens, highs, lows, closes, _ = fetch_aggvault(
    "EURUSD", "1h", "2021-04-01", "2026-03-31",
)

dt_index = pd.to_datetime(timestamps, unit="s", utc=True)
print(f"Loaded: {len(timestamps)} bars, {dt_index[0]} → {dt_index[-1]}")
print(f"Price range: {closes.min():.5f} → {closes.max():.5f}")

# ── Step 2: RSI + ATR 計算 ────────────────────────────────────────────
rsi_vals = rsi(closes, 14)
atr_vals = atr(highs, lows, closes, 14)

# ── Step 3: シグナル生成 RSI < 30 → BUY, RSI > 70 → SELL ─────────────
signal_bars = []
directions = []

for i in range(15, len(closes) - 1):
    if np.isnan(rsi_vals[i]) or np.isnan(atr_vals[i]):
        continue
    if rsi_vals[i] < 30:
        signal_bars.append(i)
        directions.append(LONG)
    elif rsi_vals[i] > 70:
        signal_bars.append(i)
        directions.append(SHORT)

signal_bars = np.array(signal_bars, dtype=np.int32)
directions  = np.array(directions, dtype=np.int8)

n_long  = np.sum(directions == LONG)
n_short = np.sum(directions == SHORT)
print(f"\nSignals: {len(signal_bars)} total ({n_long} long, {n_short} short)")

# ── Step 4: SL/TP: ATR-based, 2:1 RR ─────────────────────────────────
sl_distances = atr_vals[signal_bars] * 2.0    # 2x ATR SL
tp_distances = atr_vals[signal_bars] * 4.0    # 4x ATR TP (2:1 RR)

# ── Step 5: コスト (Tradeview ILC) ────────────────────────────────────
cost_model = BrokerCost.tradeview_ilc()
instruments = ["EURUSD"] * len(signal_bars)
entry_costs = cost_model.per_trade_cost(instruments, sl_distances)

# ── Step 6: シミュレーション ──────────────────────────────────────────
results = simulate_trades(
    highs, lows, closes,
    signal_bars, directions, sl_distances, tp_distances,
    max_hold=48,       # 48h (2 days)
    exit_mode="rr",
    open_prices=opens,
    entry_costs=entry_costs,
)

# ── Step 7: 結果表示 ──────────────────────────────────────────────────
pnl = results["pnl_r"]

print(f"\n{'='*60}")
print(f"RSI(14) Reversal — EURUSD 1h — 5 Years (from API)")
print(f"{'='*60}")
print(f"Quality Grade: {results.quality.grade}")
print(f"Trades:        {len(pnl)}")
print(f"Profit Factor: {results.profit_factor:.2f}")
print(f"Win Rate:      {results.win_rate:.1%}")
print(f"Expectancy:    {results.expectancy_r:.3f}R")
print(f"Sharpe (R):    {results.sharpe_r:.2f}")
print(f"Sortino (R):   {results.sortino_r:.2f}")
print(f"Max DD:        {results.max_drawdown_r:.1f}R")
print(f"Recovery:      {results.recovery_factor:.2f}")
print(f"Avg cost:      {np.mean(results['cost_r']):.4f}R")

# Exit type breakdown
print(f"\nExit breakdown:")
for exit_type, name in [(EXIT_SL, "SL"), (EXIT_TP, "TP"), (EXIT_TIME, "Timeout")]:
    count = np.sum(results["exit_type"] == exit_type)
    if count > 0:
        subset = pnl[results["exit_type"] == exit_type]
        avg = np.mean(subset)
        print(f"  {name:>8}: {count:>4} ({count/len(pnl)*100:>5.1f}%) | avg {avg:+.3f}R")

# Long vs Short
print(f"\nDirection breakdown:")
for dir_val, name in [(LONG, "LONG (RSI<30)"), (SHORT, "SHORT (RSI>70)")]:
    mask = directions[:len(pnl)] == dir_val
    if len(pnl) < len(directions):
        mask = mask[:len(pnl)]
    dir_pnl = pnl[mask]
    if len(dir_pnl) > 0:
        dir_wr = (dir_pnl > 0).mean()
        dir_avg = np.mean(dir_pnl)
        print(f"  {name:>18}: n={len(dir_pnl):>4}, WR={dir_wr:.1%}, avg={dir_avg:+.3f}R")

# Yearly breakdown
entry_bars_arr = results["entry_bar"]
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

print(f"\n{'='*60}")
print("Done. Data fetched live from AggVault API (no local CSV).")

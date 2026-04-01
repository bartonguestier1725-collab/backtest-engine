"""RSI Reversal Strategy — Level 1 (API版).

新規ユーザーの初回体験を想定:
  1. AggVault APIからEURUSD 1h+1mデータを取得
  2. RSI(14) 30/70 逆張りを5年バックテスト（GROSS — コストなし）
  3. シグナル: 1h足、約定シミュレーション: 1m足（正確なSL/TP判定）
  4. 結果: テクニカルエッジは存在するか？

前提:
  pip install git+ssh://git@github.com/bartonguestier1725-collab/backtest-engine.git
  export AGGVAULT_KEY=tk_あなたのキー
"""

import numpy as np
import pandas as pd
from backtest_engine import (
    fetch_aggvault, simulate_trades_hires,
    LONG, SHORT, EXIT_SL, EXIT_TP, EXIT_TIME,
    rsi, atr,
)

# ── Step 1: データ取得（シグナル用1h + 約定シミュレーション用1m）────────
print("Fetching 1h data (signals)...")
ts_1h, opens_1h, highs_1h, lows_1h, closes_1h, _ = fetch_aggvault(
    "EURUSD", "1h", "2021-04-01", "2026-03-31",
)
dt_index = pd.to_datetime(ts_1h, unit="s", utc=True)
print(f"  1h: {len(ts_1h)} bars, {dt_index[0]} → {dt_index[-1]}")

print("Fetching 1m data (execution)...")
ts_1m, opens_1m, highs_1m, lows_1m, closes_1m, _ = fetch_aggvault(
    "EURUSD", "1m", "2021-04-01", "2026-03-31",
)
print(f"  1m: {len(ts_1m)} bars")

# ── Step 2: RSI + ATR 計算（1h足で）──────────────────────────────────
rsi_vals = rsi(closes_1h, 14)
atr_vals = atr(highs_1h, lows_1h, closes_1h, 14)

# ── Step 3: シグナル生成 RSI < 30 → BUY, RSI > 70 → SELL ─────────────
signal_bars = []
directions = []

for i in range(15, len(closes_1h) - 1):
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

# ── Step 5: シミュレーション（1hシグナル × 1m約定 — GROSS）─────────────
# コストなし: テクニカルエッジの存在確認が目的
# ブローカー固有のコスト（スプレッド/手数料/スリッページ）は別ステップで検証
results = simulate_trades_hires(
    signal_timestamps=ts_1h,
    signal_bars=signal_bars,
    directions=directions,
    sl_distances=sl_distances,
    tp_distances=tp_distances,
    max_hold=48,           # 48h (2 days) in 1h bars → auto-converted to 2880 1m bars
    signal_bar_minutes=60,
    exec_timestamps=ts_1m,
    exec_opens=opens_1m,
    exec_highs=highs_1m,
    exec_lows=lows_1m,
    exec_closes=closes_1m,
)

# ── Step 6: 結果表示 ──────────────────────────────────────────────────
pnl = results["pnl_r"]

print(f"\n{'='*60}")
print(f"RSI(14) Reversal — EURUSD — 5 Years")
print(f"Signal: 1h | Execution: 1m | {results.cost_label}")
print(f"{'='*60}")
print(f"Trades:        {len(pnl)}")
print(f"Profit Factor: {results.profit_factor:.2f}")
print(f"Win Rate:      {results.win_rate:.1%}")
print(f"Expectancy:    {results.expectancy_r:.3f}R")
print(f"Sharpe (R):    {results.sharpe_r:.2f}")
print(f"Sortino (R):   {results.sortino_r:.2f}")
print(f"Max DD:        {results.max_drawdown_r:.1f}R")
print(f"Recovery:      {results.recovery_factor:.2f}")

# Exit type breakdown
print(f"\nExit breakdown:")
for exit_type, name in [(EXIT_SL, "SL"), (EXIT_TP, "TP"), (EXIT_TIME, "Timeout")]:
    count = np.sum(results["exit_type"] == exit_type)
    if count > 0:
        subset = pnl[results["exit_type"] == exit_type]
        avg = np.mean(subset)
        print(f"  {name:>8}: {count:>4} ({count/len(pnl)*100:>5.1f}%) | avg {avg:+.3f}R")

# Yearly breakdown (map 1m entry bars back to dates)
entry_bars_1m = results["entry_bar"]
entry_dates = pd.to_datetime(ts_1m[entry_bars_1m], unit="s", utc=True)
print(f"\n{'Year':>6} {'PnL(R)':>10} {'n':>5} {'WR':>6} {'PF':>6}")
for year in range(2021, 2027):
    year_mask = entry_dates.year == year
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
print(f"Done. {results.cost_label}")
print(f"To test with broker-specific costs, use BrokerCost or add your own estimates.")

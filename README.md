# backtest_engine

Numba JIT で加速したバックテストエンジン。Python の for ループベースのバックテストを **10〜50倍高速化**。

## 特徴

- **`simulate_trades()`** — SL/TP/トレーリング/カスタム決済を numba `@njit` で高速シミュレーション
- **テクニカル指標** — SMA, ATR, Bollinger Bands, RCI, expanding quantile（全て `@njit`）
- **Monte Carlo DD 分析** — 10,000 回シャッフルで最大ドローダウン分布、Kelly 基準、最適リスク%
- **Walk-Forward / CSCV** — オーバーフィッティング検証（Bailey 2015 の PBO 算出）
- **ブローカーコストモデル** — Tradeview ILC / Fundora プリセット付き

## インストール

```bash
git clone https://github.com/bartonguestier1725-collab/backtest-engine.git
cd backtest-engine
uv venv .venv && source .venv/bin/activate
uv pip install -e '.[dev]'
```

> **Note:** numba 0.63 は NumPy < 2.4 が必要です。`numpy>=2.2,<2.4` がピン留めされています。

## クイックスタート

```python
import numpy as np
from backtest_engine import simulate_trades, LONG, MonteCarloDD, atr

# OHLC データ（numpy 配列）
high, low, close = ...

# ATR ベースの SL/TP
atr_vals = atr(high, low, close, 14)

# シグナル定義
signal_bars = np.array([100, 250, 400], dtype=np.int32)
directions = np.array([LONG, LONG, LONG], dtype=np.int8)
sl_distances = atr_vals[signal_bars] * 1.5
tp_distances = atr_vals[signal_bars] * 3.0

# シミュレーション実行
results = simulate_trades(
    high, low, close,
    signal_bars, directions, sl_distances, tp_distances,
    max_hold=100,
    be_trigger_pct=0.5,  # TP 50% でブレイクイーブン
)

# 結果
print(f"Win rate: {np.mean(results['pnl_r'] > 0) * 100:.1f}%")
print(f"Avg PnL: {np.mean(results['pnl_r']):.3f}R")

# Monte Carlo DD 分析
mc = MonteCarloDD(results['pnl_r'], risk_pct=0.01)
mc.run()
print(f"95th DD: {mc.dd_percentile(95) * 100:.1f}%")
```

## simulate_trades のオプション

| パラメータ | 説明 |
|-----------|------|
| `exit_mode="rr"` | 固定 RR（SL/TP/タイムアウト/ブレイクイーブン） |
| `exit_mode="trailing"` | トレーリングストップ |
| `exit_mode="custom"` | 外部シグナルで決済 |
| `be_trigger_pct` | TP の何 % で SL をエントリー価格に移動（0=無効） |
| `retrace_pct` | 指値エントリー：TP の何 % リトレースを待つ（0=成行） |
| `trail_activation_r` | トレーリング発動の R 倍率 |
| `trail_distance_r` | トレーリング距離の R 倍率 |

## テスト

```bash
python -m pytest tests/ -v
```

71 テスト、全パス。

## ライセンス

MIT

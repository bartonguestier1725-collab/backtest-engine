# backtest_engine

Numba JIT で加速したバックテストエンジン。Python の for ループベースのバックテストを **10〜50倍高速化**。

## 特徴

- **`simulate_trades()`** — SL/TP/トレーリング/SAR トレーリング/カスタム決済を numba `@njit` で高速シミュレーション
- **BugGuard** — 過去のバックテストバグ 11 件を自動検出（BG-01〜BG-11）。未来データ混入、コスト過小評価、短期間過学習などを実行前にブロック
- **GateKeeper** — 戦略の即死判定パイプライン（Gate 0→1→2→3→4）。無駄なパラメータ探索を5分以内に打ち切る
- **テクニカル指標** — SMA, ATR, Bollinger Bands, RCI, Parabolic SAR, expanding quantile（全て `@njit`）
- **Monte Carlo DD 分析** — 10,000 回シャッフルで最大ドローダウン分布、Kelly 基準、最適リスク%
- **Walk-Forward / CSCV** — オーバーフィッティング検証（Bailey 2015 の PBO 算出）
- **ブローカーコストモデル** — 実測スプレッド+コミッション。ECN ブローカーのプリセット付き（30 ペア + XAUUSD）

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
| `exit_mode="sar_trailing"` | Parabolic SAR ベースのトレーリング（方向認識 B プラン付き） |
| `exit_mode="custom"` | 外部シグナルで決済 |
| `be_trigger_pct` | TP の何 % で SL をエントリー価格に移動（0=無効） |
| `retrace_pct` | 指値エントリー：TP の何 % リトレースを待つ（0=成行） |
| `trail_activation_r` | トレーリング発動の R 倍率 |
| `trail_distance_r` | トレーリング距離の R 倍率 |
| `open_prices` | 次バー始値でエントリー（Close エントリーの代わり） |

## BugGuard

バックテスト結果を報告する前に、過去に発生した全バグを自動チェック:

```python
from backtest_engine import bug_guard

report = bug_guard.run_all_checks(
    source_path="my_strategy.py",
    signal_bars=signal_bars,
    entry_bars=results["entry_bar"],
    exit_bars=results["exit_bar"],
    spreads_used={"USDJPY": 0.017, "EURUSD": 0.000075},
    resolution_minutes=5,
    n_bars=len(close),
    n_trades=len(signal_bars),
)
# BG-01: 未来データ参照  BG-02: コスト過小評価  BG-04: bfill リーク
# BG-05: 粗い足での SL/TP 判定  BG-06: 短期間過学習  BG-07: 全期間分位数
# BG-08: 同一バー再エントリー  BG-09: Close エントリー  ... 他
```

## GateKeeper

戦略を段階的にふるいにかけ、見込みのないものを早期に排除:

```python
from backtest_engine import GateKeeper

gk = GateKeeper(data, param_grid, strategy_fn)
gk.gate0_validate("my_strategy.py")  # BugGuard チェック
gk.gate1_quick(n_combos=20)          # 20 パラメータで即死判定
gk.gate2_screen(n_combos=100)        # 100 パラメータでスクリーニング
```

## テスト

```bash
python -m pytest tests/ -v
```

98 テスト、全パス。

## ライセンス

MIT

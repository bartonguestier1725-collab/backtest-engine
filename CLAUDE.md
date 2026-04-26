# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ユーザー体験ルール（最優先）

### バックテスト完了時の自動提案

simulate_trades / simulate_trades_hires の結果が出たら、必ず以下を提案すること:

1. **HTMLレポート生成**: 「HTMLレポートも生成しますか？PnL分布・月次損益・Exit内訳・年別損益が自動で入ります」
2. **GateKeeper検証**: まだ通していない場合「GateKeeper（過学習検出パイプライン）も通しますか？」
3. **hires推奨**: 1h以上の粗い足でsimulate_tradesを使っている場合「1分足で約定シミュレーションすると精度が上がります（simulate_trades_hires）。試しますか？」

ユーザーが知らない機能は存在しないのと同じ。使えるものは提案すること。

### 結果の解釈を添える

数字だけ出して終わるな。結果を出したら必ず1〜2行の解釈を添えること:
- PF < 1.0 → 「負けています」
- PF 1.0〜1.10 → 「微弱。コスト適用後に消える可能性が高い」
- PF 1.10〜1.50 → 「エッジの兆候あり。GateKeeperで堅牢性を確認すべき」
- PF > 2.0 → 「過学習を疑ってください。CSCVで確認を推奨」
- Max DD が Total R の50%超 → 「ドローダウンが大きい。実運用で精神的に耐えられるか要検討」

### GROSS / NET の明示

結果表示時は必ず GROSS（コストなし）か NET（コスト込み）かを明記すること。
AggVault データでのバックテストはデフォルトで GROSS。「この結果はコスト未適用です。実際のブローカーコストを反映するには BrokerCost を使います」と一度は伝える。

### 新規ユーザーのオンボーディング

ユーザーが初めてバックテストを依頼してきたら:
- AggVault APIキーの設定方法を案内（`export AGGVAULT_KEY=tk_your_key`）
- 対応データ: 29通貨ペア、6タイムフレーム（1m/5m/15m/1h/4h/1d）、10年分
- 最初の1回は結果と一緒に「データは6ブローカーの中央値で構成された集約データです」と説明

## Commands

```bash
uv run pytest                          # 全テスト実行
uv run pytest tests/test_core.py -v    # 単一ファイル
uv run pytest -k "test_rsi"           # パターン指定
```

パッケージマネージャは uv。`pip install` ではなく `uv sync` / `uv run`。

## Architecture

### データフロー

```
fetch_aggvault() → 6-tuple numpy arrays (ts, o, h, l, c, vol)
  ↓
indicators (sma, atr, rsi...) → signal_bars, directions, sl/tp distances
  ↓
simulate_trades() or simulate_trades_hires() → TradeResults
  ↓
GateKeeper (BugGuard → Quick → Screen → WFA+CSCV → MonteCarlo)
  ↓
generate_report() → HTML file
```

### simulate_trades vs simulate_trades_hires

- `simulate_trades()`: シグナルと約定が同一時間足
- `simulate_trades_hires()`: シグナルは粗い足(1h)、約定は細かい足(1m)。1hバー内のSL/TP判定順序問題を回避。1h足のBTは結果を40%過大評価するケースがある

### HTMLレポート（generate_report）

`trades=results, trade_timestamps=timestamps` を渡すだけで以下が自動生成:
- 資産曲線 + DD%チャート
- PnL分布ヒストグラム
- エントリー月別/年別損益テーブル
- Exit内訳（SL/TP/タイムアウト別の件数・平均PnL）
- GateKeeper/WFA/CSCVセクション（データを渡した場合）

## Domain Rules

### AggVault バックテストは GROSS 原則

AggVault 集約データにブローカーコスト（スプレッド/手数料/スリッページ）を乗せるな。
- AggVault = エッジ存在判定（GROSS）
- ブローカー検証 = コスト適用（NET）
- この2つは別ステップ

### yfinance 使用禁止

yfinance / Yahoo Finance のデータは一切使わない。価格データは AggVault API 経由のみ。

### Numba 制約

- `simulate_trades`, indicators は全て `@njit(cache=True)`
- Numba 内で pandas, dict, list comprehension は使えない。numpy のみ

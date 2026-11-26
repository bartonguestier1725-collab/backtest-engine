# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ユーザー体験ルール（最優先）

### バックテスト完了時の自動提案

simulate_trades / simulate_trades_hires の結果が出たら、必ず以下を提案すること:

1. **HTMLレポート生成**: 「HTMLレポートも生成しますか？PnL分布・月次損益・Exit内訳・年別損益が自動で入ります」
2. **先読みの検査（audit）**: 結果を報告する前に、戦略を `signals(data)` と `backtest(data)` の2関数のファイルにして `uv run python -m backtest_engine.audit 戦略.py --json` を回す（詳しくは下の「戦略の検証」）。fail なら PF や勝率を結果として報告しない
3. **GateKeeper検証**: まだ通していない場合「GateKeeper（過学習検出パイプライン）も通しますか？」。Gate 0 には結果を渡す（`GateKeeper(..., bar_minutes=60, resolution_minutes=約定を判定した足の分数)` と `gk.gate0_validate(trades=results, sl_distances=sl, avg_price=float(np.median(close)), timestamps=ts)`。ts はシグナルを作った時間足の全部の足の時刻（`ts[signal_bars]` ではない。hires でも1時間足の ts）。1時間足の simulate_trades なら resolution_minutes=60、hires なら 1）。渡すと取引数・前の取引の決済足でのエントリー・SLなし・始値エントリー・固定コストの検査が回り、データ期間（BG-06）は ts の最初と最後の時刻で測る（GateKeeper に n_bars も渡すなら同じ足の本数）。GK-00 の中の spreads_used の項目は GROSS なら気にしなくてよい（他の項目は埋める）。Gate 0 はシグナルのコードの中の先読みは調べない（それは audit）。Gate 1〜4 は不合格でも止まらないので `.passed` を見る（回し方は `examples/gatekeeper_pipeline.py`）
   - Gate 1・2: run_func が返す `n_trades` は `results.n_filled`（`len(results)` は約定しなかった行も数える）
   - Gate 3: WalkForward・CSCV に渡す evaluate_fn は、窓の中の本当の値（合計 R など）を返し、取引が無い窓は 0。-999 のような目印は使わない（WFA の平均と CSCV の順位がゆがむ）
4. **hires推奨**: 1h以上の粗い足でsimulate_tradesを使っている場合「1分足で約定シミュレーションすると精度が上がります（simulate_trades_hires）。試しますか？」
5. **自分のブローカーで再現するか**: Gate 3 を通ったら、次の段を提案する。① 会員のスプレッド・手数料・スリッページを聞いてコストを上乗せする（NET） ② 会員が用意した自分のブローカーのデータで回す ③ AggVault のシグナル × 自分のブローカーのデータの約定（`simulate_trades_hires` のシグナル側に AggVault の足、約定側の `exec_*` にそれより細かい会員の足を渡す。足の分数は `exec_bar_minutes`）。どれも下の「AggVault のバックテストは、まず GROSS」と「データの決まり」のとおり

ユーザーが知らない機能は存在しないのと同じ。使えるものは提案すること。

### 結果の解釈を添える

数字だけ出して終わるな。結果を出したら必ず1〜2行の解釈を添えること:
- PF < 1.0 → 「負けています」
- PF 1.0〜1.10 → 「微弱。コスト適用後に消える可能性が高い」
- PF 1.10〜1.50 → 「エッジの兆候あり。GateKeeperで堅牢性を確認すべき」
- PF 1.50〜2.0 → 「強い結果。先読みと過学習をまず疑い、audit と CSCV で確認」
- PF > 2.0 → 「過学習か先読みを疑ってください。audit と CSCV で確認を推奨」
- Total R がプラスで、Max DD がその50%超 → 「ドローダウンが大きい。実運用で精神的に耐えられるか要検討」

### GROSS / NET の明示

結果表示時は必ず GROSS（コストなし）か NET（コスト込み）かを明記すること。
AggVault データでのバックテストはデフォルトで GROSS。「この結果はコスト未適用（GROSS）です。エッジが残るなら、次は自分のブローカーで再現するかを確かめる段です（コストを上乗せする / 自分のブローカーのデータで回す / AggVault のシグナル×自分のデータの約定）」と一度は伝える。

### 新規ユーザーのオンボーディング

ユーザーが初めてバックテストを依頼してきたら:
- AggVault APIキーの設定方法を案内（`export AGGVAULT_KEY=tk_your_key`）
- 対応データ: 29銘柄（FX 28ペア + XAUUSD）、6タイムフレーム（1m/5m/15m/1h/4h/1d）、10年分
- 1分足の長い期間は、fetch_aggvault が180日ずつに分けて自動で取る（1回の上限が50万本のため）。5年分で数分かかる（同じキーで同時に取ると429の待ちが入る）。v0.5.xは1回で取っていて、1分足は最初の50万本（約1.3年分）で切れていた
- 最初の1回は結果と一緒に「データは複数ブローカーの足を中央値で合成した集約データです」と説明

### キーの扱い

- キーは環境変数 `AGGVAULT_KEY` で渡す（`.env` を使うなら `.gitignore` に入れる）。戦略ファイル・スクリプト・ノートなど、作るファイルに書き込まない（`api_key=` に文字列で渡さない）。画面や質問に貼らない
- 会員がキーをプロンプトに貼ってきたら、環境変数に入れるよう案内し、ファイルには書かない

### 取得は1回

- 取得は銘柄×足種×期間ごとに1回。`data/`（`.gitignore` で除外済み）に保存して（`np.savez` など）、2回目からはそれを読む。AggVault は本番のサービスで、取得が重なるとほかの会員が待たされる
- audit に渡す戦略ファイルの頭（import されたときに走る所）で取得しない。取るなら `if __name__ == "__main__":` の中で（audit は回すたびにファイルを読み込む。`--seed 1` の回し直しでも取り直すことになる）

### エンジン本体は書き換えない

- 会員の戦略を検証するときは、エンジン本体（`backtest_engine/`）を書き換えない。戦略は別のファイルにする（本体を書き換えると `git pull` が衝突し、audit が前提にしている関数も変わる）
- 足りない機能があれば、会員に運営への要望として伝えてもらう
- 下の「開発者向け」の節（Commands・Numba 制約）は、エンジン本体を直す人のためのもの

### 最新にする

- `git pull` → `uv sync`。以後は `uv run ...` で動かす

## 開発者向け: Commands

```bash
uv run pytest                          # 全テスト実行
uv run pytest tests/test_core.py -v    # 単一ファイル
uv run pytest -k "test_rsi"           # パターン指定
```

パッケージマネージャは uv。`pip install` ではなく `uv sync` / `uv run`。

## Architecture

### データフロー

```
fetch_aggvault() → 6-tuple numpy arrays (ts, o, h, l, c, vol)  # AggVault の vol は常に0
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
- `simulate_trades_hires()`: シグナルは粗い足(1h)、約定は細かい足(1m)。1hバーの中でSLとTPの両方に触れたとき、どちらが先かを1分足で判定する（1h足だけだとSL扱い）。エントリーはシグナル足が閉じた後の最初の1分足の始値。v0.5.xまではシグナル足が開いた1分後に入る先読みがあった（v0.6.0で修正）。差の大きさは戦略しだいで、READMEのEURUSD RSI例ではPF 0.959（1h）と0.958（1m）でほぼ同じ（コストなし）

### HTMLレポート（generate_report）

`generate_report(ReportConfig(title=..., trades=results, trade_timestamps=timestamps), 出力パス)` で以下が自動で入る:
- PnL分布ヒストグラム
- エントリー月別/年別損益テーブル
- Exit内訳（SL/TP/タイムアウト別の件数・平均PnL）

trade_timestamps は results の entry_bar で引く足の時刻（hires なら1分足の時刻。1時間足の時刻を渡すと月別・年別の表が出ない）。自動では入らないもの:
- 資産曲線 + DD%: `equity_curve=` を渡す。DD% はピークに対する比なので、累積 R ではなく残高（例: 1取引1%リスクの複利）を渡す
- GateKeeper/WFA/CSCV の表: `gates=[GateRow(...)]`・`wfa=[WfaRow(...)]`・`cscv=CscvResult(...)` を自分で作って渡す（`WalkForward.run()` などの dict はそのまま渡せない）

## Domain Rules

### AggVault のバックテストは、まず GROSS

- エッジの有無を見る段では GROSS（コストなし）。最初の結果にコストを乗せない
- 会員が自分のコストで確かめたいと言ったら（「スプレッド1.5pips・スリッページ0.5pips追加でエッジ残る？」など）、断らない。会員にスプレッド・手数料・スリッページを聞いて `BrokerCost(spreads=..., commission_per_lot=..., pip_values=..., pip_sizes=...)` を作り（スリッページはスプレッドに足す）、`entry_costs=cost.per_trade_cost(...)` で NET を出して、NET と明記する
- エッジの有無の判定（GROSS）と、自分のブローカーで再現するかの確認（NET・自分のデータ）は別の段。1つの結果の中で混ぜない
- `BrokerCost.sample_ecn()`・`BrokerCost.sample_wide_spread()` はサンプル値で、どのブローカーの値でもない。会員の数字が無いまま、これを会員のコストとして NET を出さない

### データの決まり

- シグナルを作るデータは AggVault（`fetch_aggvault`）。yfinance / Yahoo Finance などの別のデータでシグナルを作らない
- 約定の判定やコストの確認（自分のブローカーで再現するか）には、会員が用意した自分のブローカーのデータ（CSV を `load_ohlcv` で読む）を使ってよい。断らない
- 別のデータで約定を判定するときは、時刻を UTC の足の始まりにそろえる。ブローカーの書き出しはサーバー時刻のことが多く、`load_ohlcv` はタイムゾーンの無い文字列を UTC として読む。ずれてもエラーにならず、シグナルと約定が黙ってずれる

### 約定の決まり（v0.6.0）

- 時刻は足の始まりの時刻（UTC、Unix秒）。hires はミリ秒を受け付けない（エラー）
- open_prices を渡したとき（hires は常に渡している）:
  - 損切り（SL・建値・トレーリング）は、足が損切り値を越えて始まったらその始値で約定する（1Rを超えて負けることがある）
  - 足が利確値を越えて始まったら、その足で安値がSLに触れても利確（始値が先）
  - カスタム決済（exit_mode="custom"）は、シグナルの出た足の次の足の始値で約定する
- open_prices を渡さないときは、損切り値・終値で約定する（窓開けは分からない）
- 利確は利確値で約定する（窓を開けても有利な値にしない）
- 指値エントリー（`retrace_pct`）のSL/TPは約定値から測る
- モード専用の引数（rr の `retrace_pct`・`retrace_timeout`・`be_trigger_pct`、trailing の `trail_activation_r`・`trail_distance_r`、custom の `exit_signals`、sar_trailing の `sar_values`）を他のモードで渡すとエラー（v0.6.1。それまでは黙って無視していた）
- 約定しなかった行（NO_FILL）は、勝率・期待値・幾何平均・Sharpe・Sortino に数えない（v0.6.1）。MonteCarloDD・StressTest には `results.filled["pnl_r"]` を渡す
- max_hold は保有する本数。始値エントリーはエントリーの足が1本目

### 戦略の検証（audit）

`backtest_engine.audit` は、コードを読まずに動きで先読みを調べる。AI がコードを読んで確かめるのとは別に回す（問題集では、AI に中身を見せなかったライブラリの中の先読みを、これが見つけた。`bench/README.md`）。

- `lookahead`: ある足より後のデータだけを書き換えてシグナルを作り直し、それより前のシグナルが変わったら先読み。summary に、どの足が変わったかから推した原因が出る。確率的な検査で、まれにしかシグナルを変えない先読みは通ることがある
- `null_edge`: エッジのない乱数の相場で backtest を回し、同じエントリーの向きを硬貨投げにした版とも比べる。勝つ・硬貨投げと違う → fail（先読みか時刻ずれ）。負けるが硬貨投げと同じ → warn（not_run でない warn。1本の足でSLとTPの両方に触れるときSLを優先する規則で悲観側に偏っている。hires で回し直す）。負けて、硬貨投げの版は負けていない（または比べられない）→ fail（LOSS ON RANDOM DATA。エントリーの時刻ずれか、SL優先の偏り。hires と `--seed 1` で回し直す）
- null_edge の fail は問題として扱い、summary の原因を探す（問題集では、先読みのない戦略が fail になったのは360回中3回で、3回とも同じ種）。`--seed 1` でもう一度回すと分かることが増える（種が違えば乱数の相場は1本も重ならない）。2回続けて fail なら確定。fail のあと pass なら多くは偶然だが、一部の取引だけの弱い漏れも1回は通ることがあるので、summary が指す箇所のコードを確かめてから報告する
- 損益からコストを手で引いている場合、null_edge はそれも損失として数える（entry_costs で渡したコストだけを戻して検定する）
- fail の結果は報告しない。原因を直して回し直す。error は検査ができていない（合格ではない）。pass でも「先読みが無いことの証明ではない」と添える
- 終了コードは fail が1、error が2、pass と warn は0。判定は JSON の `overall` と各 `verdict` を読む
- signals と backtest の片方しか無いファイルは、無い方の検査が warn になる（details に `not_run`、summary に何を調べていないか）。両方書いて回し直す
- backtest の中ではエンジンの simulate_trades / simulate_trades_hires を使う（硬貨投げの版は、そこで向きだけを入れ替えて作る）
- 戦略ファイルの雛形（data の中身、signals の戻り値、1分足で回すときの `BAR_MINUTES`・`NULL_BARS`・`NULL_VOL`）と英語の手順は `AGENTS.md`（Claude 以外の AI も読む）。API の呼び方は README の Quick start と各節

### 開発者向け: Numba 制約

- `simulate_trades` の中のカーネル（`_sim_*_inner`）と indicators は `@njit(cache=True)`
- Numba 内で pandas, dict, list comprehension は使えない。numpy のみ

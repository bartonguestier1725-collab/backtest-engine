"""HTML report generator for backtest results.

White background, Courier New + Noto Sans JP, tooltip-enabled.
Stdlib + matplotlib only — no Jinja2.

When ``trades`` + ``trade_timestamps`` are provided, the following sections
are auto-generated from TradeResults:
  - PnL Distribution (histogram)
  - Monthly Returns
  - Exit Breakdown
  - Yearly P&L
"""
from __future__ import annotations

import base64
import datetime
import html as _html
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


def _ensure_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans Mono"]
    return plt

_EXIT_NAMES = {
    0: "損切 (SL)",
    1: "利確 (TP)",
    2: "タイムアウト",
    3: "ブレイクイーブン",
    4: "カスタム",
    5: "トレイリング",
    6: "約定不成立",
}


# ── data classes ───────────────────────────────────────────────

@dataclass
class SummaryCell:
    label: str
    value: str
    tooltip: str = ""
    color: str = ""  # "pos" | "neg" | "warn" | ""


@dataclass
class GateRow:
    name: str
    result: str  # "PASS" | "FAIL" | "SKIP"
    detail: str = ""
    tooltip: str = ""


@dataclass
class WfaRow:
    period: str
    is_value: str
    oos_value: str
    is_positive: bool = True


@dataclass
class CscvResult:
    pbo: float
    logit_mean: float | None = None
    n_combinations: int | None = None
    threshold: float = 0.40


@dataclass
class Section:
    title: str
    body_html: str
    tooltip: str = ""


@dataclass
class ReportConfig:
    title: str
    subtitle: str = ""
    verdict: str = ""
    verdict_tooltip: str = ""
    title_tooltip: str = ""

    # equity chart
    equity_curve: np.ndarray | None = None
    equity_timestamps: np.ndarray | None = None
    equity_ylabel: str = "累積損益"

    # summary grid
    summary: list[SummaryCell] = field(default_factory=list)

    # strategy
    strategy_html: str = ""
    analysis_html: str = ""

    # auto-generated from TradeResults
    trades: object | None = None  # TradeResults
    trade_timestamps: np.ndarray | None = None

    # gates
    gates: list[GateRow] = field(default_factory=list)

    # WFA / CSCV
    wfa: list[WfaRow] = field(default_factory=list)
    wfa_summary: str = ""
    cscv: CscvResult | None = None

    # free-form
    sections: list[Section] = field(default_factory=list)
    sections_after_gates: list[Section] = field(default_factory=list)

    footer: str = ""


# ── helpers ────────────────────────────────────────────────────

def _esc(s: str) -> str:
    return _html.escape(s, quote=True)


def _tip(text: str, desc: str) -> str:
    if not desc:
        return _esc(text)
    return (
        f'<span class="tip">{_esc(text)}'
        f'<span class="tip-text">{_esc(desc)}</span></span>'
    )


def _val_cls(color: str) -> str:
    safe = _html.escape(color, quote=True) if color else ""
    return f"val {safe}" if safe else "val"


def _badge(result: str) -> str:
    r = _html.escape(result.lower(), quote=True)
    return f'<span class="badge {r}">{_esc(result)}</span>'


def _pnl_color(v: float) -> str:
    return "pos" if v > 0 else "neg" if v < 0 else ""


# ── charts ─────────────────────────────────────────────────────

def equity_chart_b64(
    equity: np.ndarray,
    timestamps: np.ndarray | None = None,
    ylabel: str = "累積損益",
) -> str:
    plt = _ensure_mpl()
    fig, ax = plt.subplots(figsize=(8.4, 3), dpi=120)
    ax.set_facecolor("#fafafa")
    y = np.asarray(equity, dtype=np.float64)

    ax.fill_between(range(len(y)), y, 0, where=y >= 0, color="#060", alpha=0.05)
    ax.fill_between(range(len(y)), y, 0, where=y < 0, color="#c00", alpha=0.05)
    ax.plot(y, color="#333", linewidth=1.0)
    ax.axhline(0, color="#ccc", linewidth=0.5)

    pk = np.maximum.accumulate(y)
    dd = pk - y
    ax2 = ax.twinx()
    denom = np.where(pk > 0, pk, 1)
    dd_pct = dd / denom * 100
    ax2.fill_between(range(len(dd)), 0, -dd_pct, color="#c00", alpha=0.08)
    max_dd_pct = float(np.max(dd_pct)) if len(dd_pct) > 0 else 25
    ax2.set_ylim(-max(max_dd_pct * 1.2, 5), 0)
    ax2.set_ylabel("DD%", fontsize=9, color="#999")

    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.15, color="#ccc")
    ax.tick_params(labelsize=8)
    ax2.tick_params(labelsize=8)

    if timestamps is not None and len(timestamps) == len(y):
        yp, prev = [], None
        for i in range(len(timestamps)):
            dt = datetime.datetime.fromtimestamp(
                int(timestamps[i]), tz=datetime.timezone.utc,
            )
            if dt.year != prev:
                yp.append((i, str(dt.year)))
                prev = dt.year
        if yp:
            ax.set_xticks([p for p, _ in yp])
            ax.set_xticklabels([l for _, l in yp], fontsize=8)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _histogram_b64(pnl: np.ndarray) -> str:
    plt = _ensure_mpl()
    fig, ax = plt.subplots(figsize=(8.4, 2.4), dpi=120)
    ax.set_facecolor("#fafafa")

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    bins = np.linspace(float(pnl.min()) - 0.1, float(pnl.max()) + 0.1, 50)

    if len(wins) > 0:
        ax.hist(wins, bins=bins, color="#060", alpha=0.35, label=f"Win ({len(wins)})")
    if len(losses) > 0:
        ax.hist(losses, bins=bins, color="#c00", alpha=0.35, label=f"Loss ({len(losses)})")

    ax.axvline(0, color="#333", linewidth=0.8, linestyle="--")
    mean = float(np.mean(pnl))
    ax.axvline(mean, color="#06c", linewidth=0.8, linestyle=":")
    ax.text(mean, ax.get_ylim()[1] * 0.9, f" μ={mean:.3f}R",
            fontsize=8, color="#06c", va="top")
    ax.set_xlabel("PnL (R)", fontsize=9)
    ax.set_ylabel("件数", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.15, color="#ccc")
    ax.tick_params(labelsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── auto-generated sections from TradeResults ──────────────────

def _build_exit_breakdown(trades) -> str:
    pnl = np.asarray(trades["pnl_r"])
    et = np.asarray(trades["exit_type"])
    hb = np.asarray(trades["hold_bars"])
    rows = []
    for code in sorted(set(int(x) for x in et)):
        if code == 6:
            continue
        mask = et == code
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        name = _EXIT_NAMES.get(code, f"Exit({code})")
        avg = float(np.mean(pnl[mask]))
        pct = cnt / len(pnl) * 100
        cls = _pnl_color(avg)
        rows.append(
            f'<tr><td>{name}</td><td>{cnt}</td><td>{pct:.1f}%</td>'
            f'<td class="{cls}">{avg:+.3f}R</td></tr>'
        )
    hold_html = (
        f'<div style="margin-top:6px;color:#666;font-size:11px;">'
        f'保有期間 — 平均: {np.mean(hb):.1f} bars / '
        f'中央値: {np.median(hb):.0f} / 最大: {np.max(hb)}</div>'
    )
    return (
        '<table><thead><tr><th>Exit</th><th>件数</th><th>割合</th>'
        '<th>平均PnL</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>{hold_html}'
    )


def _build_monthly(trades, timestamps: np.ndarray) -> str:
    pnl = np.asarray(trades["pnl_r"])
    eb = np.asarray(trades["entry_bar"])
    months: dict[str, list[float]] = {}
    for i in range(len(pnl)):
        t = int(timestamps[eb[i]])
        dt = datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc)
        ym = dt.strftime("%Y-%m")
        months.setdefault(ym, []).append(float(pnl[i]))

    rows = []
    for ym in sorted(months):
        arr = np.array(months[ym])
        total = float(np.sum(arr))
        n = len(arr)
        wr = float(np.mean(arr > 0))
        cls = _pnl_color(total)
        rows.append(
            f'<tr><td>{ym}</td><td>{n}</td>'
            f'<td class="{cls}">{total:+.1f}R</td><td>{wr:.0%}</td></tr>'
        )
    black = sum(1 for ym in months if sum(months[ym]) > 0)
    total_m = len(months)
    foot = (
        f'<div style="margin-top:4px;color:#666;font-size:11px;">'
        f'月次黒字率: {black}/{total_m} ({black/total_m:.0%})</div>'
    )
    return (
        '<table><thead><tr><th>月</th><th>N</th><th>PnL(R)</th>'
        '<th>WR</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>{foot}'
    )


def _build_yearly(trades, timestamps: np.ndarray) -> str:
    pnl = np.asarray(trades["pnl_r"])
    eb = np.asarray(trades["entry_bar"])
    years: dict[int, list[float]] = {}
    for i in range(len(pnl)):
        t = int(timestamps[eb[i]])
        y = datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).year
        years.setdefault(y, []).append(float(pnl[i]))

    rows = []
    for y in sorted(years):
        arr = np.array(years[y])
        total = float(np.sum(arr))
        n = len(arr)
        wr = float(np.mean(arr > 0))
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        if len(losses) > 0 and np.sum(losses) != 0:
            pf_str = f"{float(np.sum(wins) / abs(np.sum(losses))):.2f}"
        elif len(wins) > 0:
            pf_str = "∞"
        else:
            pf_str = "N/A"
        cls = _pnl_color(total)
        rows.append(
            f'<tr><td>{y}</td><td class="{cls}">{total:+.1f}R</td>'
            f'<td>{n}</td><td>{wr:.0%}</td><td>{pf_str}</td></tr>'
        )
    return (
        '<table><thead><tr><th>年</th><th>PnL(R)</th><th>N</th>'
        '<th>WR</th><th>PF</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


# ── CSS ────────────────────────────────────────────────────────

_CSS = """\
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#fff;color:#222;font-family:'Courier New','Noto Sans JP',monospace;\
font-size:13px;line-height:1.7;max-width:860px;margin:0 auto;padding:20px 24px;}
h1{font-size:16px;font-weight:bold;margin-bottom:2px;}
h2{font-size:13px;font-weight:bold;margin:24px 0 8px;padding-bottom:4px;\
border-bottom:1px solid #ccc;}
.header{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;}
.subtitle{color:#666;font-size:12px;margin-bottom:16px;}
.verdict{font-weight:bold;font-size:18px;}
.verdict.pass{color:#060;}.verdict.fail{color:#c00;}
img.chart{width:100%;margin:8px 0;}
.param-block{background:#f7f7f7;border:1px solid #ddd;padding:12px 14px;\
font-size:12px;line-height:1.8;margin:8px 0;}
.analysis{background:#f7f7f7;border:1px solid #ddd;padding:12px 14px;\
font-size:12px;line-height:1.8;margin:8px 0;}
.summary-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:0;\
border:1px solid #ccc;font-size:12px;margin:8px 0;}
.summary-cell{padding:8px 10px;border-right:1px solid #eee;border-bottom:1px solid #eee;}
.summary-cell .lbl{color:#666;font-size:11px;}
.summary-cell .val{font-size:16px;font-weight:bold;}
.val.neg{color:#c00;}.val.pos{color:#060;}.val.warn{color:#a60;}
table{width:100%;border-collapse:collapse;font-size:12px;margin:4px 0;}
th,td{padding:4px 8px;text-align:left;border-bottom:1px solid #eee;}
th{color:#666;font-weight:normal;border-bottom:1px solid #ccc;}
td.neg{color:#c00;}td.pos{color:#060;}
.badge{font-size:11px;font-weight:bold;}
.badge.pass{color:#060;}.badge.fail{color:#c00;}.badge.skip{color:#999;}
.tip{cursor:help;border-bottom:1px dotted #999;position:relative;}
.tip-text{display:none;position:absolute;bottom:120%;left:50%;\
transform:translateX(-50%);width:300px;padding:8px 10px;background:#fff;\
color:#333;border:1px solid #ccc;font-size:11px;line-height:1.5;z-index:100;\
box-shadow:0 2px 6px rgba(0,0,0,0.12);}
.tip:hover .tip-text{display:block;}
.footer{margin-top:28px;padding-top:10px;border-top:1px solid #ccc;\
color:#999;font-size:10px;}"""


# ── builder ────────────────────────────────────────────────────

def generate_report(cfg: ReportConfig, output: str | Path) -> Path:
    parts: list[str] = []

    # --- head ---
    parts.append(
        f'<!DOCTYPE html><html lang="ja"><head>'
        f'<meta charset="UTF-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1.0">'
        f"<title>{_esc(cfg.title)}</title>"
        f"<style>{_CSS}</style></head><body>"
    )

    # --- header + verdict ---
    verdict_html = ""
    if cfg.verdict:
        v = _html.escape(cfg.verdict.lower(), quote=True)
        verdict_html = (
            f'<span class="verdict {v}">'
            f'{_tip(cfg.verdict, cfg.verdict_tooltip)}</span>'
        )
    parts.append(
        f'<div class="header">'
        f"<h1>{_tip(cfg.title, cfg.title_tooltip)}</h1>"
        f"{verdict_html}</div>"
    )
    if cfg.subtitle:
        parts.append(f'<div class="subtitle">{_esc(cfg.subtitle)}</div>')

    # --- equity chart ---
    if cfg.equity_curve is not None:
        eb64 = equity_chart_b64(
            cfg.equity_curve, cfg.equity_timestamps, cfg.equity_ylabel,
        )
        parts.append(
            f'<div class="section">'
            f'<img class="chart" src="data:image/png;base64,{eb64}" '
            f'alt="Equity Curve"></div>'
        )

    # --- summary grid ---
    if cfg.summary:
        cells = []
        for c in cfg.summary:
            cells.append(
                f'<div class="summary-cell">'
                f'<div class="lbl">{_tip(c.label, c.tooltip)}</div>'
                f'<div class="{_val_cls(c.color)}">{_esc(c.value)}</div></div>'
            )
        parts.append(
            f'<h2>Summary</h2>'
            f'<div class="summary-grid">{"".join(cells)}</div>'
        )

    # --- strategy description ---
    if cfg.strategy_html:
        parts.append(
            f'<h2>{_tip("手法", "検証対象の売買ルール。エントリー条件、利確条件、損切条件、対象銘柄、時間足を記載。")}</h2>'
            f'<div class="param-block">{cfg.strategy_html}</div>'
        )

    # --- analysis ---
    if cfg.analysis_html:
        parts.append(
            f'<h2>{_tip("分析・所感", "バックテスト結果の定性的な解釈。数字だけでは読み取れない構造的な問題や改善の方向性を記述。")}</h2>'
            f'<div class="analysis">{cfg.analysis_html}</div>'
        )

    # --- auto sections from TradeResults ---
    has_trades = (
        cfg.trades is not None
        and hasattr(cfg.trades, "__len__")
        and len(cfg.trades) > 0
    )
    has_ts = cfg.trade_timestamps is not None
    if has_trades and has_ts:
        max_eb = int(np.max(np.asarray(cfg.trades["entry_bar"])))
        if max_eb >= len(cfg.trade_timestamps):
            has_ts = False

    if has_trades:
        pnl = np.asarray(cfg.trades["pnl_r"])

        if len(pnl) > 0:
            # PnL Distribution
            hb64 = _histogram_b64(pnl)
            parts.append(
                f'<h2>{_tip("PnL Distribution", "1トレードごとの損益(R)のヒストグラム。右に厚い分布が理想。")}</h2>'
                f'<img class="chart" src="data:image/png;base64,{hb64}" alt="PnL Distribution">'
            )

        # Monthly Returns (entry month)
        if has_ts and len(pnl) > 0:
            parts.append(
                f'<h2>{_tip("エントリー月別 Returns", "エントリー月ごとのトレード数・損益・勝率。特定月だけ稼いでいる場合、環境依存の兆候。")}</h2>'
                f'{_build_monthly(cfg.trades, cfg.trade_timestamps)}'
            )

        # Exit Breakdown
        if len(pnl) > 0:
            parts.append(
                f'<h2>{_tip("Exit Breakdown", "決済理由の内訳。利確・損切・タイムアウトそれぞれの件数と平均損益。")}</h2>'
                f'{_build_exit_breakdown(cfg.trades)}'
            )

        # Yearly P&L (entry year)
        if has_ts and len(pnl) > 0:
            parts.append(
                f'<h2>{_tip("エントリー年別損益", "年ごとのPnL安定性を確認。特定年だけ稼いでいる場合は再現性に疑問。")}</h2>'
                f'{_build_yearly(cfg.trades, cfg.trade_timestamps)}'
            )

    # --- custom sections (before gates) ---
    for sec in cfg.sections:
        parts.append(f"<h2>{_tip(sec.title, sec.tooltip)}</h2>{sec.body_html}")

    # --- gatekeeper ---
    if cfg.gates:
        rows = []
        for g in cfg.gates:
            rows.append(
                f"<tr><td>{_tip(g.name, g.tooltip)}</td>"
                f"<td>{_badge(g.result)}</td>"
                f"<td>{_esc(g.detail)}</td></tr>"
            )
        parts.append(
            f'<h2>{_tip("GateKeeper", "バックテストエンジンの5段階品質検証パイプライン。Gate 0(データ品質) → Gate 1(最低収益性) → Gate 2(パラメータスクリーニング) → Gate 3(過学習検出) → Gate 4(モンテカルロ)。1つでもFAILで不合格。")}</h2>'
            f'<table><thead><tr><th>Gate</th><th>Result</th><th>詳細</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
        )

    # --- WFA ---
    if cfg.wfa:
        rows = []
        for w in cfg.wfa:
            cls = "pos" if w.is_positive else "neg"
            rows.append(
                f'<tr><td>{_esc(w.period)}</td><td>{_esc(w.is_value)}</td>'
                f'<td class="{cls}" style="font-weight:bold">{_esc(w.oos_value)}</td></tr>'
            )
        parts.append(
            f'<h2>{_tip("Walk-Forward Analysis", "ウォークフォワード分析。過去で最適化したパラメータを未来データに適用して検証。OOSが正ならリアルでも通用する可能性がある。")}</h2>'
            f'<table><thead><tr><th>期間</th><th>IS</th><th>OOS</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
        )
        if cfg.wfa_summary:
            parts.append(
                f'<div style="margin-top:4px;color:#666;font-size:11px;">'
                f'{_esc(cfg.wfa_summary)}</div>'
            )

    # --- CSCV ---
    if cfg.cscv is not None:
        c = cfg.cscv
        pbo_cls = "pos" if c.pbo <= c.threshold else "neg"
        pbo_verdict = "PASS" if c.pbo <= c.threshold else "FAIL"
        rows = [
            f'<tr><td>{_tip("PBO", "Probability of Backtest Overfitting。バックテスト結果がたまたま良く見えている確率。低いほど良い。")}</td>'
            f'<td class="{pbo_cls}" style="font-weight:bold">{c.pbo:.3f}</td>'
            f'<td>≤ {c.threshold:.2f}</td><td>{_badge(pbo_verdict)}</td></tr>',
        ]
        if c.logit_mean is not None:
            rows.append(
                f'<tr><td>Logit Mean</td><td>{c.logit_mean:.3f}</td>'
                f'<td>—</td><td>—</td></tr>'
            )
        if c.n_combinations is not None:
            rows.append(
                f'<tr><td>Combinations</td><td>{c.n_combinations}</td>'
                f'<td>—</td><td>—</td></tr>'
            )
        parts.append(
            f'<h2>{_tip("CSCV", "Combinatorially Symmetric Cross-Validation。Bailey et al.(2014)が提唱した過学習確率の定量評価。")}</h2>'
            f'<table><thead><tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Result</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
        )

    # --- custom sections (after gates) ---
    for sec in cfg.sections_after_gates:
        parts.append(f"<h2>{_tip(sec.title, sec.tooltip)}</h2>{sec.body_html}")

    # --- footer ---
    if cfg.footer:
        parts.append(f'<div class="footer">{_esc(cfg.footer)}</div>')
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        parts.append(f'<div class="footer">Backtest Engine | {now}</div>')

    parts.append("</body></html>")

    out = Path(output)
    out.write_text("".join(parts), encoding="utf-8")
    return out

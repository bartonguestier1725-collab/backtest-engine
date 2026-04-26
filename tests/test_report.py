"""Tests for backtest_engine.report — HTML report generation."""

import numpy as np
import pytest
from pathlib import Path

from backtest_engine._types import TRADE_RESULT_DTYPE, EXIT_SL, EXIT_TP, EXIT_TIME
from backtest_engine._results import TradeResults
from backtest_engine.report import (
    ReportConfig, SummaryCell, GateRow, WfaRow, CscvResult, Section,
    generate_report,
)


@pytest.fixture
def tmp_html(tmp_path):
    return tmp_path / "test_report.html"


def _make_trades(n=50):
    trades = np.zeros(n, dtype=TRADE_RESULT_DTYPE)
    rng = np.random.RandomState(42)
    trades["pnl_r"] = rng.randn(n) * 0.5
    trades["hold_bars"] = rng.randint(1, 100, n)
    trades["exit_type"] = rng.choice([EXIT_SL, EXIT_TP, EXIT_TIME], n)
    trades["mfe_r"] = np.abs(rng.randn(n))
    trades["mae_r"] = -np.abs(rng.randn(n))
    trades["entry_bar"] = np.arange(n) * 10
    trades["exit_bar"] = trades["entry_bar"] + trades["hold_bars"]
    return TradeResults(trades)


def _make_timestamps(n_bars=600):
    return np.arange(1617235200, 1617235200 + n_bars * 3600, 3600, dtype=np.int64)


class TestGenerateReport:
    def test_basic_html_output(self, tmp_html):
        cfg = ReportConfig(title="Test Report")
        out = generate_report(cfg, tmp_html)
        assert out.exists()
        html = out.read_text()
        assert "<!DOCTYPE html>" in html
        assert "Test Report" in html

    def test_all_sections_present(self, tmp_html):
        trades = _make_trades(50)
        ts = _make_timestamps(600)
        cfg = ReportConfig(
            title="Full Report",
            subtitle="Test subtitle",
            verdict="PASS",
            equity_curve=np.cumsum(trades["pnl_r"]),
            summary=[SummaryCell("Trades", "50", "tooltip")],
            strategy_html="<b>Test strategy</b>",
            analysis_html="Test analysis",
            trades=trades,
            trade_timestamps=ts,
            gates=[GateRow("BugGuard", "PASS", "14/14")],
            wfa=[WfaRow("2021-2023", "+10R", "+2R", True)],
            cscv=CscvResult(pbo=0.15, logit_mean=-0.3, n_combinations=126),
        )
        out = generate_report(cfg, tmp_html)
        html = out.read_text()
        assert "PnL Distribution" in html
        assert "エントリー月別" in html
        assert "Exit Breakdown" in html
        assert "エントリー年別" in html
        assert "GateKeeper" in html
        assert "Walk-Forward" in html
        assert "CSCV" in html
        assert "Test analysis" in html

    def test_empty_trades_no_crash(self, tmp_html):
        trades = TradeResults(np.zeros(0, dtype=TRADE_RESULT_DTYPE))
        cfg = ReportConfig(
            title="Empty",
            trades=trades,
            trade_timestamps=np.array([], dtype=np.int64),
        )
        out = generate_report(cfg, tmp_html)
        html = out.read_text()
        assert "<!DOCTYPE html>" in html
        assert "PnL Distribution" not in html

    def test_html_escape_title(self, tmp_html):
        cfg = ReportConfig(title='<script>alert("xss")</script>')
        out = generate_report(cfg, tmp_html)
        html = out.read_text()
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_html_escape_subtitle(self, tmp_html):
        cfg = ReportConfig(title="Safe", subtitle='<img src=x onerror="alert(1)">')
        out = generate_report(cfg, tmp_html)
        html = out.read_text()
        assert 'onerror="alert' not in html

    def test_verdict_pass_fail(self, tmp_html):
        for v in ("PASS", "FAIL"):
            cfg = ReportConfig(title="Test", verdict=v)
            out = generate_report(cfg, tmp_html)
            html = out.read_text()
            assert f'class="verdict {v.lower()}"' in html

    def test_custom_sections(self, tmp_html):
        cfg = ReportConfig(
            title="Custom",
            sections=[Section("MySection", "<p>content</p>", "tip")],
        )
        out = generate_report(cfg, tmp_html)
        html = out.read_text()
        assert "MySection" in html
        assert "<p>content</p>" in html

    def test_gate_rows(self, tmp_html):
        cfg = ReportConfig(
            title="Gates",
            gates=[
                GateRow("Gate0", "PASS", "ok"),
                GateRow("Gate3", "FAIL", "PBO too high"),
            ],
        )
        out = generate_report(cfg, tmp_html)
        html = out.read_text()
        assert 'badge pass">PASS' in html
        assert 'badge fail">FAIL' in html

    def test_summary_value_escaped(self, tmp_html):
        cfg = ReportConfig(
            title="Test",
            summary=[SummaryCell("X", '<img src=x>', "tip")],
        )
        out = generate_report(cfg, tmp_html)
        html = out.read_text()
        assert "<img src=x>" not in html
        assert "&lt;img" in html

    def test_gate_detail_escaped(self, tmp_html):
        cfg = ReportConfig(
            title="Test",
            gates=[GateRow("G", "PASS", '<script>alert(1)</script>')],
        )
        html = generate_report(cfg, tmp_html).read_text()
        assert "<script>" not in html

    def test_timestamps_mismatch_no_crash(self, tmp_html):
        trades = _make_trades(50)
        short_ts = np.arange(10, dtype=np.int64)
        cfg = ReportConfig(
            title="Mismatch",
            trades=trades,
            trade_timestamps=short_ts,
        )
        out = generate_report(cfg, tmp_html)
        html = out.read_text()
        assert "エントリー月別" not in html
        assert "PnL Distribution" in html

    def test_dd_chart_large_drawdown(self, tmp_html):
        equity = np.array([100.0, 50.0, 20.0, 10.0, 5.0])
        cfg = ReportConfig(title="BigDD", equity_curve=equity)
        out = generate_report(cfg, tmp_html)
        assert out.stat().st_size > 0

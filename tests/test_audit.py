"""Tests for backtest_engine.audit — look-ahead and null-data tests."""

import json

import numpy as np
import pandas as pd
import pytest

from backtest_engine import simulate_trades, rsi, atr, LONG, SHORT
from backtest_engine.audit import (
    check_lookahead, check_null_edge, random_walk_bars, audit_strategy, main,
)


def _rsi_sig(d):
    r = rsi(d["close"], 14)
    return np.where(r < 30, 1, np.where(r > 70, -1, 0))


def _bt(sig_fn, shift=0):
    def bt(d):
        sig = sig_fn(d)
        o, h, l, c = d["open"], d["high"], d["low"], d["close"]
        a = atr(h, l, c, 14)
        sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
        sb = sb[(sb >= 1) & (sb < len(c) - 1)].astype(np.int32)
        dirs = np.sign(sig[sb]).astype(np.int8)
        return simulate_trades(h, l, c, sb - shift, dirs, a[sb] * 2, a[sb] * 3,
                               max_hold=24, open_prices=o)
    return bt


class TestRandomWalkBars:
    def test_valid_ohlc(self):
        d = random_walk_bars(500, seed=3)
        assert np.all(d["high"] >= np.maximum(d["open"], d["close"]))
        assert np.all(d["low"] <= np.minimum(d["open"], d["close"]))
        np.testing.assert_allclose(d["open"][1:], d["close"][:-1])
        assert np.all(np.diff(d["time"]) == 3600)


class TestLookahead:
    def test_causal_signal_passes(self):
        r = check_lookahead(_rsi_sig, n_cuts=16)
        assert r.verdict == "pass", r.summary

    def test_next_bar_leak_fails(self):
        def leak(d):
            c = d["close"]
            return np.sign(np.r_[c[1:], c[-1]] - c)
        r = check_lookahead(leak, n_cuts=16)
        assert r.verdict == "fail"
        assert r.details["cuts_with_change"] == 16
        assert all(h["reach_bars"] == 0 for h in r.details["examples"])

    def test_centered_window_fails(self):
        def leak(d):
            m = pd.Series(d["close"]).rolling(21, center=True).mean().to_numpy()
            return np.where(d["close"] < m, 1, 0)
        assert check_lookahead(leak, n_cuts=16).verdict == "fail"

    def test_full_period_volatility_quantile_fails(self):
        """Invisible to the mirror rewrite (same volatility), caught by scale2."""
        def leak(d):
            a = atr(d["high"], d["low"], d["close"], 14)
            return np.where(a > np.nanquantile(a, 0.8), 1, 0)
        r = check_lookahead(leak, n_cuts=16)
        assert r.verdict == "fail"
        assert {h["rewrite"] for h in r.details["examples"]} != {"mirror"}

    def test_tuple_output(self):
        def tup(d):
            s = _rsi_sig(d)
            sb = np.flatnonzero(s)
            return sb, s[sb]
        assert check_lookahead(tup, n_cuts=8).verdict == "pass"

    def test_nondeterministic_is_error(self):
        rng = np.random.default_rng()
        r = check_lookahead(lambda d: rng.integers(0, 2, len(d["close"])), n_cuts=4)
        assert r.verdict == "error" and "deterministic" in r.summary

    def test_wrong_length_is_error(self):
        r = check_lookahead(lambda d: np.zeros(3), n_cuts=4)
        assert r.verdict == "error" and "one value per bar" in r.summary

    def test_accepts_6_tuple(self):
        d = random_walk_bars(1000)
        six = (d["time"], d["open"], d["high"], d["low"], d["close"], np.zeros(1000))
        assert check_lookahead(_rsi_sig, six, n_cuts=8).verdict == "pass"

    def test_does_not_mutate_input(self):
        d = random_walk_bars(800)
        before = {k: v.copy() for k, v in d.items()}
        check_lookahead(_rsi_sig, d, n_cuts=4)
        for k in d:
            np.testing.assert_array_equal(d[k], before[k])


class TestNullEdge:
    def test_causal_backtest_passes(self):
        r = check_null_edge(_bt(_rsi_sig), n_paths=12, n_bars=3000)
        assert r.verdict == "pass", r.summary

    def test_peeking_backtest_fails(self):
        def peek(d):
            c = d["close"]
            return np.sign(np.r_[c[1:], c[-1]] - c)
        r = check_null_edge(_bt(peek), n_paths=12, n_bars=3000)
        assert r.verdict == "fail"
        assert r.details["t_across_paths"] > 3

    def test_entry_before_signal_warns(self):
        r = check_null_edge(_bt(_rsi_sig, shift=1), n_paths=12, n_bars=3000)
        assert r.verdict == "warn"

    def test_plain_array_output(self):
        def pnl(d):  # zero-mean noise that differs per path
            return np.random.default_rng(int(d["close"][-1] * 1e6) % 2**32).normal(0, 1, 200)
        r = check_null_edge(pnl, n_paths=5, n_bars=500)
        assert r.verdict == "pass"

    def test_too_few_trades_is_error(self):
        r = check_null_edge(lambda d: np.array([0.1]), n_paths=3, n_bars=500)
        assert r.verdict == "error"


class TestStrategyFile:
    CLEAN = '''
import numpy as np
from backtest_engine import simulate_trades, rsi, atr

def signals(data):
    r = rsi(data["close"], 14)
    return np.where(r < 30, 1, np.where(r > 70, -1, 0))

def backtest(data):
    sig = signals(data)
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    a = atr(h, l, c, 14)
    sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
    sb = sb[sb < len(c) - 1].astype(np.int32)
    return simulate_trades(h, l, c, sb, np.sign(sig[sb]).astype(np.int8),
                           a[sb] * 2, a[sb] * 3, max_hold=24, open_prices=o)
'''

    def test_clean_file(self, tmp_path):
        f = tmp_path / "strat.py"
        f.write_text(self.CLEAN)
        rep = audit_strategy(f, n_cuts=8, n_paths=8)
        assert rep["overall"] == "pass"
        checks = {r["check"] for r in rep["results"]}
        assert {"lookahead", "null_edge", "source:BG-04"} <= checks

    def test_leaky_file_cli_json(self, tmp_path, capsys):
        f = tmp_path / "strat.py"
        f.write_text(self.CLEAN.replace(
            'r = rsi(data["close"], 14)',
            'r = rsi(data["close"], 14)\n    r = np.r_[r[1:], np.nan]'))
        code = main([str(f), "--json", "--cuts", "8", "--paths", "8"])
        out = json.loads(capsys.readouterr().out)
        assert code == 1
        assert out["overall"] == "fail"
        assert any(r["check"] == "lookahead" and r["verdict"] == "fail" for r in out["results"])

    def test_file_without_functions(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("x = 1\n")
        assert audit_strategy(f)["overall"] == "error"

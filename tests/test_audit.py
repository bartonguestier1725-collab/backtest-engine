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


def _bt(sig_fn, shift=0, sl=2.0, tp=3.0, trail=False, cost=None):
    def bt(d, sig=None):
        sig = sig_fn(d) if sig is None else sig
        o, h, l, c = d["open"], d["high"], d["low"], d["close"]
        a = atr(h, l, c, 14)
        sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
        sb = sb[(sb >= 1) & (sb < len(c) - 1)].astype(np.int32)
        dirs = np.sign(sig[sb]).astype(np.int8)
        kw = dict(exit_mode="trailing", trail_activation_r=0.5, trail_distance_r=1.0) if trail else {}
        return simulate_trades(h, l, c, sb - shift, dirs, a[sb] * sl, a[sb] * tp,
                               max_hold=24, open_prices=o,
                               entry_costs=None if cost is None else np.full(len(sb), cost), **kw)
    return bt


def _placebo(sig_fn, **kw):
    bt = _bt(sig_fn, **kw)
    def placebo(d, k):
        sig = np.asarray(sig_fn(d))
        return bt(d, sig * np.random.default_rng(k + 7).choice([-1, 1], len(sig)))
    return placebo


class TestRandomWalkBars:
    def test_valid_ohlc(self):
        d = random_walk_bars(500, seed=3)
        assert np.all(d["high"] >= np.maximum(d["open"], d["close"]))
        assert np.all(d["low"] <= np.minimum(d["open"], d["close"]))
        np.testing.assert_allclose(d["open"][1:], d["close"][:-1])
        assert np.all(np.diff(d["time"]) == 3600)


class TestLookahead:
    def test_rare_signal_leak_is_caught(self):
        """Signals on <1% of bars: cuts are also placed at signal bars."""
        def rare(d):
            r = rsi(d["close"], 14)
            r = np.r_[r[1:], np.nan]           # next bar's RSI
            return np.where(r < 25, 1, 0)
        for seed in range(5):
            assert check_lookahead(rare, n_cuts=32, seed=seed).verdict == "fail"

    def test_no_signals_is_error_not_pass(self):
        r = check_lookahead(lambda d: np.zeros(len(d["close"])), n_cuts=8)
        assert r.verdict == "error" and "too few" in r.summary

    def test_float_noise_is_not_lookahead(self):
        from scipy.signal import fftconvolve
        def causal_fft_sma(d):
            c = d["close"]
            return fftconvolve(c, np.ones(20) / 20)[: len(c)] - c   # causal, float noise
        assert check_lookahead(causal_fft_sma, n_cuts=16).verdict == "pass"

    def test_wraparound_is_diagnosed(self):
        def roll(d):
            c = d["close"]
            return np.where(c > np.roll(c, 1), 1, -1)          # bar 0 sees the last bar
        r = check_lookahead(roll, n_cuts=8)
        assert r.verdict == "fail" and "wrap-around" in r.summary

    def test_future_volume_statistic_is_caught(self):
        def vol_q(d):
            v = d["volume"]
            return np.where(v > np.quantile(v, 0.9), 1, 0)
        assert check_lookahead(vol_q, n_cuts=16).verdict == "fail"

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

    def test_entry_before_signal_fails(self):
        r = check_null_edge(_bt(_rsi_sig, shift=1), n_paths=12, n_bars=3000)
        assert r.verdict == "fail" and "LOSS ON RANDOM DATA" in r.summary

    def test_entry_before_signal_fails_against_placebo(self):
        r = check_null_edge(_bt(_rsi_sig, shift=1), n_paths=12, n_bars=3000,
                            placebo_fn=_placebo(_rsi_sig, shift=1))
        assert r.verdict == "fail" and "DIFFERS FROM RANDOM DIRECTIONS" in r.summary

    @pytest.mark.parametrize("sl,tp,trail", [(0.5, 3.0, False), (1.0, 20.0, True), (2.0, 3.0, False)])
    def test_causal_tight_stops_and_trailing_pass(self, sl, tp, trail):
        """Stops fill fairly on the bridge null (review of v0.6.0: tight SL and
        trailing stops showed an edge on the old discrete-step null)."""
        r = check_null_edge(_bt(_rsi_sig, sl=sl, tp=tp, trail=trail), n_paths=20,
                            placebo_fn=_placebo(_rsi_sig, sl=sl, tp=tp, trail=trail))
        assert r.verdict == "pass", r.summary

    def test_costs_are_added_back(self):
        r = check_null_edge(_bt(_rsi_sig, cost=0.2), n_paths=12, n_bars=3000)
        assert r.verdict == "pass", r.summary

    def test_sl_first_bias_is_a_warning_with_placebo(self):
        """SL 0.5 / TP 0.5 on coarse bars: bars often touch both; the SL-first
        rule makes the backtest pessimistic. Same for the placebo -> warn."""
        r = check_null_edge(_bt(_rsi_sig, sl=0.5, tp=0.5), n_paths=20,
                            placebo_fn=_placebo(_rsi_sig, sl=0.5, tp=0.5))
        assert r.verdict == "warn", r.summary
        assert "SL-first" in r.summary

    def test_plain_array_output(self):
        def pnl(d):  # zero-mean noise that differs per path
            return np.random.default_rng(int(d["close"][-1] * 1e6) % 2**32).normal(0, 1, 200)
        r = check_null_edge(pnl, n_paths=10, n_bars=500)
        assert r.verdict == "pass"

    def test_too_few_paths_is_error(self):
        assert check_null_edge(_bt(_rsi_sig), n_paths=5).verdict == "error"

    def test_too_few_trades_is_error(self):
        r = check_null_edge(lambda d: np.array([0.1]), n_paths=10, n_bars=500)
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
        rep = audit_strategy(f, n_cuts=8, n_paths=10)
        assert rep["overall"] == "pass"
        checks = {r["check"] for r in rep["results"]}
        assert {"lookahead", "null_edge", "source:BG-04"} <= checks

    def test_leaky_file_cli_json(self, tmp_path, capsys):
        f = tmp_path / "strat.py"
        f.write_text(self.CLEAN.replace(
            'r = rsi(data["close"], 14)',
            'r = rsi(data["close"], 14)\n    r = np.r_[r[1:], np.nan]'))
        code = main([str(f), "--json", "--cuts", "8", "--paths", "10"])
        out = json.loads(capsys.readouterr().out)
        assert code == 1
        assert out["overall"] == "fail"
        assert any(r["check"] == "lookahead" and r["verdict"] == "fail" for r in out["results"])

    def test_file_without_functions(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("x = 1\n")
        assert audit_strategy(f)["overall"] == "error"

    def test_placebo_is_used_when_backtest_calls_signals(self, tmp_path):
        f = tmp_path / "strat.py"
        f.write_text(self.CLEAN)
        rep = audit_strategy(f, n_cuts=8, n_paths=10)
        null = [r for r in rep["results"] if r["check"] == "null_edge"][0]
        assert "t_vs_placebo" in null["details"]

    def test_same_named_helpers_in_two_folders(self, tmp_path):
        """A helper cached from one audit must not be reused by the next."""
        leaky_helper = "import numpy as np\ndef sig(c):\n    return np.sign(np.r_[c[1:], c[-1]] - c)\n"
        clean_helper = ("import numpy as np\nfrom backtest_engine import rsi\n"
                        "def sig(c):\n    r = rsi(c, 14)\n    return np.where(r < 30, 1, np.where(r > 70, -1, 0))\n")
        strat = "from helpers import sig\ndef signals(data):\n    return sig(data['close'])\n"
        for name, helper in (("b", leaky_helper), ("a", clean_helper)):
            (tmp_path / name).mkdir()
            (tmp_path / name / "helpers.py").write_text(helper)
            (tmp_path / name / "strat.py").write_text(strat)
        assert audit_strategy(tmp_path / "b" / "strat.py", n_cuts=8)["overall"] == "fail"
        assert audit_strategy(tmp_path / "a" / "strat.py", n_cuts=8)["overall"] == "pass"

    def test_dataclass_with_future_annotations_loads(self, tmp_path):
        f = tmp_path / "dc.py"
        f.write_text("from __future__ import annotations\nfrom dataclasses import dataclass\n"
                     "import numpy as np\nfrom backtest_engine import rsi\n\n"
                     "@dataclass\nclass P:\n    n: int = 14\n\n"
                     "def signals(data):\n    r = rsi(data['close'], P().n)\n"
                     "    return np.where(r < 30, 1, np.where(r > 70, -1, 0))\n")
        assert audit_strategy(f, n_cuts=8)["overall"] == "pass"

    def test_load_error_is_error_exit_2(self, tmp_path, capsys):
        f = tmp_path / "broken.py"
        f.write_text("import not_a_real_module_xyz\n")
        code = main([str(f), "--json"])
        out = json.loads(capsys.readouterr().out)
        assert code == 2 and out["overall"] == "error"

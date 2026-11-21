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

    def test_inf_in_signal_does_not_hide_a_leak(self):
        def leak(d):
            c = d["close"]
            s = np.sign(np.r_[c[1:], c[-1]] - c).astype(float)
            s[0] = np.inf
            return s
        assert check_lookahead(leak, n_cuts=8).verdict == "fail"

    def test_reach_is_bars_ahead(self):
        def shift5(d):
            c = d["close"]
            return np.sign(np.r_[c[5:], np.repeat(c[-1], 5)] - c)
        r = check_lookahead(shift5, n_cuts=16)
        assert max(h["reach_bars"] for h in r.details["examples"]) <= 5
        assert "5 bars ahead" in r.summary or "next 5" in r.summary or "up to 5" in r.summary

    def test_zero_cuts_is_error(self):
        assert check_lookahead(_rsi_sig, n_cuts=0).verdict == "error"

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
        assert all(h["reach_bars"] == 1 for h in r.details["examples"])

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

    def test_entry_before_signal_fails_without_placebo(self):
        r = check_null_edge(_bt(_rsi_sig, shift=1), n_paths=12, n_bars=3000, placebo=False)
        assert r.verdict == "fail" and "LOSS ON RANDOM DATA" in r.summary

    def test_entry_before_signal_fails_against_placebo(self):
        r = check_null_edge(_bt(_rsi_sig, shift=1), n_paths=40, n_bars=3000)
        assert r.verdict == "fail" and "DIFFERS FROM RANDOM DIRECTIONS" in r.summary

    def test_direction_dependent_stops_pass(self):
        """Stop beyond the signal candle (review 3: a signal-level placebo
        recomputed these stops for the flipped side and failed every run)."""
        def bt(d):
            sig = _rsi_sig(d)
            o, h, l, c = d["open"], d["high"], d["low"], d["close"]
            a = atr(h, l, c, 14)
            sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
            sb = sb[sb < len(c) - 1].astype(np.int32)
            dirs = np.sign(sig[sb]).astype(np.int8)
            sl = np.where(dirs > 0, (c - l)[sb], (h - c)[sb]) + 0.2 * a[sb]
            return simulate_trades(h, l, c, sb, dirs, sl, 1.5 * sl, max_hold=24, open_prices=o)
        for seed in (0, 1, 2):
            r = check_null_edge(bt, n_paths=20, seed=seed)
            # Tight candle stops make the 1h fills pessimistic (warn is the
            # designed verdict); the point is that the placebo matches.
            assert r.verdict in ("pass", "warn"), r.summary
            assert abs(r.details["t_vs_placebo"]) < 3

    def test_placebo_through_partial(self):
        from functools import partial
        sig_fn = partial(_rsi_sig)
        r = check_null_edge(_bt(sig_fn, shift=1), n_paths=12, n_bars=3000)
        assert "t_vs_placebo" in r.details and r.verdict == "fail"

    def test_flip_is_thread_local_and_restored(self):
        import threading
        from backtest_engine import core
        from backtest_engine.audit import _coin_flip_directions
        d = random_walk_bars(2000, seed=5)
        bt = _bt(_rsi_sig)
        plain = np.asarray(bt(d)["pnl_r"])
        seen = {}
        def other():
            seen["pnl"] = np.asarray(bt(d)["pnl_r"])
        with _coin_flip_directions(np.random.SeedSequence(1)):
            th = threading.Thread(target=other)
            th.start(); th.join()
            flipped = np.asarray(bt(d)["pnl_r"])
        np.testing.assert_array_equal(seen["pnl"], plain)      # other thread untouched
        assert not np.array_equal(flipped, plain)
        with pytest.raises(RuntimeError):
            with _coin_flip_directions(np.random.SeedSequence(2)):
                raise RuntimeError("boom")
        assert getattr(core._DIRECTION_FLIP, "rng", None) is None

    def test_hires_is_flipped_once(self, monkeypatch):
        from backtest_engine import core, simulate_trades_hires
        from backtest_engine.audit import _coin_flip_directions
        calls = []
        real_sim = core.simulate_trades
        def spy(*a, **k):
            calls.append(np.asarray(k.get("directions", a[4] if len(a) > 4 else None)).copy())
            return real_sim(*a, **k)
        monkeypatch.setattr(core, "simulate_trades", spy)
        ts_1m = np.arange(6000, dtype=np.int64) * 60
        c = 1.1 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, 6000))
        o = np.r_[c[0], c[:-1]]; h = np.maximum(o, c) + 1e-4; l = np.minimum(o, c) - 1e-4
        ts_1h = np.arange(100, dtype=np.int64) * 3600
        dirs = np.ones(40, dtype=np.int8)
        with _coin_flip_directions(np.random.SeedSequence(3)):
            simulate_trades_hires(signal_timestamps=ts_1h, signal_bars=np.arange(5, 85, 2, dtype=np.int32),
                                  directions=dirs, sl_distances=np.full(40, 0.002),
                                  tp_distances=np.full(40, 0.004), max_hold=5, signal_bar_minutes=60,
                                  exec_timestamps=ts_1m, exec_opens=o, exec_highs=h, exec_lows=l,
                                  exec_closes=c)
        assert len(calls) == 1
        np.testing.assert_array_equal(calls[0], dirs)          # hires passes them on as given
        monkeypatch.setattr(core, "simulate_trades", real_sim)
        kw = dict(signal_timestamps=ts_1h, signal_bars=np.arange(5, 85, 2, dtype=np.int32),
                  directions=dirs, sl_distances=np.full(40, 0.002), tp_distances=np.full(40, 0.004),
                  max_hold=5, signal_bar_minutes=60, exec_timestamps=ts_1m, exec_opens=o,
                  exec_highs=h, exec_lows=l, exec_closes=c)
        plain = np.asarray(simulate_trades_hires(**kw)["pnl_r"])
        with _coin_flip_directions(np.random.SeedSequence(3)):
            flipped = np.asarray(simulate_trades_hires(**kw)["pnl_r"])
        flips = np.random.default_rng(np.random.SeedSequence(3)).choice(np.array([-1, 1]), 40)
        assert not np.array_equal(plain, flipped)
        # trades whose coin came up +1 are unchanged: flipped once, inside simulate_trades
        np.testing.assert_array_equal(flipped[flips == 1], plain[flips == 1])

    def test_seeds_share_no_paths(self):
        seen = {0: [], 1: []}
        def recorder(seed):
            def bt(d):
                seen[seed].append(d["close"][:50].copy())
                return np.random.default_rng(len(seen[seed])).normal(0, 1, 100)
            return bt
        for seed in (0, 1):
            check_null_edge(recorder(seed), n_paths=10, n_bars=200, seed=seed, placebo=False)
        assert not any(np.allclose(a, b) for a in seen[0] for b in seen[1])

    @pytest.mark.parametrize("sl,tp,trail", [(0.5, 3.0, False), (1.0, 20.0, True), (2.0, 3.0, False)])
    def test_causal_tight_stops_and_trailing_pass(self, sl, tp, trail):
        """Stops fill fairly on the bridge null (review of v0.6.0: tight SL and
        trailing stops showed an edge on the old discrete-step null)."""
        r = check_null_edge(_bt(_rsi_sig, sl=sl, tp=tp, trail=trail), n_paths=20)
        assert r.verdict == "pass", r.summary

    def test_costs_are_added_back(self):
        r = check_null_edge(_bt(_rsi_sig, cost=0.2), n_paths=12, n_bars=3000)
        assert r.verdict == "pass", r.summary

    def test_sl_first_bias_is_a_warning_with_placebo(self):
        """SL 0.5 / TP 0.5 on coarse bars: bars often touch both; the SL-first
        rule makes the backtest pessimistic. Same for the placebo -> warn."""
        r = check_null_edge(_bt(_rsi_sig, sl=0.5, tp=0.5), n_paths=20)
        assert r.verdict == "warn", r.summary
        assert "SL-first" in r.summary

    def test_plain_array_output(self):
        def pnl(d):  # zero-mean noise that differs per path
            return np.random.default_rng(int(d["close"][-1] * 1e6) % 2**32).normal(0, 1, 200)
        r = check_null_edge(pnl, n_paths=10, n_bars=500)
        assert r.verdict == "pass"
        assert "no effect" in r.details["placebo"]

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

    def test_placebo_is_used(self, tmp_path):
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

    def test_helper_package_not_reused(self, tmp_path):
        leaky = "import numpy as np\ndef sig(c):\n    return np.sign(np.r_[c[1:], c[-1]] - c)\n"
        clean = ("import numpy as np\nfrom backtest_engine import rsi\n"
                 "def sig(c):\n    r = rsi(c, 14)\n    return np.where(r < 30, 1, np.where(r > 70, -1, 0))\n")
        strat = "from helperpkg import sig\ndef signals(data):\n    return sig(data['close'])\n"
        for name, helper in (("a", clean), ("b", leaky)):
            (tmp_path / name / "helperpkg").mkdir(parents=True)
            (tmp_path / name / "helperpkg" / "__init__.py").write_text(helper)
            (tmp_path / name / "strat.py").write_text(strat)
        assert audit_strategy(tmp_path / "a" / "strat.py", n_cuts=8)["overall"] == "pass"
        assert audit_strategy(tmp_path / "b" / "strat.py", n_cuts=8)["overall"] == "fail"

    def test_namespace_helper_folder_not_reused(self, tmp_path):
        """A helper in a subfolder without __init__.py (review 5)."""
        leaky = "import numpy as np\ndef sig(c):\n    return np.sign(np.r_[c[1:], c[-1]] - c)\n"
        clean = ("import numpy as np\nfrom backtest_engine import rsi\n"
                 "def sig(c):\n    r = rsi(c, 14)\n    return np.where(r < 30, 1, np.where(r > 70, -1, 0))\n")
        strat = "from nshelp.impl import sig\ndef signals(data):\n    return sig(data['close'])\n"
        for name, helper in (("a", clean), ("b", leaky)):
            (tmp_path / name / "nshelp").mkdir(parents=True)
            (tmp_path / name / "nshelp" / "impl.py").write_text(helper)
            (tmp_path / name / "strat.py").write_text(strat)
        assert audit_strategy(tmp_path / "a" / "strat.py", n_cuts=8)["overall"] == "pass"
        assert audit_strategy(tmp_path / "b" / "strat.py", n_cuts=8)["overall"] == "fail"

    def test_callers_module_is_restored(self, tmp_path, monkeypatch):
        import sys, types
        mine = types.ModuleType("config")
        mine.__file__ = str(tmp_path / "elsewhere" / "config.py")
        monkeypatch.setitem(sys.modules, "config", mine)
        (tmp_path / "s").mkdir()
        (tmp_path / "s" / "config.py").write_text("N = 14\n")
        (tmp_path / "s" / "strat.py").write_text(
            "import numpy as np\nfrom backtest_engine import rsi\nimport config\n"
            "def signals(data):\n    r = rsi(data['close'], config.N)\n"
            "    return np.where(r < 30, 1, np.where(r > 70, -1, 0))\n")
        assert audit_strategy(tmp_path / "s" / "strat.py", n_cuts=8)["overall"] == "pass"
        assert sys.modules["config"] is mine

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

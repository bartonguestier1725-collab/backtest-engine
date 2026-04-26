"""Tests for backtest_engine.data — AggVault data fetching."""

import json
import urllib.error
from io import BytesIO
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from backtest_engine.data import fetch_aggvault, _iso_to_epoch


# ── _iso_to_epoch ────────────────────────────────────────────────────

class TestIsoToEpoch:
    def test_valid_date(self):
        assert _iso_to_epoch("2021-04-01", "start") == 1617235200

    def test_valid_date_end_of_year(self):
        assert _iso_to_epoch("2025-12-31", "end") == 1767139200

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid start date"):
            _iso_to_epoch("04-01-2021", "start")

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Invalid end date"):
            _iso_to_epoch("not-a-date", "end")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid start date"):
            _iso_to_epoch("", "start")


# ── fetch_aggvault — validation ──────────────────────────────────────

class TestFetchAggvaultValidation:
    def test_no_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                fetch_aggvault("EURUSD", "1h", "2021-04-01", "2026-03-31")

    def test_env_var_fallback(self):
        """API key from AGGVAULT_KEY env var should be used."""
        mock_bars = [{"time": 1617235200, "open": 1.17, "high": 1.18, "low": 1.16, "close": 1.175}]
        response = MagicMock()
        response.read.return_value = json.dumps(mock_bars).encode()
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)

        with patch.dict("os.environ", {"AGGVAULT_KEY": "tk_test_key"}):
            with patch("urllib.request.urlopen", return_value=response) as mock_open:
                fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-02")
                # Verify the key was used in the request
                req = mock_open.call_args[0][0]
                assert "tk_test_key" in req.get_header("Authorization")

    def test_invalid_timeframe_raises(self):
        with pytest.raises(ValueError, match="Invalid timeframe.*2h"):
            fetch_aggvault("EURUSD", "2h", "2021-04-01", "2026-03-31", api_key="tk_x")

    def test_valid_timeframes_4h_1d(self):
        """4h and 1d are valid timeframes since AggVault v2."""
        bars = [{"time": 1617235200, "open": 1.17, "high": 1.18, "low": 1.16, "close": 1.175}]
        with patch("urllib.request.urlopen", return_value=_mock_response(bars)):
            for tf in ("4h", "1d"):
                result = fetch_aggvault("EURUSD", tf, "2021-04-01", "2021-04-02", api_key="tk_x")
                assert len(result[0]) == 1

    def test_invalid_timeframe_daily_raises(self):
        with pytest.raises(ValueError, match="Invalid timeframe.*1D"):
            fetch_aggvault("EURUSD", "1D", "2021-04-01", "2026-03-31", api_key="tk_x")

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError, match="start.*must be on or before end"):
            fetch_aggvault("EURUSD", "1h", "2026-03-31", "2021-04-01", api_key="tk_x")

    def test_same_start_end_allowed(self):
        """Same-day range is valid (end is inclusive)."""
        bars = [{"time": 1617235200, "open": 1.17, "high": 1.18, "low": 1.16, "close": 1.175}]
        with patch("urllib.request.urlopen", return_value=_mock_response(bars)):
            result = fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-01", api_key="tk_x")
        assert len(result[0]) == 1

    def test_invalid_start_date_raises(self):
        with pytest.raises(ValueError, match="Invalid start date"):
            fetch_aggvault("EURUSD", "1h", "2021-13-01", "2026-03-31", api_key="tk_x")

    def test_invalid_end_date_raises(self):
        with pytest.raises(ValueError, match="Invalid end date"):
            fetch_aggvault("EURUSD", "1h", "2021-04-01", "not-a-date", api_key="tk_x")


# ── fetch_aggvault — HTTP handling ───────────────────────────────────

def _mock_response(data: list[dict]) -> MagicMock:
    """Create a mock urllib response with JSON data."""
    response = MagicMock()
    response.read.return_value = json.dumps(data).encode()
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    return response


def _mock_http_error(code: int, body: str = "") -> urllib.error.HTTPError:
    """Create a mock HTTPError."""
    return urllib.error.HTTPError(
        url="https://tick.hugen.tokyo/api/v1/historical/EURUSD",
        code=code,
        msg="Error",
        hdrs={},
        fp=BytesIO(body.encode()),
    )


class TestFetchAggvaultHTTP:
    SAMPLE_BARS = [
        {"time": 1617235200, "open": 1.17271, "high": 1.17322, "low": 1.17212, "close": 1.17294},
        {"time": 1617238800, "open": 1.17296, "high": 1.17319, "low": 1.17274, "close": 1.17298},
        {"time": 1617242400, "open": 1.17299, "high": 1.17328, "low": 1.17215, "close": 1.17244},
    ]

    def test_successful_fetch_returns_6_tuple(self):
        with patch("urllib.request.urlopen", return_value=_mock_response(self.SAMPLE_BARS)):
            result = fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-02", api_key="tk_x")

        assert len(result) == 6
        timestamps, opens, highs, lows, closes, volume = result
        assert len(timestamps) == 3
        assert timestamps.dtype == np.int64
        assert opens.dtype == np.float64

    def test_timestamps_correct(self):
        with patch("urllib.request.urlopen", return_value=_mock_response(self.SAMPLE_BARS)):
            timestamps, *_ = fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-02", api_key="tk_x")

        np.testing.assert_array_equal(timestamps, [1617235200, 1617238800, 1617242400])

    def test_ohlc_values_correct(self):
        with patch("urllib.request.urlopen", return_value=_mock_response(self.SAMPLE_BARS)):
            _, opens, highs, lows, closes, _ = fetch_aggvault(
                "EURUSD", "1h", "2021-04-01", "2021-04-02", api_key="tk_x"
            )

        assert opens[0] == pytest.approx(1.17271)
        assert highs[0] == pytest.approx(1.17322)
        assert lows[0] == pytest.approx(1.17212)
        assert closes[0] == pytest.approx(1.17294)

    def test_volume_is_zeros(self):
        with patch("urllib.request.urlopen", return_value=_mock_response(self.SAMPLE_BARS)):
            *_, volume = fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-02", api_key="tk_x")

        np.testing.assert_array_equal(volume, [0.0, 0.0, 0.0])

    def test_compatible_with_load_ohlcv_format(self):
        """Return format matches load_ohlcv: (timestamps, opens, highs, lows, closes, volume)."""
        with patch("urllib.request.urlopen", return_value=_mock_response(self.SAMPLE_BARS)):
            result = fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-02", api_key="tk_x")

        # Unpack same as load_ohlcv
        timestamps, opens, highs, lows, closes, volume = result
        assert isinstance(timestamps, np.ndarray)
        assert isinstance(volume, np.ndarray)

    def test_user_agent_header_set(self):
        with patch("urllib.request.urlopen", return_value=_mock_response(self.SAMPLE_BARS)) as mock_open:
            fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-02", api_key="tk_x")

        req = mock_open.call_args[0][0]
        assert req.get_header("User-agent") == "backtest-engine/0.5.1"

    def test_authorization_header_set(self):
        with patch("urllib.request.urlopen", return_value=_mock_response(self.SAMPLE_BARS)) as mock_open:
            fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-02", api_key="tk_mykey")

        req = mock_open.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer tk_mykey"

    def test_url_construction(self):
        with patch("urllib.request.urlopen", return_value=_mock_response(self.SAMPLE_BARS)) as mock_open:
            fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-02", api_key="tk_x")

        req = mock_open.call_args[0][0]
        assert "EURUSD" in req.full_url
        assert "tf=1h" in req.full_url
        assert "from=1617235200" in req.full_url

    def test_end_date_inclusive(self):
        """End date should include the full day (23:59:59)."""
        with patch("urllib.request.urlopen", return_value=_mock_response(self.SAMPLE_BARS)) as mock_open:
            fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-02", api_key="tk_x")

        req = mock_open.call_args[0][0]
        # 2021-04-02 00:00:00 UTC = 1617321600, +86400-1 = 1617407999 (23:59:59)
        assert "to=1617407999" in req.full_url

    def test_symbol_normalization_lowercase(self):
        """Lowercase symbols should be uppercased in URL."""
        with patch("urllib.request.urlopen", return_value=_mock_response(self.SAMPLE_BARS)) as mock_open:
            fetch_aggvault("eurusd", "1h", "2021-04-01", "2021-04-02", api_key="tk_x")

        req = mock_open.call_args[0][0]
        assert "/EURUSD?" in req.full_url

    def test_symbol_normalization_slash(self):
        """Symbols with slashes should have slashes removed."""
        with patch("urllib.request.urlopen", return_value=_mock_response(self.SAMPLE_BARS)) as mock_open:
            fetch_aggvault("EUR/USD", "1h", "2021-04-01", "2021-04-02", api_key="tk_x")

        req = mock_open.call_args[0][0]
        assert "/EURUSD?" in req.full_url

    def test_invalid_symbol_raises(self):
        with pytest.raises(ValueError, match="Invalid symbol"):
            fetch_aggvault("EUR USD!", "1h", "2021-04-01", "2021-04-02", api_key="tk_x")

    def test_custom_base_url(self):
        with patch("urllib.request.urlopen", return_value=_mock_response(self.SAMPLE_BARS)) as mock_open:
            fetch_aggvault(
                "EURUSD", "1h", "2021-04-01", "2021-04-02",
                api_key="tk_x", base_url="https://custom.example.com",
            )

        req = mock_open.call_args[0][0]
        assert req.full_url.startswith("https://custom.example.com/")


# ── fetch_aggvault — error handling ──────────────────────────────────

class TestFetchAggvaultErrors:
    def test_401_raises_with_message(self):
        with patch("urllib.request.urlopen", side_effect=_mock_http_error(401)):
            with pytest.raises(RuntimeError, match="Invalid API key"):
                fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-02", api_key="tk_bad")

    def test_403_raises_with_plan_message(self):
        with patch("urllib.request.urlopen", side_effect=_mock_http_error(403)):
            with pytest.raises(RuntimeError, match="pro.*enterprise"):
                fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-02", api_key="tk_free")

    def test_404_raises_with_symbol(self):
        with patch("urllib.request.urlopen", side_effect=_mock_http_error(404)):
            with pytest.raises(RuntimeError, match="No data found.*NZDJPY"):
                fetch_aggvault("NZDJPY", "1h", "2021-04-01", "2021-04-02", api_key="tk_x")

    def test_429_raises_with_retry_message(self):
        with patch("urllib.request.urlopen", side_effect=_mock_http_error(429)):
            with pytest.raises(RuntimeError, match="Rate limited"):
                fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-02", api_key="tk_x")

    def test_500_raises_with_code(self):
        with patch("urllib.request.urlopen", side_effect=_mock_http_error(500, "Internal Server Error")):
            with pytest.raises(RuntimeError, match="API error 500"):
                fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-02", api_key="tk_x")

    def test_empty_response_raises(self):
        with patch("urllib.request.urlopen", return_value=_mock_response([])):
            with pytest.raises(RuntimeError, match="No bars returned"):
                fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-02", api_key="tk_x")

    def test_network_error_raises(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(RuntimeError, match="Cannot reach AggVault"):
                fetch_aggvault("EURUSD", "1h", "2021-04-01", "2021-04-02", api_key="tk_x")

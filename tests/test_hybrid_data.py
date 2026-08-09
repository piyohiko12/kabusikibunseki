import pathlib
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
from unittest.mock import Mock, patch

import pandas as pd

from lib import data_fetcher, moomoo_client, moomoo_fetcher, settings_store


class _KLType:
    K_1M = "K_1M"
    K_5M = "K_5M"
    K_15M = "K_15M"
    K_60M = "K_60M"
    K_DAY = "K_DAY"
    K_WEEK = "K_WEEK"
    K_MON = "K_MON"


class _Session:
    RTH = "RTH"
    NONE = "NONE"


class _AuType:
    QFQ = "QFQ"


def _raw_history(close: float = 101.0) -> pd.DataFrame:
    return pd.DataFrame({
        "time_key": ["2026-08-07 09:30:00"],
        "open": [100.0], "high": [102.0], "low": [99.0],
        "close": [close], "volume": [1000.0],
    })


def _history_frame(close: float = 101.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": [100.0], "High": [102.0], "Low": [99.0],
         "Close": [close], "Volume": [1000.0]},
        index=pd.DatetimeIndex(["2026-08-07"], name="Date"),
    )


class _QuoteContext:
    def __init__(self, quota_data=(0, 100), quota_ret=0):
        self.quota_data = quota_data
        self.quota_ret = quota_ret
        self.quota_calls = 0
        self.history_calls = 0
        self.market_state_calls = 0
        self.closed = False

    def get_history_kl_quota(self, get_detail=False):
        self.quota_calls += 1
        return self.quota_ret, self.quota_data

    def request_history_kline(self, **_kwargs):
        self.history_calls += 1
        return 0, _raw_history(), None

    def get_market_state(self, codes):
        self.market_state_calls += 1
        return 0, pd.DataFrame({
            "code": codes, "stock_name": ["Apple"],
            "market_state": ["MORNING"],
        })

    def close(self):
        self.closed = True


def _enabled_settings(reserve=10):
    return {"enabled": True, "host": "127.0.0.1", "port": 11111,
            "history_reserve": reserve}


@contextmanager
def _enabled_history_settings(reserve=10):
    with patch.object(moomoo_fetcher, "_integration_settings",
                      return_value=_enabled_settings(reserve)), \
            patch.object(moomoo_fetcher, "history_enabled", return_value=True):
        yield


class MoomooFetcherTests(unittest.TestCase):
    def test_normalize_code(self):
        self.assertEqual(moomoo_fetcher.normalize_code("AAPL"), "US.AAPL")
        self.assertEqual(moomoo_fetcher.normalize_code("US.NVDA"), "US.NVDA")
        self.assertEqual(moomoo_fetcher.normalize_code("700.HK"), "HK.00700")
        self.assertEqual(moomoo_fetcher.normalize_code("7203.T"), "JP.7203")

    def test_yahoo_only_code_is_rejected_for_fallback(self):
        with self.assertRaises(moomoo_fetcher.MoomooError):
            moomoo_fetcher.normalize_code("^GSPC")

    def test_period_start(self):
        now = pd.Timestamp("2026-08-09")
        self.assertEqual(moomoo_fetcher._period_start("1y", now), "2025-08-09")
        self.assertEqual(moomoo_fetcher._period_start("5d", now), "2026-08-01")

    def test_normalise_history(self):
        source = pd.DataFrame({
            "time_key": ["2026-08-07 09:30:00", "2026-08-07 09:31:00"],
            "open": [100, 101], "high": [102, 103], "low": [99, 100],
            "close": [101, 102], "volume": [1000, 1200],
        })
        result = moomoo_fetcher._normalise_history(source)
        self.assertEqual(list(result.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(float(result.iloc[-1]["Close"]), 102.0)
        self.assertTrue(result.index.is_monotonic_increasing)

    def _fetch_patches(self, cache_dir: str, context: _QuoteContext,
                       now: str = "2026-08-09T00:00:00Z", reserve: int = 10):
        return (
            patch.object(moomoo_fetcher, "CACHE_DIR", Path(cache_dir)),
            patch.object(moomoo_fetcher, "_open_context", return_value=context),
            _enabled_history_settings(reserve),
            patch.object(moomoo_fetcher, "_now_utc",
                         return_value=pd.Timestamp(now)),
            patch.object(moomoo_fetcher, "KLType", _KLType),
            patch.object(moomoo_fetcher, "Session", _Session),
            patch.object(moomoo_fetcher, "AuType", _AuType),
            patch.object(moomoo_fetcher, "RET_OK", 0),
        )

    def test_existing_code_is_allowed_when_remaining_quota_is_zero(self):
        detail = {"code": "US.AAPL", "name": "Apple",
                  "request_time": "2026-08-01 12:00:00"}
        # 実機SDKは [used, remain, [detail, ...]] の入れ子を返す。
        context = _QuoteContext([100, 0, [detail]])
        with TemporaryDirectory() as cache_dir:
            patches = self._fetch_patches(cache_dir, context)
            with patches[0], patches[1], patches[2], patches[3], \
                    patches[4], patches[5], patches[6], patches[7]:
                result = moomoo_fetcher.fetch_history("AAPL", "1y")
        self.assertEqual(context.quota_calls, 1)
        self.assertEqual(context.history_calls, 1)
        self.assertEqual(result.attrs["moomoo_meta"]["remain"], 0)
        self.assertEqual(result.attrs["moomoo_meta"]["quota"], 100)

    def test_new_code_does_not_consume_reserved_quota(self):
        context = _QuoteContext((90, 10))
        with TemporaryDirectory() as cache_dir:
            patches = self._fetch_patches(cache_dir, context, reserve=10)
            with patches[0], patches[1], patches[2], patches[3], \
                    patches[4], patches[5], patches[6], patches[7]:
                with self.assertRaisesRegex(moomoo_fetcher.MoomooError, "明示許可"):
                    moomoo_fetcher.fetch_history("MSFT", "1y")
        self.assertEqual(context.quota_calls, 1)
        self.assertEqual(context.history_calls, 0)

    def test_new_code_requires_explicit_permission_even_with_many_slots(self):
        context = _QuoteContext((10, 90))
        with TemporaryDirectory() as cache_dir:
            patches = self._fetch_patches(cache_dir, context, reserve=10)
            with patches[0], patches[1], patches[2], patches[3], \
                    patches[4], patches[5], patches[6], patches[7]:
                with self.assertRaisesRegex(moomoo_fetcher.MoomooError, "明示許可"):
                    moomoo_fetcher.fetch_history("MSFT", "1y")
        self.assertEqual(context.history_calls, 0)

    def test_explicit_permission_still_preserves_reserve(self):
        context = _QuoteContext((99, 1))
        with TemporaryDirectory() as cache_dir:
            patches = self._fetch_patches(cache_dir, context, reserve=10)
            with patches[0], patches[1], patches[2], patches[3], \
                    patches[4], patches[5], patches[6], patches[7]:
                with self.assertRaisesRegex(moomoo_fetcher.MoomooError, "予約10"):
                    moomoo_fetcher.fetch_history(
                        "MSFT", "1y", allow_new_quota=True)
        self.assertEqual(context.history_calls, 0)

    def test_explicit_permission_can_use_slot_above_reserve(self):
        context = _QuoteContext((89, 11))
        with TemporaryDirectory() as cache_dir:
            patches = self._fetch_patches(cache_dir, context, reserve=10)
            with patches[0], patches[1], patches[2], patches[3], \
                    patches[4], patches[5], patches[6], patches[7]:
                result = moomoo_fetcher.fetch_history(
                    "MSFT", "1y", allow_new_quota=True)
        self.assertFalse(result.empty)
        self.assertEqual(context.history_calls, 1)
        self.assertEqual(result.attrs["moomoo_meta"]["remain"], 10)

    def test_quota_check_failure_never_calls_history_api(self):
        context = _QuoteContext("OpenD error", quota_ret=-1)
        with TemporaryDirectory() as cache_dir:
            patches = self._fetch_patches(cache_dir, context)
            with patches[0], patches[1], patches[2], patches[3], \
                    patches[4], patches[5], patches[6], patches[7]:
                with self.assertRaisesRegex(moomoo_fetcher.MoomooError, "利用枠確認"):
                    moomoo_fetcher.fetch_history("AAPL", "1y")
        self.assertEqual(context.history_calls, 0)

    def test_fresh_cache_skips_quota_and_history_calls(self):
        context = _QuoteContext((10, 90))
        with TemporaryDirectory() as cache_dir:
            patches = self._fetch_patches(cache_dir, context)
            with patches[0], patches[1], patches[2], patches[3], \
                    patches[4], patches[5], patches[6], patches[7]:
                first = moomoo_fetcher.fetch_history(
                    "AAPL", "1y", allow_new_quota=True)
                second = moomoo_fetcher.fetch_history("AAPL", "1y")
        self.assertFalse(first.empty)
        self.assertFalse(second.empty)
        self.assertEqual(context.quota_calls, 1)
        self.assertEqual(context.history_calls, 1)
        self.assertEqual(second.attrs["moomoo_meta"]["cache_status"], "fresh")

    def test_intraday_cache_expires_quickly_without_using_a_new_symbol_slot(self):
        context = _QuoteContext((10, 90))
        with TemporaryDirectory() as cache_dir:
            first_patches = self._fetch_patches(
                cache_dir, context, now="2026-08-09T00:00:00Z")
            with first_patches[0], first_patches[1], first_patches[2], \
                    first_patches[3], first_patches[4], first_patches[5], \
                    first_patches[6], first_patches[7]:
                moomoo_fetcher.fetch_history(
                    "AAPL", "5d", "1m", allow_new_quota=True)

            second_patches = self._fetch_patches(
                cache_dir, context, now="2026-08-09T00:02:00Z")
            with second_patches[0], second_patches[1], second_patches[2], \
                    second_patches[3], second_patches[4], second_patches[5], \
                    second_patches[6], second_patches[7]:
                refreshed = moomoo_fetcher.fetch_history(
                    "AAPL", "5d", "1m", allow_new_quota=True)

        self.assertEqual(context.history_calls, 2)
        self.assertEqual(refreshed.attrs["moomoo_meta"]["cache_status"], "refreshed")

    def test_stale_cache_is_used_when_reserve_blocks_refresh(self):
        first_context = _QuoteContext((10, 90))
        blocked_context = _QuoteContext((90, 10))
        with TemporaryDirectory() as cache_dir:
            first_patches = self._fetch_patches(
                cache_dir, first_context, now="2026-08-01T00:00:00Z")
            with first_patches[0], first_patches[1], first_patches[2], \
                    first_patches[3], first_patches[4], first_patches[5], \
                    first_patches[6], first_patches[7]:
                moomoo_fetcher.fetch_history(
                    "AAPL", "1y", allow_new_quota=True)

            blocked_patches = self._fetch_patches(
                cache_dir, blocked_context, now="2026-08-03T00:00:00Z")
            with blocked_patches[0], blocked_patches[1], blocked_patches[2], \
                    blocked_patches[3], blocked_patches[4], blocked_patches[5], \
                    blocked_patches[6], blocked_patches[7]:
                stale = moomoo_fetcher.fetch_history(
                    "AAPL", "1y", allow_new_quota=True)

        meta = stale.attrs["moomoo_meta"]
        self.assertEqual(meta["cache_status"], "stale")
        self.assertEqual(meta["remain"], 10)
        self.assertIn("保護", meta["fallback_reason"])
        self.assertEqual(blocked_context.history_calls, 0)

    def test_disabled_integration_does_not_use_cache_or_open_context(self):
        context_open = Mock()
        with TemporaryDirectory() as cache_dir, \
                patch.object(moomoo_fetcher, "CACHE_DIR", Path(cache_dir)), \
                patch.object(moomoo_fetcher, "_integration_settings",
                             return_value={"enabled": False, "host": "127.0.0.1",
                                           "port": 11111, "history_reserve": 10}), \
                patch.object(moomoo_fetcher, "history_enabled", return_value=False), \
                patch.object(moomoo_fetcher, "_open_context", context_open):
            moomoo_fetcher._write_history_cache(
                _history_frame(), "US.AAPL", "1y", "1d",
                {"fetched_at": "2026-08-09T00:00:00+00:00",
                 "quota": 100, "remain": 90})
            with self.assertRaisesRegex(moomoo_fetcher.MoomooError, "連携はオフ"):
                moomoo_fetcher.fetch_history("AAPL", "1y")
        context_open.assert_not_called()

    def test_market_state_is_read_only_quote_call(self):
        context = _QuoteContext()
        with patch.object(moomoo_fetcher, "_open_context", return_value=context):
            state = moomoo_fetcher.fetch_market_state("AAPL")
        self.assertEqual(state["market_state"], "MORNING")
        self.assertEqual(state["source"], "moomoo OpenAPI")
        self.assertEqual(context.market_state_calls, 1)
        self.assertFalse(context.closed)


class HybridDataTests(unittest.TestCase):
    def test_corporate_actions_are_merged_by_date(self):
        primary = pd.DataFrame(
            {"Open": [100], "High": [102], "Low": [99], "Close": [101],
             "Volume": [1000]},
            index=pd.to_datetime(["2026-08-07"]),
        )
        yahoo = pd.DataFrame(
            {"Dividends": [0.25], "Stock Splits": [0.0]},
            index=pd.DatetimeIndex(["2026-08-07"], tz="America/New_York"),
        )
        result = data_fetcher._merge_corporate_actions(primary, yahoo, "1d")
        self.assertEqual(float(result.iloc[0]["Dividends"]), 0.25)
        self.assertEqual(float(result.iloc[0]["Stock Splits"]), 0.0)

    def test_chart_history_passes_quota_permission_and_metadata(self):
        moomoo = _history_frame()
        moomoo.attrs["moomoo_meta"] = {
            "source": "moomoo OpenAPI", "code": "US.AAPL",
            "fetched_at": "2026-08-09T00:00:00+00:00",
            "cache_status": "fresh", "quota": 100, "remain": 12,
            "fallback_reason": None,
        }
        yahoo = pd.DataFrame()
        data_fetcher.fetch_chart_history.clear()
        with patch.object(moomoo_fetcher, "history_enabled", return_value=True), \
                patch.object(moomoo_fetcher, "fetch_history", return_value=moomoo) as fetcher, \
                patch.object(data_fetcher, "fetch_history", return_value=yahoo):
            result, meta = data_fetcher.fetch_chart_history(
                "AAPL", "1y", allow_new_quota=True)
        self.assertFalse(result.empty)
        fetcher.assert_called_once_with(
            "AAPL", "1y", "1d", allow_new_quota=True)
        self.assertEqual(meta["cache_status"], "fresh")
        self.assertEqual(meta["quota"], 100)
        self.assertEqual(meta["remain"], 12)
        self.assertEqual(set(meta), {
            "source", "code", "fetched_at", "cache_status", "quota", "remain",
            "fallback_reason",
        })

    def test_newer_yahoo_history_replaces_stale_moomoo_cache(self):
        moomoo = _history_frame(close=101.0)
        moomoo.attrs["moomoo_meta"] = {
            "source": "moomoo OpenAPI", "code": "US.AAPL",
            "fetched_at": "2026-08-07T00:00:00+00:00",
            "cache_status": "stale", "quota": 300, "remain": 298,
            "fallback_reason": "予約枠を保護",
        }
        yahoo = pd.DataFrame(
            {"Open": [109.0], "High": [111.0], "Low": [108.0],
             "Close": [110.0], "Volume": [2000.0]},
            index=pd.DatetimeIndex(["2026-08-08"], name="Date"),
        )
        data_fetcher.fetch_chart_history.clear()
        with patch.object(moomoo_fetcher, "history_enabled", return_value=True), \
                patch.object(moomoo_fetcher, "fetch_history", return_value=moomoo), \
                patch.object(data_fetcher, "fetch_history", return_value=yahoo):
            result, meta = data_fetcher.fetch_chart_history("AAPL", "1y")
        self.assertEqual(float(result.iloc[-1]["Close"]), 110.0)
        self.assertEqual(meta["source"], "Yahoo Finance")
        self.assertEqual(meta["cache_status"], "fallback")
        self.assertIn("期限切れ", meta["fallback_reason"])

    def test_sidebar_quota_supports_real_sdk_list_shape(self):
        context = Mock()
        context.get_history_kl_quota.return_value = (
            0, [2, 298, [{"code": "US.AAPL"}]])
        moomoo_client.history_quota.clear()
        with patch.object(moomoo_client, "_ctx", return_value=context), \
                patch.object(moomoo_client, "_ok", return_value=True):
            result = moomoo_client.history_quota()
        moomoo_client.history_quota.clear()
        self.assertEqual(result, {"used": 2, "remain": 298})

    def test_market_state_wrapper_returns_failure_metadata(self):
        data_fetcher.fetch_market_state.clear()
        with patch.object(moomoo_fetcher, "fetch_market_state",
                          side_effect=moomoo_fetcher.MoomooError("offline")):
            state = data_fetcher.fetch_market_state("AAPL")
        self.assertIsNone(state["market_state"])
        self.assertEqual(state["source"], "Unavailable")
        self.assertEqual(state["fallback_reason"], "offline")

class ConnectionIsolationTests(unittest.TestCase):
    """OpenDが応答しないときにUIを固まらせないための約束事。"""

    def test_module_never_creates_its_own_quote_context(self):
        # OpenQuoteContextはOpenD未起動だと例外を返さず無限に再接続する。
        # 接続はmoomoo_client(TCP事前確認+タイムアウト)経由に限定する。
        source = pathlib.Path(moomoo_fetcher.__file__).read_text(encoding="utf-8")
        self.assertNotIn("OpenQuoteContext(", source)

    def test_open_context_delegates_to_moomoo_client(self):
        with mock.patch.object(moomoo_client, "_ctx", return_value="CTX") as ctx:
            self.assertEqual(moomoo_fetcher._open_context(), "CTX")
        ctx.assert_called_once_with()

    def test_open_context_raises_instead_of_blocking(self):
        with mock.patch.object(moomoo_client, "_ctx", return_value=None), \
                mock.patch.object(moomoo_client, "status",
                                  return_value={"state": "no_opend",
                                                "message": "OpenDに接続できません"}):
            with self.assertRaises(moomoo_fetcher.MoomooError):
                moomoo_fetcher._open_context()

    def test_shared_context_is_not_closed(self):
        source = pathlib.Path(moomoo_fetcher.__file__).read_text(encoding="utf-8")
        self.assertNotIn("ctx.close()", source)


class HistoryQuotaOptInTests(unittest.TestCase):
    """歴史的K線クォータは30日戻らないので、既定では消費しない。"""

    def test_history_is_disabled_by_default(self):
        for settings in ({}, {"moomoo_enabled": True},
                         {"moomoo_chart_history": True}):
            with self.subTest(settings=settings):
                with mock.patch.object(settings_store, "load",
                                       return_value=settings):
                    self.assertFalse(moomoo_fetcher.history_enabled())

    def test_history_needs_both_switches(self):
        with mock.patch.object(settings_store, "load",
                               return_value={"moomoo_enabled": True,
                                             "moomoo_chart_history": True}):
            self.assertTrue(moomoo_fetcher.history_enabled())

    def test_fetch_history_refuses_before_touching_opend(self):
        with mock.patch.object(moomoo_fetcher, "history_enabled",
                               return_value=False), \
                mock.patch.object(moomoo_fetcher, "_open_context") as ctx:
            with self.assertRaises(moomoo_fetcher.MoomooError):
                moomoo_fetcher.fetch_history("AAPL", "1y", "1d")
        ctx.assert_not_called()

    def test_chart_history_skips_moomoo_when_opted_out(self):
        frame = pd.DataFrame({"Close": [1.0]},
                             index=pd.DatetimeIndex(["2026-08-07"]))
        data_fetcher.fetch_chart_history.clear()
        with mock.patch.object(moomoo_fetcher, "history_enabled",
                               return_value=False), \
                mock.patch.object(moomoo_fetcher, "fetch_history") as moomoo, \
                mock.patch.object(data_fetcher, "fetch_history",
                                  return_value=frame):
            result, meta = data_fetcher.fetch_chart_history("AAPL", "1y", "1d")
        data_fetcher.fetch_chart_history.clear()
        moomoo.assert_not_called()
        self.assertEqual(meta["source"], "Yahoo Finance")
        self.assertIsNone(meta["fallback_reason"])
        self.assertFalse(result.empty)


if __name__ == "__main__":
    unittest.main()

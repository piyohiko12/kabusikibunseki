import pathlib
import sys
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from lib import data_fetcher, moomoo_client


class _Session:
    NONE = "NONE"
    RTH = "RTH"
    ETH = "ETH"
    ALL = "ALL"
    OVERNIGHT = "OVERNIGHT"


class _SubType:
    K_1M = "K_1M"


class _KLType:
    K_1M = "K_1M"


class _AuType:
    NONE = "NONE"


def _fake_moomoo(*, all_session=True):
    module = types.ModuleType("moomoo")
    session = _Session
    if not all_session:
        session = type("SessionWithoutAll", (), {
            "NONE": "NONE", "RTH": "RTH", "ETH": "ETH",
            "OVERNIGHT": "OVERNIGHT",
        })
    module.Session = session
    module.SubType = _SubType
    module.KLType = _KLType
    module.AuType = _AuType
    return module


def _raw_klines(code: str) -> pd.DataFrame:
    base = 500.0 if code == "US.SPY" else 100.0
    return pd.DataFrame({
        "code": [code, code],
        "time_key": ["2026-08-21 09:31:00", "2026-08-21 09:30:00"],
        "open": [base + 1, base],
        "high": [base + 2, base + 1],
        "low": [base, base - 1],
        "close": [base + 1.5, base + 0.5],
        "volume": [2000, 1000],
        "turnover": [200_000, 100_000],
    })


def _raw_klines_at(code: str, times: tuple[str, ...]) -> pd.DataFrame:
    base = 500.0 if code == "US.SPY" else 100.0
    size = len(times)
    return pd.DataFrame({
        "code": [code] * size,
        "time_key": list(times),
        "open": [base + index for index in range(size)],
        "high": [base + index + 1 for index in range(size)],
        "low": [base + index - 1 for index in range(size)],
        "close": [base + index + 0.5 for index in range(size)],
        "volume": [1000 + index * 100 for index in range(size)],
        "turnover": [100_000 + index * 10_000 for index in range(size)],
    })


class _QuoteContext:
    def __init__(self, *, remain=100, quota_data=None,
                 sub_list=None, subscribe_ret=0, subscribe_exc=None,
                 record_failed_subscribe=False, unsubscribe_ret=0,
                 unsubscribe_exc=None, query_exc=None, fail_codes=(),
                 kline_times=None):
        self.remain = remain
        self.quota_data = quota_data
        self.sub_list = {
            str(kind): set(codes)
            for kind, codes in (sub_list or {}).items()
        }
        self.subscribe_ret = subscribe_ret
        self.subscribe_exc = subscribe_exc
        self.record_failed_subscribe = record_failed_subscribe
        self.unsubscribe_ret = unsubscribe_ret
        self.unsubscribe_exc = unsubscribe_exc
        self.query_exc = query_exc
        self.fail_codes = set(fail_codes)
        self.kline_times = tuple(kline_times) if kline_times else None
        self.query_calls = 0
        self.subscribe_calls = []
        self.unsubscribe_calls = []
        self.kline_calls = []

    def query_subscription(self, is_all_conn=True):
        self.query_calls += 1
        if self.query_exc is not None:
            raise self.query_exc
        if self.quota_data is not None:
            return 0, self.quota_data
        return 0, {
            "total_used": 0,
            "own_used": 0,
            "remain": self.remain,
            "sub_list": {
                subtype: sorted(codes)
                for subtype, codes in self.sub_list.items()
            },
        }

    def subscribe(self, codes, subtypes, **kwargs):
        self.subscribe_calls.append((list(codes), list(subtypes), dict(kwargs)))
        subtype = str(subtypes[0]).rsplit(".", 1)[-1]
        should_record = (
            self.record_failed_subscribe
            or (self.subscribe_ret == 0 and self.subscribe_exc is None)
        )
        if should_record:
            bucket = self.sub_list.setdefault(subtype, set())
            new_codes = set(codes) - bucket
            bucket.update(codes)
            self.remain = max(0, self.remain - len(new_codes))
        if self.subscribe_exc is not None:
            raise self.subscribe_exc
        return self.subscribe_ret, ("ok" if self.subscribe_ret == 0 else "failed")

    def unsubscribe(self, codes, subtypes):
        self.unsubscribe_calls.append((list(codes), list(subtypes)))
        if self.unsubscribe_exc is not None:
            raise self.unsubscribe_exc
        if self.unsubscribe_ret == 0:
            subtype = str(subtypes[0]).rsplit(".", 1)[-1]
            bucket = self.sub_list.setdefault(subtype, set())
            removed = bucket & set(codes)
            bucket.difference_update(codes)
            self.remain += len(removed)
        return self.unsubscribe_ret, "unsubscribe failed"

    def get_cur_kline(self, code, num, **kwargs):
        self.kline_calls.append((code, num, dict(kwargs)))
        if code in self.fail_codes:
            return -1, "permission denied"
        return 0, (_raw_klines_at(code, self.kline_times)
                   if self.kline_times else _raw_klines(code))


class CurrentKlineAdapterTests(unittest.TestCase):
    def setUp(self):
        moomoo_client._current_kline_lease.clear()
        data_fetcher.fetch_current_klines.clear()

    def tearDown(self):
        moomoo_client._current_kline_lease.clear()
        data_fetcher.fetch_current_klines.clear()

    def _call(self, context, symbols=("AAPL",), *, num=120,
              session="regular", clock=100.0, all_session=True):
        with patch.object(moomoo_client, "_ctx", return_value=context), \
                patch.object(moomoo_client.time, "monotonic", return_value=clock), \
                patch.dict(sys.modules, {
                    "moomoo": _fake_moomoo(all_session=all_session)
                }):
            return moomoo_client.current_klines(
                symbols, num=num, session=session)

    def test_subscribes_symbol_and_spy_together_and_returns_normalised_frames(self):
        context = _QuoteContext()
        result = self._call(context, ("AAPL",))

        self.assertTrue(result["meta"]["available"])
        self.assertFalse(result["meta"]["partial"])
        self.assertEqual(set(result["frames"]), {"AAPL", "SPY"})
        self.assertEqual(result["meta"]["requested_symbols"], ("AAPL",))
        self.assertEqual(result["meta"]["subscribed_symbols"], ("AAPL", "SPY"))
        self.assertEqual(
            context.subscribe_calls[0][0], ["US.AAPL", "US.SPY"])
        self.assertEqual(context.subscribe_calls[0][1], ["K_1M"])
        subscribe_kwargs = context.subscribe_calls[0][2]
        self.assertFalse(subscribe_kwargs["subscribe_push"])
        self.assertFalse(subscribe_kwargs["is_first_push"])
        self.assertTrue(subscribe_kwargs["extended_time"])
        self.assertEqual(subscribe_kwargs["session"], "ALL")
        self.assertTrue(all(call[2]["autype"] == "NONE"
                            for call in context.kline_calls))
        self.assertTrue(all(call[2]["ktype"] == "K_1M"
                            for call in context.kline_calls))
        aapl = result["frames"]["AAPL"]
        self.assertEqual(list(aapl.columns),
                         ["Open", "High", "Low", "Close", "Volume", "Turnover"])
        self.assertTrue(aapl.index.is_monotonic_increasing)
        self.assertEqual(float(aapl.iloc[-1]["Close"]), 101.5)
        fetched_at = pd.Timestamp(result["meta"]["fetched_at"])
        self.assertIsNotNone(fetched_at.tzinfo)
        decision = result["meta"]["decision_quotes"]["AAPL"]
        self.assertTrue(decision["available"])
        self.assertEqual(decision["symbol"], "AAPL")
        self.assertEqual(decision["session"], "regular")
        self.assertEqual(decision["price"], 101.5)
        self.assertIsNotNone(pd.Timestamp(decision["updated_at"]).tzinfo)
        self.assertEqual(result["meta"]["timeframe"], "K_1M")
        self.assertFalse(result["meta"]["uses_daily_bars"])
        self.assertFalse(result["meta"]["uses_history_quota"])
        self.assertEqual(result["meta"]["history_requests"], 0)

    def test_same_set_and_session_reuses_subscription(self):
        context = _QuoteContext()
        first = self._call(context, ("AAPL",), clock=100.0)
        second = self._call(context, ("US.AAPL",), clock=110.0)

        self.assertTrue(first["meta"]["available"])
        self.assertTrue(second["meta"]["available"])
        self.assertTrue(second["meta"]["subscription_reused"])
        self.assertEqual(context.query_calls, 1)
        self.assertEqual(len(context.subscribe_calls), 1)
        self.assertEqual(context.unsubscribe_calls, [])

    def test_same_set_reuses_all_session_subscription_after_session_switch(self):
        context = _QuoteContext()
        regular = self._call(context, clock=100.0, session="regular")
        context.kline_times = (
            "2026-08-21 16:00:00", "2026-08-21 16:01:00")
        after = self._call(context, clock=110.0, session="afterhours")

        self.assertTrue(regular["meta"]["available"])
        self.assertTrue(after["meta"]["available"])
        self.assertTrue(after["meta"]["subscription_reused"])
        self.assertEqual(context.query_calls, 1)
        self.assertEqual(len(context.subscribe_calls), 1)
        self.assertEqual(context.subscribe_calls[0][2]["session"], "ALL")
        self.assertTrue(context.subscribe_calls[0][2]["extended_time"])

    def test_recent_symbol_sets_coexist_and_share_existing_spy_subscription(self):
        context = _QuoteContext()
        self._call(context, ("AAPL",), clock=100.0)
        msft = self._call(context, ("MSFT",), clock=110.0)

        self.assertTrue(msft["meta"]["available"])
        self.assertEqual(context.subscribe_calls[1][0], ["US.MSFT"])
        self.assertEqual(context.unsubscribe_calls, [])
        self.assertEqual(len(moomoo_client._current_kline_leases), 2)

        reused = self._call(context, ("AAPL",), clock=115.0)
        self.assertTrue(reused["meta"]["subscription_reused"])
        self.assertEqual(len(context.subscribe_calls), 2)

    def test_only_idle_self_managed_lease_is_cleaned_on_new_acquisition(self):
        context = _QuoteContext()
        self._call(context, ("AAPL",), clock=100.0)
        self._call(context, ("MSFT",), clock=110.0)
        self._call(context, ("MSFT",), clock=220.0)
        goog = self._call(context, ("GOOG",), clock=231.0)

        self.assertTrue(goog["meta"]["available"])
        # AAPL leaseだけがidle 120秒超。利用中MSFTが使うSPYは解除しない。
        self.assertEqual(context.unsubscribe_calls, [(["US.AAPL"], ["K_1M"])])
        self.assertIn("US.SPY", context.sub_list["K_1M"])
        self.assertIn("US.MSFT", context.sub_list["K_1M"])
        self.assertEqual(context.subscribe_calls[-1][0], ["US.GOOG"])

    def test_failed_stale_cleanup_blocks_new_subscription_on_same_context(self):
        context = _QuoteContext(unsubscribe_ret=-1)
        self._call(context, ("AAPL",), clock=100.0)
        switched = self._call(context, ("MSFT",), clock=221.0)

        self.assertFalse(switched["meta"]["available"])
        self.assertIn("解除できません", switched["meta"]["errors"]["subscription"])
        self.assertEqual(len(context.subscribe_calls), 1)
        self.assertEqual(len(moomoo_client._current_kline_leases), 1)

    def test_existing_k1m_codes_do_not_consume_new_quota(self):
        both_existing = _QuoteContext(
            remain=0,
            sub_list={"SubType.K_1M": ["us.aapl", "US.SPY"]},
        )
        result = self._call(both_existing)
        self.assertTrue(result["meta"]["available"])
        self.assertEqual(result["meta"]["quota"]["required"], 0)
        # query_subscriptionはsession条件を返さないため、既存codeを一度だけ
        # ALLで再subscribeしてRTH購読の取り残しを防ぐ。新規枠は消費しない。
        self.assertEqual(len(both_existing.subscribe_calls), 1)
        self.assertEqual(
            both_existing.subscribe_calls[0][0], ["US.AAPL", "US.SPY"])
        self.assertEqual(both_existing.subscribe_calls[0][2]["session"], "ALL")
        self.assertEqual(both_existing.remain, 0)

        moomoo_client._current_kline_leases.clear()
        partly_existing = _QuoteContext(
            remain=0, sub_list={"K_1M": ("US.SPY",)})
        result = self._call(partly_existing)
        self.assertFalse(result["meta"]["available"])
        self.assertIn("不足", result["meta"]["errors"]["subscription"])
        self.assertEqual(result["meta"]["quota"]["required"], 1)
        self.assertEqual(result["meta"]["subscribed_symbols"], ("SPY",))
        self.assertEqual(partly_existing.subscribe_calls, [])

    def test_insufficient_or_unparseable_quota_fails_closed(self):
        insufficient = _QuoteContext(remain=1)
        result = self._call(insufficient)
        self.assertFalse(result["meta"]["available"])
        self.assertIn("不足", result["meta"]["errors"]["subscription"])
        self.assertEqual(insufficient.subscribe_calls, [])
        self.assertEqual(result["meta"]["quota"]["required"], 2)

        moomoo_client._current_kline_lease.clear()
        malformed = _QuoteContext(quota_data={
            "total_used": 0, "own_used": 0, "remain": "unknown", "sub_list": {},
        })
        result = self._call(malformed)
        self.assertFalse(result["meta"]["available"])
        self.assertIn("解析", result["meta"]["errors"]["subscription"])
        self.assertEqual(malformed.subscribe_calls, [])

        moomoo_client._current_kline_lease.clear()
        missing_list = _QuoteContext(quota_data={
            "total_used": 0, "own_used": 0, "remain": 100,
        })
        result = self._call(missing_list)
        self.assertFalse(result["meta"]["available"])
        self.assertIn("購読一覧", result["meta"]["errors"]["subscription"])
        self.assertEqual(missing_list.subscribe_calls, [])

        moomoo_client._current_kline_leases.clear()
        invalid_codes = _QuoteContext(quota_data={
            "total_used": 0, "own_used": 0, "remain": 100,
            "sub_list": {"K_1M": "US.AAPL"},
        })
        result = self._call(invalid_codes)
        self.assertFalse(result["meta"]["available"])
        self.assertIn("購読銘柄", result["meta"]["errors"]["subscription"])
        self.assertEqual(invalid_codes.subscribe_calls, [])

    def test_uncertain_subscription_reconciles_after_sixty_seconds(self):
        context = _QuoteContext(
            subscribe_ret=-1, record_failed_subscribe=True)
        first = self._call(context, clock=100.0)
        early = self._call(context, clock=150.0)
        reconciled = self._call(context, clock=161.0)

        self.assertFalse(first["meta"]["available"])
        self.assertFalse(early["meta"]["available"])
        self.assertTrue(reconciled["meta"]["available"])
        self.assertTrue(reconciled["meta"]["subscription_reused"])
        self.assertEqual(len(context.subscribe_calls), 1)
        lease = next(iter(moomoo_client._current_kline_leases.values()))
        self.assertEqual(lease["state"], "active")

    def test_uncertain_absence_is_cleared_and_subscription_is_retried(self):
        context = _QuoteContext(subscribe_exc=RuntimeError("timeout"))
        first = self._call(context, clock=100.0)
        context.subscribe_exc = None
        retried = self._call(context, clock=161.0)

        self.assertFalse(first["meta"]["available"])
        self.assertTrue(retried["meta"]["available"])
        self.assertEqual(len(context.subscribe_calls), 2)
        lease = next(iter(moomoo_client._current_kline_leases.values()))
        self.assertEqual(lease["state"], "active")

    def test_uncertain_existing_mode_is_retried_not_assumed_all(self):
        context = _QuoteContext(
            remain=0, sub_list={"K_1M": ("US.AAPL", "US.SPY")},
            subscribe_ret=-1)
        first = self._call(context, clock=100.0)
        early = self._call(context, clock=150.0)
        context.subscribe_ret = 0
        retried = self._call(context, clock=161.0)

        self.assertFalse(first["meta"]["available"])
        self.assertFalse(early["meta"]["available"])
        self.assertTrue(retried["meta"]["available"])
        self.assertEqual(len(context.subscribe_calls), 2)
        self.assertTrue(all(
            call[2]["session"] == "ALL" and call[2]["extended_time"]
            for call in context.subscribe_calls))

    def test_idle_uncertain_absence_does_not_block_another_set(self):
        context = _QuoteContext(
            subscribe_exc=RuntimeError("timeout"), unsubscribe_ret=-1)
        first = self._call(context, clock=100.0)
        context.subscribe_exc = None
        other = self._call(context, ("MSFT",), clock=221.0)

        self.assertFalse(first["meta"]["available"])
        self.assertTrue(other["meta"]["available"])
        self.assertEqual(context.unsubscribe_calls, [])
        self.assertEqual(context.subscribe_calls[-1][0], ["US.MSFT", "US.SPY"])

    def test_broken_old_context_does_not_poison_new_context(self):
        old = _QuoteContext(
            subscribe_exc=RuntimeError("subscribe timeout"),
            unsubscribe_exc=RuntimeError("old context closed"),
        )
        first = self._call(old, clock=100.0)
        old.query_exc = RuntimeError("old context closed")
        current = _QuoteContext()
        second = self._call(current, clock=110.0)
        self._call(current, clock=220.0)
        third = self._call(current, ("MSFT",), clock=231.0)

        self.assertFalse(first["meta"]["available"])
        self.assertTrue(second["meta"]["available"])
        self.assertTrue(third["meta"]["available"])
        self.assertEqual(len(current.subscribe_calls), 2)
        self.assertEqual(current.subscribe_calls[-1][0], ["US.MSFT"])

    def test_partial_failure_preserves_successful_frame_and_error(self):
        context = _QuoteContext(fail_codes={"US.SPY"})
        result = self._call(context)

        self.assertFalse(result["meta"]["available"])
        self.assertTrue(result["meta"]["partial"])
        self.assertEqual(set(result["frames"]), {"AAPL"})
        self.assertIn("SPY", result["meta"]["errors"])

    def test_extended_and_overnight_sessions_use_supported_kline_sessions(self):
        context = _QuoteContext(kline_times=(
            "2026-08-21 08:00:00", "2026-08-21 08:01:00"))
        pre = self._call(context, session="pre")
        kwargs = context.subscribe_calls[0][2]
        self.assertEqual(kwargs["session"], "ALL")
        self.assertTrue(kwargs["extended_time"])
        self.assertTrue(pre["meta"]["available"])
        self.assertEqual(
            pre["meta"]["decision_quotes"]["AAPL"]["session"], "premarket")

        moomoo_client._current_kline_leases.clear()
        after_context = _QuoteContext(kline_times=(
            "2026-08-21 16:00:00", "2026-08-21 16:01:00"))
        after = self._call(after_context, session="afterhours")
        self.assertTrue(after["meta"]["available"])
        self.assertEqual(after_context.subscribe_calls[0][2]["session"], "ALL")
        self.assertEqual(
            after["meta"]["decision_quotes"]["AAPL"]["session"], "afterhours")

        moomoo_client._current_kline_leases.clear()
        overnight_context = _QuoteContext(kline_times=(
            "2026-08-23 21:00:00", "2026-08-23 21:01:00"))
        overnight = self._call(overnight_context, session="overnight")
        overnight_kwargs = overnight_context.subscribe_calls[0][2]
        self.assertTrue(overnight["meta"]["available"])
        self.assertEqual(overnight_kwargs["session"], "ALL")
        self.assertNotEqual(overnight_kwargs["session"], "OVERNIGHT")
        self.assertEqual(
            overnight["meta"]["decision_quotes"]["AAPL"]["session"], "overnight")

        moomoo_client._current_kline_leases.clear()
        unsupported = self._call(
            _QuoteContext(), session="overnight", all_session=False)
        self.assertFalse(unsupported["meta"]["available"])
        self.assertIn("ALL", unsupported["meta"]["errors"]["session"])

    def test_early_close_uses_1300_afterhours_boundary(self):
        after_context = _QuoteContext(kline_times=(
            "2026-11-27 13:00:00", "2026-11-27 13:01:00"))
        after = self._call(after_context, session="afterhours")
        self.assertTrue(after["meta"]["available"])
        self.assertEqual(
            after["meta"]["decision_quotes"]["AAPL"]["session"],
            "afterhours")

        moomoo_client._current_kline_leases.clear()
        closed_context = _QuoteContext(kline_times=(
            "2026-11-27 18:00:00", "2026-11-27 18:01:00"))
        closed = self._call(closed_context, session="afterhours")
        self.assertFalse(closed["meta"]["available"])
        self.assertFalse(
            closed["meta"]["decision_quotes"]["AAPL"]["available"])

    def test_wrong_session_bars_are_fail_closed_but_frames_are_preserved(self):
        context = _QuoteContext()  # default fixture is regular 09:30 ET
        result = self._call(context, session="pre")

        self.assertFalse(result["meta"]["available"])
        self.assertTrue(result["meta"]["partial"])
        self.assertEqual(set(result["frames"]), {"AAPL", "SPY"})
        self.assertFalse(result["meta"]["decision_quotes"]["AAPL"]["available"])
        self.assertIn("指定セッション", result["meta"]["errors"]["AAPL"])

    def test_snapshot_exposes_timestamp_honest_session_quotes(self):
        context = Mock()
        context.get_market_snapshot.return_value = (0, pd.DataFrame([{
            "code": "US.AAPL", "name": "Apple", "last_price": 100.0,
            "prev_close_price": 99.0, "bid_price": 99.9, "ask_price": 100.1,
            "volume": 1000, "update_time": "2026-08-21 10:00:05",
            "pre_price": 99.5, "pre_volume": 200,
            "after_price": 100.5, "after_volume": 300,
            "overnight_price": 100.2, "overnight_volume": 100,
        }]))
        with patch.object(moomoo_client, "_ctx", return_value=context):
            result = moomoo_client.snapshot.__wrapped__(("AAPL",))

        row = result["AAPL"]
        sessions = row["session_quotes"]
        self.assertEqual(set(sessions), {
            "premarket", "regular", "afterhours", "overnight"})
        self.assertTrue(sessions["regular"]["timestamp_verified"])
        self.assertIsNotNone(pd.Timestamp(sessions["regular"]["updated_at"]).tzinfo)
        for name in ("premarket", "afterhours", "overnight"):
            self.assertTrue(sessions[name]["available"])
            self.assertIsNone(sessions[name]["updated_at"])
            self.assertFalse(sessions[name]["timestamp_verified"])
            self.assertIsNotNone(sessions[name]["snapshot_updated_at"])

    def test_snapshot_does_not_mark_regular_price_fresh_after_close(self):
        row = {
            "last_price": 100.0, "volume": 1000,
            "after_price": 100.5, "after_volume": 300,
            "update_time": "2026-08-21 16:00:05",
        }
        sessions = moomoo_client._snapshot_session_quotes(
            row, "2026-08-21T20:00:06+00:00")

        self.assertTrue(sessions["regular"]["available"])
        self.assertFalse(sessions["regular"]["timestamp_verified"])
        self.assertFalse(sessions["regular"]["actionable_from_snapshot"])
        self.assertIsNone(sessions["regular"]["updated_at"])
        self.assertTrue(sessions["afterhours"]["available"])

    def test_build_session_snapshot_keeps_regular_contract(self):
        raw = {
            "source": "moomoo OpenAPI", "price": 100.0,
            "bid": 99.9, "ask": 100.1,
            "update_time": "2026-08-21 10:00:05",
        }
        result = data_fetcher.build_session_snapshot(raw, {}, "regular")

        self.assertTrue(result["decision_ready"])
        self.assertEqual(result["price"], 100.0)
        self.assertEqual(result["bid"], 99.9)
        self.assertIsNotNone(pd.Timestamp(result["update_time"]).tzinfo)
        self.assertFalse(result["uses_daily_bars"])

    def test_build_session_snapshot_normalises_all_extended_sessions(self):
        cases = (
            ("pre", "premarket", "pre_price", "2026-08-21 08:00:20", 99.5),
            ("after", "afterhours", "after_price", "2026-08-21 16:00:20", 100.5),
            ("overnight", "overnight", "overnight_price",
             "2026-08-23 21:00:20", 100.2),
        )
        for requested, canonical, price_field, stamp, price in cases:
            with self.subTest(session=requested):
                raw = {
                    "source": "moomoo OpenAPI", "price": 100.0,
                    "code": "US.AAPL",
                    "bid": price - 0.1, "ask": price + 0.1,
                    "update_time_iso": pd.Timestamp(
                        stamp, tz="America/New_York").isoformat(),
                    "session_quotes": {
                        canonical: {
                            "available": True, "price": price,
                            "price_field": price_field,
                            "source": "moomoo OpenAPI snapshot",
                            "snapshot_updated_at": pd.Timestamp(
                                stamp, tz="America/New_York").isoformat(),
                        },
                    },
                }
                decision = {
                    "available": True, "session": canonical, "price": price,
                    "symbol": "AAPL", "source": "moomoo OpenAPI current K_1M",
                    "timestamp_verified": True, "timeframe": "K_1M",
                    "uses_daily_bars": False,
                    "bar_time": pd.Timestamp(
                        stamp, tz="America/New_York").floor("min").isoformat(),
                }
                result = data_fetcher.build_session_snapshot(
                    raw, decision, requested)
                self.assertTrue(result["decision_ready"])
                self.assertEqual(result["decision_session"], canonical)
                self.assertEqual(result["price"], price)
                self.assertEqual(result["update_time"], raw["update_time_iso"])

    def test_build_session_snapshot_uses_early_close_calendar(self):
        stamp = "2026-11-27T14:00:20-05:00"
        raw = {
            "source": "moomoo OpenAPI", "price": 100.0,
            "code": "US.AAPL", "bid": 100.4, "ask": 100.6,
            "update_time_iso": stamp,
            "session_quotes": {
                "afterhours": {
                    "available": True, "price": 100.5,
                    "price_field": "after_price",
                    "source": "moomoo OpenAPI snapshot",
                    "snapshot_updated_at": stamp,
                },
            },
        }
        decision = {
            "available": True, "session": "afterhours", "price": 100.5,
            "symbol": "AAPL", "source": "moomoo OpenAPI current K_1M",
            "timestamp_verified": True, "timeframe": "K_1M",
            "uses_daily_bars": False,
            "bar_time": "2026-11-27T14:00:00-05:00",
        }

        after = data_fetcher.build_session_snapshot(
            raw, decision, "afterhours")
        regular = data_fetcher.build_session_snapshot(raw, {}, "regular")
        self.assertTrue(after["decision_ready"])
        self.assertEqual(after["decision_session"], "afterhours")
        self.assertFalse(regular["decision_ready"])
        self.assertIsNone(regular["price"])

    def test_build_session_snapshot_rejects_stale_price_outside_tight_bbo(self):
        stamp = "2026-08-21T08:00:20-04:00"
        raw = {
            "source": "moomoo OpenAPI", "price": 100.0,
            "code": "US.AAPL", "bid": 99.0, "ask": 99.1,
            "update_time_iso": stamp,
            "session_quotes": {
                "premarket": {
                    "available": True, "price": 100.0,
                    "price_field": "pre_price",
                    "source": "moomoo OpenAPI snapshot",
                    "snapshot_updated_at": stamp,
                },
            },
        }
        decision = {
            "available": True, "session": "premarket", "price": 100.0,
            "symbol": "AAPL", "source": "moomoo OpenAPI current K_1M",
            "timestamp_verified": True, "timeframe": "K_1M",
            "uses_daily_bars": False,
            "bar_time": "2026-08-21T08:00:00-04:00",
        }

        result = data_fetcher.build_session_snapshot(
            raw, decision, "premarket")
        self.assertFalse(result["decision_ready"])
        self.assertIsNone(result["price"])
        self.assertTrue(any(
            "Bid/Ask" in error for error in result["decision_errors"]))

    def test_build_session_snapshot_clears_unverified_extended_quote(self):
        raw = {
            "source": "moomoo OpenAPI", "price": 100.0,
            "code": "US.AAPL",
            "bid": 99.9, "ask": 100.1,
            "update_time_iso": "2026-08-21T08:00:20-04:00",
            "session_quotes": {
                "premarket": {"available": True, "price": 100.0,
                               "price_field": "pre_price",
                               "source": "moomoo OpenAPI snapshot"},
            },
        }
        # K_1M証明がないため、generic regular priceを時間外へ流用しない。
        result = data_fetcher.build_session_snapshot(raw, {}, "premarket")
        self.assertFalse(result["decision_ready"])
        self.assertIsNone(result["price"])
        self.assertIsNone(result["bid"])
        self.assertIsNone(result["ask"])
        self.assertIsNone(result["update_time"])
        self.assertTrue(result["decision_errors"])

        daily = {
            "available": True, "session": "premarket", "price": 100.0,
            "symbol": "AAPL", "source": "moomoo OpenAPI current K_1M",
            "bar_time": "2026-08-21T08:00:00-04:00",
            "timestamp_verified": True, "timeframe": "K_DAY",
            "uses_daily_bars": True,
        }
        daily_result = data_fetcher.build_session_snapshot(
            raw, daily, "premarket")
        self.assertFalse(daily_result["decision_ready"])
        self.assertIsNone(daily_result["price"])
        self.assertTrue(any("K_1M" in error
                            for error in daily_result["decision_errors"]))

    def test_data_fetcher_wrapper_normalises_iterable_and_exposes_clear(self):
        expected = {"frames": {}, "meta": {"available": False}}
        with patch.object(moomoo_client, "current_klines",
                          return_value=expected) as current:
            result = data_fetcher.fetch_current_klines(["aapl"], 42, "regular")
        self.assertIs(result, expected)
        current.assert_called_once_with(("aapl",), num=42, session="regular")
        self.assertTrue(callable(data_fetcher.fetch_current_klines.clear))

    def test_adapter_source_contains_no_history_or_trading_calls(self):
        client_source = pathlib.Path(moomoo_client.__file__).read_text(encoding="utf-8")
        block = client_source.split("def current_klines", 1)[1].split(
            "# ------------------------------------------------------------------- 取得API", 1)[0]
        for forbidden in (
            "request_history_kline", "OpenSecTradeContext", "OpenFutureTradeContext",
            "place_order(", "modify_order(", "cancel_order(", "unlock_trade(",
        ):
            self.assertNotIn(forbidden, block)


if __name__ == "__main__":
    unittest.main()

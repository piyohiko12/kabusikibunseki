import inspect
import unittest
from datetime import date, time

import pandas as pd

from lib import session_intelligence as si


class CalendarAndSessionTests(unittest.TestCase):
    def test_dst_summer_and_winter_resolve_same_new_york_wall_time(self):
        summer = si.detect_current_session(pd.Timestamp("2026-07-06 13:30:00Z"))
        winter = si.detect_current_session(pd.Timestamp("2026-01-05 14:30:00Z"))
        self.assertEqual(summer["session"], "regular")
        self.assertEqual(winter["session"], "regular")
        self.assertEqual(summer["as_of"].hour, 9)
        self.assertEqual(winter["as_of"].hour, 9)
        self.assertNotEqual(summer["as_of"].utcoffset(), winter["as_of"].utcoffset())

    def test_session_boundaries_and_overnight_eligibility(self):
        cases = (
            ("2026-07-06 03:59:59-04:00", True, "overnight"),
            ("2026-07-06 04:00:00-04:00", None, "premarket"),
            ("2026-07-06 09:30:00-04:00", None, "regular"),
            ("2026-07-06 16:00:00-04:00", None, "afterhours"),
            ("2026-07-06 20:00:00-04:00", True, "overnight"),
        )
        for stamp, eligible, expected in cases:
            with self.subTest(stamp=stamp):
                result = si.detect_current_session(stamp, overnight_eligible=eligible)
                self.assertEqual(result["session"], expected)

        unknown = si.detect_current_session(
            "2026-07-06 21:00:00-04:00", overnight_eligible=None)
        self.assertEqual(unknown["session"], "unknown")
        self.assertEqual(unknown["calendar_session"], "overnight")
        self.assertIsNone(unknown["tradable"])
        self.assertEqual(unknown["data_quality"]["label"], "low")

        ineligible = si.detect_current_session(
            "2026-07-06 21:00:00-04:00", overnight_eligible=False)
        self.assertEqual(ineligible["session"], "closed")
        self.assertFalse(ineligible["tradable"])

    def test_weekend_and_sunday_night(self):
        friday_night = si.detect_current_session(
            "2026-07-10 21:00:00-04:00", overnight_eligible=True)
        sunday_night = si.detect_current_session(
            "2026-07-12 21:00:00-04:00", overnight_eligible=True)
        self.assertEqual(friday_night["session"], "closed")
        self.assertEqual(sunday_night["session"], "overnight")

    def test_market_state_is_authoritative_and_enum_text_is_supported(self):
        result = si.detect_current_session(
            "2026-07-11 12:00:00-04:00", market_state="MarketState.MORNING")
        self.assertEqual(result["session"], "regular")
        self.assertEqual(result["source"], "market_state")
        self.assertTrue(result["tradable"])
        self.assertTrue(result["warnings"])

        closed = si.detect_current_session(
            "2026-07-06 10:00:00-04:00", market_state="CLOSED")
        self.assertEqual(closed["session"], "closed")
        self.assertFalse(closed["tradable"])

    def test_naive_time_is_not_silent(self):
        result = si.detect_current_session("2026-07-06 10:00:00")
        self.assertEqual(result["session"], "regular")
        self.assertTrue(any("timezoneなし" in item for item in result["warnings"]))
        self.assertLess(result["data_quality"]["score"], 0.75)

    def test_holidays_good_friday_juneteenth_and_extra_closure(self):
        self.assertFalse(si.is_nyse_trading_day(date(2026, 4, 3)))
        self.assertFalse(si.is_nyse_trading_day(date(2026, 6, 19)))
        self.assertFalse(si.is_nyse_trading_day(date(2026, 7, 3)))
        self.assertFalse(si.is_nyse_trading_day(
            date(2026, 8, 11), extra_holidays=(date(2026, 8, 11),)))
        self.assertTrue(si.is_nyse_trading_day(date(2026, 8, 11)))

    def test_early_close_changes_regular_and_afterhours_boundary(self):
        # 2026 Thanksgiving is Nov 26; Nov 27 is a representative 13:00 close.
        self.assertEqual(si.regular_close_time(date(2026, 11, 27)), time(13, 0))
        self.assertEqual(si.afterhours_close_time(date(2026, 11, 27)), time(17, 0))
        at_noon = si.detect_current_session("2026-11-27 12:00:00-05:00")
        at_two = si.detect_current_session("2026-11-27 14:00:00-05:00")
        at_six = si.detect_current_session("2026-11-27 18:00:00-05:00")
        self.assertEqual(at_noon["session"], "regular")
        self.assertEqual(at_two["session"], "afterhours")
        self.assertEqual(at_six["session"], "closed")
        self.assertEqual(at_two["afterhours_close"], time(17, 0))

    def test_next_regular_skips_weekend_and_holiday(self):
        friday_after_close = si.next_session_open(
            "regular", "2026-07-02 17:00:00-04:00")
        # Jul 3 is the observed Independence Day holiday; next open is Monday Jul 6.
        self.assertEqual(friday_after_close["trading_date"], date(2026, 7, 6))
        self.assertEqual(friday_after_close["open_time"].time(), time(9, 30))

    def test_next_session_open_handles_same_day_early_close_and_overnight(self):
        regular = si.next_session_open(
            "regular", "2026-08-10 08:00:00-04:00")
        after = si.next_session_open(
            "afterhours", "2026-11-27 12:00:00-05:00")
        overnight = si.next_session_open(
            "overnight", "2026-07-10 21:00:00-04:00",
            overnight_eligible=True)
        self.assertEqual(regular["trading_date"], date(2026, 8, 10))
        self.assertEqual(after["open_time"].time(), time(13, 0))
        self.assertEqual(overnight["trading_date"], date(2026, 7, 13))
        self.assertEqual(overnight["open_time"].date(), date(2026, 7, 12))
        self.assertEqual(overnight["open_time"].time(), time(20, 0))

    def test_next_overnight_refuses_to_claim_availability(self):
        unknown = si.next_session_open(
            "overnight", "2026-08-10 10:00:00-04:00")
        denied = si.next_session_open(
            "overnight", "2026-08-10 10:00:00-04:00",
            overnight_eligible=False)
        self.assertEqual(unknown["status"], "unknown")
        self.assertIsNone(unknown["available"])
        self.assertIsNotNone(unknown["open_time"])
        self.assertEqual(denied["status"], "unavailable")
        self.assertFalse(denied["available"])


class SessionChangeTests(unittest.TestCase):
    def test_changes_use_explicit_previous_close_and_prior_session_chain(self):
        result = si.compute_session_changes({
            "regular": {"price": 103.0, "open": 102.0, "source": "quote"},
            "afterhours": {"price": 104.0, "source": "quote"},
            "overnight": {"price": 102.0, "source": "quote"},
            "premarket": {"price": 101.0, "source": "quote"},
        }, previous_close=100.0)
        regular = result["sessions"]["regular"]
        after = result["sessions"]["afterhours"]
        pre = result["sessions"]["premarket"]
        self.assertAlmostEqual(regular["change_vs_previous_close_pct"], 3.0)
        self.assertAlmostEqual(regular["change_vs_open_pct"], 100 / 102)
        self.assertEqual(after["prior_session"], "regular")
        self.assertAlmostEqual(after["change_vs_prior_session_pct"], 100 / 103)
        self.assertEqual(pre["prior_session"], "overnight")
        self.assertAlmostEqual(pre["change_vs_prior_session_pct"], -100 / 102)
        self.assertEqual(set(result["available"]), set(si.SESSION_NAMES))

    def test_missing_and_invalid_prices_are_never_imputed(self):
        result = si.compute_session_changes(
            {"premarket": None, "regular": 0, "afterhours": "bad"},
            previous_close=0)
        self.assertEqual(result["available"], ())
        self.assertEqual(result["data_quality"]["label"], "insufficient")
        self.assertIsNone(result["previous_close"])
        for session in si.SESSION_NAMES:
            self.assertEqual(result["sessions"][session]["status"], "missing")

    def test_explicit_reference_wins_and_metadata_is_preserved(self):
        result = si.compute_session_changes({
            "premarket": {
                "last": 102, "reference_price": 101, "volume": 5000,
                "timestamp": "2026-08-10 08:00:00-04:00", "source": "moomoo",
                "quality": 0.8,
            },
        }, previous_close=100)
        row = result["sessions"]["premarket"]
        self.assertEqual(row["prior_session"], "explicit")
        self.assertAlmostEqual(row["change_vs_prior_session_pct"], 100 / 101)
        self.assertEqual(row["volume"], 5000)
        self.assertEqual(row["source"], "moomoo")
        self.assertLessEqual(row["data_quality"]["score"], 0.8)


class NextOpenDiagnosisTests(unittest.TestCase):
    NOW = pd.Timestamp("2026-08-10 08:00:00-04:00")

    def full_features(self, sign=1.0):
        return {
            "gap_pct": 1.2 * sign,
            "futures_pct": 0.8 * sign,
            "spy_pct": 0.6 * sign,
            "qqq_pct": 0.9 * sign,
            "momentum_pct": 1.5 * sign,
            "relative_volume": 2.0,
            "event_score": 0.5 * sign,
        }

    def test_bullish_and_bearish_are_deterministic_bounded_and_transparent(self):
        bullish = si.diagnose_next_open(
            self.full_features(1), "regular", self.NOW)
        repeated = si.diagnose_next_open(
            self.full_features(1), "regular", self.NOW)
        bearish = si.diagnose_next_open(
            self.full_features(-1), "regular", self.NOW)
        self.assertEqual(bullish["direction"], "up")
        self.assertEqual(bearish["direction"], "down")
        self.assertEqual(bullish["probability_up"], repeated["probability_up"])
        self.assertGreater(bullish["probability_up"], 0.5)
        self.assertLessEqual(bullish["probability_up"], 0.8)
        self.assertAlmostEqual(
            bullish["probability_up"] + bullish["probability_down"], 1.0)
        self.assertFalse(bullish["calibrated"])
        self.assertIn("固定重み", bullish["disclaimer"])
        self.assertEqual(len(bullish["features"]), 7)
        self.assertTrue(all("contribution" in row for row in bullish["features"]))
        self.assertNotEqual(bullish["top_positive"], [])

    def test_aliases_are_accepted_but_output_is_canonical(self):
        result = si.diagnose_next_open({
            "gap_change_pct": 1.0,
            "futures_change_pct": 0.5,
            "spy_change_pct": 0.3,
        }, now=self.NOW)
        keys = {row["key"] for row in result["features"]}
        self.assertIn("gap_pct", keys)
        self.assertIn("futures_pct", keys)
        self.assertIn("spy_pct", keys)
        self.assertIsNotNone(result["probability_up"])

    def test_insufficient_features_return_unknown_without_fake_probability(self):
        empty = si.diagnose_next_open({}, now=self.NOW)
        one = si.diagnose_next_open({"futures_pct": 3.0}, now=self.NOW)
        volume_only = si.diagnose_next_open(
            {"relative_volume": 5.0}, now=self.NOW)
        for result in (empty, one, volume_only):
            self.assertEqual(result["direction"], "unknown")
            self.assertIsNone(result["probability_up"])
            self.assertIsNone(result["probability_down"])
            self.assertEqual(result["confidence"], "low")

    def test_overnight_target_requires_symbol_eligibility(self):
        unknown = si.diagnose_next_open(
            self.full_features(), "overnight", self.NOW,
            overnight_eligible=None)
        denied = si.diagnose_next_open(
            self.full_features(), "overnight", self.NOW,
            overnight_eligible=False)
        allowed = si.diagnose_next_open(
            self.full_features(), "overnight", self.NOW,
            overnight_eligible=True)
        self.assertEqual(unknown["direction"], "unknown")
        self.assertEqual(denied["direction"], "unknown")
        self.assertIsNone(unknown["probability_up"])
        self.assertIsNotNone(allowed["probability_up"])

    def test_stale_and_future_timestamp_degrade_or_exclude_features(self):
        features = self.full_features()
        features["futures_pct"] = {
            "value": 1.0, "timestamp": "2026-08-09 00:00:00-04:00",
            "source": "stale",
        }
        features["spy_pct"] = {
            "value": 1.0, "timestamp": "2026-08-10 12:30:00-04:00",
            "source": "future",
        }
        result = si.diagnose_next_open(features, now=self.NOW)
        rows = {row["key"]: row for row in result["features"]}
        self.assertEqual(rows["futures_pct"]["status"], "stale")
        self.assertEqual(rows["spy_pct"]["status"], "invalid_time")
        self.assertEqual(rows["spy_pct"]["quality"], 0.0)
        self.assertTrue(rows["spy_pct"]["warnings"])

    def test_event_risk_without_direction_reduces_quality_not_direction(self):
        base = {"gap_pct": 1.0, "futures_pct": 0.5, "spy_pct": 0.4}
        normal = si.diagnose_next_open(base, now=self.NOW)
        risky = si.diagnose_next_open(
            {**base, "event_risk": True}, now=self.NOW)
        self.assertEqual(risky["direction"], normal["direction"])
        self.assertTrue(risky["event_risk"])
        self.assertLess(risky["data_quality"]["score"], normal["data_quality"]["score"])

    def test_target_weights_are_session_specific(self):
        regular = si.diagnose_next_open(self.full_features(), "regular", self.NOW)
        after = si.diagnose_next_open(self.full_features(), "afterhours", self.NOW)
        regular_rows = {row["key"]: row for row in regular["features"]}
        after_rows = {row["key"]: row for row in after["features"]}
        self.assertNotEqual(
            regular_rows["event_score"]["weight"], after_rows["event_score"]["weight"])
        self.assertNotEqual(regular["raw_logit"], after["raw_logit"])

    def test_integrated_facade_derives_gap_and_declares_no_api_usage(self):
        result = si.analyze_session_intelligence(
            now=self.NOW,
            market_state="PRE_MARKET",
            session_prices={
                "premarket": {
                    "price": 102.0, "timestamp": self.NOW, "source": "snapshot",
                },
            },
            previous_close=100.0,
            target_session="regular",
            features={"futures_pct": 0.5, "spy_pct": 0.3},
        )
        diagnosis = result["next_open_diagnosis"]
        rows = {row["key"]: row for row in diagnosis["features"]}
        self.assertEqual(result["current_session"]["session"], "premarket")
        self.assertAlmostEqual(rows["gap_pct"]["value"], 2.0)
        self.assertEqual(rows["gap_pct"]["source"], "snapshot")
        self.assertFalse(result["calibrated"])
        self.assertTrue(result["read_only"])
        self.assertFalse(result["uses_opend_history_quota"])

    def test_invalid_target_is_rejected_explicitly(self):
        with self.assertRaises(ValueError):
            si.diagnose_next_open({}, "tomorrow", self.NOW)
        with self.assertRaises(ValueError):
            si.next_session_open("tomorrow", self.NOW)


class SafetyTests(unittest.TestCase):
    def test_module_has_no_network_broker_or_random_dependencies(self):
        source = inspect.getsource(si)
        for forbidden in (
            "import requests", "import yfinance", "moomoo_client",
            "OpenQuoteContext", "OpenSecTradeContext", "OpenFutureTradeContext",
            "random.",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()

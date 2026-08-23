import unittest
from datetime import date

import pandas as pd

from lib import today_inputs


class TodayInputTests(unittest.TestCase):
    def test_session_prices_keep_missing_sessions_missing(self):
        result = today_inputs.session_prices({
            "source": "moomoo OpenAPI", "price": 100, "pre_price": 99,
            "after_price": None, "overnight_price": 0, "update_time": "2026-08-10 09:00:00",
        })
        self.assertEqual(set(result), {"premarket", "regular"})
        self.assertIsNone(result["premarket"]["timestamp"])
        self.assertFalse(result["premarket"]["timestamp_verified"])
        self.assertLess(result["premarket"]["quality"], 1.0)

    def test_session_prices_only_use_verified_per_session_timestamp(self):
        result = today_inputs.session_prices({
            "source": "moomoo OpenAPI", "price": 100, "after_price": 102,
            "update_time": "2026-08-10 16:05:00",
            "session_quotes": {
                "regular": {
                    "price": 100, "timestamp_verified": False,
                    "updated_at": None,
                },
                "afterhours": {
                    "price": 102, "timestamp_verified": False,
                    "updated_at": None,
                },
            },
        })
        self.assertIsNone(result["regular"]["timestamp"])
        self.assertIsNone(result["afterhours"]["timestamp"])
        self.assertEqual(result["regular"]["quality"], 0.65)
        self.assertEqual(result["afterhours"]["quality"], 0.65)

    def test_daily_fallback_only_populates_regular(self):
        bar = pd.Series({"Open": 98, "Close": 100, "Volume": 10},
                        name=pd.Timestamp("2026-08-07"))
        result = today_inputs.session_prices({}, bar)
        self.assertEqual(set(result), {"regular"})
        self.assertIn("確定日足", result["regular"]["source"])

    def test_overnight_missing_is_unknown_not_false(self):
        self.assertIsNone(today_inputs.overnight_eligibility({}))
        self.assertTrue(today_inputs.overnight_eligibility({"overnight_price": 101}))

    def test_event_risk_requires_upcoming_near_high_impact(self):
        report = {"events": [{
            "status": "UPCOMING", "event_date": "2026-08-11",
            "impact_level": "HIGH", "impact_score": 68,
        }]}
        self.assertTrue(today_inputs.imminent_event_risk(
            report, as_of=date(2026, 8, 10)))

    def test_low_32_point_event_is_not_high_impact(self):
        report = {"events": [{
            "status": "UPCOMING", "event_date": "2026-08-11",
            "impact_level": "LOW", "impact_score": 32,
        }]}
        self.assertFalse(today_inputs.imminent_event_risk(
            report, as_of=date(2026, 8, 10)))

    def test_legacy_event_without_level_uses_zero_to_100_threshold(self):
        low = {"events": [{
            "status": "UPCOMING", "event_date": "2026-08-11", "impact_score": 59,
        }]}
        high = {"events": [{
            "status": "UPCOMING", "event_date": "2026-08-11", "impact_score": 60,
        }]}
        self.assertFalse(today_inputs.imminent_event_risk(
            low, as_of=date(2026, 8, 10)))
        self.assertTrue(today_inputs.imminent_event_risk(
            high, as_of=date(2026, 8, 10)))

    def test_opening_features_do_not_add_direction_from_event(self):
        idx = pd.date_range("2026-08-01", periods=7, freq="D")
        hist = pd.DataFrame({"Close": range(100, 107)}, index=idx)
        features = today_inputs.opening_features(
            hist, {"volume_ratio": 1.2, "update_time": "2026-08-07"},
            {"spy_pct": 0.4}, {"events": []})
        self.assertIn("momentum_pct", features)
        self.assertIn("event_risk", features)
        self.assertNotIn("event_score", features)


if __name__ == "__main__":
    unittest.main()

import unittest

from lib import daily_decision


class IntradayTrendTests(unittest.TestCase):
    def test_realtime_uptrend_uses_four_explainable_checks(self):
        result = daily_decision.intraday_trend({
            "source": "moomoo OpenAPI", "price": 105, "open": 101,
            "high": 106, "low": 100, "average_price": 102,
            "change_percent": 2.0,
        })
        self.assertEqual(result["direction"], "up")
        self.assertEqual(result["data_quality"], "realtime")
        self.assertEqual(len(result["checks"]), 4)

    def test_daily_fallback_is_not_labelled_realtime(self):
        result = daily_decision.intraday_trend({}, {
            "Open": 100, "High": 102, "Low": 98, "Close": 99,
        })
        self.assertEqual(result["direction"], "down")
        self.assertEqual(result["data_quality"], "close_only")
        self.assertEqual(result["source"], "直近確定日足")

    def test_missing_values_return_unknown(self):
        result = daily_decision.intraday_trend({}, {})
        self.assertEqual(result["direction"], "unknown")
        self.assertIsNone(result["strength"])


class NearestLevelTests(unittest.TestCase):
    def test_returns_nearest_edges_and_reward_risk(self):
        result = daily_decision.nearest_levels([
            {"type": "サポート", "price": 95, "zone_low": 94,
             "zone_high": 96, "strength": 4},
            {"type": "抵抗線", "price": 110, "zone_low": 109,
             "zone_high": 111, "strength": 3},
        ], 100)
        self.assertEqual(result["support"]["edge_price"], 96)
        self.assertEqual(result["resistance"]["edge_price"], 109)
        self.assertAlmostEqual(result["reward_risk"], 2.25)

    def test_marks_current_battle_zone(self):
        result = daily_decision.nearest_levels([
            {"type": "サポート", "price": 100, "zone_low": 99,
             "zone_high": 101, "strength": 5},
        ], 100)
        self.assertEqual(len(result["in_zones"]), 1)


if __name__ == "__main__":
    unittest.main()

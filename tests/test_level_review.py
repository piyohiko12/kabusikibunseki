import unittest

import pandas as pd

from lib import level_review


def sample_prices(rows: int = 80) -> pd.DataFrame:
    index = pd.bdate_range("2025-01-02", periods=rows)
    close = pd.Series([100 + i * 0.05 for i in range(rows)], index=index)
    return pd.DataFrame({
        "Open": close - 0.1,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": 1000,
    }, index=index)


def base_level(kind: str, price: float, strength: int = 3) -> dict:
    return {
        "type": kind,
        "price": price,
        "zone_low": price - 0.4,
        "zone_high": price + 0.4,
        "distance_pct": price / 103.95 * 100 - 100,
        "bounces": 2,
        "breaks": 1,
        "touches": 3,
        "rejects": 1,
        "swings": 2,
        "bounce_rate": 66.7,
        "raw_score": 1.8,
        "vol_share": 0.08,
        "last_touch": "2025-04-01",
        "basis": "スイング",
        "confluence": [],
        "last_event": None,
        "strength": strength,
    }


class MoomooLevelReviewTests(unittest.TestCase):
    def setUp(self):
        self.prices = sample_prices()
        self.levels = [
            base_level("抵抗線", 105.0),
            base_level("サポート", 103.0),
        ]

    def test_live_demand_raises_support_relative_score(self):
        book = pd.DataFrame({
            "売気配値": [105.0, 105.2, 105.4],
            "売数量": [100, 100, 100],
            "買気配値": [103.0, 102.8, 102.6],
            "買数量": [900, 100, 100],
        })
        ticks = pd.DataFrame({
            "ticker_direction": ["BUY", "BUY", "SELL"],
            "volume": [500, 300, 100],
        })
        capital = {"net": 1_000_000, "tiers": []}

        result = level_review.review_levels(
            self.prices, self.levels, book, capital, ticks,
            reviewed_at="2026-08-09T12:00:00+09:00",
        )
        support = next(level for level in result["levels"]
                       if level["type"] == "サポート" and not level.get("temporary"))
        resistance = next(level for level in result["levels"]
                          if level["type"] == "抵抗線" and not level.get("temporary"))

        self.assertGreater(support["moomoo_score"], resistance["moomoo_score"])
        self.assertEqual(support["base_strength"], 3)
        self.assertIn("板情報", result["sources"])
        self.assertIn("歩み値", result["sources"])
        self.assertIn("資金フロー", result["sources"])
        self.assertEqual(result["reviewed_at"], "2026-08-09T12:00:00+09:00")
        self.assertNotIn("base_strength", self.levels[1])

    def test_unmatched_large_wall_becomes_temporary_candidate(self):
        book = pd.DataFrame({
            "売気配値": [104.1, 104.2, 104.3],
            "売数量": [100, 100, 100],
            "買気配値": [101.0, 100.9, 100.8],
            "買数量": [1000, 100, 100],
        })
        result = level_review.review_levels(self.prices, self.levels, book)
        candidates = [level for level in result["levels"] if level.get("temporary")]

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["type"], "サポート")
        self.assertEqual(candidates[0]["price"], 101.0)
        self.assertEqual(candidates[0]["base_strength"], 0)

    def test_no_live_inputs_keeps_review_traceable(self):
        result = level_review.review_levels(self.prices, self.levels)

        self.assertEqual(result["sources"], [])
        self.assertEqual(len(result["levels"]), len(self.levels))
        self.assertTrue(all(level["moomoo_reason"] == "OHLCV基礎評価のみ"
                            for level in result["levels"]))

    def test_neutral_only_ticks_are_not_counted_as_live_evidence(self):
        ticks = pd.DataFrame({
            "ticker_direction": ["NEUTRAL", "NEUTRAL", None],
            "volume": [500, 300, 100],
        })
        baseline = level_review.review_levels(self.prices, self.levels)
        result = level_review.review_levels(self.prices, self.levels, ticks=ticks)

        self.assertNotIn("歩み値", result["sources"])
        self.assertNotIn("直近約定", result["summary"])
        for actual, expected in zip(result["levels"], baseline["levels"]):
            self.assertEqual(actual["moomoo_score"], expected["moomoo_score"])
            self.assertEqual(actual["moomoo_confidence"],
                             expected["moomoo_confidence"])
            self.assertEqual(actual["moomoo_reason"], "OHLCV基礎評価のみ")


if __name__ == "__main__":
    unittest.main()

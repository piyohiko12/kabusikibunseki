import unittest

import numpy as np
import pandas as pd

from lib import data_fetcher, sector_scan


def series_frame(drift: float, seed: int, rows: int = 260,
                 volume_multiplier: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-08-11", periods=rows)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.011, rows))),
                      index=idx)
    volume = rng.integers(1_000_000, 5_000_000, rows).astype(float)
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": volume * volume_multiplier}, index=idx)


class ScoreTests(unittest.TestCase):
    def test_neutral_input_scores_near_fifty(self):
        """ベンチマークと同じ値動きなら、相対力の各要素は中立になる。"""
        frame = series_frame(0.0005, seed=1)
        close, volume = frame["Close"], frame["Volume"]
        scored = sector_scan._score_frame(close, volume, close)
        for name in ("相対力(5日)", "相対力(20日)", "当日"):
            self.assertAlmostEqual(scored["parts"][name],
                                   {"相対力(5日)": 12.5, "相対力(20日)": 10.0,
                                    "当日": 10.0}[name], places=1)

    def test_outperformer_scores_above_underperformer(self):
        bench = series_frame(0.0002, seed=2)["Close"]
        strong = series_frame(0.0002, seed=2)["Close"] * np.linspace(1, 1.15, 260)
        weak = series_frame(0.0002, seed=2)["Close"] * np.linspace(1, 0.85, 260)
        volume = series_frame(0.0002, seed=2)["Volume"]
        high = sector_scan._score_frame(strong, volume, bench)
        low = sector_scan._score_frame(weak, volume, bench)
        self.assertGreater(high["score"], low["score"])
        self.assertGreater(high["rel5"], 0)
        self.assertLess(low["rel5"], 0)

    def test_strength_labels_are_symmetric_around_fifty(self):
        """50を中立として、±同じだけ離れたスコアは対になるラベルになる。"""
        self.assertEqual(sector_scan._label(50.0), ("中立", "flat"))
        mirror = {"非常に強い": "非常に弱い", "強い": "弱い", "やや強い": "やや弱い"}
        for offset in (5, 12, 25, 40):
            with self.subTest(offset=offset):
                high, high_tone = sector_scan._label(50 + offset)
                low, low_tone = sector_scan._label(50 - offset)
                self.assertEqual(mirror[high], low)
                self.assertEqual((high_tone, low_tone), ("up", "down"))

    def test_short_history_is_rejected(self):
        short = series_frame(0.0, seed=3, rows=10)
        self.assertIsNone(sector_scan._score_frame(
            short["Close"], short["Volume"], short["Close"]))

    def test_every_sector_has_an_etf_and_members(self):
        etfs = [s["etf"] for s in sector_scan.SECTORS]
        self.assertEqual(len(etfs), len(set(etfs)))
        self.assertEqual(len(sector_scan.SECTORS), 11)
        for sector in sector_scan.SECTORS:
            with self.subTest(etf=sector["etf"]):
                self.assertGreaterEqual(len(sector["members"]), 6)
                self.assertEqual(len(sector["members"]),
                                 len(set(sector["members"])))


class RankingTests(unittest.TestCase):
    def setUp(self):
        frames = {"SPY": series_frame(0.0004, seed=0)}
        for i, sector in enumerate(sector_scan.SECTORS):
            frames[sector["etf"]] = series_frame(0.0010 - i * 0.0002, seed=10 + i)
            for j, member in enumerate(sector["members"]):
                frames[member] = series_frame(0.0008 - j * 0.0002,
                                              seed=200 + i * 20 + j)
        self._frames = frames
        self._original = data_fetcher.fetch_daily_batch
        data_fetcher.fetch_daily_batch = lambda tickers, period="1y": {
            t: frames[t].copy() for t in tickers if t in frames}

    def tearDown(self):
        data_fetcher.fetch_daily_batch = self._original

    def test_ranking_returns_every_sector_sorted_by_score(self):
        ranking = sector_scan.sector_ranking.__wrapped__()
        self.assertEqual(len(ranking), 11)
        scores = [r["score"] for r in ranking]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for row in ranking:
            self.assertIn(row["tone"], {"up", "down", "flat"})
            self.assertTrue(0 <= row["score"] <= 100)

    def test_members_are_limited_and_sorted(self):
        members = sector_scan.sector_members.__wrapped__("XLK", limit=4)
        self.assertEqual(len(members), 4)
        scores = [m["score"] for m in members]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_unknown_sector_returns_empty(self):
        self.assertEqual(sector_scan.sector_members.__wrapped__("NOPE"), [])

    def test_top_picks_span_requested_sectors(self):
        ranking = sector_scan.sector_ranking.__wrapped__()
        picks = sector_scan.top_picks(ranking, sectors=3, per_sector=2)
        self.assertEqual(len(picks), 6)
        self.assertEqual(len({p["etf"] for p in picks}), 3)


if __name__ == "__main__":
    unittest.main()

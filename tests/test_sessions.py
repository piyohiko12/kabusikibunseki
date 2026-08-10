import unittest

import numpy as np
import pandas as pd

from lib import gap, intraday, sessions


def intraday_frame(day: str = "2026-08-06", drift: float = 0.0,
                   seed: int = 0) -> pd.DataFrame:
    """04:00〜19:55 ETの5分足を1日ぶん作る(時間外を含む)。"""
    idx = pd.date_range(f"{day} 04:00", f"{day} 19:55", freq="5min",
                        tz="America/New_York")
    rng = np.random.default_rng(seed)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.0011, len(idx)))),
                      index=idx)
    open_ = close.shift(1).fillna(100.0)
    volume = rng.integers(200_000, 900_000, len(idx)).astype(float)
    volume[(sessions.classify(idx) != sessions.REGULAR).to_numpy()] *= 0.1
    return pd.DataFrame({
        "Open": open_, "High": np.maximum(open_, close) * 1.001,
        "Low": np.minimum(open_, close) * 0.999, "Close": close,
        "Volume": volume}, index=idx)


class SessionClassificationTests(unittest.TestCase):
    def test_boundaries(self):
        cases = {
            "03:59": sessions.OVERNIGHT,
            "04:00": sessions.PRE,
            "09:29": sessions.PRE,
            "09:30": sessions.REGULAR,
            "15:59": sessions.REGULAR,
            "16:00": sessions.AFTER,
            "19:59": sessions.AFTER,
            "20:00": sessions.OVERNIGHT,
        }
        for hhmm, expected in cases.items():
            with self.subTest(time=hhmm):
                ts = pd.Timestamp(f"2026-08-06 {hhmm}", tz="America/New_York")
                self.assertEqual(sessions.classify([ts]).iloc[0], expected)

    def test_weekend_is_overnight(self):
        ts = pd.Timestamp("2026-08-08 11:00", tz="America/New_York")  # 土曜
        self.assertEqual(sessions.classify([ts]).iloc[0], sessions.OVERNIGHT)

    def test_naive_index_is_treated_as_et(self):
        naive = pd.DatetimeIndex(["2026-08-06 10:00"])
        self.assertEqual(sessions.classify(naive).iloc[0], sessions.REGULAR)

    def test_trading_day_rolls_over_at_20et(self):
        idx = pd.DatetimeIndex([
            pd.Timestamp("2026-08-06 19:30", tz="America/New_York"),
            pd.Timestamp("2026-08-06 20:30", tz="America/New_York"),
            pd.Timestamp("2026-08-07 05:00", tz="America/New_York"),
        ])
        days = [str(d)[:10] for d in sessions.trading_day(idx)]
        self.assertEqual(days, ["2026-08-06", "2026-08-07", "2026-08-07"])

    def test_rangebreaks_depend_on_extended_hours(self):
        regular = sessions.rangebreaks("5m", show_extended=False)
        extended = sessions.rangebreaks("5m", show_extended=True)
        self.assertIn({"bounds": [16, 9.5], "pattern": "hour"}, regular)
        self.assertIn({"bounds": [20, 4], "pattern": "hour"}, extended)
        # 日足には時間帯のrangebreakを付けない
        self.assertEqual(sessions.rangebreaks("1d", True),
                         [{"bounds": ["sat", "mon"]}])

    def test_session_summary_chains_from_previous_close(self):
        rows = sessions.session_summary(intraday_frame(), prev_close=100.0)
        names = [r["session"] for r in rows]
        self.assertEqual(names, [sessions.PRE, sessions.REGULAR, sessions.AFTER])
        # 各セッションの変化率は1つ前の終値が基準
        self.assertAlmostEqual(rows[0]["change_pct"],
                               (rows[0]["close"] / 100.0 - 1) * 100, places=6)
        self.assertAlmostEqual(
            rows[1]["change_pct"],
            (rows[1]["close"] / rows[0]["close"] - 1) * 100, places=6)

    def test_filter_sessions_drops_extended(self):
        frame = intraday_frame()
        only_regular = sessions.filter_sessions(frame, show_extended=False)
        self.assertTrue((sessions.classify(only_regular.index)
                         == sessions.REGULAR).all())
        self.assertLess(len(only_regular), len(frame))


class IntradayTrendTests(unittest.TestCase):
    def test_direction_is_ordered_and_symmetric(self):
        up = intraday.analyze(intraday_frame(drift=0.0010, seed=1), 100.0)
        flat = intraday.analyze(intraday_frame(drift=0.0, seed=1), 100.0)
        down = intraday.analyze(intraday_frame(drift=-0.0010, seed=1), 100.0)
        self.assertGreater(up["score"], flat["score"])
        self.assertGreater(flat["score"], down["score"])
        self.assertGreater(up["score"], 55)
        self.assertLess(down["score"], -55)

    def test_swing_structure_is_symmetric(self):
        rising = np.concatenate([np.arange(40, 60.0), np.arange(60, 80.0)])
        falling = rising[::-1].copy()
        self.assertEqual(intraday._swing_structure(rising, 20), 1.0)
        self.assertEqual(intraday._swing_structure(falling, 20), -1.0)
        # レンジ拡大(高値切り上げ+安値切り下げ)は打ち消し合って0
        prior = np.full(20, 50.0)
        expanding = np.concatenate([prior, np.array([40.0] * 10 + [60.0] * 10)])
        self.assertEqual(intraday._swing_structure(expanding, 20), 0.0)

    def test_opening_range_detects_breakout(self):
        frame = intraday_frame(drift=0.0015, seed=2)
        orb = intraday.opening_range(frame)
        self.assertTrue(orb["complete"])
        self.assertTrue(orb["broke_up"])
        self.assertGreater(orb["high"], orb["low"])

    def test_vwap_resets_each_trading_day(self):
        two = pd.concat([intraday_frame("2026-08-05", seed=3),
                         intraday_frame("2026-08-06", seed=4)])
        vw = intraday.vwap(two)
        self.assertEqual(len(vw), len(two))
        self.assertTrue(vw.notna().all())
        # 2日目の最初のVWAPはその足の価格に近い(前日を引きずらない)
        day2 = sessions.latest_day_slice(two)
        first = day2.index[0]
        typical = float((day2["High"].iloc[0] + day2["Low"].iloc[0]
                         + day2["Close"].iloc[0]) / 3)
        self.assertAlmostEqual(float(vw.loc[first]), typical, places=6)

    def test_falls_back_to_previous_day_when_new_day_is_thin(self):
        frame = pd.concat([
            intraday_frame("2026-08-06", drift=0.001, seed=5),
            # 20:05〜20:15 の3本だけ(新しい取引日に転がる)
            intraday_frame("2026-08-06", seed=6).iloc[-3:].set_index(
                pd.date_range("2026-08-06 20:05", periods=3, freq="5min",
                              tz="America/New_York")),
        ])
        result = intraday.analyze(frame, 100.0)
        self.assertIsNotNone(result)
        self.assertTrue(result["stale"])
        self.assertGreater(result["regular_bars"], 0)

    def test_returns_none_without_enough_bars(self):
        self.assertIsNone(intraday.analyze(pd.DataFrame()))
        self.assertIsNone(intraday.analyze(intraday_frame().head(3)))


def daily_frame(rows: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=rows)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.014, rows)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.008, rows))
    return pd.DataFrame({
        "Open": open_, "High": np.maximum(open_, close) * 1.006,
        "Low": np.minimum(open_, close) * 0.994, "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, rows).astype(float),
    }, index=idx)


class GapTests(unittest.TestCase):
    def setUp(self):
        self.table = gap.gap_table(daily_frame())

    def test_gap_table_columns_and_definitions(self):
        for column in ("gap_pct", "open_to_close", "filled", "follow", "fade"):
            self.assertIn(column, self.table.columns)
        # 上窓の行は、安値が前日終値以下のときだけ「窓埋め」
        up = self.table[self.table["gap_pct"] > 0]
        expected = up["low"] <= up["prev_close"]
        self.assertTrue((up["filled"] == expected).all())
        # follow と fade は同時に立たない
        self.assertFalse((self.table["follow"] & self.table["fade"]).any())

    def test_conditional_widens_until_enough_samples(self):
        cond = gap.conditional(self.table, 0.6)
        self.assertIsNotNone(cond)
        self.assertGreaterEqual(cond["n"], gap.MIN_SAMPLES)
        self.assertIn("fill_rate", cond)

    def test_conditional_flags_fallback_for_extreme_gap(self):
        cond = gap.conditional(self.table, 25.0)
        self.assertIsNotNone(cond)
        self.assertTrue(cond.get("widened_to_side"))
        self.assertIsNone(cond["width"])

    def test_conditional_returns_none_without_history(self):
        self.assertIsNone(gap.conditional(pd.DataFrame(), 1.0))
        self.assertIsNone(gap.conditional(self.table, None))

    def test_baseline_buckets_cover_every_gap(self):
        for value in (-9.0, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 9.0):
            with self.subTest(gap=value):
                self.assertIsNotNone(gap.baseline_for(value))

    def test_projected_open_uses_prev_close(self):
        proj = gap.projected_open(200.0, 1.5, gap.conditional(self.table, 1.5))
        self.assertAlmostEqual(proj["open_price"], 203.0, places=6)
        self.assertAlmostEqual(proj["gap_amount"], 3.0, places=6)
        self.assertLess(proj["close_low"], proj["close_high"])

    def test_verdict_stays_neutral_without_a_clear_bias(self):
        balanced = {"follow_rate": 51.0, "fade_rate": 49.0, "fill_rate": 60.0,
                    "n": 40}
        self.assertEqual(gap.verdict(balanced, 1.0)["tone"], "neutral")
        biased = {"follow_rate": 75.0, "fade_rate": 25.0, "fill_rate": 30.0,
                  "n": 40}
        self.assertEqual(gap.verdict(biased, 1.0)["tone"], "follow")

    def test_extended_reference_picks_latest_extended_session(self):
        frame = intraday_frame(seed=7)
        ref = gap.extended_reference(frame, prev_close=100.0)
        self.assertEqual(ref["session"], sessions.AFTER)
        self.assertIsNotNone(ref["gap_pct"])
        # 立会だけのフレームなら手がかりなし
        regular_only = sessions.session_slice(frame, [sessions.REGULAR])
        self.assertIsNone(gap.extended_reference(regular_only, 100.0))


if __name__ == "__main__":
    unittest.main()

import unittest
from datetime import date

import pandas as pd

from lib import signal_context


def _daily_frame(periods=240, end="2026-08-07", volume=100_000):
    idx = pd.bdate_range(end=end, periods=periods)
    close = pd.Series(range(100, 100 + periods), index=idx, dtype=float)
    return pd.DataFrame({
        "Open": close - 0.5,
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": float(volume),
    }, index=idx)


class CompletedBarTests(unittest.TestCase):
    def test_drops_current_live_bar(self):
        df = _daily_frame(periods=3, end="2026-08-10")
        out, meta = signal_context.completed_daily_bars(
            df, "US.AAPL", "MORNING",
            now=pd.Timestamp("2026-08-10 12:00", tz="America/New_York"),
        )
        self.assertEqual(len(out), 2)
        self.assertTrue(meta["dropped_incomplete"])

    def test_keeps_current_bar_after_market_close(self):
        df = _daily_frame(periods=3, end="2026-08-10")
        out, meta = signal_context.completed_daily_bars(
            df, "US.AAPL", "CLOSED",
            now=pd.Timestamp("2026-08-10 18:00", tz="America/New_York"),
        )
        self.assertEqual(len(out), 3)
        self.assertFalse(meta["dropped_incomplete"])

    def test_unknown_state_uses_local_close_time(self):
        df = _daily_frame(periods=3, end="2026-08-10")
        before, _ = signal_context.completed_daily_bars(
            df, "US.AAPL", None,
            now=pd.Timestamp("2026-08-10 15:00", tz="America/New_York"),
        )
        after, _ = signal_context.completed_daily_bars(
            df, "US.AAPL", None,
            now=pd.Timestamp("2026-08-10 17:00", tz="America/New_York"),
        )
        self.assertEqual(len(before), 2)
        self.assertEqual(len(after), 3)


class GateTests(unittest.TestCase):
    def test_safe_context_passes_required_gates(self):
        df = _daily_frame(volume=200_000)
        gates = signal_context.build_external_gates(
            df, code="US.AAPL", bar_meta={"bar_complete": True},
            snapshot={"bid": 338.9, "ask": 339.1},
            earnings_date="2026-09-01", as_of=date(2026, 8, 10),
        )
        self.assertTrue(signal_context.gate_summary(gates)["passed"])

    def test_earnings_blackout_blocks(self):
        df = _daily_frame(volume=200_000)
        gates = signal_context.build_external_gates(
            df, code="US.AAPL", bar_meta={"bar_complete": True},
            snapshot={}, earnings_date="2026-08-11", as_of=date(2026, 8, 10),
        )
        summary = signal_context.gate_summary(gates)
        self.assertFalse(summary["passed"])
        self.assertIn("earnings", {item["key"] for item in summary["blocked"]})

    def test_unknown_spread_is_warning_not_block(self):
        df = _daily_frame(volume=200_000)
        gates = signal_context.build_external_gates(
            df, code="US.AAPL", bar_meta={"bar_complete": True},
            snapshot={}, earnings_date="2026-09-01", as_of=date(2026, 8, 10),
        )
        spread = next(g for g in gates if g["key"] == "spread")
        self.assertIsNone(spread["passed"])
        self.assertFalse(spread["required"])


if __name__ == "__main__":
    unittest.main()

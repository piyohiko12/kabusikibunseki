import unittest
from unittest.mock import patch

import pandas as pd

from lib import trading_context


def _history(periods=800, end="2026-08-10"):
    index = pd.bdate_range(end=end, periods=periods)
    close = pd.Series(range(100, 100 + periods), index=index, dtype=float)
    return pd.DataFrame({
        "Open": close - 0.5,
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": 1_000_000.0,
    }, index=index)


class TradingContextTests(unittest.TestCase):
    def test_safety_config_keeps_defaults_and_overrides_saved_values(self):
        config = trading_context.safety_config({
            "safety": {"max_spread_pct": 0.25},
        })
        self.assertEqual(config["max_spread_pct"], 0.25)
        self.assertEqual(config["min_history"], 220)

    @patch("lib.signal_context._market_now",
           return_value=pd.Timestamp("2026-08-10 12:00", tz="America/New_York"))
    @patch("lib.trading_context.levels.find_levels", return_value=[{"price": 10}])
    def test_prepare_uses_completed_two_year_history_without_network(
            self, find_levels, _market_now):
        context = trading_context.prepare_from_history(
            _history(),
            source_meta={"code": "US.AAPL"},
            market_meta={"market_state": "MORNING"},
            snapshot={"price": 900.0, "previous_close": 899.0},
            canonical_years=2,
        )
        self.assertIsNotNone(context)
        self.assertTrue(context["bar_meta"]["dropped_incomplete"])
        self.assertLess(len(context["df"]), 530)
        self.assertEqual(context["price"], 900.0)
        self.assertEqual(context["levels"], [{"price": 10}])
        self.assertLessEqual(len(find_levels.call_args.args[0]), 182)

    def test_external_gates_uses_rule_safety(self):
        context = trading_context.prepare_from_history(
            _history(periods=260, end="2026-08-07"),
            source_meta={"code": "US.AAPL"},
            market_meta={"market_state": "CLOSED"},
            snapshot={"bid": 358.0, "ask": 360.0},
            canonical_years=None,
        )
        gates = trading_context.external_gates(
            context, {"safety": {"max_spread_pct": 0.10}})
        spread = next(item for item in gates if item["key"] == "spread")
        self.assertFalse(spread["passed"])
        self.assertIn("0.10%", spread["reason"])


if __name__ == "__main__":
    unittest.main()

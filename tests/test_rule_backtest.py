import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from lib import rule_backtest


def synthetic_ohlcv(rows: int = 24, *, freq: str = "B") -> pd.DataFrame:
    """Deterministic, internally consistent OHLCV without network access."""
    index = pd.date_range("2022-01-03", periods=rows, freq=freq)
    opens = 100.0 + np.arange(rows, dtype=float) * 0.25
    closes = opens + 0.10
    return pd.DataFrame({
        "Open": opens,
        "High": closes + 1.0,
        "Low": opens - 1.0,
        "Close": closes,
        "Volume": 1_000_000 + np.arange(rows) * 1_000,
    }, index=index)


LOOSE_ACCEPTANCE = {
    "min_trades": 1,
    "min_folds": 1,
    "min_profit_factor": 0.0,
    "min_expectancy": -1.0,
    "max_drawdown": 1.0,
    "min_positive_fold_ratio": 0.0,
}


class FoldTests(unittest.TestCase):
    def test_build_folds_keeps_training_out_of_test_and_partial_tail(self):
        folds = rule_backtest.build_folds(20, train_bars=6, test_bars=5,
                                          step_bars=5)
        self.assertEqual(
            [(f["train_start"], f["train_end"], f["test_start"], f["test_end"])
             for f in folds],
            [(0, 6, 6, 11), (5, 11, 11, 16), (10, 16, 16, 20)],
        )
        self.assertTrue(all(f["train_end"] == f["test_start"] for f in folds))

    def test_insufficient_data_is_reported_without_fetching(self):
        result = rule_backtest.run_walk_forward(
            synthetic_ohlcv(6), {}, train_bars=5, test_bars=3, step_bars=3)
        self.assertEqual(result["summary"]["status"], "INSUFFICIENT_DATA")
        self.assertEqual(result["summary"]["trade_count"], 0)
        self.assertTrue(result["trades"].empty)
        self.assertTrue(any("不足" in reason for reason in result["summary"]["reasons"]))

    def test_overlapping_test_folds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "step_bars"):
            rule_backtest.run_walk_forward(
                synthetic_ohlcv(30), {}, train_bars=10,
                test_bars=10, step_bars=5)


class ExecutionTests(unittest.TestCase):
    def test_profit_factor_uses_returns_not_share_price_pnl(self):
        trades = pd.DataFrame({
            "net_pnl": [1.0, -1.0],
            "net_return": [0.10, -0.05],
        })
        stats = rule_backtest._stats(trades)
        self.assertAlmostEqual(stats["profit_factor"], 2.0)

    def test_signal_at_close_fills_next_open_and_applies_both_costs(self):
        prices = synthetic_ohlcv(14)
        index = prices.index
        evaluated = []
        level_windows = []

        def fake_levels(frame):
            level_windows.append(frame.index.copy())
            return []

        def fake_evaluate(history, _rule, _levels, *, position_mode,
                          external_gates):
            self.assertIsNone(external_gates)
            evaluated.append((history.index[0], history.index[-1], position_mode))
            if history.index[-1] == index[5] and position_mode == "entry":
                return {
                    "verdict": "BUY",
                    "risk_plan": {"stop": 50.0, "target": 200.0},
                }
            if history.index[-1] == index[7] and position_mode == "holding":
                return {"verdict": "RISK_EXIT", "risk_plan": {}}
            return {"verdict": "HOLD" if position_mode == "holding" else "NEUTRAL"}

        with patch("lib.rule_backtest.levels.find_levels", side_effect=fake_levels), \
                patch("lib.rule_backtest.rules.evaluate", side_effect=fake_evaluate):
            result = rule_backtest.run_walk_forward(
                prices, {}, train_bars=5, test_bars=8, step_bars=8,
                one_way_bps=100, slippage_bps=50,
                max_holding_days=10, acceptance=LOOSE_ACCEPTANCE,
            )

        self.assertEqual(len(result["trades"]), 1)
        trade = result["trades"].iloc[0]
        self.assertEqual(trade["entry_signal_time"], index[5])
        self.assertEqual(trade["entry_time"], index[6])
        self.assertEqual(trade["exit_signal_time"], index[7])
        self.assertEqual(trade["exit_time"], index[8])
        self.assertEqual(trade["exit_reason"], "risk_exit")

        entry = float(prices.loc[index[6], "Open"])
        exit_ = float(prices.loc[index[8], "Open"])
        expected = (exit_ * 0.995 * 0.99) / (entry * 1.005 * 1.01) - 1
        self.assertAlmostEqual(float(trade["net_return"]), expected, places=12)
        self.assertLess(float(trade["net_return"]), float(trade["gross_return"]))
        self.assertEqual(evaluated[0][1], index[5])
        self.assertTrue(all(window.max() <= seen[1]
                            for window, seen in zip(level_windows, evaluated)))

    def test_same_bar_stop_and_target_uses_stop_first(self):
        prices = synthetic_ohlcv(10)
        index = prices.index
        # The entry session crosses both ATR boundaries.
        prices.loc[index[5], "High"] = 120.0
        prices.loc[index[5], "Low"] = 80.0

        def fake_evaluate(history, _rule, _levels, **kwargs):
            if history.index[-1] == index[4] and kwargs["position_mode"] == "entry":
                return {"verdict": "BUY", "risk_plan": {}}
            return {"verdict": "HOLD"}

        with patch("lib.rule_backtest.levels.find_levels", return_value=[]), \
                patch("lib.rule_backtest.rules.evaluate", side_effect=fake_evaluate):
            result = rule_backtest.run_walk_forward(
                prices, {}, train_bars=4, test_bars=5, step_bars=5,
                atr_period=3, atr_stop_mult=2, atr_target_mult=2,
                one_way_bps=0, slippage_bps=0,
                acceptance=LOOSE_ACCEPTANCE,
            )

        trade = result["trades"].iloc[0]
        self.assertEqual(trade["entry_time"], index[5])
        self.assertEqual(trade["exit_time"], index[5])
        self.assertEqual(trade["exit_reason"], "atr_stop")
        self.assertAlmostEqual(float(trade["exit_price"]),
                               float(trade["stop_price"]))

    def test_old_sell_verdict_maps_to_holding_risk_exit(self):
        prices = synthetic_ohlcv(14)
        index = prices.index

        # Deliberately use the old three-argument signature.
        def old_evaluate(history, _rule, _levels):
            if history.index[-1] == index[5]:
                return {"verdict": "BUY"}
            if history.index[-1] == index[7]:
                return {"verdict": "SELL"}
            return {"verdict": "NEUTRAL"}

        with patch("lib.rule_backtest.levels.find_levels", return_value=[]), \
                patch("lib.rule_backtest.rules.evaluate", new=old_evaluate):
            result = rule_backtest.run_walk_forward(
                prices, {}, train_bars=5, test_bars=8, step_bars=8,
                atr_period=3, atr_stop_mult=20, atr_target_mult=20,
                one_way_bps=0, slippage_bps=0,
                acceptance=LOOSE_ACCEPTANCE,
            )

        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["trades"].iloc[0]["exit_reason"], "risk_exit")
        self.assertEqual(result["trades"].iloc[0]["exit_time"], index[8])

    def test_max_holding_exit_is_filled_at_following_open(self):
        prices = synthetic_ohlcv(14)
        index = prices.index

        def fake_evaluate(history, _rule, _levels, **kwargs):
            if history.index[-1] == index[5] and kwargs["position_mode"] == "entry":
                return {
                    "verdict": "BUY",
                    "risk_plan": {"stop": 50.0, "target": 200.0},
                }
            return {"verdict": "HOLD"}

        with patch("lib.rule_backtest.levels.find_levels", return_value=[]), \
                patch("lib.rule_backtest.rules.evaluate", side_effect=fake_evaluate):
            result = rule_backtest.run_walk_forward(
                prices, {}, train_bars=5, test_bars=8, step_bars=8,
                max_holding_days=2, one_way_bps=0, slippage_bps=0,
                acceptance=LOOSE_ACCEPTANCE,
            )

        trade = result["trades"].iloc[0]
        self.assertEqual(trade["entry_time"], index[6])
        self.assertEqual(trade["exit_time"], index[8])
        self.assertEqual(trade["holding_days"], 2)
        self.assertEqual(trade["exit_reason"], "max_holding")

    def test_max_drawdown_includes_open_position_mark_to_market(self):
        prices = synthetic_ohlcv(16)
        index = prices.index
        prices.loc[index[7], "Close"] = 60.0
        prices.loc[index[7], "Low"] = 59.0

        def fake_evaluate(history, _rule, _levels, **kwargs):
            if history.index[-1] == index[5] and kwargs["position_mode"] == "entry":
                return {
                    "verdict": "BUY",
                    "risk_plan": {"stop": 1.0, "target": 1000.0},
                }
            if history.index[-1] == index[8] and kwargs["position_mode"] == "holding":
                return {"verdict": "RISK_EXIT", "risk_plan": {}}
            return {"verdict": "HOLD" if kwargs["position_mode"] == "holding"
                    else "NEUTRAL", "risk_plan": {}}

        with patch("lib.rule_backtest.levels.find_levels", return_value=[]), \
                patch("lib.rule_backtest.rules.evaluate", side_effect=fake_evaluate):
            result = rule_backtest.run_walk_forward(
                prices, {}, train_bars=5, test_bars=10, step_bars=10,
                one_way_bps=0, slippage_bps=0,
                max_holding_days=10, acceptance=LOOSE_ACCEPTANCE,
            )

        self.assertEqual(len(result["trades"]), 1)
        self.assertGreater(float(result["summary"]["max_drawdown"]), 0.35)


class PointInTimeTests(unittest.TestCase):
    def test_level_window_never_contains_future_and_is_limited_to_182_days(self):
        prices = synthetic_ohlcv(215, freq="D")
        calls = []

        def fake_levels(frame):
            calls.append(frame.index.copy())
            return []

        def fake_evaluate(history, _rule, _levels, **_kwargs):
            self.assertEqual(history.index[-1], calls[-1].max())
            return {"verdict": "NEUTRAL"}

        with patch("lib.rule_backtest.levels.find_levels", side_effect=fake_levels), \
                patch("lib.rule_backtest.rules.evaluate", side_effect=fake_evaluate):
            result = rule_backtest.run_walk_forward(
                prices, {}, train_bars=200, test_bars=10, step_bars=10,
                acceptance=LOOSE_ACCEPTANCE,
            )

        self.assertGreater(len(calls), 0)
        for window in calls:
            self.assertLessEqual(window.max() - window.min(), pd.Timedelta(days=182))
        self.assertEqual(result["summary"]["trade_count"], 0)
        self.assertEqual(result["summary"]["status"], "INSUFFICIENT_DATA")


if __name__ == "__main__":
    unittest.main()

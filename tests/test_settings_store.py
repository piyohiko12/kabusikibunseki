import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from lib import settings_store


class SettingsStoreSafetyTests(unittest.TestCase):
    def test_non_mapping_json_is_ignored(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            with mock.patch.object(settings_store, "DATA_FILE", path):
                for value in (None, [], "bad"):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    self.assertEqual(settings_store.load(), {})

    def test_invalid_runtime_values_are_removed(self):
        raw = {
            "moomoo_enabled": "yes",
            "moomoo_port": "bad",
            "moomoo_history_reserve": -1,
            "advanced_chart": [],
            "default_ticker": " skhy ",
        }
        cleaned = settings_store._normalize(raw)
        self.assertEqual(cleaned, {"default_ticker": "SKHY"})

    def test_numeric_strings_are_normalized_and_saved_safely(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            with mock.patch.object(settings_store, "DATA_FILE", path):
                saved = settings_store.save(
                    moomoo_port="11111", moomoo_history_reserve="10",
                    advanced_chart={"preset": "スイング"})
                self.assertEqual(saved["moomoo_port"], 11111)
                self.assertEqual(saved["moomoo_history_reserve"], 10)
                self.assertIsInstance(settings_store.load()["advanced_chart"], dict)

    def test_malformed_nested_chart_values_are_made_ui_safe(self):
        cleaned = settings_store._normalize({
            "advanced_chart": {
                "preset": "スイング",
                "chart_type": 123,
                "overlays": "移動平均線(SMA)",
                "oscillators": ["RSI", 123, "未知"],
                "indicator_params": {
                    "sma_periods": [10],
                    "ema_periods": "20,50",
                    "boll_period": "bad",
                    "boll_std": float("inf"),
                    "rsi_period": -1,
                    "macd_fast": 50,
                    "macd_slow": 3,
                },
                "height": "999",
                "events": "true",
            },
        })["advanced_chart"]
        self.assertEqual(cleaned["preset"], "スイング")
        self.assertNotIn("chart_type", cleaned)
        self.assertNotIn("overlays", cleaned)
        self.assertEqual(cleaned["oscillators"], ["RSI"])
        self.assertNotIn("height", cleaned)
        self.assertNotIn("events", cleaned)
        params = cleaned["indicator_params"]
        self.assertEqual(params["sma_periods"], [20, 50, 200])
        self.assertEqual(params["ema_periods"], [20, 50])
        self.assertEqual(params["boll_period"], 20)
        self.assertEqual(params["boll_std"], 2.0)
        self.assertEqual(params["rsi_period"], 14)
        self.assertGreater(params["macd_slow"], params["macd_fast"])
        # stock_analysis.pyの既存添字参照が常に安全。
        self.assertEqual(len(params["sma_periods"]), 3)
        self.assertEqual(len(params["ema_periods"]), 2)

    def test_valid_nested_chart_values_are_normalized_without_string_arrays(self):
        cleaned = settings_store._normalize({
            "advanced_chart": {
                "overlays": ["ボリンジャーバンド", "ボリンジャーバンド"],
                "benchmarks": ["S&P500", "UNKNOWN"],
                "indicator_params": {
                    "sma_periods": ["10", "25", "100"],
                    "ema_periods": ["9", "21"],
                    "boll_std": "2.5",
                },
                "height": "640",
            },
        })["advanced_chart"]
        self.assertEqual(cleaned["overlays"], ["ボリンジャーバンド"])
        self.assertEqual(cleaned["benchmarks"], ["S&P500"])
        self.assertEqual(cleaned["indicator_params"]["sma_periods"], [10, 25, 100])
        self.assertEqual(cleaned["indicator_params"]["ema_periods"], [9, 21])
        self.assertEqual(cleaned["indicator_params"]["boll_std"], 2.5)
        self.assertEqual(cleaned["height"], 640)


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from lib import charts, indicators


def sample_prices(rows: int = 260) -> pd.DataFrame:
    index = pd.bdate_range("2025-01-02", periods=rows)
    close = pd.Series([100 + i * 0.15 + (i % 7 - 3) * 0.2
                       for i in range(rows)], index=index)
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close + 1.2,
        "Low": close - 1.1,
        "Close": close,
        "Volume": [1000 + i * 5 for i in range(rows)],
    }, index=index)


class IndicatorParameterTests(unittest.TestCase):
    def test_custom_indicator_parameters(self):
        result = indicators.add_indicators(sample_prices(), {
            "sma_periods": (10, 30, 100),
            "ema_periods": (8, 21),
            "boll_period": 15,
            "boll_std": 2.5,
            "rsi_period": 9,
            "macd_fast": 8,
            "macd_slow": 21,
            "macd_signal": 5,
            "volume_ma": 12,
        })
        for column in ("SMA10", "SMA30", "SMA100", "EMA8", "EMA21",
                       "BB_mid", "BB_up", "BB_low", "RSI", "MACD",
                       "MACD_signal", "VOL_MA"):
            self.assertIn(column, result.columns)
        self.assertNotIn("SMA200", result.columns)
        self.assertTrue(result["VOL_MA"].iloc[-1] > 0)


class AdvancedChartTests(unittest.TestCase):
    def setUp(self):
        self.prices = indicators.add_indicators(sample_prices())

    def test_all_chart_types(self):
        expected = {
            "ローソク足": "candlestick",
            "平均足": "candlestick",
            "OHLCバー": "ohlc",
            "ライン": "scatter",
            "エリア": "scatter",
        }
        for chart_type, trace_type in expected.items():
            with self.subTest(chart_type=chart_type):
                figure = charts.price_chart(self.prices, "TEST", {
                    "chart_type": chart_type,
                    "overlays": [], "oscillators": [],
                })
                self.assertEqual(figure.data[0].type, trace_type)

    def test_interaction_and_theme_options(self):
        figure = charts.price_chart(self.prices, "TEST", {
            "interaction": "移動", "theme": "ダーク", "grid": True,
            "range_slider": True, "range_selector": True,
            "overlays": ["移動平均線(SMA)"],
            "oscillators": ["RSI", "MACD"],
        })
        self.assertEqual(figure.layout.dragmode, "pan")
        self.assertEqual(figure.layout.paper_bgcolor, "#111827")
        self.assertEqual(figure.layout.uirevision, "TEST-1d-ローソク足")
        self.assertTrue(figure.layout.xaxis.showgrid)
        self.assertGreaterEqual(len(figure.data), 8)

    def test_multi_timeframe_mini_chart(self):
        figure = charts.mini_price_chart(self.prices, "日足・1年")
        self.assertEqual(figure.layout.height, 310)
        self.assertEqual(figure.data[0].type, "candlestick")
        self.assertFalse(figure.layout.xaxis.rangeslider.visible)


if __name__ == "__main__":
    unittest.main()


class SubplotIsolationTests(unittest.TestCase):
    """サブチャートが価格パネルに侵食されないことを確認する。"""

    def setUp(self):
        self.prices = indicators.add_indicators(sample_prices())

    def test_rangeslider_is_off_on_every_row_by_default(self):
        # ローソク足はrangesliderの既定がTrueで、消さないと1行目のスライダーが
        # 出来高パネルの位置に価格チャートの縮小版を描いてしまう。
        figure = charts.price_chart(self.prices, "TEST", {
            "oscillators": ["出来高", "RSI", "MACD"], "range_slider": False,
        })
        for name in ("xaxis", "xaxis2", "xaxis3", "xaxis4"):
            self.assertFalse(bool(figure.layout[name].rangeslider.visible), name)

    def test_rangeslider_only_on_bottom_row_when_enabled(self):
        figure = charts.price_chart(self.prices, "TEST", {
            "oscillators": ["出来高", "RSI"], "range_slider": True,
        })
        self.assertFalse(bool(figure.layout.xaxis.rangeslider.visible))
        self.assertFalse(bool(figure.layout.xaxis2.rangeslider.visible))
        self.assertTrue(bool(figure.layout.xaxis3.rangeslider.visible))

    def test_volume_profile_does_not_hijack_a_subplot_axis(self):
        # 重ね描き用の軸がサブプロットの軸(x2, x3...)と衝突すると、
        # その行のトレースが別の座標系に飛んでしまう。
        for oscillators in ([], ["出来高"], ["出来高", "RSI", "MACD"]):
            with self.subTest(oscillators=oscillators):
                figure = charts.price_chart(self.prices, "TEST", {
                    "overlays": ["出来高プロファイル"], "oscillators": oscillators,
                })
                rows = 1 + len(oscillators)
                for i in range(1, rows + 1):
                    name = "xaxis" if i == 1 else f"xaxis{i}"
                    self.assertIsNone(figure.layout[name].overlaying, name)
                overlay = figure.layout[f"xaxis{rows + 1}"]
                self.assertEqual(overlay.overlaying, "x")
                self.assertEqual(overlay.anchor, "y")
                profile = [t for t in figure.data if t.name == "出来高プロファイル"]
                self.assertEqual(len(profile), 1)
                self.assertEqual(profile[0].xaxis, f"x{rows + 1}")
                self.assertEqual(profile[0].yaxis, "y")

    def test_volume_traces_stay_on_their_own_row(self):
        figure = charts.price_chart(self.prices, "TEST", {
            "overlays": ["移動平均線(SMA)", "出来高プロファイル"],
            "oscillators": ["出来高", "RSI"],
        })
        volume = [t for t in figure.data if t.name in ("出来高", "出来高MA20")]
        self.assertEqual(len(volume), 2)
        for trace in volume:
            self.assertEqual((trace.xaxis, trace.yaxis), ("x2", "y2"))
        price = [t for t in figure.data if t.type == "candlestick"]
        self.assertEqual(len(price), 1)
        self.assertIn(price[0].xaxis, (None, "x"))
        self.assertIn(price[0].yaxis, (None, "y"))


class VolumeMovingAverageTests(unittest.TestCase):
    """VOL_MA20はチャートの期間設定に左右されない固定の20日平均。"""

    def test_vol_ma20_is_always_the_20_period_mean(self):
        prices = sample_prices()
        expected = prices["Volume"].rolling(20).mean()
        for volume_ma in (5, 20, 60):
            with self.subTest(volume_ma=volume_ma):
                result = indicators.add_indicators(prices, {"volume_ma": volume_ma})
                self.assertIn("VOL_MA20", result.columns)
                self.assertIn("VOL_MA", result.columns)
                pd.testing.assert_series_equal(
                    result["VOL_MA20"], expected, check_names=False)
                pd.testing.assert_series_equal(
                    result["VOL_MA"],
                    prices["Volume"].rolling(volume_ma).mean(), check_names=False)

    def test_consumers_fall_back_to_vol_ma(self):
        from lib import levels, rules
        result = indicators.add_indicators(sample_prices(), {"volume_ma": 10})
        without = result.drop(columns=["VOL_MA20"])
        self.assertIsNotNone(rules.METRICS["vol_ratio"]["fn"]({"df": without}))
        self.assertEqual(len(levels.find_levels(without)),
                         len(levels.find_levels(result)))

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

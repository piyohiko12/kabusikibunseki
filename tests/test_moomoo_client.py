import unittest
from unittest.mock import Mock, patch

import pandas as pd

from lib import moomoo_client


class SymbolNormalizationTests(unittest.TestCase):
    def test_class_share_dot_is_not_mistaken_for_market_prefix(self):
        self.assertEqual(moomoo_client.to_code("BRK.B"), "US.BRK.B")
        self.assertEqual(moomoo_client.to_code("brk.b"), "US.BRK.B")

    def test_supported_market_prefixes_are_preserved(self):
        for code in ("US.BRK.B", "HK.00700", "SH.600519", "SZ.000001",
                     "SG.D05", "MY.1155", "JP.7203", "CC.BTCUSD"):
            with self.subTest(code=code):
                self.assertEqual(moomoo_client.to_code(code), code)

    def test_plain_us_and_yahoo_only_symbols_keep_existing_contract(self):
        self.assertEqual(moomoo_client.to_code("AAPL"), "US.AAPL")
        self.assertIsNone(moomoo_client.to_code("^GSPC"))
        self.assertIsNone(moomoo_client.to_code("ES=F"))

    def test_realtime_snapshot_queries_class_share_with_normalized_code(self):
        context = Mock()
        context.get_market_snapshot.return_value = (0, pd.DataFrame([{
            "code": "US.BRK.B", "name": "Berkshire Hathaway",
            "last_price": 500.0, "prev_close_price": 499.0,
        }]))
        with patch.object(moomoo_client, "_ctx", return_value=context):
            result = moomoo_client.snapshot.__wrapped__(("BRK.B",))

        context.get_market_snapshot.assert_called_once_with(["US.BRK.B"])
        self.assertEqual(result["BRK.B"]["price"], 500.0)


class UnderlyingVolatilityTests(unittest.TestCase):
    def test_underlying_apis_are_used_and_contract_api_is_never_called(self):
        context = Mock()
        context.get_option_underlying_overview.return_value = (0, pd.DataFrame([{
            "code": "US.AAPL", "iv": 35.0, "pre_iv": 34.0,
            "hv_30d": 25.0, "iv_rank": 70.0,
        }]))
        first = pd.DataFrame([
            {"time": "2026-08-21", "iv": 35.0, "hv": 25.0},
        ])
        second = pd.DataFrame([
            {"time": "2026-08-20", "iv": 32.0, "hv": 24.0},
        ])
        context.get_option_underlying_his_volatility.side_effect = [
            (0, first, "next"), (0, second, None),
        ]
        context.get_option_volatility.side_effect = AssertionError(
            "option contract API must not receive an underlying")

        with patch.object(moomoo_client, "_ctx", return_value=context):
            result = moomoo_client.option_volatility.__wrapped__("AAPL")

        self.assertEqual(result["code"], "US.AAPL")
        self.assertEqual(result["source"], "moomoo 原資産オプション統計")
        self.assertEqual(list(result["series"]["IV"]), [32.0, 35.0])
        self.assertEqual(list(result["series"]["IVプレミアム"]), [8.0, 10.0])
        self.assertAlmostEqual(result["average_iv"], 33.5)
        context.get_option_underlying_overview.assert_called_once_with(["US.AAPL"])
        self.assertEqual(context.get_option_underlying_his_volatility.call_count, 2)
        context.get_option_volatility.assert_not_called()
        self.assertFalse(context.close.called)

    def test_overview_only_sdk_returns_current_snapshot(self):
        class OverviewOnlyContext:
            def __init__(self):
                self.calls = []

            def get_option_underlying_overview(self, codes):
                self.calls.append(codes)
                return 0, pd.DataFrame([{"code": codes[0], "iv": 30.0,
                                         "hv_30d": 22.0}])

        context = OverviewOnlyContext()
        with patch.object(moomoo_client, "_ctx", return_value=context):
            result = moomoo_client.option_volatility.__wrapped__("BRK.B")

        self.assertEqual(context.calls, [["US.BRK.B"]])
        self.assertEqual(result["series"].iloc[0]["IVプレミアム"], 8.0)

    def test_old_sdk_does_not_fall_back_to_contract_volatility_api(self):
        class OldContext:
            def __init__(self):
                self.contract_api_called = False

            def get_option_volatility(self, _code):
                self.contract_api_called = True
                return 0, pd.DataFrame()

        context = OldContext()
        with patch.object(moomoo_client, "_ctx", return_value=context):
            result = moomoo_client.option_volatility.__wrapped__("AAPL")

        self.assertIsNone(result)
        self.assertFalse(context.contract_api_called)


if __name__ == "__main__":
    unittest.main()

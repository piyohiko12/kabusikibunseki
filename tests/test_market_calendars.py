import sys
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from lib import moomoo_client


def _fake_moomoo_module():
    module = types.ModuleType("moomoo")
    module.Market = types.SimpleNamespace(US="US")
    module.EconomicImportance = types.SimpleNamespace(HIGH="HIGH")
    module.EarningsCalendarSortType = types.SimpleNamespace(MARKET_CAP="MARKET_CAP")
    return module


class MarketCalendarTests(unittest.TestCase):
    def test_economic_calendar_uses_read_only_calendar_endpoint(self):
        context = Mock()
        context.get_economic_calendar.return_value = (0, pd.DataFrame([{
            "timestamp": 1, "country": "US", "title": "CPI", "star": "HIGH",
            "previous": "2.5", "consensus": "2.4", "actual": None,
        }]), None, False)
        with patch.object(moomoo_client, "_ctx", return_value=context), \
                patch.object(moomoo_client, "_ok", return_value=True), \
                patch.dict(sys.modules, {"moomoo": _fake_moomoo_module()}):
            result = moomoo_client.economic_calendar.__wrapped__(7)
        self.assertEqual(result.iloc[0]["title"], "CPI")
        context.get_economic_calendar.assert_called_once()

    def test_earnings_calendar_normalises_ticker(self):
        context = Mock()
        context.get_earnings_calendar.return_value = (0, pd.DataFrame([{
            "security": "US.AAPL", "name": "Apple", "earnings_date": "2026-08-12",
            "pub_type": "AFTER", "eps_predict": 1.5, "market_cap": 3e12,
        }]))
        with patch.object(moomoo_client, "_ctx", return_value=context), \
                patch.object(moomoo_client, "_ok", return_value=True), \
                patch.dict(sys.modules, {"moomoo": _fake_moomoo_module()}):
            result = moomoo_client.earnings_calendar.__wrapped__(7, 50)
        self.assertEqual(result.iloc[0]["ticker"], "AAPL")

    def test_calendar_failure_isolated_as_empty(self):
        context = Mock()
        context.get_economic_calendar.side_effect = RuntimeError("offline")
        with patch.object(moomoo_client, "_ctx", return_value=context), \
                patch.dict(sys.modules, {"moomoo": _fake_moomoo_module()}):
            result = moomoo_client.economic_calendar.__wrapped__(7)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()

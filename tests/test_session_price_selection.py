import unittest

from lib import data_fetcher


def snapshot():
    return {
        "source": "moomoo OpenAPI",
        "price": 163.41,
        "pre_price": 164.25,
        "after_price": 163.77,
        "overnight_price": 166.67,
        "bid": 163.40,
        "ask": 163.42,
        "update_time": "2026-08-21 16:01:00",
        "fetched_at": "2026-08-21T20:01:03+00:00",
        "session_quotes": {
            "premarket": {
                "available": True, "price": 164.25,
                "source": "moomoo OpenAPI snapshot",
                "timestamp_verified": False, "updated_at": None,
                "fetched_at": "2026-08-21T20:01:03+00:00",
            },
            "regular": {
                "available": True, "price": 163.41,
                "source": "moomoo OpenAPI snapshot",
                "timestamp_verified": True,
                "updated_at": "2026-08-21T15:59:55-04:00",
            },
            "afterhours": {
                "available": True, "price": 163.77,
                "source": "moomoo OpenAPI snapshot",
                "timestamp_verified": False, "updated_at": None,
                "fetched_at": "2026-08-21T20:01:03+00:00",
            },
            "overnight": {
                "available": True, "price": 166.67,
                "source": "moomoo OpenAPI snapshot",
                "timestamp_verified": False, "updated_at": None,
                "fetched_at": "2026-08-21T20:01:03+00:00",
            },
        },
    }


class SessionPriceSelectionTests(unittest.TestCase):
    def test_each_active_session_uses_its_own_price(self):
        expected = {
            "premarket": (164.25, "pre_price", "プレ価格"),
            "regular": (163.41, "price", "立会価格"),
            "afterhours": (163.77, "after_price", "アフター価格"),
            "overnight": (166.67, "overnight_price", "夜間価格"),
        }
        for session, (price, field, label) in expected.items():
            with self.subTest(session=session):
                selected = data_fetcher.select_session_price(
                    snapshot(), {"session": session})
                self.assertEqual(selected["price"], price)
                self.assertEqual(selected["decision_price"], price)
                self.assertEqual(selected["price_field"], field)
                self.assertEqual(selected["label_ja"], label)

    def test_extended_price_does_not_borrow_generic_quote_timestamp(self):
        selected = data_fetcher.select_session_price(
            snapshot(), {"session": "afterhours"})
        self.assertFalse(selected["timestamp_verified"])
        self.assertIsNone(selected["as_of"])
        self.assertIsNotNone(selected["observed_at"])

        safe = data_fetcher.snapshot_for_session(snapshot(), selected)
        self.assertEqual(safe["price"], 163.77)
        self.assertIsNone(safe["bid"])
        self.assertIsNone(safe["ask"])
        self.assertIsNone(safe["update_time"])

    def test_regular_price_keeps_verified_bbo(self):
        selected = data_fetcher.select_session_price(
            snapshot(), {"session": "regular"})
        safe = data_fetcher.snapshot_for_session(snapshot(), selected)
        self.assertTrue(selected["timestamp_verified"])
        self.assertEqual(safe["price"], 163.41)
        self.assertEqual(safe["bid"], 163.40)
        self.assertEqual(safe["ask"], 163.42)

    def test_missing_extended_price_is_display_only_fallback(self):
        raw = snapshot()
        raw["after_price"] = None
        raw["session_quotes"]["afterhours"].update({
            "available": False, "price": None,
        })
        selected = data_fetcher.select_session_price(
            raw, {"session": "afterhours"},
            fallback_price=160.0, fallback_as_of="2026-08-20")
        self.assertEqual(selected["price"], 160.0)
        self.assertIsNone(selected["decision_price"])
        self.assertFalse(selected["decision_available"])
        self.assertTrue(selected["fallback_used"])

    def test_unknown_session_uses_calendar_session_without_guessing_regular(self):
        selected = data_fetcher.select_session_price(
            snapshot(), {"session": "unknown", "calendar_session": "overnight"})
        self.assertEqual(selected["session"], "overnight")
        self.assertEqual(selected["price"], 166.67)


if __name__ == "__main__":
    unittest.main()

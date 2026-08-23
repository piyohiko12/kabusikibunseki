import json
import unittest
from unittest import mock

from lib import news_fetcher


def _item(title, symbols=None):
    content = {
        "title": title,
        "summary": "summary",
        "pubDate": "2026-08-22T12:00:00Z",
        "provider": {"displayName": "Test"},
        "canonicalUrl": {"url": "https://example.com/article"},
    }
    if symbols is not None:
        content["finance"] = {"stockTickers": symbols}
    return {"content": content}


class YahooNewsRelevanceTests(unittest.TestCase):
    def setUp(self):
        news_fetcher.fetch_news.clear()

    def tearDown(self):
        news_fetcher.fetch_news.clear()

    def test_explicitly_unrelated_recommendations_are_removed(self):
        rows = [
            _item("Apple update", ["AAPL"]),
            _item("Unrelated sports tax story", ["BABA", "DIS"]),
            _item("Untagged provider story", None),
        ]
        with mock.patch.object(news_fetcher.yf, "Ticker") as ticker_class:
            ticker_class.return_value.news = rows
            result = news_fetcher.fetch_news("AAPL", "Apple Inc.")
        self.assertEqual(
            [row["title"] for row in result],
            ["Apple update"],
        )

    def test_symbol_normalization_accepts_us_prefix_and_share_class(self):
        item = {"relatedTickers": [{"symbol": "US.BRK-B"}],
                "content": _item("Berkshire update")["content"]}
        with mock.patch.object(news_fetcher.yf, "Ticker") as ticker_class:
            ticker_class.return_value.news = [item]
            result = news_fetcher.fetch_news("BRK.B")
        self.assertEqual(len(result), 1)

    def test_untagged_recommendations_need_company_or_ticker_mention(self):
        rows = [
            _item("Apple launches a new product", None),
            _item("Anthropic plans a market debut", None),
            _item("AAPL valuation update", None),
        ]
        with mock.patch.object(news_fetcher.yf, "Ticker") as ticker_class:
            ticker_class.return_value.news = rows
            result = news_fetcher.fetch_news("AAPL", "Apple Inc.")
        self.assertEqual(
            [row["title"] for row in result],
            ["Apple launches a new product", "AAPL valuation update"],
        )

    def test_schema_variants_and_malformed_rows_do_not_stop_all_news(self):
        rows = [
            None,
            "bad row",
            {"content": "unexpected string"},
            {"content": {
                "title": "AAPL modern schema",
                "provider": "Direct Provider",
                "canonicalUrl": "https://example.com/modern",
                "pubDate": 123,
                "finance": {"stockTickers": ["AAPL"]},
            }},
            {
                "title": "AAPL legacy schema", "publisher": "Legacy Provider",
                "link": "https://example.com/legacy",
                "providerPublishTime": 1_786_291_200,
                "relatedTickers": ["AAPL"],
            },
        ]
        with mock.patch.object(news_fetcher.yf, "Ticker") as ticker_class:
            ticker_class.return_value.news = rows
            result = news_fetcher.fetch_news("AAPL")
        self.assertEqual([row["title"] for row in result], [
            "AAPL modern schema", "AAPL legacy schema"])
        self.assertEqual(result[0]["provider"], "Direct Provider")
        self.assertEqual(result[0]["url"], "https://example.com/modern")
        self.assertEqual(result[0]["pub_date"], "")
        self.assertEqual(result[1]["provider"], "Legacy Provider")
        self.assertRegex(result[1]["pub_date"], r"^2026-")

    def test_non_list_yahoo_payload_fails_safe(self):
        with mock.patch.object(news_fetcher.yf, "Ticker") as ticker_class:
            ticker_class.return_value.news = {"unexpected": "mapping"}
            self.assertEqual(news_fetcher.fetch_news("AAPL"), [])

    def test_sec_share_class_uses_dash_ticker(self):
        recent = {
            "filings": {"recent": {
                "form": ["10-K"], "filingDate": ["2026-08-20"],
                "accessionNumber": ["0001-02-03"],
                "primaryDocument": ["brk-2026.htm"],
            }},
        }
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            recent).encode("utf-8")
        with mock.patch.object(news_fetcher, "_sec_cik_map",
                               return_value={"BRK-B": 1067983}), \
                mock.patch.object(news_fetcher.urllib.request, "urlopen",
                                  return_value=response):
            result = news_fetcher.fetch_sec_filings.__wrapped__("US.BRK.B")
        self.assertEqual(len(result), 1)
        self.assertIn("10-K", result[0]["title"])


if __name__ == "__main__":
    unittest.main()

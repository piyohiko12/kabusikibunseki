import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from lib import derivatives_context as dc


def yahoo_batch(symbols=("ES=F", "NQ=F"), rows=25) -> pd.DataFrame:
    index = pd.bdate_range("2026-06-01", periods=rows)
    columns = pd.MultiIndex.from_product([symbols, ["Close"]])
    values = np.column_stack([
        np.linspace(100.0 + offset * 20, 120.0 + offset * 20, rows)
        for offset, _ in enumerate(symbols)
    ])
    return pd.DataFrame(values, index=index, columns=columns)


class FakeResponse:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, handler):
        self.handler = handler
        self.headers = {}
        self.calls = []
        self.closed = False

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self.handler(url, params or {})

    def close(self):
        self.closed = True


def okx_payload(data, code="0", msg=""):
    return {"code": code, "msg": msg, "data": [data] if data is not None else []}


def okx_handler(fail_endpoint=None, asset_prices=None, endpoint_ts_offsets_ms=None):
    asset_prices = asset_prices or {asset: 100.0 for asset in dc.PERP_ASSETS}
    endpoint_ts_offsets_ms = endpoint_ts_offsets_ms or {}
    base_ts = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000) - 10_000
    funding_time = base_ts
    next_funding_time = funding_time + 8 * 60 * 60 * 1000

    def handle(url, params):
        endpoint = url.split("/api/v5/", 1)[-1]
        if endpoint == fail_endpoint:
            return FakeResponse(error=requests.Timeout("timeout"))
        instrument = params["instId"]
        asset = instrument.split("-", 1)[0]
        price = asset_prices[asset]
        endpoint_ts = base_ts + endpoint_ts_offsets_ms.get(endpoint, 0)
        if endpoint == "market/ticker":
            data = {"instId": instrument, "last": str(price),
                    "open24h": str(price * 0.8), "high24h": str(price * 1.1),
                    "low24h": str(price * 0.7), "volCcy24h": "123.5",
                    "ts": str(endpoint_ts)}
        elif endpoint == "public/mark-price":
            data = {"instId": instrument, "markPx": str(price * 1.01),
                    "ts": str(endpoint_ts)}
        elif endpoint == "public/open-interest":
            data = {"instId": instrument, "oiUsd": "5000000",
                    "ts": str(endpoint_ts)}
        elif endpoint == "public/funding-rate":
            data = {"instId": instrument, "fundingRate": "0.0001",
                    "premium": "0.0002", "fundingTime": str(funding_time),
                    "nextFundingTime": str(next_funding_time),
                    "ts": str(endpoint_ts)}
        else:
            raise AssertionError(f"unexpected endpoint: {endpoint}")
        return FakeResponse(okx_payload(data))

    return handle


class CatalogAndMappingTests(unittest.TestCase):
    def test_futures_catalog_contains_required_contracts_and_metadata(self):
        expected = {"ES=F", "NQ=F", "YM=F", "RTY=F", "ZN=F",
                    "CL=F", "NG=F", "GC=F", "HG=F"}
        self.assertEqual(set(dc.FUTURES_CATALOG), expected)
        for spec in dc.FUTURES_CATALOG.values():
            self.assertTrue(spec["name"])
            self.assertTrue(spec["category"])
            self.assertTrue(spec["relation"])
            self.assertGreaterEqual(spec["digits"], 2)
            self.assertTrue(spec["unit"])

    def test_suggestions_always_start_with_es_and_are_limited_to_three(self):
        self.assertEqual(
            dc.suggested_future_symbols("Energy", "Oil & Gas E&P", "XOM"),
            ("ES=F", "CL=F", "NG=F"),
        )
        technology = dc.suggested_future_symbols(
            "Technology", "Semiconductors", "NVDA")
        self.assertEqual(technology, ("ES=F", "NQ=F", "ZN=F"))
        self.assertLessEqual(len(technology), 3)
        self.assertEqual(
            dc.suggested_future_symbols("エネルギー", None, "US.XOM"),
            ("ES=F", "CL=F", "NG=F"),
        )
        self.assertEqual(dc.suggested_future_symbols(None, None, "UNKNOWN"), ("ES=F",))

    def test_industry_and_ticker_overrides_are_explicit_context_not_direction(self):
        self.assertEqual(
            dc.suggested_future_symbols("Basic Materials", "Gold Mining", "NEM"),
            ("ES=F", "GC=F", "HG=F"),
        )
        notes = " ".join(item["relation"] for item in dc.FUTURES_CATALOG.values())
        self.assertNotIn("買い判定", notes)
        self.assertNotIn("売り判定", notes)

    def test_perp_assets_and_crypto_stock_whitelist_are_explicit(self):
        self.assertEqual(dc.PERP_ASSETS, ("BTC", "ETH", "SOL", "XRP", "DOGE"))
        self.assertEqual(set(dc.PERP_PRICE_DIGITS), set(dc.PERP_ASSETS))
        self.assertEqual(dc.related_perp_assets("mstr"), ("BTC",))
        self.assertEqual(dc.related_perp_assets("US.MSTR"), ("BTC",))
        self.assertEqual(dc.related_perp_assets("COIN"), ("BTC", "ETH"))
        self.assertEqual(dc.related_perp_assets("AAPL"), ())
        self.assertIs(dc.CRYPTO_RELATED_STOCKS, dc.CRYPTO_RELATED_TICKERS)


class YahooFuturesTests(unittest.TestCase):
    def test_batch_parsing_summary_normalization_and_utc_metadata(self):
        raw = yahoo_batch()
        with patch.object(dc.yf, "download", return_value=raw) as download:
            summary, normalized, meta = dc._fetch_yahoo_futures_uncached(
                ("ES=F", "NQ=F"), "3mo")

        download.assert_called_once()
        self.assertEqual(list(summary.columns), dc.FUTURES_SUMMARY_COLUMNS)
        self.assertEqual(list(summary["symbol"]), ["ES=F", "NQ=F"])
        self.assertEqual(list(normalized.columns), ["ES=F", "NQ=F"])
        self.assertIsInstance(normalized.index, pd.DatetimeIndex)
        self.assertIsNotNone(normalized.index.tz)
        self.assertAlmostEqual(float(normalized["ES=F"].dropna().iloc[0]), 100.0)
        self.assertAlmostEqual(float(summary.loc[0, "change_5d_pct"]),
                               (120 / raw[("ES=F", "Close")].iloc[-6] - 1) * 100)
        self.assertTrue(all(stamp.tzinfo is not None for stamp in summary["as_of"]))
        self.assertEqual(meta["status"], "ok")
        self.assertEqual(meta["errors"], {})
        self.assertIsNotNone(meta["fetched_at"].tzinfo)

    def test_single_symbol_flat_columns_and_nan_values_are_safe(self):
        index = pd.bdate_range("2026-07-01", periods=6)
        raw = pd.DataFrame({"Close": [100.0, np.nan, np.inf, 103.0, 104.0, 105.0]},
                           index=index)
        with patch.object(dc.yf, "download", return_value=raw):
            summary, normalized, meta = dc._fetch_yahoo_futures_uncached(("ES=F",))
        self.assertEqual(len(summary), 1)
        numeric = summary.select_dtypes(include="number").to_numpy()
        self.assertFalse(np.isinf(numeric).any())
        self.assertTrue(np.isfinite(normalized.to_numpy()).all())
        self.assertEqual(meta["status"], "ok")

    def test_partial_batch_failure_keeps_successful_symbol(self):
        raw = yahoo_batch(("ES=F",))
        with patch.object(dc.yf, "download", return_value=raw):
            summary, normalized, meta = dc._fetch_yahoo_futures_uncached(
                ("ES=F", "NQ=F"))
        self.assertEqual(list(summary["symbol"]), ["ES=F"])
        self.assertEqual(list(normalized.columns), ["ES=F"])
        self.assertEqual(meta["status"], "partial")
        self.assertEqual(meta["failed"], ("NQ=F",))
        self.assertIn("NQ=F", meta["errors"])

    def test_empty_unknown_and_download_failure_are_safe(self):
        with patch.object(dc.yf, "download") as download:
            summary, normalized, meta = dc._fetch_yahoo_futures_uncached(())
        download.assert_not_called()
        self.assertTrue(summary.empty)
        self.assertTrue(normalized.empty)
        self.assertIsInstance(normalized.index, pd.DatetimeIndex)
        self.assertIsNotNone(normalized.index.tz)
        self.assertEqual(meta["status"], "empty")

        with patch.object(dc.yf, "download", side_effect=RuntimeError("offline")):
            summary, normalized, meta = dc._fetch_yahoo_futures_uncached(("ES=F",))
        self.assertTrue(summary.empty)
        self.assertTrue(normalized.empty)
        self.assertIsInstance(normalized.index, pd.DatetimeIndex)
        self.assertIsNotNone(normalized.index.tz)
        self.assertEqual(meta["status"], "unavailable")
        self.assertIn("offline", meta["errors"]["ES=F"])

        with patch.object(dc.yf, "download") as download:
            _, _, meta = dc._fetch_yahoo_futures_uncached(("BAD",))
        download.assert_not_called()
        self.assertEqual(meta["failed"], ("BAD",))

    def test_public_fetchers_expose_streamlit_cache_clear(self):
        self.assertTrue(callable(dc.fetch_yahoo_futures.clear))
        self.assertTrue(callable(dc.fetch_okx_perpetuals.clear))


class OkxPerpetualTests(unittest.TestCase):
    def test_okx_payload_parser_rejects_error_and_empty_data(self):
        error_session = FakeSession(
            lambda _url, _params: FakeResponse(okx_payload({}, code="51000", msg="bad")))
        with self.assertRaisesRegex(dc.DerivativesContextError, "51000"):
            dc._okx_get(error_session, "market/ticker", {"instId": "BTC-USDT-SWAP"})

        empty_session = FakeSession(
            lambda _url, _params: FakeResponse({"code": "0", "msg": "", "data": []}))
        with self.assertRaisesRegex(dc.DerivativesContextError, "dataが空"):
            dc._okx_get(empty_session, "market/ticker", {"instId": "BTC-USDT-SWAP"})

        empty_item = FakeSession(
            lambda _url, _params: FakeResponse(okx_payload({})))
        with self.assertRaisesRegex(dc.DerivativesContextError, "instIdが不一致"):
            dc._okx_get(empty_item, "market/ticker", {"instId": "BTC-USDT-SWAP"})

        bad_fields = FakeSession(lambda _url, _params: FakeResponse(okx_payload({
            "instId": "BTC-USDT-SWAP", "last": "bad", "open24h": "100",
            "high24h": "101", "low24h": "99", "volCcy24h": "10", "ts": "1",
        })))
        with self.assertRaisesRegex(dc.DerivativesContextError, "必須フィールド"):
            dc._okx_get(bad_fields, "market/ticker", {"instId": "BTC-USDT-SWAP"})

    def test_full_perp_response_parses_units_funding_and_utc_times(self):
        session = FakeSession(okx_handler())
        frame, meta = dc._fetch_okx_perpetuals_uncached(("BTC",), session=session)

        self.assertEqual(list(frame.columns), dc.PERP_COLUMNS)
        row = frame.iloc[0]
        self.assertEqual(row["instrument"], "BTC-USDT-SWAP")
        self.assertEqual(row["venue"], "OKX")
        self.assertEqual(row["status"], "ok")
        self.assertAlmostEqual(float(row["change_24h_pct"]), 25.0)
        self.assertAlmostEqual(float(row["volume_base_24h"]), 123.5)
        self.assertAlmostEqual(float(row["funding_rate_pct"]), 0.01)
        self.assertAlmostEqual(float(row["funding_interval_hours"]), 8.0)
        self.assertAlmostEqual(float(row["funding_annualized_pct"]), 10.95)
        self.assertAlmostEqual(float(row["funding_premium_pct"]), 0.02)
        self.assertAlmostEqual(float(row["open_interest_usd"]), 5_000_000)
        self.assertIsNotNone(row["as_of"].tzinfo)
        self.assertIsNotNone(row["latest_as_of"].tzinfo)
        self.assertEqual(row["freshness_status"], "fresh")
        self.assertEqual(row["metric_freshness"]["open_interest_usd"]["status"],
                         "fresh")
        self.assertIsNotNone(row["funding_time"].tzinfo)
        self.assertIsNotNone(row["next_funding_time"].tzinfo)
        self.assertEqual(len(session.calls), 4)
        self.assertTrue(all(call["timeout"] == 6 for call in session.calls))
        self.assertIn("read-only", session.headers["User-Agent"])
        self.assertEqual(meta["status"], "ok")
        self.assertFalse(meta["authenticated"])
        self.assertTrue(meta["read_only"])
        self.assertEqual(meta["errors"], {})
        self.assertEqual(meta["freshness"]["BTC"]["mark_price"]["status"],
                         "fresh")

    def test_stale_open_interest_prevents_overall_ok_even_when_mark_is_fresh(self):
        session = FakeSession(okx_handler(endpoint_ts_offsets_ms={
            "public/open-interest": -(dc.PERP_FRESHNESS_MAX_AGE_SECONDS + 60) * 1000,
        }))
        frame, meta = dc._fetch_okx_perpetuals_uncached(("BTC",), session=session)

        row = frame.iloc[0]
        self.assertEqual(row["metric_freshness"]["mark_price"]["status"], "fresh")
        self.assertEqual(
            row["metric_freshness"]["open_interest_usd"]["status"], "stale")
        self.assertEqual(row["freshness_status"], "partial")
        self.assertEqual(row["status"], "partial")
        self.assertEqual(meta["status"], "partial")
        self.assertEqual(row["as_of"], row["open_interest_as_of"])
        self.assertGreater(row["latest_as_of"], row["as_of"])
        self.assertIn("open_interest", row["error"])

    def test_partial_endpoint_failure_keeps_row_and_error(self):
        session = FakeSession(okx_handler(fail_endpoint="public/mark-price"))
        frame, meta = dc._fetch_okx_perpetuals_uncached(("ETH",), session=session)
        row = frame.iloc[0]
        self.assertEqual(row["status"], "partial")
        self.assertTrue(np.isnan(row["mark_price"]))
        self.assertIn("mark_price", row["error"])
        self.assertEqual(meta["status"], "partial")
        self.assertEqual(meta["partial"], ("ETH",))
        self.assertIn("ETH", meta["errors"])

    def test_all_failures_and_unsupported_asset_are_unavailable(self):
        def fail_all(_url, _params):
            return FakeResponse(error=requests.ConnectionError("offline"))

        session = FakeSession(fail_all)
        frame, meta = dc._fetch_okx_perpetuals_uncached(("SOL", "BAD"), session=session)
        self.assertEqual(list(frame["status"]), ["unavailable", "unavailable"])
        self.assertEqual(meta["status"], "unavailable")
        self.assertEqual(meta["unavailable"], ("SOL", "BAD"))
        self.assertEqual(len(session.calls), 4)
        self.assertIn("未対応", frame.loc[frame["asset"] == "BAD", "error"].iloc[0])

    def test_empty_assets_do_not_create_session(self):
        with patch.object(dc.requests, "Session") as session:
            frame, meta = dc._fetch_okx_perpetuals_uncached(())
        session.assert_not_called()
        self.assertTrue(frame.empty)
        self.assertEqual(meta["status"], "empty")
        self.assertEqual(meta["errors"], {})

    def test_session_creation_failure_is_returned_as_unavailable(self):
        with patch.object(dc.requests, "Session", side_effect=RuntimeError("no session")):
            frame, meta = dc._fetch_okx_perpetuals_uncached(("BTC",))
        self.assertEqual(frame.loc[0, "status"], "unavailable")
        self.assertIn("no session", frame.loc[0, "error"])
        self.assertEqual(meta["status"], "unavailable")
        self.assertIn("BTC", meta["errors"])
        self.assertIsNotNone(meta["fetched_at"].tzinfo)

    def test_owned_session_close_failure_does_not_discard_data(self):
        session = FakeSession(okx_handler())
        session.close = Mock(side_effect=RuntimeError("close failed"))
        with patch.object(dc.requests, "Session", return_value=session):
            frame, meta = dc._fetch_okx_perpetuals_uncached(("BTC",))
        self.assertEqual(frame.loc[0, "status"], "ok")
        self.assertEqual(meta["status"], "ok")
        session.close.assert_called_once_with()

    def test_five_supported_assets_make_no_more_than_twenty_requests(self):
        session = FakeSession(okx_handler())
        frame, meta = dc._fetch_okx_perpetuals_uncached(dc.PERP_ASSETS,
                                                        session=session)
        self.assertEqual(len(frame), 5)
        self.assertEqual(len(session.calls), 20)
        self.assertEqual(meta["status"], "ok")


if __name__ == "__main__":
    unittest.main()

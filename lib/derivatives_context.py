"""銘柄ページ向けの先物・暗号資産PERP市場コンテキスト。

読み取り専用の公開市場データだけを取得する。注文・認証・口座APIは扱わず、
取得結果を売買判定へ自動加点する責務も持たない。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


FUTURES_CATALOG: dict[str, dict[str, object]] = {
    "ES=F": {
        "name": "E-mini S&P 500先物",
        "category": "株価指数",
        "relation": "米国株市場全体のリスク選好を確認する基準",
        "digits": 2,
        "unit": "指数pt",
    },
    "NQ=F": {
        "name": "E-mini NASDAQ 100先物",
        "category": "株価指数",
        "relation": "大型グロース・テクノロジー株の市場環境を確認する参考",
        "digits": 2,
        "unit": "指数pt",
    },
    "YM=F": {
        "name": "E-miniダウ先物",
        "category": "株価指数",
        "relation": "大型バリュー・資本財を含むダウ構成株の市場環境を確認する参考",
        "digits": 2,
        "unit": "指数pt",
    },
    "RTY=F": {
        "name": "E-mini Russell 2000先物",
        "category": "株価指数",
        "relation": "米国内需・小型株の市場環境を確認する参考",
        "digits": 2,
        "unit": "指数pt",
    },
    "ZN=F": {
        "name": "米10年国債先物",
        "category": "金利",
        "relation": "金利感応度の参考。先物価格と国債利回りは通常逆方向",
        "digits": 5,
        "unit": "価格pt",
    },
    "CL=F": {
        "name": "WTI原油先物",
        "category": "エネルギー",
        "relation": "エネルギー企業の販売価格や運輸・製造業のコスト環境を確認する参考",
        "digits": 2,
        "unit": "USD/バレル",
    },
    "NG=F": {
        "name": "天然ガス先物",
        "category": "エネルギー",
        "relation": "天然ガス関連企業・公益企業の販売価格や燃料コストを確認する参考",
        "digits": 3,
        "unit": "USD/MMBtu",
    },
    "GC=F": {
        "name": "金先物",
        "category": "貴金属",
        "relation": "金鉱株・素材株と安全資産需要の市場環境を確認する参考",
        "digits": 2,
        "unit": "USD/トロイオンス",
    },
    "HG=F": {
        "name": "銅先物",
        "category": "産業用金属",
        "relation": "鉱山・素材・資本財と世界景気の市場環境を確認する参考",
        "digits": 4,
        "unit": "USD/ポンド",
    },
}


_SECTOR_FUTURES: dict[str, tuple[str, ...]] = {
    "technology": ("NQ=F", "ZN=F"),
    "テクノロジー": ("NQ=F", "ZN=F"),
    "communication services": ("NQ=F",),
    "通信サービス": ("NQ=F",),
    "consumer cyclical": ("RTY=F", "CL=F"),
    "一般消費財": ("RTY=F", "CL=F"),
    "consumer defensive": ("ZN=F",),
    "生活必需品": ("ZN=F",),
    "energy": ("CL=F", "NG=F"),
    "エネルギー": ("CL=F", "NG=F"),
    "financial services": ("ZN=F", "YM=F"),
    "金融": ("ZN=F", "YM=F"),
    "healthcare": ("RTY=F", "ZN=F"),
    "ヘルスケア": ("RTY=F", "ZN=F"),
    "industrials": ("YM=F", "HG=F"),
    "資本財": ("YM=F", "HG=F"),
    "basic materials": ("HG=F", "GC=F"),
    "素材": ("HG=F", "GC=F"),
    "real estate": ("ZN=F", "RTY=F"),
    "不動産": ("ZN=F", "RTY=F"),
    "utilities": ("ZN=F", "NG=F"),
    "公益": ("ZN=F", "NG=F"),
}

# 業種名はYahoo Financeの表記揺れを含むため、部分一致で上から適用する。
_INDUSTRY_FUTURES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("semiconductor", "software", "internet content", "electronic"),
     ("NQ=F", "ZN=F")),
    (("oil & gas", "oil and gas", "energy equipment", "refining"),
     ("CL=F", "NG=F")),
    (("gold", "precious metal"), ("GC=F", "HG=F")),
    (("copper", "industrial metal", "mining"), ("HG=F", "GC=F")),
    (("airline", "transportation", "trucking", "railroad"), ("CL=F", "YM=F")),
    (("bank", "insurance", "capital market", "mortgage"), ("ZN=F", "YM=F")),
    (("reit", "real estate"), ("ZN=F", "RTY=F")),
    (("utility",), ("ZN=F", "NG=F")),
)

_TICKER_FUTURES: dict[str, tuple[str, ...]] = {
    "XOM": ("CL=F", "NG=F"), "CVX": ("CL=F", "NG=F"),
    "COP": ("CL=F", "NG=F"), "SLB": ("CL=F", "NG=F"),
    "FCX": ("HG=F", "GC=F"), "NEM": ("GC=F", "HG=F"),
    "GOLD": ("GC=F", "HG=F"),
    "AAL": ("CL=F", "YM=F"), "DAL": ("CL=F", "YM=F"),
    "UAL": ("CL=F", "YM=F"), "LUV": ("CL=F", "YM=F"),
}


def suggested_future_symbols(sector: str | None, industry: str | None,
                             ticker: str | None) -> tuple[str, ...]:
    """ESを基準に、セクター・業種との関連が説明できる先物を最大3本返す。

    関連は市場コンテキストの選択にだけ使い、価格変化を固定的な強気・弱気へ
    変換しない。
    """
    sector_key = str(sector or "").strip().casefold()
    industry_key = str(industry or "").strip().casefold()
    ticker_key = str(ticker or "").strip().upper()
    if ticker_key.startswith("US."):
        ticker_key = ticker_key[3:]

    candidates: list[str] = ["ES=F"]
    candidates.extend(_TICKER_FUTURES.get(ticker_key, ()))
    for needles, symbols in _INDUSTRY_FUTURES:
        if industry_key and any(needle in industry_key for needle in needles):
            candidates.extend(symbols)
            break
    candidates.extend(_SECTOR_FUTURES.get(sector_key, ()))

    selected = []
    for symbol in candidates:
        if symbol in FUTURES_CATALOG and symbol not in selected:
            selected.append(symbol)
        if len(selected) == 3:
            break
    return tuple(selected)


FUTURES_SUMMARY_COLUMNS = [
    "symbol", "name", "category", "price", "unit", "change_1d_pct",
    "change_5d_pct", "change_1m_pct", "as_of", "source", "relation",
]


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _as_utc(value) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(stamp):
        return pd.NaT
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _safe_number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _safe_pct(current: float, previous: float) -> float:
    if not np.isfinite(current) or not np.isfinite(previous) or previous == 0:
        return np.nan
    return (current / previous - 1) * 100


def _empty_futures_result(requested: tuple[str, ...], failed: tuple[str, ...],
                          error: str | None = None):
    meta = {
        "source": "Yahoo Finance",
        "status": "unavailable" if requested else "empty",
        "requested": requested,
        "succeeded": (),
        "failed": failed,
        "fetched_at": _utc_now(),
        "error": error,
        "errors": ({symbol: error or "取得できませんでした" for symbol in failed}
                   if failed else {}),
        "read_only": True,
    }
    normalized = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    return pd.DataFrame(columns=FUTURES_SUMMARY_COLUMNS), normalized, meta


def _unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    out = []
    for value in values or ():
        item = str(value).strip().upper()
        if item and item not in out:
            out.append(item)
    return tuple(out)


def _extract_yahoo_frame(raw: pd.DataFrame, symbol: str,
                         valid_symbols: tuple[str, ...]) -> pd.DataFrame:
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        for level in range(raw.columns.nlevels):
            if symbol in raw.columns.get_level_values(level):
                try:
                    frame = raw.xs(symbol, axis=1, level=level, drop_level=True)
                except (KeyError, ValueError):
                    continue
                return frame if isinstance(frame, pd.DataFrame) else frame.to_frame()
        return pd.DataFrame()
    return raw.copy() if len(valid_symbols) == 1 else pd.DataFrame()


def _close_series(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    column = next((name for name in ("Close", "Adj Close") if name in frame.columns), None)
    if column is None:
        return pd.Series(dtype=float)
    close = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    close = close.dropna()
    if close.empty:
        return close
    close = close[~close.index.duplicated(keep="last")].sort_index()
    index = pd.DatetimeIndex([_as_utc(value) for value in close.index])
    valid = ~index.isna()
    return pd.Series(close.to_numpy()[valid], index=index[valid], name="Close")


def _fetch_yahoo_futures_uncached(
        symbols: tuple[str, ...], period: str = "3mo") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    requested = _unique_strings(symbols)
    valid = tuple(symbol for symbol in requested if symbol in FUTURES_CATALOG)
    invalid = tuple(symbol for symbol in requested if symbol not in FUTURES_CATALOG)
    if not requested or not valid:
        return _empty_futures_result(requested, requested if requested else (),
                                     "対応する先物シンボルがありません" if requested else None)

    try:
        raw = yf.download(
            list(valid), period=period, interval="1d", group_by="ticker",
            auto_adjust=True, progress=False, threads=True,
        )
    except Exception as exc:
        return _empty_futures_result(requested, requested, str(exc))

    rows, normalized = [], {}
    failed = list(invalid)
    for symbol in valid:
        close = _close_series(_extract_yahoo_frame(raw, symbol, valid))
        if close.empty:
            failed.append(symbol)
            continue
        first, last = float(close.iloc[0]), float(close.iloc[-1])
        if not np.isfinite(first) or first == 0 or not np.isfinite(last):
            failed.append(symbol)
            continue
        spec = FUTURES_CATALOG[symbol]
        rows.append({
            "symbol": symbol,
            "name": spec["name"],
            "category": spec["category"],
            "price": last,
            "unit": spec["unit"],
            "change_1d_pct": _safe_pct(last, float(close.iloc[-2]))
            if len(close) >= 2 else np.nan,
            "change_5d_pct": _safe_pct(last, float(close.iloc[-6]))
            if len(close) >= 6 else np.nan,
            "change_1m_pct": _safe_pct(last, float(close.iloc[-22]))
            if len(close) >= 22 else np.nan,
            "as_of": _as_utc(close.index[-1]),
            "source": "Yahoo Finance",
            "relation": spec["relation"],
        })
        normalized[symbol] = close / first * 100

    summary = pd.DataFrame(rows, columns=FUTURES_SUMMARY_COLUMNS)
    if not summary.empty:
        summary["as_of"] = pd.to_datetime(summary["as_of"], utc=True, errors="coerce")
    normalized_frame = (pd.DataFrame(normalized).sort_index() if normalized else
                        pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC")))
    succeeded = tuple(row["symbol"] for row in rows)
    failed_tuple = tuple(symbol for symbol in requested if symbol not in succeeded)
    status = "ok" if succeeded and not failed_tuple else "partial" if succeeded else "unavailable"
    meta = {
        "source": "Yahoo Finance",
        "status": status,
        "requested": requested,
        "succeeded": succeeded,
        "failed": failed_tuple,
        "fetched_at": _utc_now(),
        "error": None if succeeded else "先物データを取得できませんでした",
        "errors": {symbol: "取得できませんでした" for symbol in failed_tuple},
        "read_only": True,
    }
    return summary, normalized_frame, meta


@st.cache_data(ttl=300, show_spinner=False)
def fetch_yahoo_futures(
        symbols: tuple[str, ...], period: str = "3mo") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Yahoo Financeから関連先物を一括取得する（300秒キャッシュ）。"""
    return _fetch_yahoo_futures_uncached(symbols, period)


PERP_ASSETS = ("BTC", "ETH", "SOL", "XRP", "DOGE")
PERP_PRICE_DIGITS = {"BTC": 2, "ETH": 2, "SOL": 3, "XRP": 5, "DOGE": 6}

# 「株式と同名の暗号資産」を誤対応しないよう、直接関連は明示リストだけに限定する。
CRYPTO_RELATED_TICKERS: dict[str, tuple[str, ...]] = {
    "MSTR": ("BTC",),
    "COIN": ("BTC", "ETH"),
    "MARA": ("BTC",),
    "RIOT": ("BTC",),
    "CLSK": ("BTC",),
    "HUT": ("BTC",),
    "IREN": ("BTC",),
    "WULF": ("BTC",),
    "BTDR": ("BTC",),
    "CIFR": ("BTC",),
    "CORZ": ("BTC",),
    "HOOD": ("BTC", "ETH"),
    "XYZ": ("BTC",),
}

# 初期実装時の名称を参照する呼び出し側に備えたalias。
CRYPTO_RELATED_STOCKS = CRYPTO_RELATED_TICKERS


def related_perp_assets(ticker: str | None) -> tuple[str, ...]:
    """暗号資産へ直接関連付けた株式だけ、明示済みPERP候補を返す。"""
    ticker_key = str(ticker or "").strip().upper()
    if ticker_key.startswith("US."):
        ticker_key = ticker_key[3:]
    return CRYPTO_RELATED_TICKERS.get(ticker_key, ())


PERP_COLUMNS = [
    "asset", "instrument", "venue", "last", "mark_price",
    "change_24h_pct", "high_24h", "low_24h", "volume_base_24h",
    "funding_rate_pct", "funding_interval_hours", "funding_annualized_pct",
    "funding_premium_pct", "open_interest_usd", "funding_time",
    "next_funding_time", "as_of",
    "status", "error",
]

OKX_BASE_URL = "https://www.okx.com/api/v5"
OKX_USER_AGENT = "kabusikibunseki/1.0 (read-only market context)"
OKX_TIMEOUT_SECONDS = 6
OKX_REQUIRED_NUMERIC_FIELDS = {
    "market/ticker": ("last", "open24h", "high24h", "low24h", "volCcy24h", "ts"),
    "public/mark-price": ("markPx", "ts"),
    "public/open-interest": ("oiUsd", "ts"),
    "public/funding-rate": ("fundingRate", "fundingTime", "nextFundingTime", "ts"),
}


class DerivativesContextError(RuntimeError):
    """公開市場データの応答形式またはHTTP取得エラー。"""


def _ms_to_utc(value) -> pd.Timestamp:
    number = _safe_number(value)
    if not np.isfinite(number):
        return pd.NaT
    try:
        return pd.to_datetime(number, unit="ms", utc=True)
    except (OverflowError, TypeError, ValueError):
        return pd.NaT


def _okx_get(session: requests.Session, endpoint: str,
              params: dict[str, str]) -> dict:
    endpoint_key = endpoint.lstrip("/")
    response = session.get(
        f"{OKX_BASE_URL}/{endpoint_key}", params=params,
        timeout=OKX_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise DerivativesContextError("OKXの応答がオブジェクトではありません")
    if str(payload.get("code", "")) != "0":
        message = str(payload.get("msg") or "unknown error")
        raise DerivativesContextError(f"OKX code={payload.get('code')}: {message}")
    data = payload.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise DerivativesContextError("OKXのdataが空です")
    item = data[0]
    expected_instrument = params.get("instId")
    if expected_instrument and item.get("instId") != expected_instrument:
        raise DerivativesContextError(
            f"OKXのinstIdが不一致です: expected={expected_instrument}")
    invalid = [
        field for field in OKX_REQUIRED_NUMERIC_FIELDS.get(endpoint_key, ())
        if not np.isfinite(_safe_number(item.get(field)))
    ]
    if invalid:
        raise DerivativesContextError(
            f"OKX {endpoint_key} の必須フィールドが欠損/不正です: {', '.join(invalid)}")
    return item


def _funding_interval_hours(funding: dict) -> float:
    current = _ms_to_utc(funding.get("fundingTime"))
    following = _ms_to_utc(funding.get("nextFundingTime"))
    if pd.isna(current) or pd.isna(following) or following <= current:
        return np.nan
    hours = (following - current).total_seconds() / 3600
    return hours if np.isfinite(hours) and hours > 0 else np.nan


def _empty_perp_row(asset: str) -> dict:
    row = {column: np.nan for column in PERP_COLUMNS}
    row.update({
        "asset": asset,
        "instrument": f"{asset}-USDT-SWAP",
        "venue": "OKX",
        "funding_time": pd.NaT,
        "next_funding_time": pd.NaT,
        "as_of": pd.NaT,
        "status": "unavailable",
        "error": "",
    })
    return row


def _parse_okx_perp(asset: str, responses: dict[str, dict],
                     errors: dict[str, str], fetched_at: pd.Timestamp) -> dict:
    row = _empty_perp_row(asset)
    ticker = responses.get("ticker", {})
    mark = responses.get("mark_price", {})
    interest = responses.get("open_interest", {})
    funding = responses.get("funding_rate", {})

    last = _safe_number(ticker.get("last"))
    open_24h = _safe_number(ticker.get("open24h"))
    funding_decimal = _safe_number(funding.get("fundingRate"))
    interval = _funding_interval_hours(funding)
    funding_pct = funding_decimal * 100 if np.isfinite(funding_decimal) else np.nan
    annualized = (funding_pct * (24 / interval) * 365
                  if np.isfinite(funding_pct) and np.isfinite(interval) and interval > 0
                  else np.nan)

    timestamps = [
        _ms_to_utc(ticker.get("ts")), _ms_to_utc(mark.get("ts")),
        _ms_to_utc(interest.get("ts")), _ms_to_utc(funding.get("ts")),
    ]
    timestamps = [stamp for stamp in timestamps if not pd.isna(stamp)]
    as_of = max(timestamps) if timestamps else fetched_at

    row.update({
        "last": last,
        "mark_price": _safe_number(mark.get("markPx")),
        "change_24h_pct": _safe_pct(last, open_24h),
        "high_24h": _safe_number(ticker.get("high24h")),
        "low_24h": _safe_number(ticker.get("low24h")),
        "volume_base_24h": _safe_number(ticker.get("volCcy24h")),
        "funding_rate_pct": funding_pct,
        "funding_interval_hours": interval,
        # 現在の率を単純反復した参考値で、将来の率を予測するものではない。
        "funding_annualized_pct": annualized,
        # OKX funding-rateのpremium index。Mark-Index乖離ではないため名称を分ける。
        "funding_premium_pct": (_safe_number(funding.get("premium")) * 100
                                if np.isfinite(_safe_number(funding.get("premium")))
                                else np.nan),
        "open_interest_usd": _safe_number(interest.get("oiUsd")),
        "funding_time": _ms_to_utc(funding.get("fundingTime")),
        "next_funding_time": _ms_to_utc(funding.get("nextFundingTime")),
        "as_of": as_of,
        "status": "ok" if len(responses) == 4 else "partial" if responses else "unavailable",
        "error": "; ".join(f"{key}: {value}" for key, value in errors.items()),
    })
    return row


def _empty_perp_result(requested: tuple[str, ...], error: str | None = None):
    meta = {
        "source": "OKX Public API",
        "venue": "OKX",
        "status": "unavailable" if requested else "empty",
        "requested": requested,
        "ok": (),
        "partial": (),
        "unavailable": requested,
        "fetched_at": _utc_now(),
        "errors": ({"request": error} if error else {}),
        "authenticated": False,
        "read_only": True,
    }
    return pd.DataFrame(columns=PERP_COLUMNS), meta


def _finalize_perp_rows(rows: list[dict], requested: tuple[str, ...],
                        fetched_at: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    frame = pd.DataFrame(rows, columns=PERP_COLUMNS)
    for column in ("funding_time", "next_funding_time", "as_of"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    ok = tuple(frame.loc[frame["status"] == "ok", "asset"])
    partial = tuple(frame.loc[frame["status"] == "partial", "asset"])
    unavailable = tuple(frame.loc[frame["status"] == "unavailable", "asset"])
    status = "ok" if len(ok) == len(requested) else (
        "partial" if len(ok) + len(partial) > 0 else "unavailable")
    meta = {
        "source": "OKX Public API",
        "venue": "OKX",
        "status": status,
        "requested": requested,
        "ok": ok,
        "partial": partial,
        "unavailable": unavailable,
        "fetched_at": fetched_at,
        "errors": {row["asset"]: row["error"] for row in rows if row["error"]},
        "authenticated": False,
        "read_only": True,
    }
    return frame, meta


def _fetch_okx_perpetuals_uncached(
        assets: tuple[str, ...], session: requests.Session | None = None) -> tuple[pd.DataFrame, dict]:
    requested = _unique_strings(assets)
    if not requested:
        return _empty_perp_result(())

    fetched_at = _utc_now()
    owned_session = session is None
    http = session
    try:
        if http is None:
            http = requests.Session()
        http.headers.update({"User-Agent": OKX_USER_AGENT, "Accept": "application/json"})
    except Exception as exc:
        if owned_session and http is not None:
            try:
                http.close()
            except Exception:
                pass
        rows = []
        for asset in requested:
            row = _empty_perp_row(asset)
            row["as_of"] = fetched_at
            row["error"] = f"HTTPセッションを開始できませんでした: {exc}"
            rows.append(row)
        return _finalize_perp_rows(rows, requested, fetched_at)
    rows_by_asset: dict[str, dict] = {}
    try:
        supported = []
        jobs = []
        for asset in requested:
            if asset not in PERP_ASSETS:
                row = _empty_perp_row(asset)
                row["error"] = "未対応のPERP資産です"
                rows_by_asset[asset] = row
                continue
            supported.append(asset)
            instrument = f"{asset}-USDT-SWAP"
            specs = (
                ("ticker", "market/ticker", {"instId": instrument}),
                ("mark_price", "public/mark-price",
                 {"instType": "SWAP", "instId": instrument}),
                ("open_interest", "public/open-interest",
                 {"instType": "SWAP", "instId": instrument}),
                ("funding_rate", "public/funding-rate", {"instId": instrument}),
            )
            for key, endpoint, params in specs:
                jobs.append((asset, key, endpoint, params))

        responses_by_asset = {asset: {} for asset in supported}
        errors_by_asset = {asset: {} for asset in supported}
        if jobs:
            # UIの最大3資産では12件、公開上限でも5資産×4件=20件。
            # endpointを同時取得し、障害時も待ち時間をtimeout 1回分に抑える。
            with ThreadPoolExecutor(max_workers=min(20, len(jobs))) as executor:
                futures = {
                    executor.submit(_okx_get, http, endpoint, params): (asset, key)
                    for asset, key, endpoint, params in jobs
                }
                for future in as_completed(futures):
                    asset, key = futures[future]
                    try:
                        responses_by_asset[asset][key] = future.result()
                    except Exception as exc:
                        errors_by_asset[asset][key] = str(exc)

        for asset in supported:
            rows_by_asset[asset] = _parse_okx_perp(
                asset, responses_by_asset[asset], errors_by_asset[asset], fetched_at)
    finally:
        if owned_session:
            try:
                http.close()
            except Exception:
                # 取得済みデータをclose失敗だけで破棄しない。
                pass

    rows = [rows_by_asset[asset] for asset in requested]
    return _finalize_perp_rows(rows, requested, fetched_at)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_okx_perpetuals(assets: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    """OKXの認証不要な公開APIからUSDT建てPERPを取得する（60秒キャッシュ）。"""
    return _fetch_okx_perpetuals_uncached(assets)

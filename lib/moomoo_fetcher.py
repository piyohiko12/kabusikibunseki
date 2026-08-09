"""moomoo OpenAPIから読み取り専用の市場データを取得する。

OpenDまたはSDKが利用できない場合は例外を返し、呼び出し側がyfinanceへ
フォールバックできるようにする。取引コンテキストや注文APIは扱わない。
"""

from __future__ import annotations

import os
import socket
from contextlib import closing

import pandas as pd

_SDK_IMPORT_ERROR = None
try:
    from moomoo import (AuType, KLType, OpenQuoteContext, RET_OK, Session,
                        SubType)
except Exception as exc:  # SDKやログ初期化の失敗でもyfinanceで動かす
    AuType = KLType = OpenQuoteContext = Session = SubType = None
    RET_OK = 0
    _SDK_IMPORT_ERROR = str(exc)


class MoomooError(Exception):
    """OpenD接続、権限、シンボル、API応答に関する取得失敗。"""


INTERVAL_MAP = {
    "1m": "K_1M",
    "5m": "K_5M",
    "15m": "K_15M",
    "1h": "K_60M",
    "1d": "K_DAY",
    "1wk": "K_WEEK",
    "1mo": "K_MON",
}

SUPPORTED_PREFIXES = {"US", "HK", "SH", "SZ", "SG", "MY", "JP", "CC"}


def _connection() -> tuple[str, int]:
    host = os.environ.get("FUTU_OPEND_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("FUTU_OPEND_PORT", "11111"))
    except ValueError:
        port = 11111
    return host, port


def normalize_code(ticker: str) -> str:
    """画面のティッカーをmoomoo形式へ変換する。

    このアプリの主対象は米国株なので、接頭辞のないティッカーはUS銘柄として
    扱う。Yahoo固有の指数・為替・先物コードは変換せず、フォールバックさせる。
    """
    raw = str(ticker).strip().upper()
    if not raw:
        raise MoomooError("ティッカーが空です")

    prefix = raw.split(".", 1)[0]
    if prefix in SUPPORTED_PREFIXES and "." in raw:
        return raw
    if raw.endswith(".HK"):
        return f"HK.{raw[:-3].zfill(5)}"
    if raw.endswith(".T"):
        return f"JP.{raw[:-2]}"
    if raw.startswith("^") or "=" in raw:
        raise MoomooError(f"moomoo形式へ変換できないYahooコードです: {raw}")
    return f"US.{raw}"


def connection_status(timeout: float = 0.4) -> dict:
    """SDKとOpenDの利用可否をpickle可能なdictで返す。"""
    sdk_installed = OpenQuoteContext is not None
    host, port = _connection()
    reachable = False
    if sdk_installed:
        try:
            with closing(socket.create_connection((host, port), timeout=timeout)):
                reachable = True
        except OSError:
            pass

    if not sdk_installed:
        message = ("moomoo-api SDKが未導入です" if not _SDK_IMPORT_ERROR else
                   f"moomoo-api SDKを読み込めません: {_SDK_IMPORT_ERROR}")
    elif not reachable:
        message = f"OpenDに接続できません({host}:{port})"
    else:
        message = f"OpenD接続中({host}:{port})"
    return {
        "available": sdk_installed and reachable,
        "sdk_installed": sdk_installed,
        "opend_reachable": reachable,
        "host": host,
        "port": port,
        "message": message,
    }


def _ensure_available() -> None:
    status = connection_status()
    if not status["available"]:
        raise MoomooError(status["message"])


def _open_context():
    _ensure_available()
    host, port = _connection()
    try:
        return OpenQuoteContext(host=host, port=port, ai_type=1)
    except TypeError:
        # 古いSDKではai_typeが未実装。読み取り専用接続は従来引数で継続する。
        return OpenQuoteContext(host=host, port=port)


def _period_start(period: str, now: pd.Timestamp | None = None) -> str:
    """yfinance形式のperiodをOpenAPI用の開始日に変換する。"""
    end = (now or pd.Timestamp.now()).normalize()
    offsets = {
        "5d": pd.DateOffset(days=8),
        "1mo": pd.DateOffset(months=1),
        "3mo": pd.DateOffset(months=3),
        "6mo": pd.DateOffset(months=6),
        "1y": pd.DateOffset(years=1),
        "2y": pd.DateOffset(years=2),
        "3y": pd.DateOffset(years=3),
        "5y": pd.DateOffset(years=5),
        "10y": pd.DateOffset(years=10),
        "max": pd.DateOffset(years=20),
    }
    if period == "ytd":
        start = pd.Timestamp(year=end.year, month=1, day=1)
    else:
        start = end - offsets.get(period, pd.DateOffset(years=2))
    return start.strftime("%Y-%m-%d")


def _as_float(value, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if pd.notna(number) else default


def _normalise_history(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty or "time_key" not in data:
        return pd.DataFrame()
    index = pd.DatetimeIndex(pd.to_datetime(data["time_key"], errors="coerce"))
    out = pd.DataFrame({
        "Open": pd.to_numeric(data.get("open"), errors="coerce").to_numpy(),
        "High": pd.to_numeric(data.get("high"), errors="coerce").to_numpy(),
        "Low": pd.to_numeric(data.get("low"), errors="coerce").to_numpy(),
        "Close": pd.to_numeric(data.get("close"), errors="coerce").to_numpy(),
        "Volume": pd.to_numeric(data.get("volume"), errors="coerce").fillna(0).to_numpy(),
    }, index=index)
    out.index.name = "Date"
    out = out[~out.index.isna()].dropna(subset=["Open", "High", "Low", "Close"])
    return out[~out.index.duplicated(keep="last")].sort_index()


def fetch_history(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
    """moomooの前方復権済みローソク足を取得する。"""
    if interval not in INTERVAL_MAP:
        raise MoomooError(f"未対応の足種です: {interval}")
    _ensure_available()
    code = normalize_code(ticker)
    ktype = getattr(KLType, INTERVAL_MAP[interval])
    session = (Session.RTH if code.startswith("US.") and interval in
               {"1m", "5m", "15m", "1h"} else Session.NONE)
    start = _period_start(period)
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    ctx = None
    try:
        ctx = _open_context()
        frames = []
        page_key = None
        for _ in range(30):
            kwargs = {
                "code": code,
                "start": start,
                "end": end,
                "ktype": ktype,
                "autype": AuType.QFQ,
                "max_count": 1000,
                "session": session,
            }
            if page_key is not None:
                kwargs["page_req_key"] = page_key
            ret, data, page_key = ctx.request_history_kline(**kwargs)
            if ret != RET_OK:
                raise MoomooError(str(data))
            if data is not None and not data.empty:
                frames.append(data)
            if page_key is None:
                break
        if not frames:
            return pd.DataFrame()
        return _normalise_history(pd.concat(frames, ignore_index=True))
    except MoomooError:
        raise
    except Exception as exc:
        raise MoomooError(str(exc)) from exc
    finally:
        if ctx is not None:
            ctx.close()


def fetch_snapshot(ticker: str) -> dict:
    """最新価格・前日終値・最良気配などのスナップショットを取得する。"""
    code = normalize_code(ticker)
    ctx = None
    try:
        ctx = _open_context()
        ret, data = ctx.get_market_snapshot([code])
        if ret != RET_OK:
            raise MoomooError(str(data))
        if data is None or data.empty:
            return {}
        row = data.iloc[0]
        return {
            "code": code,
            "name": row.get("name") or ticker,
            "price": _as_float(row.get("last_price")),
            "previous_close": _as_float(row.get("prev_close_price")),
            "bid": _as_float(row.get("bid_price")),
            "ask": _as_float(row.get("ask_price")),
            "volume": _as_float(row.get("volume"), 0.0),
            "update_time": str(row.get("update_time") or ""),
            "source": "moomoo OpenAPI",
        }
    except MoomooError:
        raise
    except Exception as exc:
        raise MoomooError(str(exc)) from exc
    finally:
        if ctx is not None:
            ctx.close()


def _book_value(item) -> tuple[float | None, float | None]:
    if isinstance(item, dict):
        return (_as_float(item.get("price")),
                _as_float(item.get("volume"), 0.0))
    try:
        return _as_float(item[0]), _as_float(item[1], 0.0)
    except (IndexError, TypeError):
        return None, None


def fetch_order_book(ticker: str, num: int = 10) -> pd.DataFrame:
    """板情報を最良気配から最大num段取得する。取引は一切行わない。"""
    code = normalize_code(ticker)
    ctx = None
    try:
        ctx = _open_context()
        ret, message = ctx.subscribe([code], [SubType.ORDER_BOOK])
        if ret != RET_OK:
            raise MoomooError(str(message))
        ret, data = ctx.get_order_book(code, num=num)
        if ret != RET_OK:
            raise MoomooError(str(data))
        bids = data.get("Bid", []) if isinstance(data, dict) else []
        asks = data.get("Ask", []) if isinstance(data, dict) else []
        rows = []
        for i in range(max(len(bids), len(asks))):
            bid_price, bid_volume = _book_value(bids[i]) if i < len(bids) else (None, None)
            ask_price, ask_volume = _book_value(asks[i]) if i < len(asks) else (None, None)
            rows.append({
                "気配": i + 1,
                "売数量": ask_volume,
                "売気配値": ask_price,
                "買気配値": bid_price,
                "買数量": bid_volume,
            })
        return pd.DataFrame(rows)
    except MoomooError:
        raise
    except Exception as exc:
        raise MoomooError(str(exc)) from exc
    finally:
        if ctx is not None:
            ctx.close()

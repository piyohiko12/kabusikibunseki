"""moomoo OpenAPIから読み取り専用の市場データを取得する。

OpenDまたはSDKが利用できない場合は例外を返し、呼び出し側がyfinanceへ
フォールバックできるようにする。取引コンテキストや注文APIは扱わない。

接続そのものは lib.moomoo_client に一本化している。OpenQuoteContext は
OpenDが起動していないと例外を返さず無限に再接続を試みるため、
moomoo_client 側のTCP事前確認+デーモンスレッド+タイムアウト+失敗キャッシュを
必ず経由する。このモジュールでcontextを直接生成してはいけない。
"""

from __future__ import annotations

import pandas as pd

from lib import moomoo_client

_SDK_IMPORT_ERROR = None
try:
    from moomoo import AuType, KLType, RET_OK, Session, SubType
except Exception as exc:  # SDKやログ初期化の失敗でもyfinanceで動かす
    AuType = KLType = Session = SubType = None
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


HISTORY_DISABLED_MESSAGE = (
    "moomooの履歴K線取得はオフです(歴史的K線クォータを消費するため既定でオフ)"
)


def _connection() -> tuple[str, int]:
    cfg = moomoo_client._settings()
    return cfg["host"], cfg["port"]


def history_enabled() -> bool:
    """チャート履歴をmoomooから取るかどうか。

    request_history_kline は口座資産に応じた「歴史的K線クォータ」を消費し、
    一度使うと30日間戻らない。既定はオフにして、チャートはYahoo Financeから
    取得する。設定画面で明示的にオンにしたときだけmoomooを使う。
    """
    from lib import settings_store
    s = settings_store.load()
    return bool(s.get("moomoo_enabled", False)) and bool(
        s.get("moomoo_chart_history", False))


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


def connection_status() -> dict:
    """SDKとOpenDの利用可否をpickle可能なdictで返す。

    判定は moomoo_client.status() に委譲する(TCP確認・接続タイムアウト・
    失敗キャッシュを含む)。
    """
    state = moomoo_client.status()
    host, port = _connection()
    return {
        "available": state["state"] == moomoo_client._State.OK,
        "sdk_installed": state["state"] != moomoo_client._State.NOT_INSTALLED,
        "opend_reachable": state["state"] in (moomoo_client._State.OK,
                                              moomoo_client._State.TIMEOUT),
        "host": host,
        "port": port,
        "message": state["message"],
    }


def _open_context():
    """moomoo_clientが保持する共有contextを返す。

    呼び出し側はこのcontextを close してはいけない(使い回すため)。
    """
    ctx = moomoo_client._ctx()
    if ctx is None:
        raise MoomooError(moomoo_client.status()["message"])
    return ctx


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


def fetch_history(ticker: str, period: str, interval: str = "1d",
                  prepost: bool = False) -> pd.DataFrame:
    """moomooの前方復権済みローソク足を取得する。

    prepost=Trueなら米国株の分足に時間外セッションを含める(Session.ALL)。
    """
    if interval not in INTERVAL_MAP:
        raise MoomooError(f"未対応の足種です: {interval}")
    if not history_enabled():
        raise MoomooError(HISTORY_DISABLED_MESSAGE)
    code = normalize_code(ticker)
    ktype = getattr(KLType, INTERVAL_MAP[interval])
    if code.startswith("US.") and interval in {"1m", "5m", "15m", "1h"}:
        # Session.ALL は古いSDKに無いことがあるのでgetattrで退避する。
        session = (getattr(Session, "ALL", Session.RTH) if prepost
                   else Session.RTH)
    else:
        session = Session.NONE
    start = _period_start(period)
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
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


def fetch_snapshot(ticker: str) -> dict:
    """最新価格・前日終値・最良気配などのスナップショットを取得する。"""
    code = normalize_code(ticker)
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
    try:
        ctx = _open_context()
        # 購読数には口座ごとの上限があるので、moomoo_client側の購読管理を通す。
        if not moomoo_client._ensure_subscribed(ctx, code, [SubType.ORDER_BOOK]):
            raise MoomooError(f"板情報の購読に失敗しました: {code}")
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

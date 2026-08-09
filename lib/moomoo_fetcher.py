"""moomoo OpenAPIから読み取り専用の市場データを取得する。

OpenDまたはSDKが利用できない場合は例外を返し、呼び出し側がyfinanceへ
フォールバックできるようにする。取引コンテキストや注文APIは扱わない。
"""

from __future__ import annotations

import json
import os
import re
import socket
from contextlib import closing
from pathlib import Path

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

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_MAX_AGES = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=2),
    "15m": pd.Timedelta(minutes=5),
    "1h": pd.Timedelta(minutes=15),
    "1d": pd.Timedelta(minutes=5),
    "1wk": pd.Timedelta(hours=6),
    "1mo": pd.Timedelta(hours=24),
}
QUOTA_LOOKBACK = pd.Timedelta(days=30)
QUOTA_RESERVE_ENV = "MOOMOO_HISTORY_QUOTA_RESERVE"
DEFAULT_QUOTA_RESERVE = 10


def _integration_settings() -> dict:
    """サイドバーと同じ保存設定を返す。環境変数は未設定項目の補助に使う。"""
    try:
        from lib import settings_store
        saved = settings_store.load()
    except Exception:
        saved = {}
    host = saved.get("moomoo_host") or os.environ.get(
        "FUTU_OPEND_HOST", "127.0.0.1")
    raw_port = saved.get("moomoo_port") or os.environ.get("FUTU_OPEND_PORT", "11111")
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        port = 11111
    return {
        "enabled": bool(saved.get("moomoo_enabled", False)),
        "host": host,
        "port": port,
        "history_reserve": saved.get("moomoo_history_reserve"),
    }


def _connection() -> tuple[str, int]:
    settings = _integration_settings()
    return settings["host"], settings["port"]


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
    settings = _integration_settings()
    sdk_installed = OpenQuoteContext is not None
    host, port = settings["host"], settings["port"]
    reachable = False
    if settings["enabled"] and sdk_installed:
        try:
            with closing(socket.create_connection((host, port), timeout=timeout)):
                reachable = True
        except OSError:
            pass

    if not settings["enabled"]:
        message = "moomoo連携はオフ"
    elif not sdk_installed:
        message = ("moomoo-api SDKが未導入です" if not _SDK_IMPORT_ERROR else
                   f"moomoo-api SDKを読み込めません: {_SDK_IMPORT_ERROR}")
    elif not reachable:
        message = f"OpenDに接続できません({host}:{port})"
    else:
        message = f"OpenD接続中({host}:{port})"
    return {
        "available": settings["enabled"] and sdk_installed and reachable,
        "enabled": settings["enabled"],
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


def _as_int(value, default: int | None = None) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _quota_reserve() -> int:
    """新規銘柄の取得で使わず残す過去K線枠。無効値は既定値へ戻す。"""
    configured = _integration_settings().get("history_reserve")
    raw = configured if configured is not None else os.environ.get(
        QUOTA_RESERVE_ENV,
        os.environ.get("FUTU_HISTORY_QUOTA_RESERVE", str(DEFAULT_QUOTA_RESERVE)))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_QUOTA_RESERVE


def _safe_cache_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "unknown"


def _history_cache_paths(code: str, period: str, interval: str) -> tuple[Path, Path]:
    stem = "history_{}_{}_{}".format(
        _safe_cache_part(code), _safe_cache_part(period), _safe_cache_part(interval))
    return CACHE_DIR / f"{stem}.csv", CACHE_DIR / f"{stem}.json"


def _parse_utc(value) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _read_history_cache(code: str, period: str, interval: str,
                        now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict] | None:
    """永続キャッシュを読み、fresh/staleを付けて返す。壊れたキャッシュは無視。"""
    csv_path, meta_path = _history_cache_paths(code, period, interval)
    if not csv_path.exists():
        return None
    try:
        frame = pd.read_csv(csv_path, index_col="Date", parse_dates=["Date"])
        required = {"Open", "High", "Low", "Close", "Volume"}
        if frame.empty or not required.issubset(frame.columns):
            return None
        frame = frame[["Open", "High", "Low", "Close", "Volume"]].rename_axis("Date")
        metadata = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        fetched_at = _parse_utc(metadata.get("fetched_at"))
        if fetched_at is None:
            fetched_at = pd.Timestamp(csv_path.stat().st_mtime, unit="s", tz="UTC")
        current = now or _now_utc()
        if current.tzinfo is None:
            current = current.tz_localize("UTC")
        else:
            current = current.tz_convert("UTC")
        max_age = CACHE_MAX_AGES.get(interval, pd.Timedelta(minutes=5))
        cache_status = "fresh" if current - fetched_at <= max_age else "stale"
        meta = {
            "source": "moomoo OpenAPI",
            "code": code,
            "fetched_at": fetched_at.isoformat(),
            "cache_status": cache_status,
            "quota": _as_int(metadata.get("quota")),
            "remain": _as_int(metadata.get("remain")),
            "fallback_reason": None,
        }
        frame.attrs["moomoo_meta"] = meta
        return frame, meta
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _write_history_cache(frame: pd.DataFrame, code: str, period: str,
                         interval: str, metadata: dict) -> None:
    """正常取得した過去K線をCSVとJSONへ保存する。保存失敗は取得結果を壊さない。"""
    csv_path, meta_path = _history_cache_paths(code, period, interval)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index_label="Date")
        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def _quota_parts(data) -> tuple[int | None, int | None, list[dict]]:
    """SDKのtuple/dict/DataFrame各形式から利用枠と詳細を正規化する。"""
    used = remain = None
    details = []
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return None, None, []
        row = data.iloc[0]
        used = _as_int(row.get("used_quota"))
        remain = _as_int(row.get("remain_quota"))
        raw_details = row.get("detail_list", [])
    elif isinstance(data, dict):
        used = _as_int(data.get("used_quota", data.get("usedQuota")))
        remain = _as_int(data.get("remain_quota", data.get("remainQuota")))
        raw_details = data.get("detail_list", data.get("detailList", []))
    elif isinstance(data, (tuple, list)) and len(data) >= 2:
        used = _as_int(data[0])
        remain = _as_int(data[1])
        raw_details = list(data[2:])
    else:
        return None, None, []

    if isinstance(raw_details, dict):
        raw_details = [raw_details]
    if isinstance(raw_details, (tuple, list)):
        for item in raw_details:
            if isinstance(item, (tuple, list)):
                details.extend(v for v in item if isinstance(v, dict))
            elif isinstance(item, dict):
                details.append(item)
    return used, remain, details


def _detail_code(item: dict) -> str:
    code = item.get("code")
    if not code and isinstance(item.get("security"), dict):
        security = item["security"]
        market = str(security.get("market") or "").upper()
        raw_code = str(security.get("code") or "").upper()
        code = f"{market}.{raw_code}" if market and not market.isdigit() else raw_code
    return str(code or "").upper()


def _history_quota_decision(ctx, code: str, allow_new_quota: bool,
                            now: pd.Timestamp | None = None) -> dict:
    """履歴APIを呼んでよいか判定する。確認に失敗した場合は必ず拒否する。"""
    current = now or _now_utc()
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    try:
        ret, data = ctx.get_history_kl_quota(get_detail=True)
    except Exception as exc:
        return {"allowed": False, "quota": None, "remain": None,
                "reason": f"過去K線の利用枠確認に失敗しました: {exc}"}
    if ret != RET_OK:
        return {"allowed": False, "quota": None, "remain": None,
                "reason": f"過去K線の利用枠確認に失敗しました: {data}"}

    used, remain, details = _quota_parts(data)
    if used is None or remain is None:
        return {"allowed": False, "quota": None, "remain": None,
                "reason": "過去K線の利用枠応答を解釈できませんでした"}
    total = used + remain
    cutoff = current - QUOTA_LOOKBACK
    already_requested = False
    for item in details:
        if _detail_code(item) != code.upper():
            continue
        requested = _parse_utc(item.get("request_time", item.get("requestTime")))
        if requested is not None and requested >= cutoff:
            already_requested = True
            break

    if already_requested:
        return {"allowed": True, "quota": total, "remain": remain,
                "reason": None, "already_requested": True}
    if remain <= 0:
        return {"allowed": False, "quota": total, "remain": remain,
                "reason": "過去K線の残り利用枠がありません"}

    reserve = _quota_reserve()
    if not allow_new_quota:
        return {
            "allowed": False,
            "quota": total,
            "remain": remain,
            "reason": ("過去30日以内に未取得の銘柄です。新規の過去K線枠は"
                       "明示許可がある場合だけ使用します"),
        }
    if remain <= reserve:
        return {
            "allowed": False,
            "quota": total,
            "remain": remain,
            "reason": (f"新規銘柄用の過去K線枠を保護しました"
                       f"(残り{remain}、予約{reserve})"),
        }
    return {"allowed": True, "quota": total, "remain": remain,
            "reason": None, "already_requested": False}


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
                  allow_new_quota: bool = False) -> pd.DataFrame:
    """moomooの前方復権済みローソク足を、利用枠を保護して取得する。

    足種に応じた有効期間内の永続キャッシュはOpenDへ接続せず返す。過去30日以内に
    未取得の銘柄は ``allow_new_quota=True`` の明示指定がなければ取得しない。
    明示指定時も ``MOOMOO_HISTORY_QUOTA_RESERVE``（既定10）以下の予約枠は消費しない。
    """
    if interval not in INTERVAL_MAP:
        raise MoomooError(f"未対応の足種です: {interval}")
    code = normalize_code(ticker)
    if not _integration_settings()["enabled"]:
        # 設定OFFは明示的な利用停止。OpenDだけでなくローカルキャッシュも使わない。
        raise MoomooError("moomoo連携はオフ")
    cached = _read_history_cache(code, period, interval)
    if cached is not None and cached[1]["cache_status"] == "fresh":
        return cached[0]
    stale = cached[0] if cached is not None else None
    ctx = None
    quota = {"allowed": False, "quota": None, "remain": None,
             "reason": None}
    try:
        ctx = _open_context()
        ktype = getattr(KLType, INTERVAL_MAP[interval])
        session = (Session.RTH if code.startswith("US.") and interval in
                   {"1m", "5m", "15m", "1h"} else Session.NONE)
        start = _period_start(period)
        end = pd.Timestamp.now().strftime("%Y-%m-%d")
        quota = _history_quota_decision(ctx, code, allow_new_quota)
        if not quota["allowed"]:
            if stale is not None:
                meta = dict(stale.attrs.get("moomoo_meta", {}))
                meta.update({
                    "source": "moomoo OpenAPI",
                    "code": code,
                    "cache_status": "stale",
                    "quota": quota.get("quota"),
                    "remain": quota.get("remain"),
                    "fallback_reason": quota.get("reason"),
                })
                stale.attrs["moomoo_meta"] = meta
                return stale
            raise MoomooError(quota["reason"])

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
            if stale is not None:
                meta = dict(stale.attrs.get("moomoo_meta", {}))
                meta.update({
                    "cache_status": "stale", "quota": quota.get("quota"),
                    "remain": quota.get("remain"),
                    "fallback_reason": "moomooから過去K線を取得できませんでした",
                })
                stale.attrs["moomoo_meta"] = meta
                return stale
            empty = pd.DataFrame()
            empty.attrs["moomoo_meta"] = {
                "source": "moomoo OpenAPI", "code": code,
                "fetched_at": None, "cache_status": "miss",
                "quota": quota.get("quota"), "remain": quota.get("remain"),
                "fallback_reason": "moomooから過去K線を取得できませんでした",
            }
            return empty

        result = _normalise_history(pd.concat(frames, ignore_index=True))
        fetched_at = _now_utc().isoformat()
        displayed_remain = quota.get("remain")
        if (displayed_remain is not None
                and quota.get("already_requested") is False):
            displayed_remain = max(int(displayed_remain) - 1, 0)
        meta = {
            "source": "moomoo OpenAPI",
            "code": code,
            "fetched_at": fetched_at,
            "cache_status": "refreshed",
            "quota": quota.get("quota"),
            "remain": displayed_remain,
            "fallback_reason": None,
        }
        result.attrs["moomoo_meta"] = meta
        if not result.empty:
            _write_history_cache(result, code, period, interval, meta)
        return result
    except MoomooError as exc:
        if stale is not None:
            meta = dict(stale.attrs.get("moomoo_meta", {}))
            meta.update({
                "cache_status": "stale", "quota": quota.get("quota"),
                "remain": quota.get("remain"),
                "fallback_reason": (quota.get("reason") or str(exc)),
            })
            stale.attrs["moomoo_meta"] = meta
            return stale
        raise
    except Exception as exc:
        if stale is not None:
            meta = dict(stale.attrs.get("moomoo_meta", {}))
            meta.update({
                "cache_status": "stale", "quota": quota.get("quota"),
                "remain": quota.get("remain"), "fallback_reason": str(exc),
            })
            stale.attrs["moomoo_meta"] = meta
            return stale
        raise MoomooError(str(exc)) from exc
    finally:
        if ctx is not None:
            ctx.close()


def _enum_text(value) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    text = str(value or "")
    return text.rsplit(".", 1)[-1] if "." in text else text


def fetch_market_state(ticker: str) -> dict:
    """銘柄が属する市場の取引状態を読み取り専用で取得する。"""
    code = normalize_code(ticker)
    ctx = None
    try:
        ctx = _open_context()
        ret, data = ctx.get_market_state([code])
        if ret != RET_OK:
            raise MoomooError(str(data))
        if data is None or data.empty:
            return {}
        row = data.iloc[0]
        return {
            "code": str(row.get("code") or code),
            "stock_name": str(row.get("stock_name") or ""),
            "market_state": _enum_text(row.get("market_state")),
            "source": "moomoo OpenAPI",
            "fallback_reason": None,
        }
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


def fetch_recent_ticks(ticker: str, num: int = 60) -> pd.DataFrame:
    """直近の歩み値を取得する。取引方向と出来高を再評価に利用する。"""
    code = normalize_code(ticker)
    ctx = None
    try:
        ctx = _open_context()
        ret, message = ctx.subscribe([code], [SubType.TICKER])
        if ret != RET_OK:
            raise MoomooError(str(message))
        ret, data = ctx.get_rt_ticker(code, num=num)
        if ret != RET_OK:
            raise MoomooError(str(data))
        if data is None or data.empty:
            return pd.DataFrame()
        columns = [column for column in
                   ("time", "price", "volume", "turnover", "ticker_direction", "type")
                   if column in data.columns]
        return data[columns].copy().reset_index(drop=True)
    except MoomooError:
        raise
    except Exception as exc:
        raise MoomooError(str(exc)) from exc
    finally:
        if ctx is not None:
            ctx.close()


def fetch_capital_distribution(ticker: str) -> dict | None:
    """大口・中口・小口の当日純流入を取得する。"""
    code = normalize_code(ticker)
    ctx = None
    try:
        ctx = _open_context()
        ret, data = ctx.get_capital_distribution(code)
        if ret != RET_OK:
            raise MoomooError(str(data))
        if data is None or data.empty:
            return None
        row = data.iloc[0]

        def number(key: str) -> float:
            return _as_float(row.get(key), 0.0) or 0.0

        tiers = [
            ("大口", number("capital_in_big") - number("capital_out_big"),
             number("capital_in_big"), number("capital_out_big")),
            ("中口", number("capital_in_mid") - number("capital_out_mid"),
             number("capital_in_mid"), number("capital_out_mid")),
            ("小口", number("capital_in_small") - number("capital_out_small"),
             number("capital_in_small"), number("capital_out_small")),
        ]
        if not any(item[2] or item[3] for item in tiers):
            return None
        return {
            "code": code,
            "tiers": tiers,
            "net": sum(item[1] for item in tiers),
            "update_time": str(row.get("update_time") or ""),
        }
    except MoomooError:
        raise
    except Exception as exc:
        raise MoomooError(str(exc)) from exc
    finally:
        if ctx is not None:
            ctx.close()

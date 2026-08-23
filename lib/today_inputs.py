"""今日のセッション診断へ渡す入力を、取得データから安全に整形する。"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping

import pandas as pd


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _positive(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def session_prices(snapshot: Mapping[str, Any] | None,
                   daily_bar: Mapping[str, Any] | pd.Series | None = None) -> dict:
    """moomoo snapshotを4セッション入力へ変換する。

    セッション値がない箇所は欠損のままにし、日足終値でプレ/アフター/夜間価格を
    補完しない。日足フォールバックはregular欄だけへ入れる。
    """
    snap = dict(snapshot or {})
    stamp = snap.get("update_time")
    live = snap.get("source") == "moomoo OpenAPI"
    result: dict[str, dict] = {}

    if live:
        session_quotes = (snap.get("session_quotes")
                          if isinstance(snap.get("session_quotes"), Mapping)
                          else {})
        fields = {
            "premarket": ("pre_price", "pre_volume"),
            "regular": ("price", "volume"),
            "afterhours": ("after_price", "after_volume"),
            "overnight": ("overnight_price", "overnight_volume"),
        }
        for session, (price_key, volume_key) in fields.items():
            quote = (session_quotes.get(session)
                     if isinstance(session_quotes.get(session), Mapping) else {})
            price = _positive(quote.get("price")) or _positive(snap.get(price_key))
            if price is None:
                continue
            # moomoo SDKの汎用update_timeは、時間外価格ごとの更新時刻ではない。
            # session_quotesがある場合はtimestamp_verified=Trueの時刻だけ公開し、
            # 未検証の時間外値を「最新」と見せない。
            if quote:
                timestamp_verified = bool(quote.get("timestamp_verified"))
                timestamp = quote.get("updated_at") if timestamp_verified else None
                quality = 1.0 if timestamp_verified else 0.65
            else:
                timestamp_verified = session == "regular" and stamp is not None
                timestamp = stamp if timestamp_verified else None
                quality = 1.0 if timestamp_verified else 0.65
            result[session] = {
                "price": price,
                "open": (_positive(snap.get("open"))
                         if session == "regular" else None),
                "volume": (_number(quote.get("volume"))
                           if quote.get("volume") is not None
                           else _number(snap.get(volume_key))),
                "timestamp": timestamp,
                "source": (str(quote.get("source")) if quote.get("source")
                           else "moomoo OpenAPI snapshot"),
                "quality": quality,
                "timestamp_verified": timestamp_verified,
            }
    elif daily_bar is not None:
        bar = dict(daily_bar)
        close = _positive(bar.get("Close"))
        if close is not None:
            result["regular"] = {
                "price": close, "open": _positive(bar.get("Open")),
                "volume": _number(bar.get("Volume")),
                "timestamp": getattr(daily_bar, "name", None),
                "source": "Yahoo Finance 直近確定日足", "quality": 0.45,
                "timestamp_verified": True,
            }
    return result


def overnight_eligibility(snapshot: Mapping[str, Any] | None) -> bool | None:
    """実際の夜間値があるときだけ対象と確認し、欠損から対象外を推測しない。"""
    return True if _positive(dict(snapshot or {}).get("overnight_price")) else None


def return_pct(history: pd.DataFrame | None, sessions: int) -> float | None:
    if history is None or "Close" not in history or len(history) <= sessions:
        return None
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if len(close) <= sessions:
        return None
    start, end = _positive(close.iloc[-sessions - 1]), _positive(close.iloc[-1])
    return None if start is None or end is None else (end / start - 1) * 100


def imminent_event_risk(report: Mapping[str, Any] | None, *,
                        as_of: date | None = None, window_days: int = 2) -> bool:
    current = as_of or date.today()
    for event in dict(report or {}).get("events", []) or []:
        if event.get("status") != "UPCOMING":
            continue
        event_date = pd.to_datetime(event.get("event_date"), errors="coerce")
        if pd.isna(event_date):
            continue
        distance = (event_date.date() - current).days
        level = str(event.get("impact_level") or "").strip().upper()
        if level in {"HIGH", "CRITICAL"}:
            high_impact = True
        elif level in {"LOW", "MEDIUM"}:
            high_impact = False
        else:
            # 旧データにimpact_levelがない場合だけ0〜100点を使う。
            # 現行生成値はMEDIUMが最大58、HIGHが最小68なので60を境界とする。
            score = _number(event.get("impact_score"))
            high_impact = score is not None and score >= 60
        if 0 <= distance <= max(0, int(window_days)) and high_impact:
            return True
    return False


def opening_features(stock_history: pd.DataFrame | None,
                     snapshot: Mapping[str, Any] | None,
                     market_features: Mapping[str, Any] | None = None,
                     event_report: Mapping[str, Any] | None = None) -> dict:
    """寄付き診断用の透明な特徴量辞書を作る。

    ``market_features`` は futures_pct / spy_pct / qqq_pct と任意のtimestamp/sourceを
    受ける。イベントは方向を加点せず、直近高影響イベントのリスクだけを渡す。
    """
    features: dict[str, Any] = {}
    momentum = return_pct(stock_history, 5)
    if momentum is not None:
        stamp = stock_history.index[-1] if stock_history is not None and not stock_history.empty else None
        features["momentum_pct"] = {
            "value": momentum, "timestamp": stamp,
            "source": "Yahoo Finance 日足", "quality": 0.75,
        }
    snap = dict(snapshot or {})
    volume_ratio = _number(snap.get("volume_ratio"))
    if volume_ratio is not None and volume_ratio > 0:
        features["relative_volume"] = {
            "value": volume_ratio, "timestamp": snap.get("update_time"),
            "source": "moomoo OpenAPI snapshot", "quality": 1.0,
        }
    for key in ("futures_pct", "spy_pct", "qqq_pct"):
        value = dict(market_features or {}).get(key)
        if isinstance(value, Mapping):
            if _number(value.get("value")) is not None:
                features[key] = dict(value)
        elif _number(value) is not None:
            features[key] = {"value": _number(value), "source": "Yahoo Finance"}
    features["event_risk"] = {
        "value": imminent_event_risk(event_report),
        "source": "イベント影響分析", "quality": 1.0,
    }
    return features


__all__ = [
    "imminent_event_risk", "opening_features", "overnight_eligibility",
    "return_pct", "session_prices",
]

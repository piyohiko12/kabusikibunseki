"""売買判定に渡す市場データの安全性を検査する補助関数。

APIアクセスはこのモジュールでは行わない。取得済みの日足、スナップショット、
市場状態、決算予定日を受け取り、確定足への切り替えと必須ゲートを構築する。
"""

from __future__ import annotations

from datetime import date
from datetime import time as clock_time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


MARKET_TIMEZONES = {
    "US": "America/New_York",
    "HK": "Asia/Hong_Kong",
    "SH": "Asia/Shanghai",
    "SZ": "Asia/Shanghai",
    "SG": "Asia/Singapore",
    "MY": "Asia/Kuala_Lumpur",
    "JP": "Asia/Tokyo",
    "CC": "UTC",
}

MARKET_CLOSE_TIMES = {
    "US": clock_time(16, 15),
    "HK": clock_time(16, 15),
    "SH": clock_time(15, 15),
    "SZ": clock_time(15, 15),
    "SG": clock_time(17, 15),
    "MY": clock_time(17, 15),
    "JP": clock_time(15, 45),
}

# moomoo MarketStateの文字列表現。通常取引中だけでなく、時間外で当日足が
# 更新され得る状態も「未確定」として扱う。
LIVE_MARKET_STATES = {
    "MORNING", "AFTERNOON", "FUTURE_DAY_OPEN", "FUTURE_OPEN",
    "PRE_MARKET_BEGIN", "PRE_MARKET", "AFTER_HOURS_BEGIN", "AFTER_HOURS",
    "OVERNIGHT", "NIGHT_OPEN", "NIGHT_OPENING",
}


def market_prefix(code: str) -> str:
    """moomoo/Yahoo形式のコードから市場接頭辞を推定する。"""
    raw = str(code or "").strip().upper()
    if "." in raw and raw.split(".", 1)[0] in MARKET_TIMEZONES:
        return raw.split(".", 1)[0]
    if raw.endswith(".HK"):
        return "HK"
    if raw.endswith(".T"):
        return "JP"
    return "US"


def market_today(code: str, now: pd.Timestamp | None = None) -> date:
    """銘柄市場の現地日付を返す。テストではnowを固定できる。"""
    tz = ZoneInfo(MARKET_TIMEZONES[market_prefix(code)])
    current = now if now is not None else pd.Timestamp.now(tz=tz)
    if current.tzinfo is None:
        current = current.tz_localize(tz)
    else:
        current = current.tz_convert(tz)
    return current.date()


def _market_now(code: str, now: pd.Timestamp | None = None) -> pd.Timestamp:
    tz = ZoneInfo(MARKET_TIMEZONES[market_prefix(code)])
    current = now if now is not None else pd.Timestamp.now(tz=tz)
    if current.tzinfo is None:
        return current.tz_localize(tz)
    return current.tz_convert(tz)


def normalise_market_state(value: object) -> str:
    """Enum文字列を比較しやすい大文字の末尾名へ変換する。"""
    text = str(value or "").strip().upper()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def completed_daily_bars(
    df: pd.DataFrame,
    code: str,
    market_state: object = None,
    now: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """形成途中の当日足を除き、判定用の確定日足と説明メタデータを返す。"""
    if df is None or df.empty:
        return pd.DataFrame(), {
            "bar_complete": False,
            "dropped_incomplete": False,
            "last_bar": None,
            "reason": "日足データがありません",
        }

    out = df.copy().sort_index()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    valid_dates = pd.Series(idx.date, index=out.index)
    last_date = valid_dates.iloc[-1]
    local_now = _market_now(code, now)
    today = local_now.date()
    state = normalise_market_state(market_state)
    is_live = state in LIVE_MARKET_STATES

    # 市場状態を取得できない場合は、現地の通常取引終了+15分を目安にする。
    # 暗号資産は24時間取引のため、その日の足は常に形成途中として扱う。
    prefix = market_prefix(code)
    close_time = MARKET_CLOSE_TIMES.get(prefix)
    unknown_live = (not state and (prefix == "CC" or close_time is None
                                    or local_now.time() < close_time))
    should_drop = last_date >= today and (is_live or unknown_live)
    if should_drop:
        out = out.iloc[:-1]

    last_complete = None if out.empty else str(pd.Timestamp(out.index[-1]).date())
    return out, {
        "bar_complete": not out.empty,
        "dropped_incomplete": should_drop,
        "last_bar": last_complete,
        "market_state": state or "UNKNOWN",
        "reason": ("形成途中の当日足を除外しました" if should_drop
                   else "最新の確定日足を使用しています"),
    }


def _business_day_distance(start: date, end: date) -> int:
    """startからendまでのおおよその営業日数（土日除外、符号付き）。"""
    if start == end:
        return 0
    if end > start:
        return int(np.busday_count(start, end))
    return -int(np.busday_count(end, start))


def parse_earnings_date(value: object) -> date | None:
    """yfinance/moomoo由来の日付表現をdateへ正規化する。"""
    if value in (None, "", "N/A"):
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.date()


def build_external_gates(
    df: pd.DataFrame,
    *,
    code: str,
    bar_meta: dict | None = None,
    snapshot: dict | None = None,
    earnings_date: object = None,
    as_of: date | None = None,
    min_history: int = 220,
    max_stale_business_days: int = 5,
    min_median_dollar_volume: float = 5_000_000,
    max_spread_pct: float = 0.50,
    earnings_blackout_days: int = 2,
) -> list[dict]:
    """判定前に確認する安全ゲートを、画面表示可能な形で返す。

    ``passed=None`` は取得不能のため注意表示に留める。``required=True`` かつ
    ``passed=False`` のゲートだけが売買判定をWAITへ変える。
    """
    bar_meta = bar_meta or {}
    snapshot = snapshot or {}
    today = as_of or market_today(code)
    gates: list[dict] = []

    enough = df is not None and len(df) >= min_history
    gates.append({
        "key": "history",
        "label": "指標計算に必要な履歴",
        "passed": enough,
        "required": True,
        "reason": f"確定日足 {0 if df is None else len(df)}本 / 必要 {min_history}本",
    })

    complete = bool(bar_meta.get("bar_complete")) and df is not None and not df.empty
    gates.append({
        "key": "completed_bar",
        "label": "確定日足",
        "passed": complete,
        "required": True,
        "reason": bar_meta.get("reason") or "足の確定状態を確認できません",
    })

    if df is None or df.empty:
        fresh = False
        stale_days = None
    else:
        last_day = pd.Timestamp(df.index[-1]).date()
        stale_days = max(_business_day_distance(last_day, today), 0)
        fresh = stale_days <= max_stale_business_days
    gates.append({
        "key": "freshness",
        "label": "データ鮮度",
        "passed": fresh,
        "required": True,
        "reason": ("最新確定足を確認できません" if stale_days is None else
                   f"最終確定足から約{stale_days}営業日"),
    })

    liquidity = None
    if df is not None and not df.empty and {"Close", "Volume"}.issubset(df.columns):
        dollar_volume = (pd.to_numeric(df["Close"], errors="coerce")
                         * pd.to_numeric(df["Volume"], errors="coerce"))
        liquidity = float(dollar_volume.tail(20).median())
    liquid_ok = liquidity is not None and np.isfinite(liquidity) and (
        liquidity >= min_median_dollar_volume)
    gates.append({
        "key": "liquidity",
        "label": "流動性",
        "passed": bool(liquid_ok),
        "required": True,
        "reason": ("売買代金を計算できません" if liquidity is None else
                   f"20日中央値 {liquidity:,.0f} / 必要 {min_median_dollar_volume:,.0f}"),
    })

    bid, ask = snapshot.get("bid"), snapshot.get("ask")
    spread = None
    try:
        bid, ask = float(bid), float(ask)
        if bid > 0 and ask >= bid:
            spread = (ask - bid) / ((ask + bid) / 2) * 100
    except (TypeError, ValueError):
        spread = None
    gates.append({
        "key": "spread",
        "label": "スプレッド",
        "passed": None if spread is None else spread <= max_spread_pct,
        "required": spread is not None,
        "reason": ("気配値を取得できないため警告のみ" if spread is None else
                   f"{spread:.3f}% / 上限 {max_spread_pct:.2f}%"),
    })

    earnings = parse_earnings_date(earnings_date)
    distance = None if earnings is None else _business_day_distance(today, earnings)
    in_blackout = distance is not None and abs(distance) <= earnings_blackout_days
    gates.append({
        "key": "earnings",
        "label": "決算イベント",
        "passed": None if earnings is None else not in_blackout,
        "required": earnings is not None,
        "reason": ("次回決算日を取得できないため警告のみ" if earnings is None else
                   f"決算日 {earnings.isoformat()}（約{distance:+d}営業日）"),
    })

    return gates


def gate_summary(gates: list[dict]) -> dict:
    """必須ゲートの成否と、WAIT理由をまとめる。"""
    blocked = [g for g in gates if g.get("required") and g.get("passed") is False]
    unknown = [g for g in gates if g.get("passed") is None]
    return {
        "passed": not blocked,
        "blocked": blocked,
        "unknown": unknown,
        "reason": " / ".join(str(g.get("reason") or g.get("label")) for g in blocked),
    }

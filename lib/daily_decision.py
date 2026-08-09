"""「今日の判断」画面で使う小さな説明可能分析。

売買指示を生成するモジュールではない。現在値が当日始値・平均価格・高安レンジの
どこにあるかを整理し、既存の支持抵抗レベルから最寄りの価格帯を選ぶ。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _comparison(value: float | None, reference: float | None,
                tolerance_pct: float = 0.05) -> int | None:
    if value is None or reference in (None, 0):
        return None
    diff_pct = (value / reference - 1) * 100
    if diff_pct > tolerance_pct:
        return 1
    if diff_pct < -tolerance_pct:
        return -1
    return 0


def intraday_trend(snapshot: Mapping[str, Any] | None,
                   daily_bar: Mapping[str, Any] | pd.Series | None = None) -> dict:
    """現在セッションの方向を、独立した4つの観測点から要約する。

    moomoo snapshotがない場合は直近日足へフォールバックするが、品質を
    ``close_only`` として明示し「リアルタイム」とは表示しない。
    """
    snap = dict(snapshot or {})
    live = snap.get("source") == "moomoo OpenAPI" and _number(snap.get("price")) is not None
    # pandas.Series は真偽値評価できないため、``daily_bar or {}`` を使わない。
    fallback = dict(daily_bar) if daily_bar is not None else {}
    source = snap if live else fallback

    price = _number(source.get("price") if live else source.get("Close"))
    open_price = _number(source.get("open") if live else source.get("Open"))
    high = _number(source.get("high") if live else source.get("High"))
    low = _number(source.get("low") if live else source.get("Low"))
    average = _number(source.get("average_price")) if live else None
    change_pct = _number(snap.get("change_percent")) if live else None

    checks: list[dict] = []
    votes: list[int] = []

    def add(label: str, vote: int | None, actual: str) -> None:
        status = "unknown" if vote is None else ("up" if vote > 0 else "down" if vote < 0 else "flat")
        checks.append({"label": label, "status": status, "actual": actual})
        if vote is not None:
            votes.append(vote)

    vote = _comparison(price, open_price)
    add("現在値 vs 当日始値", vote,
        "—" if price is None or open_price is None else f"{(price / open_price - 1) * 100:+.2f}%")

    vote = _comparison(price, average)
    add("現在値 vs 当日平均価格", vote,
        "—" if price is None or average is None else f"{(price / average - 1) * 100:+.2f}%")

    change_vote = None if change_pct is None else (1 if change_pct > 0.05 else -1 if change_pct < -0.05 else 0)
    add("前日終値比", change_vote,
        "—" if change_pct is None else f"{change_pct:+.2f}%")

    range_position = None
    if price is not None and high is not None and low is not None and high > low:
        range_position = (price - low) / (high - low)
    range_vote = (None if range_position is None else
                  1 if range_position >= 0.60 else -1 if range_position <= 0.40 else 0)
    add("当日レンジ内の位置", range_vote,
        "—" if range_position is None else f"下値から{range_position * 100:.0f}%")

    if not votes:
        ratio = None
        direction = "unknown"
    else:
        ratio = sum(votes) / len(votes)
        direction = "up" if ratio >= 0.35 else "down" if ratio <= -0.35 else "sideways"

    labels = {"up": "上昇", "down": "下降", "sideways": "もみ合い", "unknown": "判定不能"}
    return {
        "direction": direction,
        "label": labels[direction],
        "strength": None if ratio is None else round(abs(ratio) * 100),
        "score": ratio,
        "price": price,
        "open": open_price,
        "average_price": average,
        "range_position": range_position,
        "checks": checks,
        "data_quality": "realtime" if live else "close_only" if price is not None else "unavailable",
        "source": snap.get("source") if live else "直近確定日足" if price is not None else "Unavailable",
    }


def nearest_levels(level_rows: Iterable[Mapping[str, Any]] | None,
                   price: float | None, min_strength: int = 1) -> dict:
    """最寄り支持帯・抵抗帯と、それらから見た単純な上下余地を返す。"""
    current = _number(price)
    if current is None or current <= 0:
        return {"support": None, "resistance": None, "in_zones": [],
                "reward_risk": None, "data_quality": "unavailable"}

    supports, resistances, in_zones = [], [], []
    for original in level_rows or []:
        row = dict(original)
        strength = int(_number(row.get("strength")) or 0)
        if strength < min_strength:
            continue
        low = _number(row.get("zone_low"))
        high = _number(row.get("zone_high"))
        center = _number(row.get("price"))
        if low is None or high is None or center is None:
            continue
        if low <= current <= high:
            in_zones.append(row)
        kind = str(row.get("type") or "")
        if kind == "サポート" or "サポート" in kind:
            row["edge_price"] = high
            row["distance_pct"] = (high / current - 1) * 100
            supports.append(row)
        elif kind == "抵抗線" or "抵抗" in kind:
            row["edge_price"] = low
            row["distance_pct"] = (low / current - 1) * 100
            resistances.append(row)

    support = min(supports, key=lambda item: abs(item["distance_pct"])) if supports else None
    resistance = min(resistances, key=lambda item: abs(item["distance_pct"])) if resistances else None
    reward_risk = None
    if support and resistance:
        downside = max(0.0, -float(support["distance_pct"]))
        upside = max(0.0, float(resistance["distance_pct"]))
        if downside > 0:
            reward_risk = upside / downside
    return {
        "support": support,
        "resistance": resistance,
        "in_zones": in_zones,
        "reward_risk": reward_risk,
        "data_quality": "available" if support or resistance else "unavailable",
    }

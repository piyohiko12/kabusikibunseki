"""moomooのリアルタイム需給でサポート/レジスタンスを再評価する。

価格水準そのものはOHLCVから再現可能な ``levels.find_levels`` を基準とし、
変化しやすい板・歩み値・当日資金フローは「ライブ確度」の補助材料に限定する。
生成AIやmoomooアプリ内AIの非公開APIは呼び出さず、判断根拠を追跡できる
決定論的な採点を行う。
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


def _number(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return 0.0
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    value = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return _number(value)


def _book_rows(book: pd.DataFrame | None, price_col: str,
               volume_col: str) -> list[tuple[float, float]]:
    if book is None or book.empty or price_col not in book or volume_col not in book:
        return []
    rows = []
    for price, volume in zip(book[price_col], book[volume_col]):
        p, v = _number(price), _number(volume)
        if p > 0 and v > 0:
            rows.append((p, v))
    return rows


def _wall(rows: list[tuple[float, float]]) -> dict | None:
    if not rows:
        return None
    volumes = np.asarray([volume for _, volume in rows], dtype=float)
    price, volume = max(rows, key=lambda item: item[1])
    median = float(np.median(volumes)) if len(volumes) else 0.0
    total = float(volumes.sum())
    return {
        "price": price,
        "volume": volume,
        "ratio": volume / median if median > 0 else 1.0,
        "share": volume / total if total > 0 else 0.0,
    }


def _tick_imbalance(ticks: pd.DataFrame | None) -> tuple[float, int]:
    """買い約定を+、売り約定を-とした出来高加重の偏りを返す。"""
    if ticks is None or ticks.empty or "ticker_direction" not in ticks:
        return 0.0, 0
    volume_values = (ticks["volume"] if "volume" in ticks
                     else pd.Series(1.0, index=ticks.index))
    volumes = pd.to_numeric(volume_values, errors="coerce").fillna(0).clip(lower=0)
    directions = ticks["ticker_direction"].astype(str).str.upper()
    buy = float(volumes[directions.str.contains("BUY", na=False)].sum())
    sell = float(volumes[directions.str.contains("SELL", na=False)].sum())
    total = buy + sell
    return ((buy - sell) / total if total > 0 else 0.0), int(len(ticks))


def _capital_net(capital: dict | None) -> float:
    if not capital:
        return 0.0
    if capital.get("net") is not None:
        return _number(capital.get("net"))
    return sum(_number(row[1]) for row in capital.get("tiers", [])
               if isinstance(row, (tuple, list)) and len(row) > 1)


def _confidence(score: float, evidence_count: int) -> str:
    if evidence_count >= 2 and score >= 72:
        return "高"
    if evidence_count >= 1 and score >= 54:
        return "中"
    return "低"


def _live_strength(score: float) -> int:
    return int(np.clip(round(score / 20), 1, 5))


def _summary(book_imbalance: float, tick_imbalance: float,
             capital_net: float, sources: list[str]) -> str:
    parts = []
    if "板情報" in sources:
        if book_imbalance > 0.15:
            parts.append("買い板が優勢")
        elif book_imbalance < -0.15:
            parts.append("売り板が優勢")
        else:
            parts.append("板の買売は概ね均衡")
    if "歩み値" in sources:
        if tick_imbalance > 0.15:
            parts.append("直近約定は買い優勢")
        elif tick_imbalance < -0.15:
            parts.append("直近約定は売り優勢")
        else:
            parts.append("直近約定は中立")
    if "資金フロー" in sources:
        parts.append("当日資金は純流入" if capital_net > 0 else
                     ("当日資金は純流出" if capital_net < 0 else "当日資金は均衡"))
    return "、".join(parts) if parts else "追加のリアルタイム需給データはありません"


def review_levels(df: pd.DataFrame, base_levels: list[dict],
                  order_book: pd.DataFrame | None = None,
                  capital: dict | None = None,
                  ticks: pd.DataFrame | None = None,
                  reviewed_at: str | None = None) -> dict:
    """基礎レベルをmoomooのライブ需給で再採点し、更新候補を返す。

    基礎の ``strength`` は ``base_strength`` に保存する。戻り値の ``strength`` は
    ライブ材料を加味した表示用の★であり、元のバックテスト較正値とは区別する。
    """
    if df is None or df.empty or not base_levels:
        return {"levels": [], "sources": [], "summary": "評価対象がありません"}

    current = _number(df["Close"].iloc[-1])
    atr = _atr(df)
    bids = _book_rows(order_book, "買気配値", "買数量")
    asks = _book_rows(order_book, "売気配値", "売数量")
    bid_wall, ask_wall = _wall(bids), _wall(asks)
    bid_volume = sum(volume for _, volume in bids)
    ask_volume = sum(volume for _, volume in asks)
    book_total = bid_volume + ask_volume
    book_imbalance = ((bid_volume - ask_volume) / book_total
                      if book_total > 0 else 0.0)
    tick_imbalance, tick_count = _tick_imbalance(ticks)
    capital_net = _capital_net(capital)

    sources = []
    if bids or asks:
        sources.append("板情報")
    if tick_count:
        sources.append("歩み値")
    if capital:
        sources.append("資金フロー")

    reviewed = []
    for original in base_levels:
        level = dict(original)
        base_strength = int(level.get("base_strength", level.get("strength", 1)))
        score = 34.0 + base_strength * 8.5
        evidence = []
        live_signals = 0
        wall = ask_wall if level.get("type") == "抵抗線" else bid_wall
        tolerance = max(
            abs(_number(level.get("zone_high")) - _number(level.get("zone_low"))) / 2,
            atr * 0.45,
            current * 0.003,
        )
        if wall and abs(_number(level.get("price")) - wall["price"]) <= tolerance:
            wall_bonus = min(20.0, 10.0 + max(0.0, wall["ratio"] - 1) * 4.0)
            score += wall_bonus
            live_signals += 1
            side = "売り" if level.get("type") == "抵抗線" else "買い"
            evidence.append(
                f"{wall['price']:,.2f}付近の{side}板が中央値の{wall['ratio']:.1f}倍")

        direction = 1 if level.get("type") == "サポート" else -1
        if capital:
            capital_alignment = direction * np.sign(capital_net)
            score += 5.0 * capital_alignment
            live_signals += 1
            if capital_net > 0:
                evidence.append("当日資金は純流入(サポート寄り)")
            elif capital_net < 0:
                evidence.append("当日資金は純流出(抵抗寄り)")
            else:
                evidence.append("当日資金フローは中立")
        if tick_count:
            tick_alignment = direction * tick_imbalance
            score += 7.0 * tick_alignment
            live_signals += 1
            evidence.append(f"直近{tick_count}件の約定偏り {tick_imbalance:+.0%}")
        if sources and not evidence:
            evidence.append("この価格帯と一致する厚い板は未確認")

        score = float(np.clip(score, 0, 100))
        level.update({
            "base_strength": base_strength,
            "strength": _live_strength(score),
            "moomoo_score": score,
            "moomoo_confidence": _confidence(score, live_signals),
            "moomoo_reason": " / ".join(evidence) if evidence else "OHLCV基礎評価のみ",
            "moomoo_reviewed": True,
        })
        reviewed.append(level)

    # 非常に厚い最良候補の板が既存ゾーンから離れている場合だけ、一時候補を追加する。
    for wall, kind, side_label in ((bid_wall, "サポート", "買い"),
                                   (ask_wall, "抵抗線", "売り")):
        if not wall or wall["ratio"] < 1.8 or wall["share"] < 0.25:
            continue
        if abs(wall["price"] / current - 1) > 0.05:
            continue
        tolerance = max(atr * 0.45, current * 0.003)
        if any(abs(_number(level.get("price")) - wall["price"]) <= tolerance
               for level in reviewed):
            continue
        score = float(np.clip(52 + min(24, (wall["ratio"] - 1) * 8), 0, 100))
        half_width = max(atr * 0.20, current * 0.001)
        reviewed.append({
            "type": kind,
            "price": wall["price"],
            "zone_low": wall["price"] - half_width,
            "zone_high": wall["price"] + half_width,
            "distance_pct": (wall["price"] / current - 1) * 100,
            "bounces": 0, "breaks": 0, "touches": 0, "rejects": 0,
            "swings": 0, "bounce_rate": None, "raw_score": 0.0,
            "vol_share": 0.0, "last_touch": "リアルタイム",
            "basis": "moomoo板厚候補(一時)", "confluence": [],
            "last_event": None, "base_strength": 0,
            "strength": _live_strength(score), "moomoo_score": score,
            "moomoo_confidence": _confidence(score, 1),
            "moomoo_reason": (f"{wall['price']:,.2f}付近の{side_label}板が"
                              f"中央値の{wall['ratio']:.1f}倍・全{side_label}板の"
                              f"{wall['share']:.0%}"),
            "moomoo_reviewed": True, "temporary": True,
        })

    reviewed.sort(key=lambda level: -_number(level.get("price")))
    return {
        "levels": reviewed,
        "sources": sources,
        "summary": _summary(book_imbalance, tick_imbalance, capital_net, sources),
        "metrics": {
            "book_imbalance": book_imbalance,
            "tick_imbalance": tick_imbalance,
            "capital_net": capital_net,
            "bid_wall": bid_wall,
            "ask_wall": ask_wall,
        },
        "reviewed_at": reviewed_at or datetime.now().astimezone().isoformat(timespec="seconds"),
    }

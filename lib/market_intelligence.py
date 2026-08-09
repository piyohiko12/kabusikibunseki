"""米国市場・11セクター・代表銘柄候補の読み取り専用分析。

このモジュールが返すのは、日足トレンドに基づく候補選定とその根拠であり、
売買注文や利益を保証する判定ではない。履歴は既存のYahoo Finance取得層だけを
使い、moomooは任意の最新スナップショットに限定する。そのため、moomooの
``request_history_kline`` と過去K線の新規銘柄クォータを消費しない。

ネットワークアクセスは公開取得関数を呼んだ時にだけ発生する。履歴は15分間
キャッシュし、個別銘柄候補は指定されたセクターだけを遅延取得する。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from lib import data_fetcher, moomoo_client, v6_signals


MARKET_BENCHMARK = "SPY"
MARKET_NAME = "米国株式市場（S&P 500 ETF）"
CACHE_TTL_SECONDS = 900
DEFAULT_PERIOD = "1y"
MAX_HISTORY_WORKERS = 6

TREND_UP_SCORE = 65.0
TREND_DOWN_SCORE = 35.0
MIN_TREND_ROWS = 50
# 候補への昇格はSMA200を含む全トレンド条件が揃うことを要求する。
MIN_CANDIDATE_ROWS = 200
MIN_DOLLAR_VOLUME = 25_000_000.0
SNAPSHOT_FRESH_SECONDS = 300.0

TREND_LABELS = {
    "up": "上昇",
    "down": "下降",
    "neutral": "中立",
    "unavailable": "判定不能",
}

# V6の既存ユニバースを唯一の定義元にする。候補は各セクターの代表4銘柄であり、
# セクター全構成銘柄を網羅したスクリーナーではない。
SECTOR_CATALOG: tuple[dict[str, Any], ...] = tuple(
    {
        "key": str(item["name"]),
        "name_ja": str(item["jp"]),
        "etf": str(item["etf"]),
        "candidates": tuple(str(symbol) for symbol in item["members"]),
    }
    for item in v6_signals.SECTORS
)


def _utc_now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _timestamp_iso(value: Any) -> str | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    try:
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
    except (TypeError, ValueError):
        return str(value)
    return stamp.isoformat()


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _find_column(frame: pd.DataFrame, wanted: str) -> Any | None:
    for column in frame.columns:
        label = column[-1] if isinstance(column, tuple) else column
        if str(label).strip().lower() == wanted.lower():
            return column
    return None


def _column_series(frame: pd.DataFrame, column: Any) -> pd.Series:
    values = frame[column]
    # 重複列名を持つ壊れた入力でも例外にせず、最後の系列だけを採用する。
    if isinstance(values, pd.DataFrame):
        return values.iloc[:, -1]
    return values


def _clean_history(history: pd.DataFrame | None) -> pd.DataFrame:
    """必要なClose/Volumeだけを有限値に正規化する。入力は変更しない。"""
    if not isinstance(history, pd.DataFrame) or history.empty:
        return pd.DataFrame(columns=["Close", "Volume"])
    close_column = _find_column(history, "Close")
    if close_column is None:
        return pd.DataFrame(columns=["Close", "Volume"])
    clean = pd.DataFrame(index=history.index)
    clean["Close"] = pd.to_numeric(
        _column_series(history, close_column), errors="coerce")
    volume_column = _find_column(history, "Volume")
    if volume_column is not None:
        clean["Volume"] = pd.to_numeric(
            _column_series(history, volume_column), errors="coerce")
    else:
        clean["Volume"] = np.nan
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=["Close"])
    clean = clean[clean["Close"] > 0]
    clean = clean[~clean.index.duplicated(keep="last")]
    try:
        clean = clean.sort_index()
    except (TypeError, ValueError):
        # 混在型indexは比較できないことがある。取得順を維持して安全に続行する。
        pass
    return clean


def _session_return(close: pd.Series, sessions: int) -> float | None:
    if len(close) < sessions + 1:
        return None
    start = _finite_float(close.iloc[-(sessions + 1)])
    end = _finite_float(close.iloc[-1])
    if start is None or end is None or start <= 0:
        return None
    return end / start - 1.0


def _relative_return(history: pd.DataFrame, benchmark: pd.DataFrame,
                     sessions: int) -> float | None:
    left = _clean_history(history)["Close"].rename("asset")
    right = _clean_history(benchmark)["Close"].rename("benchmark")
    if left.empty or right.empty:
        return None
    try:
        aligned = pd.concat([left, right], axis=1, join="inner").dropna()
    except (TypeError, ValueError):
        return None
    if len(aligned) < sessions + 1:
        return None
    asset_return = _session_return(aligned["asset"], sessions)
    benchmark_return = _session_return(aligned["benchmark"], sessions)
    if asset_return is None or benchmark_return is None:
        return None
    return asset_return - benchmark_return


def _component(key: str, label: str, weight: float,
               passed: bool | None, detail: str) -> dict[str, Any]:
    available = passed is not None
    points = float(weight if passed else 0.0) if available else None
    return {
        "key": key,
        "label": label,
        "weight": float(weight),
        "available": available,
        "available_weight": float(weight) if available else 0.0,
        "passed": passed,
        "points": points,
        "detail": detail,
    }


def _comparison_component(key: str, label: str, weight: float,
                          value: float | None, detail_name: str,
                          *, threshold: float = 0.0) -> dict[str, Any]:
    if value is None:
        return _component(key, label, weight, None, f"{detail_name}: データ不足")
    passed = value > threshold
    return _component(
        key, label, weight, passed,
        f"{detail_name}: {value * 100:+.2f}% "
        f"({'条件達成' if passed else '条件未達'})",
    )


def score_trend(history: pd.DataFrame | None,
                benchmark_history: pd.DataFrame | None = None) -> dict[str, Any]:
    """日足トレンドを0～100点で評価し、全採点根拠を返す純粋関数。

    ``benchmark_history`` が ``None`` の場合は市場自身の評価として相対強度を
    採点対象外にする。空DataFrameを渡した場合は、相対強度を評価したかったが
    ベンチマーク取得に失敗したものとして部分データ扱いにする。
    """
    clean = _clean_history(history)
    close = clean["Close"]
    rows = len(clean)
    as_of = _timestamp_iso(clean.index[-1]) if rows else None
    last = _finite_float(close.iloc[-1]) if rows else None

    sma50 = _finite_float(close.tail(50).mean()) if rows >= 50 else None
    sma200 = _finite_float(close.tail(200).mean()) if rows >= 200 else None
    return_20 = _session_return(close, 20)
    return_63 = _session_return(close, 63)
    relative_20 = None
    compare_requested = benchmark_history is not None
    if compare_requested:
        relative_20 = _relative_return(clean, benchmark_history, 20)

    if last is None or sma50 is None:
        price_vs_sma = _component(
            "price_above_sma50", "終値 > 50日移動平均", 25, None,
            "終値/50日移動平均: データ不足",
        )
    else:
        passed = last > sma50
        price_vs_sma = _component(
            "price_above_sma50", "終値 > 50日移動平均", 25, passed,
            f"終値 {last:.2f} / SMA50 {sma50:.2f} "
            f"({'条件達成' if passed else '条件未達'})",
        )

    if sma50 is None or sma200 is None:
        ma_alignment = _component(
            "sma50_above_sma200", "50日移動平均 > 200日移動平均", 20, None,
            "SMA50/SMA200: データ不足",
        )
    else:
        passed = sma50 > sma200
        ma_alignment = _component(
            "sma50_above_sma200", "50日移動平均 > 200日移動平均", 20,
            passed, f"SMA50 {sma50:.2f} / SMA200 {sma200:.2f} "
            f"({'条件達成' if passed else '条件未達'})",
        )

    components = [
        price_vs_sma,
        ma_alignment,
        _comparison_component(
            "return_20d_positive", "20営業日リターン > 0%", 20,
            return_20, "20営業日リターン"),
        _comparison_component(
            "return_63d_positive", "63営業日リターン > 0%", 15,
            return_63, "63営業日リターン"),
    ]
    if compare_requested:
        components.append(_comparison_component(
            "relative_20d_positive", "20営業日でSPYを上回る", 20,
            relative_20, "SPY比相対リターン"))

    expected_points = float(sum(item["weight"] for item in components))
    available_points = float(sum(
        item["weight"] for item in components if item["available"]))
    raw_points = float(sum(
        item["points"] or 0.0 for item in components if item["available"]))
    score = round(raw_points / available_points * 100.0, 1) if available_points else None
    coverage = round(available_points / expected_points * 100.0, 1) if expected_points else 0.0
    history_coverage = min(rows / 200.0, 1.0) * 100.0
    confidence = round(min(coverage, history_coverage), 1)

    enough_for_direction = rows >= MIN_TREND_ROWS and available_points >= 45.0
    if score is None or not enough_for_direction:
        trend = "unavailable"
    elif score >= TREND_UP_SCORE:
        trend = "up"
    elif score <= TREND_DOWN_SCORE:
        trend = "down"
    else:
        trend = "neutral"

    if trend == "unavailable":
        data_status = "unavailable"
    elif available_points < expected_points:
        data_status = "partial"
    else:
        data_status = "ok"

    reasons = [item["detail"] for item in components if item["available"]]
    warnings = [item["detail"] for item in components if not item["available"]]
    if rows < MIN_TREND_ROWS:
        warnings.append(f"判定に必要な日足が不足しています ({rows}/{MIN_TREND_ROWS}本)")

    volume = pd.to_numeric(clean["Volume"], errors="coerce")
    dollar_volume = (close * volume).replace([np.inf, -np.inf], np.nan)
    median_dollar_volume = _finite_float(dollar_volume.tail(20).median())
    rolling_high = _finite_float(close.tail(252).max()) if rows else None
    drawdown = (last / rolling_high - 1.0
                if last is not None and rolling_high and rolling_high > 0 else None)

    return {
        "score": score,
        "raw_points": round(raw_points, 2),
        "available_points": round(available_points, 2),
        "expected_points": round(expected_points, 2),
        "coverage_pct": coverage,
        "confidence_pct": confidence,
        "trend": trend,
        "trend_label": TREND_LABELS[trend],
        "data_status": data_status,
        "observations": rows,
        "as_of": as_of,
        "metrics": {
            "close": last,
            "sma50": sma50,
            "sma200": sma200,
            "return_20d_pct": (round(return_20 * 100.0, 4)
                               if return_20 is not None else None),
            "return_63d_pct": (round(return_63 * 100.0, 4)
                               if return_63 is not None else None),
            "relative_20d_pct": (round(relative_20 * 100.0, 4)
                                 if relative_20 is not None else None),
            "drawdown_252d_pct": (round(drawdown * 100.0, 4)
                                  if drawdown is not None else None),
            "median_dollar_volume_20d": median_dollar_volume,
        },
        "components": components,
        "reasons": reasons,
        "warnings": warnings,
    }


def _resolve_sector(sector: str) -> dict[str, Any]:
    needle = str(sector or "").strip().upper()
    for item in SECTOR_CATALOG:
        aliases = {item["key"].upper(), item["name_ja"].upper(), item["etf"].upper()}
        if needle in aliases:
            return item
    valid = ", ".join(item["key"] for item in SECTOR_CATALOG)
    raise ValueError(f"不明なセクターです: {sector!r} (有効値: {valid})")


def _safe_snapshot(snapshot: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not snapshot:
        return None
    price = _finite_float(snapshot.get("price"))
    if price is None or price <= 0:
        return None
    raw_update_time = snapshot.get("update_time")
    update_time = None
    timezone_assumption = None
    freshness_status = "unknown"
    age_seconds = None
    if raw_update_time is not None:
        try:
            stamp = pd.Timestamp(raw_update_time)
            if not pd.isna(stamp):
                if stamp.tzinfo is None:
                    # このモジュールの対象は米国株/米国ETFに限定される。
                    stamp = stamp.tz_localize("America/New_York")
                    timezone_assumption = "America/New_York"
                stamp = stamp.tz_convert("UTC")
                update_time = stamp.isoformat()
                age_seconds = float((pd.Timestamp.now(tz="UTC") - stamp).total_seconds())
                if age_seconds < -60.0:
                    freshness_status = "future"
                elif age_seconds <= SNAPSHOT_FRESH_SECONDS:
                    freshness_status = "fresh"
                else:
                    freshness_status = "stale"
        except (TypeError, ValueError):
            pass
    return {
        "price": price,
        "change_percent": _finite_float(snapshot.get("change_percent")),
        "update_time": update_time,
        "update_time_raw": (str(raw_update_time)
                            if raw_update_time is not None else None),
        "timezone_assumption": timezone_assumption,
        "age_seconds": round(age_seconds, 1) if age_seconds is not None else None,
        "freshness_status": freshness_status,
        "source": "moomoo OpenAPI snapshot",
    }


def _valid_snapshot_count(snapshots: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(_safe_snapshot(snapshot) is not None
               for snapshot in snapshots.values())


def _snapshot_freshness_counts(
    snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    counts = {"fresh": 0, "stale": 0, "future": 0, "unknown": 0}
    for snapshot in snapshots.values():
        clean = _safe_snapshot(snapshot)
        if clean:
            counts[clean["freshness_status"]] += 1
    return counts


def _enrich_result(result: dict[str, Any], symbol: str,
                   history_source: str,
                   snapshots: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    snapshot = _safe_snapshot(snapshots.get(symbol))
    sources = [history_source]
    if snapshot:
        sources.append("moomoo OpenAPI snapshot")
    return {
        **result,
        "symbol": symbol,
        "source": " + ".join(sources),
        "freshness": {
            "history_as_of": result.get("as_of"),
            "snapshot_update_time": snapshot.get("update_time") if snapshot else None,
            "snapshot_status": snapshot.get("freshness_status") if snapshot else None,
        },
        "realtime_snapshot": snapshot,
    }


def analyze_market_frames(
    histories: Mapping[str, pd.DataFrame],
    *,
    snapshots: Mapping[str, Mapping[str, Any]] | None = None,
    history_source: str = "Yahoo Finance",
    fetched_at: str | None = None,
    errors: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """取得済みDataFrameから市場全体と11セクターを分析する純粋関数。"""
    snapshots = snapshots or {}
    valid_snapshot_count = _valid_snapshot_count(snapshots)
    errors = dict(errors or {})
    benchmark = histories.get(MARKET_BENCHMARK)
    market = _enrich_result(
        score_trend(benchmark), MARKET_BENCHMARK, history_source, snapshots)
    market["name"] = MARKET_NAME

    sectors: list[dict[str, Any]] = []
    for spec in SECTOR_CATALOG:
        symbol = spec["etf"]
        # benchmarkが未取得なら、空DataFrameを渡して相対強度が欠損したことを明示する。
        comparison = benchmark if isinstance(benchmark, pd.DataFrame) else pd.DataFrame()
        scored = _enrich_result(
            score_trend(histories.get(symbol), comparison),
            symbol, history_source, snapshots)
        scored.update({
            "sector_key": spec["key"],
            "sector_name_ja": spec["name_ja"],
            "etf": symbol,
            "candidate_universe": spec["candidates"],
        })
        sectors.append(scored)

    ranked = sorted(
        (item for item in sectors if item["trend"] != "unavailable"),
        key=lambda item: (-float(item["score"]), item["sector_key"]),
    )
    rank_by_key = {item["sector_key"]: rank for rank, item in enumerate(ranked, 1)}
    for item in sectors:
        item["rank"] = rank_by_key.get(item["sector_key"])

    counts = {key: 0 for key in TREND_LABELS}
    for item in sectors:
        counts[item["trend"]] += 1
    decided = counts["up"] + counts["down"] + counts["neutral"]
    breadth = {
        "up": counts["up"],
        "down": counts["down"],
        "neutral": counts["neutral"],
        "unavailable": counts["unavailable"],
        "up_ratio_pct": round(counts["up"] / decided * 100.0, 1) if decided else None,
    }

    requested = (MARKET_BENCHMARK,) + tuple(item["etf"] for item in SECTOR_CATALOG)
    succeeded = tuple(symbol for symbol in requested
                      if isinstance(histories.get(symbol), pd.DataFrame)
                      and not histories[symbol].empty)
    failed = tuple(symbol for symbol in requested if symbol not in succeeded)
    analysis_by_symbol = {
        MARKET_BENCHMARK: market,
        **{item["etf"]: item for item in sectors},
    }
    analysis_unavailable = tuple(
        symbol for symbol in requested
        if analysis_by_symbol[symbol]["trend"] == "unavailable")
    analysis_partial = tuple(
        symbol for symbol in requested
        if analysis_by_symbol[symbol]["data_status"] == "partial")
    if len(analysis_unavailable) == len(requested):
        status = "unavailable"
    elif failed or analysis_unavailable or analysis_partial:
        status = "partial"
    else:
        status = "ok"

    return {
        "market": market,
        "sectors": sectors,
        "breadth": breadth,
        "meta": {
            "status": status,
            "requested": requested,
            "succeeded": succeeded,
            "failed": failed,
            "errors": {symbol: errors.get(symbol, "データを取得できませんでした")
                       for symbol in failed},
            "analysis_unavailable": analysis_unavailable,
            "analysis_partial": analysis_partial,
            "fetched_at": fetched_at or _utc_now_iso(),
            "history_source": history_source,
            "realtime_source": ("moomoo OpenAPI snapshot"
                                if valid_snapshot_count else None),
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
            "history_quota_consumed": False,
            "allow_new_history_quota": False,
            "read_only": True,
        },
        "disclaimer": (
            "日足トレンドの説明可能な機械的評価です。BUY/SELLや利益を保証せず、"
            "投資判断は利用者自身で行ってください。"
        ),
    }


def _candidate_component(key: str, label: str, weight: float,
                         passed: bool | None, detail: str,
                         *, points: float | None = None,
                         available_weight: float | None = None) -> dict[str, Any]:
    item = _component(key, label, weight, passed, detail)
    if passed is not None and points is not None:
        item["points"] = round(max(0.0, min(float(weight), float(points))), 2)
    if available_weight is not None:
        item["available_weight"] = round(
            max(0.0, min(float(weight), float(available_weight))), 2)
    return item


def _candidate_score(symbol: str, history: pd.DataFrame | None,
                     benchmark: pd.DataFrame | None,
                     sector_history: pd.DataFrame | None) -> dict[str, Any]:
    comparison = benchmark if isinstance(benchmark, pd.DataFrame) else pd.DataFrame()
    trend = score_trend(history, comparison)
    trend_score = trend.get("score")
    if trend_score is None:
        trend_component = _candidate_component(
            "daily_trend", "日足トレンド", 50, None,
            "日足トレンド: データ不足")
    else:
        expected = float(trend["expected_points"]) or 100.0
        trend_points = float(trend["raw_points"]) / expected * 50.0
        trend_available_weight = float(trend["available_points"]) / expected * 50.0
        trend_component = _candidate_component(
            "daily_trend", "日足トレンド", 50, trend_score >= TREND_UP_SCORE,
            f"日足トレンド {trend_score:.1f}/100、"
            f"基礎データ網羅率 {trend['coverage_pct']:.1f}%",
            points=trend_points, available_weight=trend_available_weight)

    clean = _clean_history(history)
    sector_clean = (_clean_history(sector_history)
                    if isinstance(sector_history, pd.DataFrame) else pd.DataFrame())
    relative_20 = _relative_return(clean, sector_clean, 20)
    relative_63 = _relative_return(clean, sector_clean, 63)
    median_dollar_volume = trend["metrics"]["median_dollar_volume_20d"]
    drawdown_pct = trend["metrics"]["drawdown_252d_pct"]

    components = [
        trend_component,
        _comparison_component(
            "relative_20d_vs_sector", "20営業日でセクターETFを上回る", 20,
            relative_20, "セクターETF比20営業日リターン"),
        _comparison_component(
            "relative_63d_vs_sector", "63営業日でセクターETFを上回る", 15,
            relative_63, "セクターETF比63営業日リターン"),
    ]
    if median_dollar_volume is None:
        liquidity = _candidate_component(
            "liquidity", "20日中央値ドル出来高 >= 2,500万ドル", 10,
            None, "ドル出来高: データ不足")
    else:
        passed = median_dollar_volume >= MIN_DOLLAR_VOLUME
        liquidity = _candidate_component(
            "liquidity", "20日中央値ドル出来高 >= 2,500万ドル", 10,
            passed, f"20日中央値ドル出来高 ${median_dollar_volume:,.0f} "
            f"({'条件達成' if passed else '条件未達'})")
    components.append(liquidity)

    if drawdown_pct is None:
        drawdown = _candidate_component(
            "drawdown", "52週高値からの下落 >= -20%", 5,
            None, "52週高値からの下落: データ不足")
    else:
        passed = drawdown_pct >= -20.0
        drawdown = _candidate_component(
            "drawdown", "52週高値からの下落 >= -20%", 5,
            passed, f"52週高値から {drawdown_pct:+.2f}% "
            f"({'条件達成' if passed else '条件未達'})")
    components.append(drawdown)

    available_points = float(sum(
        item["available_weight"] for item in components))
    raw_points = float(sum(
        item["points"] or 0.0 for item in components if item["available"]))
    # 候補スコアは100点満点の生点。欠損分を再正規化して過大評価しない。
    score = round(raw_points, 1) if available_points else None
    coverage = round(available_points, 1)  # 満点100なのでそのままカバレッジ率。
    confidence = round(min(coverage, float(trend["confidence_pct"])), 1)
    context_complete = relative_20 is not None and relative_63 is not None
    market_context_complete = any(
        item["key"] == "relative_20d_positive" and item["available"]
        for item in trend["components"])
    data_complete = (
        score is not None
        and trend["observations"] >= MIN_CANDIDATE_ROWS
        and available_points >= 80.0
        and context_complete
        and market_context_complete
        and confidence >= 60.0
        and median_dollar_volume is not None
        and trend["data_status"] == "ok"
    )
    liquidity_ok = (median_dollar_volume is not None
                    and median_dollar_volume >= MIN_DOLLAR_VOLUME)
    eligible = data_complete and liquidity_ok
    if not data_complete:
        candidate_status = "データ不足"
    elif not liquidity_ok:
        candidate_status = "流動性基準外"
    elif score >= 70.0 and trend["trend"] == "up":
        candidate_status = "候補"
    elif score >= 55.0 and trend["trend"] in {"up", "neutral"}:
        candidate_status = "監視候補"
    else:
        candidate_status = "優先度低"

    return {
        "symbol": symbol,
        "recommendation_score": score,
        "raw_points": round(raw_points, 2),
        "available_points": round(available_points, 2),
        "coverage_pct": coverage,
        "confidence_pct": confidence,
        "data_complete": data_complete,
        "eligible": eligible,
        "candidate_status": candidate_status,
        "trend": trend,
        "components": components,
        "reasons": [item["detail"] for item in components if item["available"]],
        "warnings": [item["detail"] for item in components if not item["available"]],
    }


def rank_sector_candidates(
    sector: str,
    histories: Mapping[str, pd.DataFrame],
    *,
    top_n: int = 3,
    snapshots: Mapping[str, Mapping[str, Any]] | None = None,
    history_source: str = "Yahoo Finance",
    fetched_at: str | None = None,
    errors: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """指定セクターの代表4銘柄を採点し、上位候補と全順位を返す純粋関数。"""
    if top_n < 1:
        raise ValueError("top_nは1以上で指定してください")
    spec = _resolve_sector(sector)
    snapshots = snapshots or {}
    valid_snapshot_count = _valid_snapshot_count(snapshots)
    errors = dict(errors or {})
    benchmark = histories.get(MARKET_BENCHMARK)
    sector_history = histories.get(spec["etf"])

    ranking: list[dict[str, Any]] = []
    for symbol in spec["candidates"]:
        candidate = _candidate_score(
            symbol, histories.get(symbol), benchmark, sector_history)
        enriched = _enrich_result(
            candidate["trend"], symbol, history_source, snapshots)
        candidate["trend"] = enriched
        candidate["source"] = enriched["source"]
        candidate["freshness"] = enriched["freshness"]
        candidate["realtime_snapshot"] = enriched["realtime_snapshot"]
        ranking.append(candidate)

    ranking.sort(key=lambda item: (
        not item["eligible"],
        -(float(item["recommendation_score"])
          if item["recommendation_score"] is not None else -1.0),
        item["symbol"],
    ))
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank

    candidates = [
        item for item in ranking
        if item["candidate_status"] in {"候補", "監視候補"}
    ][:top_n]
    requested = (MARKET_BENCHMARK, spec["etf"], *spec["candidates"])
    succeeded = tuple(symbol for symbol in requested
                      if isinstance(histories.get(symbol), pd.DataFrame)
                      and not histories[symbol].empty)
    failed = tuple(symbol for symbol in requested if symbol not in succeeded)
    scored_count = sum(item["recommendation_score"] is not None for item in ranking)
    data_complete_count = sum(bool(item["data_complete"]) for item in ranking)
    eligible_count = sum(bool(item["eligible"]) for item in ranking)
    if scored_count == 0:
        status = "unavailable"
    elif failed or data_complete_count < len(ranking):
        status = "partial"
    else:
        status = "ok"

    return {
        "sector": {
            "key": spec["key"],
            "name_ja": spec["name_ja"],
            "etf": spec["etf"],
            "universe": spec["candidates"],
        },
        "candidates": candidates,
        "ranking": ranking,
        "meta": {
            "status": status,
            "requested": requested,
            "succeeded": succeeded,
            "failed": failed,
            "errors": {symbol: errors.get(symbol, "データを取得できませんでした")
                       for symbol in failed},
            "evaluated_count": len(ranking),
            "scored_count": scored_count,
            "data_complete_count": data_complete_count,
            "eligible_count": eligible_count,
            "analysis_unavailable": tuple(
                item["symbol"] for item in ranking if not item["data_complete"]),
            "selection_status": "found" if candidates else "none",
            "fetched_at": fetched_at or _utc_now_iso(),
            "history_source": history_source,
            "realtime_source": ("moomoo OpenAPI snapshot"
                                if valid_snapshot_count else None),
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
            "history_quota_consumed": False,
            "allow_new_history_quota": False,
            "read_only": True,
        },
        "disclaimer": (
            "各セクターの代表4銘柄だけを、日足トレンド・相対強度・流動性で"
            "並べた候補リストです。BUY判定や利益を保証するものではありません。"
        ),
    }


def _normalise_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))


def _load_history_bundle_uncached(
    symbols: tuple[str, ...], period: str = DEFAULT_PERIOD,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """既存Yahoo取得層を並列利用する。moomoo過去K線には到達しない。"""
    symbols = _normalise_symbols(symbols)
    histories: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    if not symbols:
        return histories, {
            "status": "unavailable", "requested": (), "succeeded": (),
            "failed": (), "errors": {}, "fetched_at": _utc_now_iso(),
            "source": "Yahoo Finance", "history_quota_consumed": False,
        }

    workers = min(MAX_HISTORY_WORKERS, len(symbols))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            symbol: executor.submit(data_fetcher.fetch_history, symbol, period, "1d")
            for symbol in symbols
        }
        for symbol in symbols:
            try:
                frame = futures[symbol].result()
            except Exception as exc:  # 1銘柄の失敗で全体を失敗させない。
                errors[symbol] = f"{type(exc).__name__}: {exc}"
                continue
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                errors[symbol] = "空の履歴が返されました"
                continue
            histories[symbol] = frame

    succeeded = tuple(symbol for symbol in symbols if symbol in histories)
    failed = tuple(symbol for symbol in symbols if symbol not in histories)
    status = "ok" if not failed else ("partial" if succeeded else "unavailable")
    return histories, {
        "status": status,
        "requested": symbols,
        "succeeded": succeeded,
        "failed": failed,
        "errors": errors,
        "fetched_at": _utc_now_iso(),
        "source": "Yahoo Finance",
        "history_quota_consumed": False,
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _load_history_bundle(
    symbols: tuple[str, ...], period: str = DEFAULT_PERIOD,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    return _load_history_bundle_uncached(symbols, period)


def _load_snapshots(symbols: tuple[str, ...], enabled: bool) -> tuple[dict, str | None]:
    if not enabled or not symbols:
        return {}, None
    try:
        snapshots = moomoo_client.snapshot(symbols)
    except Exception as exc:  # moomoo障害はYahoo日足分析を止めない。
        return {}, f"{type(exc).__name__}: {exc}"
    if not isinstance(snapshots, dict):
        return {}, "moomoo snapshotの応答形式が不正です"
    if not snapshots:
        return {}, (
            "moomoo snapshotを取得できませんでした（連携OFF・接続・権限・"
            "対象銘柄データのいずれかを確認してください）")
    return snapshots, None


def _snapshot_delivery_meta(
    symbols: tuple[str, ...],
    snapshots: Mapping[str, Mapping[str, Any]],
    *,
    requested: bool,
    load_error: str | None,
) -> dict[str, Any]:
    requested_symbols = symbols if requested else ()
    received_symbols = tuple(
        symbol for symbol in requested_symbols
        if _safe_snapshot(snapshots.get(symbol)) is not None)
    missing_symbols = tuple(
        symbol for symbol in requested_symbols if symbol not in received_symbols)
    if not requested:
        status = "not_requested"
        fallback_reason = None
    elif not received_symbols:
        status = "unavailable"
        fallback_reason = load_error or "moomoo snapshotを取得できませんでした"
    elif missing_symbols:
        status = "partial"
        fallback_reason = load_error or (
            f"moomoo snapshotは{len(received_symbols)}/{len(requested_symbols)}銘柄のみ取得")
    else:
        status = "available"
        fallback_reason = load_error
    return {
        "realtime_requested": requested,
        "realtime_received": len(received_symbols),
        "realtime_error": fallback_reason,
        "snapshot_status": status,
        "snapshot_requested_symbols": requested_symbols,
        "snapshot_received_symbols": received_symbols,
        "snapshot_missing_symbols": missing_symbols,
        "snapshot_freshness_counts": _snapshot_freshness_counts({
            symbol: snapshots[symbol] for symbol in received_symbols
        }),
    }


def get_us_market_overview(*, include_realtime: bool = True,
                           period: str = DEFAULT_PERIOD) -> dict[str, Any]:
    """米国市場全体と11セクターを取得・分析する遅延ロードAPI。"""
    symbols = (MARKET_BENCHMARK,) + tuple(item["etf"] for item in SECTOR_CATALOG)
    histories, load_meta = _load_history_bundle(symbols, period)
    snapshots, snapshot_error = _load_snapshots(symbols, include_realtime)
    result = analyze_market_frames(
        histories, snapshots=snapshots, history_source=load_meta["source"],
        fetched_at=load_meta["fetched_at"], errors=load_meta["errors"])
    result["meta"].update({"period": period})
    result["meta"].update(_snapshot_delivery_meta(
        symbols, snapshots, requested=bool(include_realtime),
        load_error=snapshot_error))
    return result


def get_sector_candidates(sector: str, *, top_n: int = 3,
                          include_realtime: bool = True,
                          period: str = DEFAULT_PERIOD) -> dict[str, Any]:
    """指定セクターだけを遅延取得し、説明可能な上位銘柄候補を返す。"""
    if top_n < 1:
        raise ValueError("top_nは1以上で指定してください")
    spec = _resolve_sector(sector)
    symbols = (MARKET_BENCHMARK, spec["etf"], *spec["candidates"])
    histories, load_meta = _load_history_bundle(symbols, period)
    snapshots, snapshot_error = _load_snapshots(symbols, include_realtime)
    result = rank_sector_candidates(
        spec["key"], histories, top_n=top_n, snapshots=snapshots,
        history_source=load_meta["source"], fetched_at=load_meta["fetched_at"],
        errors=load_meta["errors"])
    result["meta"].update({"period": period})
    result["meta"].update(_snapshot_delivery_meta(
        symbols, snapshots, requested=bool(include_realtime),
        load_error=snapshot_error))
    return result


def get_market_intelligence(
    *,
    candidate_sectors: Sequence[str] = (),
    top_n: int = 3,
    include_realtime: bool = True,
    period: str = DEFAULT_PERIOD,
) -> dict[str, Any]:
    """市場・11セクターと、明示指定セクターの候補をまとめて返す。

    ``candidate_sectors`` の既定値は空である。44銘柄を暗黙に一括取得せず、UIで
    開かれたセクターだけを遅延ロードするための安全な既定値である。
    """
    if top_n < 1:
        raise ValueError("top_nは1以上で指定してください")
    requested_sectors = (() if candidate_sectors is None else
                         ((candidate_sectors,) if isinstance(candidate_sectors, str)
                          else candidate_sectors))
    resolved: list[dict[str, Any]] = []
    for requested in requested_sectors:
        spec = _resolve_sector(requested)
        if not any(item["key"] == spec["key"] for item in resolved):
            resolved.append(spec)

    # 入力検証が済んでから初めてネットワーク境界へ進む。
    overview = get_us_market_overview(
        include_realtime=include_realtime, period=period)
    candidate_results: dict[str, dict[str, Any]] = {}
    for spec in resolved:
        candidate_results[spec["key"]] = get_sector_candidates(
            spec["key"], top_n=top_n, include_realtime=include_realtime,
            period=period)
    return {
        **overview,
        "candidate_sectors": candidate_results,
    }


def clear_market_intelligence_cache() -> None:
    """このモジュール固有の15分履歴バンドルキャッシュだけを消去する。"""
    _load_history_bundle.clear()


__all__ = [
    "CACHE_TTL_SECONDS",
    "MARKET_BENCHMARK",
    "SECTOR_CATALOG",
    "analyze_market_frames",
    "clear_market_intelligence_cache",
    "get_market_intelligence",
    "get_sector_candidates",
    "get_us_market_overview",
    "rank_sector_candidates",
    "score_trend",
]

"""米国株向けの説明可能なイベント影響コンテキスト。

このモジュールは、決算・配当・株式分割・重要企業ニュースと、FOMC・CPI・
雇用統計を同じ形式に正規化する。結果はあくまでシナリオ分析であり、方向を
保証したり、既存の売買スコアへ自動加点したりしない。

ネットワーク取得は :func:`fetch_event_intelligence` を明示的に呼んだ時だけ行う。
既存のYahoo Finance系取得関数を再利用し、moomooの過去K線・注文・口座APIは
呼び出さない。
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from hashlib import sha1
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from lib import data_fetcher, news_fetcher, sensitivity


CACHE_TTL_SECONDS = 900
CALENDAR_VERIFIED_AT = "2026-08-10"
NO_TRADE_SCORE_EFFECT = 0
DISCLAIMER = (
    "イベント前後は上昇・下落の両方があり得ます。表示は過去の値動きと条件別"
    "シナリオであり、方向を保証せず、売買判定スコアへ自動加点しません。"
)

FED_FOMC_CALENDAR_URL = (
    "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
)
BLS_CPI_CALENDAR_URL = "https://www.bls.gov/schedule/news_release/cpi.htm"
BLS_EMPLOYMENT_CALENDAR_URL = (
    "https://www.bls.gov/cps/publications/release-calendar.htm"
)

# BLS公表日程。時刻はいずれも米東部時間08:30。日程変更に備えて、各イベントに
# 検証日と公式URLを付け、build_event_intelligenceのmacro_events引数で差替可能。
_CPI_RELEASE_DATES_2026 = (
    "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10",
    "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-11", "2026-10-14", "2026-11-10", "2026-12-10",
)
_EMPLOYMENT_RELEASE_DATES_2026 = (
    "2026-01-09", "2026-02-11", "2026-03-06", "2026-04-03",
    "2026-05-08", "2026-06-05", "2026-07-02", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
)

EVENT_FIELDS = (
    "event_id", "kind", "name", "event_date", "event_time_et", "status",
    "session", "session_label", "impact_level", "impact_score",
    "directional_bias", "scenarios", "historical_sensitivity", "confidence",
    "confidence_label", "evidence", "source", "url", "freshness",
    "score_effect", "automatic_trade_score", "disclaimer",
)

_SESSION_LABELS = {
    "PRE_MARKET": "プレマーケット",
    "REGULAR": "立会時間",
    "REGULAR_OPEN": "立会寄付き",
    "AFTER_MARKET": "アフターマーケット",
    "OVERNIGHT": "夜間・24時間取引帯",
    "DATE_ONLY": "日付のみ（時刻不明）",
    "UNKNOWN": "セッション不明",
}

# 表示専用の日本語辞書。保存・判定に使うコード値は変更せず、
# ``localize_event_for_display`` でUI向け表示だけを組み立てる。
_KIND_LABELS_JA = {
    "earnings": "決算発表",
    "fomc": "FOMC政策金利発表",
    "cpi": "米国消費者物価指数（CPI）",
    "employment": "米国雇用統計",
    "dividend": "配当・権利落ち",
    "split": "株式分割・併合",
    "merger": "合併・買収",
    "regulatory": "規制・訴訟",
    "guidance": "業績見通し",
    "filing": "企業開示",
    "company_news": "企業ニュース",
}

_STATUS_LABELS_JA = {
    "UPCOMING": "今後の予定",
    "RECENT": "最近のイベント",
    "HISTORICAL": "過去のイベント",
    "UNKNOWN": "時期不明",
}

_DIRECTIONAL_BIAS_LABELS_JA = {
    "UP_HISTORY_BIASED": "過去は上昇寄り",
    "DOWN_HISTORY_BIASED": "過去は下落寄り",
    "TWO_SIDED": "過去は上下両方向",
    "UNKNOWN": "方向性の材料不足",
}

_CONFIDENCE_LABELS_JA = {
    "HIGH": "十分",
    "MEDIUM": "一部あり",
    "LOW": "少ない",
    "UNKNOWN": "判定材料不足",
}

_SCENARIO_LABELS_JA = {
    "up": "上振れシナリオ",
    "down": "下振れシナリオ",
    "two_sided": "上下に振れるシナリオ",
}

# impact_score（1〜100）をおおむね20点刻みで★1〜5へ変換する。0・欠損は
# 「判定材料なし」とし、低影響（★1）と区別する。星の数はimpact_scoreに対して
# 単調非減少となり、判定ロジックそのものには影響しない。
IMPACT_STAR_BASIS = (
    "影響スコア0・欠損: 判定材料なし、1〜19: ★1、20〜39: ★2、40〜59: ★3、"
    "60〜79: ★4、80〜100: ★5"
)

HISTORICAL_IMPACT_STAR_BASIS = (
    "過去イベント時の変動÷平常時の変動が4倍以上: ★5、3倍以上: ★4、"
    "2倍以上: ★3、1.3倍以上: ★2、1.3倍未満: ★1"
)

_SCENARIOS = {
    "earnings": {
        "up": "売上・EPS・会社見通しが市場予想を上回り、時間外の出来高を伴う場合。",
        "down": "売上・EPSの未達、見通し引下げ、利益率悪化が意識される場合。",
        "two_sided": "実績と見通しが逆方向、または初動と経営陣説明後の評価が反転する場合。",
    },
    "fomc": {
        "up": "政策・声明・会見が織込みよりハト派で、金利低下と指数先物上昇が続く場合。",
        "down": "織込みよりタカ派で、金利上昇・バリュエーション低下が意識される場合。",
        "two_sided": "声明直後と会見後で金利解釈が変わり、初動が反転する場合。",
    },
    "cpi": {
        "up": "総合・コア物価が予想を下回り、利下げ期待と指数先物がともに強まる場合。",
        "down": "物価が予想を上回り、金利上昇と金融引締め長期化が意識される場合。",
        "two_sided": "総合とコア、前年比と前月比が食い違い、金利と株価の初動が定まらない場合。",
    },
    "employment": {
        "up": "雇用が緩やかに減速しつつ景気後退懸念が強まらない、いわゆる軟着陸の場合。",
        "down": "賃金・雇用の過熱で金利が上がる、または急減速で景気懸念が強まる場合。",
        "two_sided": "雇用者数・失業率・賃金が異なる方向を示し、金利と株価の反応が交錯する場合。",
    },
    "dividend": {
        "up": "増配や持続可能性の確認が資本還元への評価改善につながる場合。",
        "down": "減配・無配、または配当維持への資金負担が意識される場合。",
        "two_sided": "権利落ちによる機械的な価格調整と企業価値への評価を分けて見る必要がある場合。",
    },
    "split": {
        "up": "流動性・個人投資家の参加期待が高まる場合（分割自体は企業価値を変えません）。",
        "down": "材料出尽くしや同時発表の弱い見通しが優先される場合。",
        "two_sided": "分割への短期反応と、業績に基づく中期評価が異なる場合。",
    },
    "merger": {
        "up": "買収条件・相乗効果・成立確度が市場想定を上回る場合。",
        "down": "希薄化、過大な買収価格、規制・資金調達リスクが意識される場合。",
        "two_sided": "買い手と対象会社で反応が分かれ、成立確率の変化で価格が往復する場合。",
    },
    "regulatory": {
        "up": "承認・訴訟解消・規制リスク後退が確認される場合。",
        "down": "調査、提訴、承認遅延、罰金など将来キャッシュフローへの懸念が増す場合。",
        "two_sided": "見出しと開示詳細の解釈が異なり、初動が修正される場合。",
    },
    "guidance": {
        "up": "会社見通し・受注・製品需要が市場予想を上回る場合。",
        "down": "見通し引下げ、需要鈍化、供給制約や利益率低下が示される場合。",
        "two_sided": "短期コストと中長期成長期待が競合し、時間軸で評価が分かれる場合。",
    },
    "filing": {
        "up": "開示内容が財務・ガバナンス上の不確実性を低下させる場合。",
        "down": "希薄化、継続企業、訴訟、内部統制などのリスクが判明する場合。",
        "two_sided": "定型開示と重要な新情報が混在し、精査後に評価が変わる場合。",
    },
    "company_news": {
        "up": "業績・需要・提携など将来キャッシュフローを改善する新情報が確認される場合。",
        "down": "需要鈍化・コスト増・競争・経営上の不確実性が増す場合。",
        "two_sided": "見出しの印象と定量的な影響が一致せず、詳細確認後に初動が反転する場合。",
    },
}

_NEWS_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("earnings", ("earnings", "quarterly results", "eps", "revenue", "決算")),
    ("guidance", ("guidance", "outlook", "forecast", "見通し", "業績予想")),
    ("dividend", ("dividend", "distribution", "配当", "増配", "減配")),
    ("split", ("stock split", "reverse split", "株式分割", "併合")),
    ("merger", ("acquire", "acquisition", "merger", "takeover", "買収", "合併")),
    ("regulatory", ("fda", "doj", "sec investigation", "antitrust", "lawsuit",
                    "approval", "regulator", "訴訟", "規制", "承認")),
    ("filing", ("10-k", "10-q", "8-k", "6-k", "20-f", "s-1", "13d", "13g")),
    ("company_news", ("launch", "partnership", "contract", "order", "recall",
                      "ceo", "cfo", "製品", "提携", "契約", "リコール", "退任")),
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _us_market_date() -> date:
    return datetime.now(ZoneInfo("America/New_York")).date()


def _as_date(value: object) -> date | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.date()


def _round_or_none(value: object, digits: int = 2) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def impact_score_to_stars(impact_score: object) -> int:
    """影響スコアを表示用の★0〜5へ単調変換する。

    スコアは0〜100へ丸め、1〜19を★1、20〜39を★2、40〜59を
    ★3、60〜79を★4、80〜100を★5とする。0・欠損・非数値・負数は
    「判定材料なし」の★0とし、低影響の★1と区別する。元のイベント判定や
    売買スコアは変更しない。
    """
    numeric = _round_or_none(impact_score, 8)
    if numeric is None or numeric <= 0:
        return 0
    score = min(100.0, numeric)
    return min(5, int(score // 20) + 1)


def _contains_japanese(value: str) -> bool:
    return re.search(r"[\u3040-\u30ff\u3400-\u9fff]", value) is not None


def _sample_size(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _historical_impact_stars(study: object) -> int | None:
    """旧イベント感応度と同じ閾値で実測比率を星へ変換する。"""
    if not isinstance(study, Mapping) or _sample_size(study.get("sample_size")) < 3:
        return None
    ratio = _round_or_none(study.get("sensitivity_ratio"), 8)
    if ratio is None or ratio < 0:
        return None
    for threshold, stars in ((4.0, 5), (3.0, 4), (2.0, 3), (1.3, 2)):
        if ratio >= threshold:
            return stars
    return 1


def localize_event_for_display(event: Mapping[str, object]) -> dict:
    """イベントを日本語UI向けに投影する純粋な公開ヘルパー。

    元のコード値と全フィールドを保持したコピーに、``*_label_ja`` と
    ★表示を追加する。日本語名はそのまま主見出しに使うが、英語だけの名称・
    ニュース見出しは機械翻訳せず、イベント種別の日本語名を主見出し、原文を
    ``original_name`` に分ける。これによりUIは英語原文を補助欄へ表示できる。
    入力オブジェクトとその入れ子は変更しない。
    """
    if not isinstance(event, Mapping):
        raise TypeError("eventは辞書形式で指定してください")

    localized = deepcopy(dict(event))
    kind = str(event.get("kind") or "").strip().casefold()
    status = str(event.get("status") or "UNKNOWN").strip().upper()
    session = str(event.get("session") or "UNKNOWN").strip().upper()
    directional_bias = str(
        event.get("directional_bias") or "UNKNOWN").strip().upper()
    confidence_label = str(
        event.get("confidence_label") or "UNKNOWN").strip().upper()

    kind_label = _KIND_LABELS_JA.get(kind, "その他のイベント")
    raw_name = str(event.get("name") or "").strip()
    if raw_name and _contains_japanese(raw_name):
        display_name = raw_name
        original_name = None
    else:
        # 原文が英語だけの場合は翻訳を生成せず、既知の種別名を主見出しにする。
        display_name = kind_label
        original_name = raw_name or None

    historical_sensitivity = event.get("historical_sensitivity")
    historical_stars = _historical_impact_stars(historical_sensitivity)
    if historical_stars is not None:
        stars = historical_stars
        star_source = "過去実測（平常時比）"
        star_basis = HISTORICAL_IMPACT_STAR_BASIS
    else:
        stars = impact_score_to_stars(event.get("impact_score"))
        star_source = "イベント種類別の目安" if stars else "判定材料なし"
        star_basis = IMPACT_STAR_BASIS
    display_impact_label = (
        "影響大" if stars >= 4 else "影響中" if stars == 3
        else "影響小" if stars else "判定材料不足"
    )

    sample_size = (
        _sample_size(historical_sensitivity.get("sample_size"))
        if isinstance(historical_sensitivity, Mapping) else 0
    )
    scenario_rows = []
    scenarios = event.get("scenarios")
    if isinstance(scenarios, Mapping):
        for key in ("up", "down", "two_sided"):
            if key in scenarios:
                scenario_rows.append({
                    "key": key,
                    "label_ja": _SCENARIO_LABELS_JA[key],
                    "description": deepcopy(scenarios[key]),
                })

    labels = {
        "kind": kind_label,
        "status": _STATUS_LABELS_JA.get(status, "時期不明"),
        "session": _SESSION_LABELS.get(session, _SESSION_LABELS["UNKNOWN"]),
        # 主表示の言葉と★数を一致させる。内部impact_levelは元データに保持する。
        "impact": display_impact_label,
        "directional_bias": (
            "過去データ不足で方向不明"
            if directional_bias == "TWO_SIDED" and sample_size < 5
            else _DIRECTIONAL_BIAS_LABELS_JA.get(
                directional_bias, "方向性の材料不足")
        ),
        "confidence": _CONFIDENCE_LABELS_JA.get(
            confidence_label, "判定材料不足"),
    }
    localized.update({
        "display_name_ja": display_name,
        "original_name": original_name,
        "original_name_label_ja": "原文見出し",
        "kind_label_ja": labels["kind"],
        "status_label_ja": labels["status"],
        "session_label_ja": labels["session"],
        "impact_label_ja": labels["impact"],
        "directional_bias_label_ja": labels["directional_bias"],
        "confidence_label_ja": labels["confidence"],
        "impact_stars": stars,
        "impact_stars_text": "★" * stars + "☆" * (5 - stars),
        "impact_stars_label": "—" if stars == 0 else "★" * stars,
        "impact_stars_accessible_ja": (
            "判定材料なし" if stars == 0
            else f"{'★' * stars}{'☆' * (5 - stars)}（{stars}/5・{labels['impact']}）"
        ),
        "impact_available": stars > 0,
        "impact_star_source_ja": star_source,
        "impact_star_basis": star_basis,
        "scenario_labels_ja": dict(_SCENARIO_LABELS_JA),
        "scenario_rows_ja": scenario_rows,
        "labels_ja": labels,
    })
    return localized


def normalize_ticker(ticker: str) -> str:
    """米国株コードをYahoo形式へ正規化する。"""
    value = str(ticker or "").strip().upper()
    if value.startswith("US."):
        value = value[3:]
    if not value or not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,14}", value):
        raise ValueError("有効な米国株ティッカーを指定してください")
    return value


def default_macro_events() -> list[dict]:
    """内蔵の公式マクロ日程を正規化して返す（呼出し側で変更可能）。"""
    rows: list[dict] = []
    for event_date in sensitivity.FOMC_DATES:
        rows.append({
            "kind": "fomc", "name": "FOMC政策金利発表",
            "event_date": event_date, "event_time_et": "14:00",
            "session": "REGULAR", "source": "Federal Reserve",
            "url": FED_FOMC_CALENDAR_URL, "verified_at": CALENDAR_VERIFIED_AT,
        })
    for event_date in _CPI_RELEASE_DATES_2026:
        rows.append({
            "kind": "cpi", "name": "米国消費者物価指数（CPI）",
            "event_date": event_date, "event_time_et": "08:30",
            "session": "PRE_MARKET", "source": "U.S. Bureau of Labor Statistics",
            "url": BLS_CPI_CALENDAR_URL, "verified_at": CALENDAR_VERIFIED_AT,
        })
    for event_date in _EMPLOYMENT_RELEASE_DATES_2026:
        rows.append({
            "kind": "employment", "name": "米国雇用統計",
            "event_date": event_date, "event_time_et": "08:30",
            "session": "PRE_MARKET", "source": "U.S. Bureau of Labor Statistics",
            "url": BLS_EMPLOYMENT_CALENDAR_URL, "verified_at": CALENDAR_VERIFIED_AT,
        })
    return sorted(rows, key=lambda row: (row["event_date"], row["kind"]))


def _clean_history(history: pd.DataFrame | None) -> pd.DataFrame:
    if history is None or not isinstance(history, pd.DataFrame) or history.empty:
        return pd.DataFrame()
    if "Close" not in history.columns:
        return pd.DataFrame()
    out = history.copy()
    try:
        index = pd.DatetimeIndex(out.index)
    except (TypeError, ValueError):
        return pd.DataFrame()
    if index.tz is not None:
        index = index.tz_localize(None)
    out.index = index.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for column in ("Open", "Close", "Dividends", "Stock Splits"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace(
                [math.inf, -math.inf], float("nan"))
    return out.dropna(subset=["Close"])


def _volatility_summary(history: pd.DataFrame) -> dict:
    frame = _clean_history(history)
    if len(frame) < 2:
        return {
            "price_through": None,
            "baseline_median_abs_daily_move_pct": None,
            "recent_median_abs_daily_move_pct": None,
            "volatility_20d_annualized_pct": None,
            "volatility_60d_annualized_pct": None,
        }
    returns = frame["Close"].pct_change().dropna()

    def annualized(window: int) -> float | None:
        sample = returns.tail(window)
        if len(sample) < min(10, window):
            return None
        return _round_or_none(sample.std(ddof=1) * math.sqrt(252) * 100)

    return {
        "price_through": frame.index[-1].date().isoformat(),
        "baseline_median_abs_daily_move_pct": _round_or_none(
            returns.abs().median() * 100),
        "recent_median_abs_daily_move_pct": _round_or_none(
            returns.tail(60).abs().median() * 100),
        "volatility_20d_annualized_pct": annualized(20),
        "volatility_60d_annualized_pct": annualized(60),
    }


def event_study(history: pd.DataFrame, event_dates: Sequence[object],
                as_of: date | None = None) -> dict:
    """イベント日と翌営業日の反応から、方向を保証しない感応度を計算する。

    公表セッションが不明な決算日にも対応するため、イベント当日と翌営業日の
    日次リターンのうち絶対値が大きい方を採用する。これは因果推定ではない。
    """
    cutoff = as_of or _us_market_date()
    frame = _clean_history(history)
    if not frame.empty:
        # 当日足は途中経過の可能性があるため、完了済み日足だけを使う。
        frame = frame[frame.index.date < cutoff]
    volatility = _volatility_summary(frame)
    empty = {
        "sample_size": 0,
        "window": "イベント当日・翌営業日のうち絶対値が大きい日次リターン",
        "median_abs_move_pct": None,
        "mean_abs_move_pct": None,
        "max_abs_move_pct": None,
        "median_signed_move_pct": None,
        "up_rate_pct": None,
        "sensitivity_ratio": None,
        **volatility,
        "method_note": "相関的なイベントスタディであり、イベントの因果効果ではありません。",
    }
    if len(frame) < 3:
        return empty

    returns = frame["Close"].pct_change() * 100
    index = frame.index
    moves: list[float] = []
    seen: set[date] = set()
    for raw_date in event_dates or ():
        event_date = _as_date(raw_date)
        if event_date is None or event_date >= cutoff or event_date in seen:
            continue
        seen.add(event_date)
        positions = [i for i, stamp in enumerate(index) if stamp.date() >= event_date]
        if not positions:
            continue
        first = positions[0]
        if (index[first].date() - event_date).days > 5:
            continue
        candidates = []
        for position in (first, first + 1):
            if position >= len(returns):
                continue
            value = _round_or_none(returns.iloc[position], 8)
            if value is not None:
                candidates.append(value)
        if candidates:
            moves.append(max(candidates, key=abs))
    if not moves:
        return empty

    series = pd.Series(moves, dtype=float)
    baseline = volatility["baseline_median_abs_daily_move_pct"]
    median_abs = float(series.abs().median())
    ratio = median_abs / baseline if baseline and baseline > 0 else None
    return {
        **empty,
        "sample_size": len(series),
        "median_abs_move_pct": _round_or_none(median_abs),
        "mean_abs_move_pct": _round_or_none(series.abs().mean()),
        "max_abs_move_pct": _round_or_none(series.abs().max()),
        "median_signed_move_pct": _round_or_none(series.median()),
        "up_rate_pct": _round_or_none((series > 0).mean() * 100, 1),
        "sensitivity_ratio": _round_or_none(ratio),
    }


def _impact(kind: str, study: Mapping[str, object]) -> tuple[str, int, str]:
    sample_size = int(study.get("sample_size") or 0)
    ratio = study.get("sensitivity_ratio")
    if sample_size >= 3 and ratio is not None:
        numeric = float(ratio)
        if numeric >= 2.0:
            return "HIGH", 85, "過去のイベント時変動が平常時の2倍以上"
        if numeric >= 1.35:
            return "HIGH", 72, "過去のイベント時変動が平常時を明確に上回る"
        if numeric >= 0.85:
            return "MEDIUM", 55, "過去のイベント時変動は平常時と同程度"
        return "LOW", 32, "過去のイベント時変動は平常時を下回る"

    qualitative = {
        "earnings": ("HIGH", 78), "fomc": ("HIGH", 70),
        "cpi": ("HIGH", 68), "employment": ("MEDIUM", 58),
        "merger": ("HIGH", 75), "regulatory": ("HIGH", 70),
        "guidance": ("HIGH", 68), "filing": ("MEDIUM", 55),
        "split": ("MEDIUM", 50), "dividend": ("LOW", 35),
        "company_news": ("MEDIUM", 50),
    }
    level, score = qualitative.get(kind, ("UNKNOWN", 0))
    return level, score, "十分な個別サンプルがないためイベント種別による定性的評価"


def _directional_bias(study: Mapping[str, object]) -> str:
    sample_size = int(study.get("sample_size") or 0)
    up_rate = study.get("up_rate_pct")
    if sample_size < 5 or up_rate is None:
        return "TWO_SIDED"
    if float(up_rate) >= 65:
        return "UP_HISTORY_BIASED"
    if float(up_rate) <= 35:
        return "DOWN_HISTORY_BIASED"
    return "TWO_SIDED"


def _confidence(study: Mapping[str, object], source_quality: str = "primary") -> tuple[int, str]:
    sample_size = int(study.get("sample_size") or 0)
    if sample_size >= 12:
        value = 82
    elif sample_size >= 8:
        value = 72
    elif sample_size >= 5:
        value = 62
    elif sample_size >= 3:
        value = 50
    elif sample_size:
        value = 38
    else:
        value = 28
    if source_quality == "secondary":
        value = max(20, value - 8)
    label = "HIGH" if value >= 70 else "MEDIUM" if value >= 45 else "LOW"
    return value, label


def _event_status(event_date: date | None, as_of: date) -> str:
    if event_date is None:
        return "UNKNOWN"
    if event_date >= as_of:
        return "UPCOMING"
    return "RECENT" if event_date >= as_of - timedelta(days=90) else "HISTORICAL"


def _event_id(kind: str, event_date: date | None, name: str) -> str:
    safe_name = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:30]
    suffix = safe_name or sha1(name.encode("utf-8")).hexdigest()[:10]
    return f"{kind}:{event_date.isoformat() if event_date else 'unknown'}:{suffix}"


def _make_event(*, kind: str, name: str, event_date: date | None,
                event_time_et: str | None, session: str, study: Mapping[str, object],
                evidence: Sequence[str], source: str, url: str | None,
                fetched_at: datetime, source_as_of: str | None = None,
                source_quality: str = "primary", as_of: date) -> dict:
    level, impact_score, impact_reason = _impact(kind, study)
    confidence, confidence_label = _confidence(study, source_quality)
    session_key = session if session in _SESSION_LABELS else "UNKNOWN"
    all_evidence = [str(item) for item in evidence if item]
    ratio = study.get("sensitivity_ratio")
    if study.get("sample_size"):
        all_evidence.append(
            f"過去{study['sample_size']}回の中央値±{study['median_abs_move_pct']}%、"
            f"平常時比{ratio}倍" if ratio is not None else
            f"過去{study['sample_size']}回の中央値±{study['median_abs_move_pct']}%"
        )
    all_evidence.append(impact_reason)
    return {
        "event_id": _event_id(kind, event_date, name),
        "kind": kind,
        "name": name,
        "event_date": event_date.isoformat() if event_date else None,
        "event_time_et": event_time_et,
        "status": _event_status(event_date, as_of),
        "session": session_key,
        "session_label": _SESSION_LABELS[session_key],
        "impact_level": level,
        "impact_score": impact_score,
        "directional_bias": _directional_bias(study),
        "scenarios": dict(_SCENARIOS.get(kind, _SCENARIOS["company_news"])),
        "historical_sensitivity": dict(study),
        "confidence": confidence,
        "confidence_label": confidence_label,
        "evidence": all_evidence,
        "source": source,
        "url": url,
        "freshness": {
            "analysis_fetched_at": fetched_at.isoformat(),
            "source_as_of": source_as_of,
            "price_through": study.get("price_through"),
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
        },
        "score_effect": NO_TRADE_SCORE_EFFECT,
        "automatic_trade_score": False,
        "disclaimer": DISCLAIMER,
    }


def _infer_news_session(title: str) -> tuple[str, str | None]:
    text = title.casefold()
    if any(word in text for word in ("before market", "pre-market", "premarket", "寄り前")):
        return "PRE_MARKET", None
    if any(word in text for word in ("after hours", "after-hours", "after market",
                                     "after the bell", "引け後", "時間外")):
        return "AFTER_MARKET", None
    if any(word in text for word in ("overnight", "24-hour", "24 hour", "夜間")):
        return "OVERNIGHT", None
    return "UNKNOWN", None


def _earnings_session(analyst: Mapping[str, object]) -> tuple[str, str | None]:
    raw = str(analyst.get("earnings_session") or analyst.get("pub_type") or "").casefold()
    if "before" in raw or raw in {"bmo", "pre", "pre_market"}:
        return "PRE_MARKET", None
    if "after" in raw or raw in {"amc", "post", "after_market"}:
        return "AFTER_MARKET", None
    if "during" in raw or "regular" in raw:
        return "REGULAR", None

    value = analyst.get("earnings_date")
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return "DATE_ONLY", None
    if pd.isna(stamp) or (stamp.hour == 0 and stamp.minute == 0):
        return "DATE_ONLY", None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York")
    minutes = stamp.hour * 60 + stamp.minute
    event_time = f"{stamp.hour:02d}:{stamp.minute:02d}"
    if minutes < 9 * 60 + 30:
        return "PRE_MARKET", event_time
    if minutes < 16 * 60:
        return "REGULAR", event_time
    if minutes < 20 * 60:
        return "AFTER_MARKET", event_time
    return "OVERNIGHT", event_time


def classify_news_event(item: Mapping[str, object]) -> str | None:
    """重要イベントに該当するニュース種別を返す。通常の記事・論評は除外。"""
    text = f"{item.get('title') or ''} {item.get('summary') or ''}".casefold()
    for kind, keywords in _NEWS_PATTERNS:
        if any(_contains_keyword(text, keyword) for keyword in keywords):
            return kind
    return None


def _contains_keyword(text: str, keyword: str) -> bool:
    if re.fullmatch(r"[a-z0-9][a-z0-9 .&-]*", keyword):
        pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
        return re.search(pattern, text) is not None
    return keyword in text


def _recent_action_events(history: pd.DataFrame, as_of: date) -> list[dict]:
    frame = _clean_history(history)
    rows = []
    for column, kind, name in (
        ("Dividends", "dividend", "配当・権利落ち"),
        ("Stock Splits", "split", "株式分割・併合"),
    ):
        if column not in frame.columns:
            continue
        actions = frame[pd.to_numeric(frame[column], errors="coerce").fillna(0) != 0]
        if actions.empty:
            continue
        recent = actions[actions.index.date >= as_of - timedelta(days=90)]
        if recent.empty:
            continue
        stamp = recent.index[-1]
        value = float(recent.loc[stamp, column])
        rows.append({
            "kind": kind, "name": name, "event_date": stamp.date(),
            "value": value,
            "all_dates": [value.date() for value in actions.index],
        })
    return rows


def _valid_macro_events(events: Iterable[Mapping[str, object]]) -> list[dict]:
    out = []
    for item in events:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("kind") or "").strip().casefold()
        event_date = _as_date(item.get("event_date"))
        if kind not in {"fomc", "cpi", "employment"} or event_date is None:
            continue
        session = str(item.get("session") or "UNKNOWN").upper()
        out.append({
            "kind": kind,
            "name": str(item.get("name") or kind.upper()),
            "event_date": event_date,
            "event_time_et": str(item.get("event_time_et") or "") or None,
            "session": session if session in _SESSION_LABELS else "UNKNOWN",
            "source": str(item.get("source") or ""),
            "url": str(item.get("url") or "") or None,
            "verified_at": str(item.get("verified_at") or "") or None,
        })
    return sorted(out, key=lambda row: (row["event_date"], row["kind"]))


def build_event_intelligence(
    ticker: str,
    history: pd.DataFrame | None,
    info: Mapping[str, object] | None = None,
    analyst: Mapping[str, object] | None = None,
    earnings_dates: Sequence[object] = (),
    news: Sequence[Mapping[str, object]] = (),
    macro_events: Iterable[Mapping[str, object]] | None = None,
    *,
    as_of: date | None = None,
    fetched_at: datetime | None = None,
    horizon_days: int = 120,
    max_news_events: int = 5,
) -> dict:
    """取得済みデータから、イベント影響レポートを構築する純粋な公開API。

    ``macro_events`` を指定すると内蔵日程を完全に差し替えられる。UI側で既に
    info/history/newsを取得済みなら、この関数を使うことで追加通信を避けられる。
    """
    symbol = normalize_ticker(ticker)
    current_date = as_of or _us_market_date()
    now = fetched_at or _utc_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    horizon_days = max(1, min(int(horizon_days), 366))
    info = dict(info or {})
    analyst = dict(analyst or {})
    frame = _clean_history(history)
    if not frame.empty:
        frame = frame[frame.index.date < current_date]
    volatility = _volatility_summary(frame)
    company = str(info.get("name") or symbol)
    events: list[dict] = []
    warnings: list[str] = []

    earnings_study = event_study(frame, earnings_dates, current_date)
    next_earnings = _as_date(analyst.get("earnings_date"))
    if (next_earnings is not None
            and current_date <= next_earnings <= current_date + timedelta(days=horizon_days)):
        eps = analyst.get("eps_estimate")
        evidence = ["Yahoo Finance calendarの次回決算予定日"]
        if _round_or_none(eps) is not None:
            evidence.append(f"市場予想EPS: {_round_or_none(eps)}")
        earnings_session, earnings_time = _earnings_session(analyst)
        events.append(_make_event(
            kind="earnings", name=f"{company} 決算発表", event_date=next_earnings,
            event_time_et=earnings_time, session=earnings_session, study=earnings_study,
            evidence=evidence, source="Yahoo Finance calendar", url=None,
            fetched_at=now, source_as_of=now.isoformat(), source_quality="secondary",
            as_of=current_date,
        ))

    macros = _valid_macro_events(
        default_macro_events() if macro_events is None else macro_events)
    for kind in ("fomc", "cpi", "employment"):
        kind_rows = [row for row in macros if row["kind"] == kind]
        history_dates = [row["event_date"] for row in kind_rows
                         if row["event_date"] < current_date]
        upcoming = next((row for row in kind_rows
                         if current_date <= row["event_date"]
                         <= current_date + timedelta(days=horizon_days)), None)
        if upcoming is None:
            continue
        study = event_study(frame, history_dates, current_date)
        events.append(_make_event(
            kind=kind, name=upcoming["name"], event_date=upcoming["event_date"],
            event_time_et=upcoming["event_time_et"], session=upcoming["session"],
            study=study,
            evidence=[f"{upcoming['source']}の公式公表日程",
                      "発表値と市場予想の差で反応が変わります"],
            source=upcoming["source"], url=upcoming["url"], fetched_at=now,
            source_as_of=upcoming["verified_at"], as_of=current_date,
        ))

    for action in _recent_action_events(frame, current_date):
        study = event_study(frame, action["all_dates"], current_date + timedelta(days=1))
        unit = "1株当たり" if action["kind"] == "dividend" else "分割比率"
        events.append(_make_event(
            kind=action["kind"], name=f"{company} {action['name']}",
            event_date=action["event_date"], event_time_et=None,
            session="REGULAR_OPEN", study=study,
            evidence=[f"Yahoo Finance corporate actions: {unit} {action['value']:g}"],
            source="Yahoo Finance corporate actions", url=None, fetched_at=now,
            source_as_of=action["event_date"].isoformat(), source_quality="secondary",
            as_of=current_date,
        ))

    seen_news: set[str] = set()
    news_count = 0
    news_limit = max(0, int(max_news_events))
    for item in news or ():
        if news_count >= news_limit:
            break
        if not isinstance(item, Mapping):
            continue
        kind = classify_news_event(item)
        title = str(item.get("title") or "").strip()
        if kind is None or not title:
            continue
        url = str(item.get("url") or "")
        dedupe_key = re.sub(r"\s+", " ", title.casefold()).strip()
        if dedupe_key in seen_news:
            continue
        seen_news.add(dedupe_key)
        event_date = _as_date(item.get("pub_date"))
        if event_date and event_date < current_date - timedelta(days=30):
            continue
        session, event_time = _infer_news_session(title)
        study = {**event_study(frame, (), current_date)}
        evidence = [title]
        summary = str(item.get("summary") or "").strip()
        if summary:
            evidence.append(summary[:240])
        provider = str(item.get("provider") or "ニュース")
        news_event = _make_event(
            kind=kind, name=title, event_date=event_date, event_time_et=event_time,
            session=session, study=study, evidence=evidence, source=provider,
            url=url or None, fetched_at=now,
            source_as_of=str(item.get("pub_date") or "") or None,
            source_quality="secondary", as_of=current_date,
        )
        # ニュースは取得時点ですでに公表済み。同日記事を「今後の予定」へ
        # 混ぜると、予定イベントと既知材料を取り違えるため明示的に分ける。
        news_event["status"] = "RECENT" if event_date is not None else "UNKNOWN"
        events.append(news_event)
        news_count += 1

    def sort_key(item: Mapping[str, object]):
        parsed = _as_date(item.get("event_date"))
        ordinal = parsed.toordinal() if parsed else date.max.toordinal()
        if item.get("status") == "UPCOMING":
            return 0, ordinal, str(item.get("name") or "")
        # 過去イベントは新しい順。日付不明は最後に置く。
        return 1, -ordinal if parsed else 0, str(item.get("name") or "")

    events.sort(key=sort_key)

    if frame.empty:
        warnings.append("株価履歴を取得できず、過去ボラティリティと感応度は未算出です。")
    if macro_events is None and current_date > date(2026, 12, 31):
        warnings.append("内蔵マクロ日程の対象期間外です。最新の公式日程を指定してください。")
    return {
        "ticker": symbol,
        "company": company,
        "as_of": current_date.isoformat(),
        "events": events,
        "event_count": len(events),
        "volatility": volatility,
        "warnings": warnings,
        "status": "ok" if events else "empty",
        "score_effect": NO_TRADE_SCORE_EFFECT,
        "automatic_trade_score": False,
        "disclaimer": DISCLAIMER,
        "meta": {
            "fetched_at": now.isoformat(),
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
            "calendar_verified_at": CALENDAR_VERIFIED_AT,
            "moomoo_history_quota_used": False,
            "read_only": True,
        },
    }


def _safe_source(name: str, function, default, errors: dict[str, str]):
    try:
        return function()
    except Exception as exc:
        errors[name] = str(exc) or exc.__class__.__name__
        return default


def _fetch_event_intelligence_uncached(
    ticker: str,
    as_of_date: str | None = None,
    include_news: bool = True,
    horizon_days: int = 120,
) -> dict:
    """既存取得関数を部分失敗に耐える形で遅延実行する。"""
    symbol = normalize_ticker(ticker)
    current_date = _as_date(as_of_date) if as_of_date else _us_market_date()
    if current_date is None:
        raise ValueError("as_of_dateはYYYY-MM-DD形式で指定してください")
    errors: dict[str, str] = {}
    info = _safe_source(
        "company_info", lambda: data_fetcher.fetch_info(symbol), {}, errors)
    analyst = _safe_source(
        "calendar", lambda: data_fetcher.fetch_analyst(symbol), {}, errors)
    earnings_dates = _safe_source(
        "earnings_history", lambda: data_fetcher.fetch_earnings_history(symbol), [], errors)
    # fetch_historyはYahoo Finance専用。fetch_chart_historyを使わないため、moomooの
    # 新規過去K線枠を消費しない。
    history = _safe_source(
        "price_history", lambda: data_fetcher.fetch_history(symbol, "2y", "1d"),
        pd.DataFrame(), errors)

    news: list[dict] = []
    if include_news:
        yahoo_news = _safe_source(
            "yahoo_news",
            lambda: news_fetcher.fetch_news(symbol, info.get("name")), [], errors)
        sec_news = _safe_source(
            "sec_filings", lambda: news_fetcher.fetch_sec_filings(symbol), [], errors)
        news = list(yahoo_news or []) + list(sec_news or [])

    report = build_event_intelligence(
        symbol, history, info, analyst, earnings_dates, news,
        as_of=current_date, fetched_at=_utc_now(), horizon_days=horizon_days,
    )
    report["source_status"] = {
        name: ("failed" if name in errors else "ok")
        for name in ("company_info", "calendar", "earnings_history", "price_history",
                     *(("yahoo_news", "sec_filings") if include_news else ()))
    }
    report["errors"] = errors
    if errors:
        report["warnings"].append(
            "一部データを取得できませんでした: " + ", ".join(sorted(errors)))
        report["status"] = "partial" if report["events"] else "unavailable"
    return report


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_event_intelligence(
    ticker: str,
    as_of_date: str | None = None,
    include_news: bool = True,
    horizon_days: int = 120,
) -> dict:
    """イベント影響データを明示的に取得するキャッシュ付き公開API。

    Streamlit UIではタブ選択だけで呼ばず、「読み込む/更新」操作から呼ぶことで
    ネットワークを完全に遅延ロードできる。キャッシュTTLは15分。
    """
    return _fetch_event_intelligence_uncached(
        ticker, as_of_date=as_of_date, include_news=include_news,
        horizon_days=horizon_days,
    )

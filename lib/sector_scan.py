"""セクターの強弱ランキングと、セクター内の銘柄スキャン(順張り向け)。

market_mood のセクター判定は「売られすぎたセクターを拾う」逆張りの見方だが、
デイトレ・スイングでは「いま強いセクターに乗る」順張りの見方も要る。
このモジュールは後者を担当する。両者は目的が違うので別に持つ。

セクタースコア(0〜100)の内訳:
    対SPY相対力(5日)   25点   直近1週間でSPYにどれだけ勝っているか
    対SPY相対力(20日)  20点   1ヶ月の勝ち負け
    当日の値動き        20点   今日SPYにどれだけ勝っているか
    トレンド            20点   EMA20>EMA50 と 200日線の上かどうか
    出来高              15点   直近の出来高が平常時より増えているか

銘柄スコアも同じ考え方で、所属セクターに対する相対力で採点する。
「上がりやすい」の根拠は過去の相対力の継続性であって、将来の保証ではない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lib import data_fetcher

BENCHMARK = "SPY"

# 検証結果(重要):
# S&P500・486銘柄 / 2013-2018 の日足 47,976サンプルで、このスコアの
# 5分位別に「その後5日・20日の市場平均に対する超過リターン」を測った。
#   5日 : 最弱 -0.027% → 最強 +0.029%(差 +0.056pt, t=1.15, p=0.25, 単調でない)
#   20日: 最弱 -0.001% → 最強 -0.053%(差 -0.052pt, t=-0.54, p=0.59)
#   期間を前半・後半で割ると後半は +0.002pt(実質ゼロ)
# つまり「スコアが高い=今後上がりやすい」という関係は確認できなかった。
# このスコアは "いま相対的に強いのはどれか" を並べる記述的な指標であって、
# 将来の値上がりを予測するものではない。UIでも必ずそう表示すること。
EDGE_NOTE = ("このスコアは『いま強い順』の記述であって、値上がりの予測ではありません。"
             "S&P500・2013-2018の47,976サンプルで検証したところ、スコア上位の"
             "その後5日の超過リターンは +0.06pt(p=0.25)、20日では −0.05pt で、"
             "統計的に意味のある差は確認できませんでした。")

# セクターETFと代表銘柄。時価総額上位を中心に、値動きの大きい銘柄も混ぜている。
SECTORS = [
    {"etf": "XLK", "jp": "テクノロジー", "members": [
        "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "CRM", "ORCL", "ADBE",
        "QCOM", "INTC", "MU", "PANW"]},
    {"etf": "XLC", "jp": "通信サービス", "members": [
        "GOOGL", "META", "NFLX", "DIS", "TMUS", "CMCSA", "EA", "WBD"]},
    {"etf": "XLY", "jp": "一般消費財", "members": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "ABNB"]},
    {"etf": "XLP", "jp": "生活必需品", "members": [
        "WMT", "COST", "PG", "KO", "PEP", "PM", "MDLZ", "CL"]},
    {"etf": "XLE", "jp": "エネルギー", "members": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY"]},
    {"etf": "XLF", "jp": "金融", "members": [
        "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "AXP"]},
    {"etf": "XLV", "jp": "ヘルスケア", "members": [
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "PFE", "AMGN", "ISRG"]},
    {"etf": "XLI", "jp": "資本財", "members": [
        "GE", "CAT", "RTX", "HON", "UNP", "BA", "DE", "LMT", "UPS"]},
    {"etf": "XLB", "jp": "素材", "members": [
        "LIN", "SHW", "FCX", "APD", "ECL", "NEM", "DOW", "NUE"]},
    {"etf": "XLRE", "jp": "不動産", "members": [
        "PLD", "AMT", "EQIX", "WELL", "SPG", "O", "PSA", "CCI"]},
    {"etf": "XLU", "jp": "公益", "members": [
        "NEE", "SO", "DUK", "AEP", "SRE", "D", "EXC", "XEL"]},
]

SECTOR_BY_ETF = {s["etf"]: s for s in SECTORS}

# 強さの区分(スコア下限, ラベル, 色)。
# 各要素は中立で満点の半分が入るので、合計の中立点は50。50を軸に対称にする。
STRENGTH = [
    (72, "非常に強い", "up"),
    (60, "強い", "up"),
    (53, "やや強い", "up"),
    (47, "中立", "flat"),
    (40, "やや弱い", "down"),
    (28, "弱い", "down"),
    (0, "非常に弱い", "down"),
]


def _label(score: float) -> tuple[str, str]:
    return next((lb, tone) for th, lb, tone in STRENGTH if score >= th)


def _pct(series: pd.Series, days: int) -> float | None:
    """days営業日前からの変化率(%)。"""
    if series is None or len(series) <= days:
        return None
    prior = float(series.iloc[-days - 1])
    if not prior:
        return None
    return (float(series.iloc[-1]) / prior - 1) * 100


def _scale(value: float | None, span: float, points: float) -> float:
    """value を ±span で ±points に線形変換して 0〜points に載せ替える。"""
    if value is None or not np.isfinite(value):
        return points / 2
    return float(np.clip(value / span, -1, 1) + 1) / 2 * points


def _trend_points(close: pd.Series, points: float) -> tuple[float, str]:
    if len(close) < 60:
        return points / 2, "データ不足"
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    above_fast = float(ema20.iloc[-1]) > float(ema50.iloc[-1])
    sma200 = close.rolling(200).mean()
    above_long = (len(close) >= 200 and pd.notna(sma200.iloc[-1])
                  and float(close.iloc[-1]) > float(sma200.iloc[-1]))
    got = (points * 0.6 if above_fast else 0.0) + (points * 0.4 if above_long else 0.0)
    note = ("EMA20>EMA50" if above_fast else "EMA20<EMA50")
    note += " / 200日線" + ("上" if above_long else "下")
    return got, note


def _volume_points(volume: pd.Series, points: float) -> tuple[float, float]:
    if volume is None or len(volume) < 25 or volume.tail(20).sum() <= 0:
        return points / 2, 1.0
    recent = float(volume.tail(5).mean())
    base = float(volume.tail(20).mean())
    ratio = recent / base if base else 1.0
    return _scale((ratio - 1) * 100, 40, points), ratio


def _relative(close: pd.Series, bench: pd.Series, days: int) -> float | None:
    a, b = _pct(close, days), _pct(bench, days)
    if a is None or b is None:
        return None
    return a - b


def _score_frame(close: pd.Series, volume: pd.Series,
                 bench: pd.Series) -> dict | None:
    """ETF・個別株に共通の採点。ベンチマークに対する相対力で測る。"""
    if close is None or len(close) < 30:
        return None
    rel5 = _relative(close, bench, 5)
    rel20 = _relative(close, bench, 20)
    rel1 = _relative(close, bench, 1)
    trend_pts, trend_note = _trend_points(close, 20)
    vol_pts, vol_ratio = _volume_points(volume, 15)
    parts = {
        "相対力(5日)": _scale(rel5, 5.0, 25),
        "相対力(20日)": _scale(rel20, 10.0, 20),
        "当日": _scale(rel1, 2.0, 20),
        "トレンド": trend_pts,
        "出来高": vol_pts,
    }
    return {
        "score": round(float(sum(parts.values())), 1),
        "parts": {k: round(v, 1) for k, v in parts.items()},
        "rel1": rel1, "rel5": rel5, "rel20": rel20,
        "ret1": _pct(close, 1), "ret5": _pct(close, 5), "ret20": _pct(close, 20),
        "trend_note": trend_note,
        "vol_ratio": vol_ratio,
        "price": float(close.iloc[-1]),
    }


def _closes(frames: dict, ticker: str) -> tuple[pd.Series, pd.Series] | None:
    df = frames.get(ticker)
    if df is None or df.empty or "Close" not in df:
        return None
    close = df["Close"].dropna()
    volume = df["Volume"] if "Volume" in df else pd.Series(dtype=float)
    return (close, volume) if len(close) >= 30 else None


@st.cache_data(ttl=900, show_spinner="セクターの強弱を計算中...")
def sector_ranking() -> list[dict]:
    """11セクターETFを強い順に並べる。"""
    tickers = tuple([BENCHMARK] + [s["etf"] for s in SECTORS])
    try:
        frames = data_fetcher.fetch_daily_batch(tickers, period="1y")
    except data_fetcher.FetchError:
        return []
    bench = _closes(frames, BENCHMARK)
    if bench is None:
        return []
    bench_close = bench[0]

    rows = []
    for sector in SECTORS:
        pair = _closes(frames, sector["etf"])
        if pair is None:
            continue
        scored = _score_frame(pair[0], pair[1], bench_close)
        if not scored:
            continue
        label, tone = _label(scored["score"])
        rows.append({"etf": sector["etf"], "jp": sector["jp"],
                     "label": label, "tone": tone, **scored})
    return sorted(rows, key=lambda r: -r["score"])


@st.cache_data(ttl=900, show_spinner="銘柄をスキャン中...")
def sector_members(etf: str, limit: int = 6) -> list[dict]:
    """1セクター内の銘柄を強い順に並べる。

    採点の基準はセクターETF。「セクターの中で相対的に強い銘柄」を出すので、
    セクター自体が弱ければ、その中の1位も市場全体では弱いことがある。
    """
    sector = SECTOR_BY_ETF.get(etf)
    if not sector:
        return []
    tickers = tuple([etf] + sector["members"])
    try:
        frames = data_fetcher.fetch_daily_batch(tickers, period="1y")
    except data_fetcher.FetchError:
        return []
    base = _closes(frames, etf)
    if base is None:
        return []
    base_close = base[0]

    rows = []
    for ticker in sector["members"]:
        pair = _closes(frames, ticker)
        if pair is None:
            continue
        scored = _score_frame(pair[0], pair[1], base_close)
        if not scored:
            continue
        label, tone = _label(scored["score"])
        rows.append({"ticker": ticker, "label": label, "tone": tone,
                     "etf": etf, "sector_jp": sector["jp"], **scored})
    rows.sort(key=lambda r: -r["score"])
    return rows[:limit]


def top_picks(ranking: list[dict], sectors: int = 3,
              per_sector: int = 3) -> list[dict]:
    """上位セクターから、それぞれ上位銘柄を拾ってまとめる。"""
    picks = []
    for sector in ranking[:sectors]:
        for member in sector_members(sector["etf"], per_sector):
            picks.append({**member, "sector_score": sector["score"],
                          "sector_label": sector["label"]})
    return picks

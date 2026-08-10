"""当日のトレンド判定(デイトレ向け)。

分足から「今日は上か、下か、レンジか」を、根拠を並べて示す。
判定は5つの観測を足し合わせたスコアで、隠れたモデルは使わない。
各観測が何点入ったかをそのまま返すので、UI側で内訳を出せる。

    VWAP        価格がVWAPの上か下か(日中の平均取得単価との位置関係)
    EMA         短期EMA(9)と中期EMA(21)の並びと傾き
    オープニングレンジ  寄り後30分の高安を抜けたか(ORB)
    高値安値     直近の押し安値・戻り高値を切り上げているか
    出来高       上昇バーと下降バーのどちらに出来高が乗っているか

スコアは −100(強い下降)〜 +100(強い上昇)。閾値はUIの表示を分けるための
区切りで、勝率や期待値を意味しない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lib import sessions

OPENING_RANGE_MINUTES = 30

# 判定ラベル(スコア下限, ラベル, 色, 短い説明)。
# shortはメトリクス表示用。長いラベルは幅の狭い列で省略されてしまう。
VERDICTS = [
    (55, "強い上昇トレンド", "up",
     "押し目待ちの買いが機能しやすい地合い。戻り売りは分が悪い"),
    (20, "上昇トレンド", "up", "上向き。ただし勢いは限定的"),
    (-20, "レンジ・方向感なし", "flat",
     "トレンドフォローは空振りしやすい。レンジ上下限での逆張り向き"),
    (-55, "下降トレンド", "down", "下向き。戻り売りが機能しやすい"),
    (-101, "強い下降トレンド", "down",
     "戻り売りが機能しやすい地合い。押し目買いは分が悪い"),
]
SHORT_LABELS = {
    "強い上昇トレンド": "強い上昇",
    "上昇トレンド": "上昇",
    "レンジ・方向感なし": "レンジ",
    "下降トレンド": "下降",
    "強い下降トレンド": "強い下降",
}


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()


def vwap(df: pd.DataFrame) -> pd.Series:
    """当日の出来高加重平均価格(取引日ごとにリセット)。"""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0) if "Volume" in df else pd.Series(1.0, index=df.index)
    day = sessions.trading_day(df.index)
    pv = (typical * vol).groupby(day.to_numpy()).cumsum()
    cv = vol.groupby(day.to_numpy()).cumsum()
    return (pv / cv.replace(0, np.nan)).ffill()


def opening_range(df: pd.DataFrame,
                  minutes: int = OPENING_RANGE_MINUTES) -> dict | None:
    """立会開始からminutes分の高値・安値(オープニングレンジ)。"""
    regular = sessions.session_slice(sessions.latest_day_slice(df),
                                     [sessions.REGULAR])
    if regular is None or regular.empty:
        return None
    start = sessions.to_et(regular.index)[0]
    cutoff = start + pd.Timedelta(minutes=minutes)
    window = regular[sessions.to_et(regular.index) < cutoff]
    if window.empty:
        return None
    after = regular[sessions.to_et(regular.index) >= cutoff]
    high, low = float(window["High"].max()), float(window["Low"].min())
    broke_up = bool(not after.empty and after["High"].max() > high)
    broke_down = bool(not after.empty and after["Low"].min() < low)
    return {
        "high": high,
        "low": low,
        "minutes": minutes,
        "bars": int(len(window)),
        "complete": not after.empty,
        "broke_up": broke_up,
        "broke_down": broke_down,
        "open": float(window["Open"].iloc[0]),
    }


def _swing_structure(close: np.ndarray, lookback: int = 20) -> float:
    """直近の高値・安値の切り上げ/切り下げを −1〜+1 で表す。"""
    if len(close) < lookback * 2:
        return 0.0
    recent, prior = close[-lookback:], close[-lookback * 2:-lookback]
    # 高値・安値それぞれの切り上げ/切り下げを数え、差をとる。
    # 「高値切り上げ+安値切り下げ」(レンジ拡大)のような打ち消し合う形は0になる。
    up = int(recent.max() > prior.max()) + int(recent.min() > prior.min())
    down = int(recent.max() < prior.max()) + int(recent.min() < prior.min())
    return (up - down) / 2


def _volume_bias(df: pd.DataFrame) -> float:
    """上昇バーと下降バーのどちらに出来高が乗っているかを −1〜+1 で表す。"""
    if "Volume" not in df or df["Volume"].sum() <= 0:
        return 0.0
    up = df["Close"] >= df["Open"]
    up_vol = float(df.loc[up, "Volume"].sum())
    down_vol = float(df.loc[~up, "Volume"].sum())
    total = up_vol + down_vol
    if total <= 0:
        return 0.0
    return (up_vol - down_vol) / total


def analyze(intraday: pd.DataFrame, prev_close: float | None = None) -> dict | None:
    """当日のトレンド判定と、その根拠の内訳を返す。

    intraday は分足(時間外を含んでいてもよい)。判定は立会のバーで行い、
    時間外はセッション別の騰落としてのみ扱う。
    """
    if intraday is None or intraday.empty:
        return None
    day = sessions.latest_day_slice(intraday)
    # 20:00 ETを回った直後は新しい取引日のバーがまだ数本しかない。
    # その場合は直前の取引日を対象にして、stale=Trueで「いつの分か」を伝える。
    stale = False
    if len(day) < 6:
        days = sessions.trading_day(intraday.index)
        previous = days[days < days.iloc[-1]]
        if previous.empty:
            return None
        day = intraday[(days == previous.iloc[-1]).to_numpy()]
        stale = True
    if day.empty:
        return None
    regular = sessions.session_slice(day, [sessions.REGULAR])
    # 立会がまだ始まっていない時間帯は、時間外のバーで代用して傾きだけ見る
    target = regular if len(regular) >= 6 else day
    if len(target) < 6:
        return None

    close = target["Close"].to_numpy(float)
    price = float(close[-1])
    vw = vwap(target)
    vwap_now = float(vw.iloc[-1]) if not vw.empty else price
    ema9, ema21 = _ema(close, 9), _ema(close, 21)

    parts = []

    # ① VWAP との位置関係(±25)
    vwap_dev = (price / vwap_now - 1) * 100 if vwap_now else 0.0
    vwap_score = float(np.clip(vwap_dev / 0.6, -1, 1) * 25)
    parts.append({
        "name": "VWAPとの位置",
        "score": vwap_score,
        "max": 25,
        "value": f"{vwap_dev:+.2f}%",
        "note": ("VWAPの上=買い方優勢" if vwap_dev > 0.05 else
                 "VWAPの下=売り方優勢" if vwap_dev < -0.05 else "VWAP付近で拮抗"),
    })

    # ② EMAの並びと傾き(±25)
    spread = (ema9[-1] / ema21[-1] - 1) * 100 if ema21[-1] else 0.0
    slope = (ema9[-1] / ema9[max(0, len(ema9) - 7)] - 1) * 100 if len(ema9) > 7 else 0.0
    ema_score = float(np.clip(spread / 0.4, -1, 1) * 15
                      + np.clip(slope / 0.5, -1, 1) * 10)
    parts.append({
        "name": "EMA9 / EMA21",
        "score": ema_score,
        "max": 25,
        "value": f"乖離 {spread:+.2f}% / 傾き {slope:+.2f}%",
        "note": ("短期が上=上昇の並び" if spread > 0 else
                 "短期が下=下降の並び" if spread < 0 else "並びは中立"),
    })

    # ③ オープニングレンジのブレイク(±20)
    orb = opening_range(day)
    if orb and orb["complete"]:
        if orb["broke_up"] and not orb["broke_down"]:
            orb_score, orb_note = 20.0, "寄り30分の高値を上抜け"
        elif orb["broke_down"] and not orb["broke_up"]:
            orb_score, orb_note = -20.0, "寄り30分の安値を下抜け"
        elif orb["broke_up"] and orb["broke_down"]:
            orb_score, orb_note = 0.0, "上下とも抜けており方向が定まらない"
        else:
            orb_score, orb_note = 0.0, "寄り30分のレンジ内で推移"
        orb_value = f"${orb['low']:,.2f}〜${orb['high']:,.2f}"
    else:
        orb_score, orb_note = 0.0, "寄り30分がまだ確定していません"
        orb_value = "—"
    parts.append({"name": "オープニングレンジ", "score": orb_score, "max": 20,
                  "value": orb_value, "note": orb_note})

    # ④ 高値・安値の切り上げ / 切り下げ(±15)
    struct = _swing_structure(close)
    parts.append({
        "name": "高値・安値の推移",
        "score": struct * 15,
        "max": 15,
        "value": {1.0: "切り上げ", 0.5: "やや切り上げ", 0.0: "横ばい",
                  -0.5: "やや切り下げ", -1.0: "切り下げ"}[struct],
        "note": ("高値・安値とも切り上げ" if struct == 1 else
                 "高値・安値とも切り下げ" if struct == -1 else
                 "明確な切り上げ / 切り下げなし"),
    })

    # ⑤ 出来高がどちらに乗っているか(±15)
    bias = _volume_bias(target)
    parts.append({
        "name": "出来高の偏り",
        "score": bias * 15,
        "max": 15,
        "value": f"上昇バー {50 + bias * 50:.0f}%",
        "note": ("上昇バーに出来高が乗っている" if bias > 0.1 else
                 "下降バーに出来高が乗っている" if bias < -0.1 else
                 "出来高の偏りは小さい"),
    })

    score = float(sum(p["score"] for p in parts))
    label, tone, advice = next((lb, tn, ad) for th, lb, tn, ad in VERDICTS
                               if score >= th)

    day_open = float(day["Open"].iloc[0])
    reg_open = float(regular["Open"].iloc[0]) if not regular.empty else day_open
    return {
        "score": round(score, 1),
        "label": label,
        "short": SHORT_LABELS.get(label, label),
        "tone": tone,
        "advice": advice,
        "parts": parts,
        "price": price,
        "vwap": vwap_now,
        "vwap_dev": vwap_dev,
        "opening_range": orb,
        "session": str(sessions.classify([target.index[-1]]).iloc[0]),
        "bars": int(len(target)),
        "stale": stale,
        "regular_bars": int(len(regular)),
        "regular_open": reg_open,
        "from_open_pct": (price / reg_open - 1) * 100 if reg_open else 0.0,
        "from_prev_pct": ((price / prev_close - 1) * 100
                          if prev_close else None),
        "sessions": sessions.session_summary(day, prev_close),
        "day_high": float(day["High"].max()),
        "day_low": float(day["Low"].min()),
        "as_of": day.index[-1],
    }


def market_trend(index_frames: dict[str, pd.DataFrame],
                 prev_closes: dict[str, float] | None = None) -> dict | None:
    """主要指数の当日トレンドをまとめて「市場全体の向き」を出す。

    index_frames: {表示名: 分足DataFrame}
    個別銘柄と同じ判定を各指数にかけ、平均スコアで市場の向きを決める。
    """
    if not index_frames:
        return None
    prev_closes = prev_closes or {}
    rows = []
    for name, frame in index_frames.items():
        res = analyze(frame, prev_closes.get(name))
        if res:
            rows.append({"name": name, **res})
    if not rows:
        return None
    score = float(np.mean([r["score"] for r in rows]))
    label, tone, advice = next((lb, tn, ad) for th, lb, tn, ad in VERDICTS
                               if score >= th)
    agree = sum(1 for r in rows if (r["score"] > 0) == (score > 0))
    return {
        "score": round(score, 1),
        "label": label,
        "short": SHORT_LABELS.get(label, label),
        "tone": tone,
        "advice": advice,
        "members": rows,
        "agreement": f"{agree}/{len(rows)}",
        "aligned": agree == len(rows),
    }

"""寄付(ギャップ)の実測統計と、そこから引く条件付きの見通し。

やっていることは予測モデルではなく **過去の頻度集計** です。
「今と同じくらいのギャップが過去に何回あって、そのうち何回どうなったか」を
そのまま出します。将来を当てるものではないことを、UI側でも明記すること。

用語:
    ギャップ率   当日始値 ÷ 前日終値 − 1
    窓埋め       ギャップアップなら当日安値が前日終値まで戻ること(ダウンは逆)
    寄り天/寄り底 始値が当日の高値/安値に近く、そこから逆行して引けること
    寄り後リターン 当日終値 ÷ 当日始値 − 1(寄付で入って引けで返した場合の値動き)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lib import sessions

# 条件付き統計に使う「似たギャップ」の幅。狭すぎるとサンプルが枯れる。
BUCKET_HALF_WIDTH = 0.5     # ±0.5%を「似たギャップ」とみなす
MIN_SAMPLES = 8             # これ未満は統計として出さない
WIDEN_STEPS = (0.5, 0.75, 1.0, 1.5, 2.0)   # サンプルが足りなければ順に広げる

# ギャップ率の区分(表示用)
BUCKETS = [
    (-np.inf, -2.0, "大きく下窓(−2%超)"),
    (-2.0, -1.0, "下窓(−1〜−2%)"),
    (-1.0, -0.3, "小さい下窓(−0.3〜−1%)"),
    (-0.3, 0.3, "ほぼ窓なし(±0.3%)"),
    (0.3, 1.0, "小さい上窓(+0.3〜+1%)"),
    (1.0, 2.0, "上窓(+1〜+2%)"),
    (2.0, np.inf, "大きく上窓(+2%超)"),
]

# 市場平均の実測ベースライン。
# S&P500構成505銘柄・2013-02〜2018-02の日足 617,298日ぶんを集計した実績値。
# 個別銘柄の統計がこれとどれだけ違うかを見るための基準として使う。
# fill=窓埋め率, follow=ギャップと同方向に引けた率, otc=寄り後リターンの平均(%)
MARKET_BASELINE = {
    "大きく下窓(−2%超)": {"n": 8962, "fill": 24.5, "follow": 42.2, "otc": +0.45},
    "下窓(−1〜−2%)": {"n": 24745, "fill": 40.6, "follow": 46.1, "otc": +0.08},
    "小さい下窓(−0.3〜−1%)": {"n": 110079, "fill": 65.8, "follow": 47.5, "otc": -0.01},
    "ほぼ窓なし(±0.3%)": {"n": 305193, "fill": 82.8, "follow": 46.2, "otc": +0.03},
    "小さい上窓(+0.3〜+1%)": {"n": 135086, "fill": 63.4, "follow": 51.4, "otc": +0.06},
    "上窓(+1〜+2%)": {"n": 25288, "fill": 38.0, "follow": 51.9, "otc": +0.09},
    "大きく上窓(+2%超)": {"n": 7945, "fill": 25.4, "follow": 46.5, "otc": -0.27},
}
BASELINE_NOTE = ("S&P500・505銘柄 / 2013-02〜2018-02 / 617,298日の実測。"
                 "ギャップが大きいほど窓埋めしにくい(±0.3%で83%→2%超で25%)。"
                 "一方で寄り後の方向はどの区分でもほぼ五分五分。")


def baseline_for(gap_pct: float | None) -> dict | None:
    """そのギャップ率が属する区分の市場平均ベースラインを返す。"""
    if gap_pct is None:
        return None
    for lo, hi, label in BUCKETS:
        if lo < gap_pct <= hi:
            base = MARKET_BASELINE.get(label)
            return {"label": label, **base} if base else None
    return None


def gap_table(daily: pd.DataFrame, lookback: int = 504) -> pd.DataFrame:
    """日足から1日1行のギャップ実測テーブルを作る。

    戻り値の列:
        gap_pct       ギャップ率(%)
        open_to_close 寄り後リターン(%)
        filled        窓を当日中に埋めたか
        follow        ギャップと同じ方向に寄り後も動いたか
        fade          ギャップと逆方向に寄り後動いたか(寄り天/寄り底)
        range_pct     当日の高安レンジ(始値比%)
        excursion     ギャップ方向への最大到達(始値比%、上窓なら高値まで)
    """
    if daily is None or len(daily) < 30:
        return pd.DataFrame()
    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(daily.columns):
        return pd.DataFrame()

    df = daily.tail(lookback + 1).copy()
    prev_close = df["Close"].shift(1)
    out = pd.DataFrame(index=df.index)
    out["prev_close"] = prev_close
    out["open"] = df["Open"]
    out["close"] = df["Close"]
    out["high"] = df["High"]
    out["low"] = df["Low"]
    out["gap_pct"] = (df["Open"] / prev_close - 1) * 100
    out["open_to_close"] = (df["Close"] / df["Open"] - 1) * 100
    out["range_pct"] = (df["High"] - df["Low"]) / df["Open"] * 100

    up = out["gap_pct"] > 0
    down = out["gap_pct"] < 0
    # 窓埋め: 上窓なら安値が前日終値以下、下窓なら高値が前日終値以上
    out["filled"] = np.where(up, out["low"] <= out["prev_close"],
                             np.where(down, out["high"] >= out["prev_close"],
                                      False))
    out["follow"] = np.where(up, out["open_to_close"] > 0,
                             np.where(down, out["open_to_close"] < 0, False))
    out["fade"] = np.where(up, out["open_to_close"] < 0,
                           np.where(down, out["open_to_close"] > 0, False))
    # ギャップ方向への最大到達(始値からどこまで伸びたか)
    out["excursion"] = np.where(
        up, (out["high"] / out["open"] - 1) * 100,
        np.where(down, (out["low"] / out["open"] - 1) * 100, 0.0))
    return out.dropna(subset=["gap_pct"])


def _stats(sample: pd.DataFrame) -> dict:
    """1グループぶんの集計。"""
    n = len(sample)
    if n == 0:
        return {}
    otc = sample["open_to_close"]
    return {
        "n": n,
        "fill_rate": float(sample["filled"].mean() * 100),
        "follow_rate": float(sample["follow"].mean() * 100),
        "fade_rate": float(sample["fade"].mean() * 100),
        "otc_mean": float(otc.mean()),
        "otc_median": float(otc.median()),
        "otc_std": float(otc.std(ddof=1)) if n > 1 else 0.0,
        "otc_p10": float(otc.quantile(0.10)),
        "otc_p90": float(otc.quantile(0.90)),
        "up_rate": float((otc > 0).mean() * 100),
        "range_mean": float(sample["range_pct"].mean()),
        "excursion_mean": float(sample["excursion"].abs().mean()),
    }


def bucket_summary(table: pd.DataFrame) -> list[dict]:
    """ギャップ率の区分ごとの実績。"""
    if table is None or table.empty:
        return []
    rows = []
    for lo, hi, label in BUCKETS:
        part = table[(table["gap_pct"] > lo) & (table["gap_pct"] <= hi)]
        if part.empty:
            continue
        rows.append({"label": label, "lo": lo, "hi": hi, **_stats(part)})
    return rows


def conditional(table: pd.DataFrame, gap_pct: float) -> dict | None:
    """「今回と同じくらいのギャップ」の過去実績を返す。

    ±BUCKET_HALF_WIDTH から始めて、サンプルがMIN_SAMPLESに満たなければ
    段階的に幅を広げる。広げた事実はwidthとして返し、UIで明示する。
    """
    if table is None or table.empty or gap_pct is None:
        return None
    same_side = table[table["gap_pct"] > 0] if gap_pct > 0 else \
        (table[table["gap_pct"] < 0] if gap_pct < 0 else table)
    for width in WIDEN_STEPS:
        part = same_side[(same_side["gap_pct"] >= gap_pct - width)
                         & (same_side["gap_pct"] <= gap_pct + width)]
        if len(part) >= MIN_SAMPLES:
            return {"width": width, "gap_pct": gap_pct, **_stats(part)}
    # 最大幅でも足りないときは、同方向のすべてで代用する
    if len(same_side) >= MIN_SAMPLES:
        return {"width": None, "gap_pct": gap_pct, "widened_to_side": True,
                **_stats(same_side)}
    return None


def overall(table: pd.DataFrame) -> dict:
    """全期間のギャップの傾向(頻度と大きさ)。"""
    if table is None or table.empty:
        return {}
    g = table["gap_pct"]
    return {
        "n": int(len(g)),
        "mean_abs": float(g.abs().mean()),
        "median_abs": float(g.abs().median()),
        "up_rate": float((g > 0).mean() * 100),
        "big_up_rate": float((g >= 1.0).mean() * 100),
        "big_down_rate": float((g <= -1.0).mean() * 100),
        "quiet_rate": float((g.abs() < 0.3).mean() * 100),
        "p05": float(g.quantile(0.05)),
        "p95": float(g.quantile(0.95)),
    }


def implied_gap(reference_price: float | None,
                prev_close: float | None) -> float | None:
    """時間外の現在値から想定ギャップ率(%)を出す。"""
    if not reference_price or not prev_close:
        return None
    return (float(reference_price) / float(prev_close) - 1) * 100


def extended_reference(intraday: pd.DataFrame,
                       prev_close: float | None) -> dict | None:
    """時間外バーから「寄付の手がかりになる価格」を取り出す。

    直近取引日の pre / after / overnight のうち、いちばん新しいセッションの
    終値を採用する。立会中ならNone(寄付はもう済んでいる)。
    """
    if intraday is None or intraday.empty:
        return None
    day = sessions.latest_day_slice(intraday)
    if day.empty:
        return None
    sess = sessions.classify(day.index)
    ext = day[sess.isin(sessions.EXTENDED).to_numpy()]
    if ext.empty:
        return None
    last_name = str(sessions.classify(ext.index).iloc[-1])
    part = ext[(sessions.classify(ext.index) == last_name).to_numpy()]
    price = float(part["Close"].iloc[-1])
    return {
        "session": last_name,
        "label": sessions.SESSION_LABELS[last_name],
        "price": price,
        "high": float(part["High"].max()),
        "low": float(part["Low"].min()),
        "volume": float(part["Volume"].sum()) if "Volume" in part else 0.0,
        "bars": int(len(part)),
        "gap_pct": implied_gap(price, prev_close),
        "at": part.index[-1],
    }


def projected_open(prev_close: float | None, gap_pct: float | None,
                   cond: dict | None) -> dict | None:
    """想定ギャップと過去実績から、寄付価格と寄り後の目安レンジを組み立てる。

    レンジは寄り後リターンの10〜90パーセンタイル。8割がこの中に収まったという
    実績であって、8割の確率で収まるという予測ではない。
    """
    if not prev_close or gap_pct is None:
        return None
    open_price = float(prev_close) * (1 + gap_pct / 100)
    out = {
        "prev_close": float(prev_close),
        "gap_pct": gap_pct,
        "open_price": open_price,
        "gap_amount": open_price - float(prev_close),
    }
    if cond:
        out.update({
            "close_low": open_price * (1 + cond["otc_p10"] / 100),
            "close_high": open_price * (1 + cond["otc_p90"] / 100),
            "close_typical": open_price * (1 + cond["otc_median"] / 100),
        })
    return out


def verdict(cond: dict | None, gap_pct: float | None) -> dict | None:
    """条件付き実績を一言でまとめる(判断そのものはユーザーに委ねる)。

    偏りが小さいときは無理に方向を出さず「はっきりした偏りなし」と言い切る。
    """
    if not cond or gap_pct is None:
        return None
    direction = "上窓" if gap_pct > 0 else ("下窓" if gap_pct < 0 else "窓なし")
    follow, fade, fill = cond["follow_rate"], cond["fade_rate"], cond["fill_rate"]
    lead = max(follow, fade)
    if lead < 58:
        tone, headline = "neutral", f"{direction}後の方向にはっきりした偏りなし"
    elif follow >= fade:
        tone = "follow"
        headline = (f"{direction}のあと、同じ方向に伸びたのが{follow:.0f}%"
                    if gap_pct > 0 else
                    f"{direction}のあと、さらに下げたのが{follow:.0f}%")
    else:
        tone = "fade"
        headline = (f"{direction}のあと、寄り天になったのが{fade:.0f}%"
                    if gap_pct > 0 else
                    f"{direction}のあと、寄り底から戻したのが{fade:.0f}%")
    return {
        "tone": tone,
        "headline": headline,
        "fill_note": f"窓埋めは{fill:.0f}%({cond['n']}回中)",
        "n": cond["n"],
    }

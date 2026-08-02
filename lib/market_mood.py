"""市場センチメントの合成スコアと買い場判定。

CNNのFear & Greed指数を参考に、無料データで取れる5要素を0〜100に正規化して平均する。
0=極度の恐怖、100=極度の強欲。恐怖側ほど逆張りの買い場になりやすいという
歴史的傾向に基づく参考評価で、将来を保証するものではない。

要素:
1. VIX水準       … 過去1年レンジ内の位置(高いほど恐怖)
2. 株価モメンタム … S&P500の125日移動平均線からの乖離
3. 市場の幅       … 50日線を上回るセクターETF(11本)の割合
4. 安全資産需要   … 株式(S&P500)と長期債(TLT)の直近20日リターン差
5. 短期過熱感     … S&P500のRSI(14)
"""

import pandas as pd
import streamlit as st

from lib import data_fetcher

SECTOR_SYMS = ["XLK", "XLC", "XLY", "XLF", "XLV", "XLI",
               "XLP", "XLE", "XLB", "XLRE", "XLU"]

SECTOR_LABELS = {
    "XLK": "テクノロジー", "XLC": "通信サービス", "XLY": "一般消費財",
    "XLF": "金融", "XLV": "ヘルスケア", "XLI": "資本財",
    "XLP": "生活必需品", "XLE": "エネルギー", "XLB": "素材",
    "XLRE": "不動産", "XLU": "公益",
}

ZONES = [
    (0, 25, "極度の恐怖", "#d03b3b"),
    (25, 45, "恐怖", "#ec835a"),
    (45, 55, "中立", "#898781"),
    (55, 75, "強欲", "#54a054"),
    (75, 101, "極度の強欲", "#0ca30c"),
]


def _naive(c: pd.Series) -> pd.Series:
    idx = c.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    c = c.copy()
    c.index = idx.normalize()
    return c


def _close(ticker: str, period: str) -> pd.Series:
    """終値をタイムゾーンなしのindexで返す。

    fetch_history は取得失敗やレート制限のとき例外ではなく空のDataFrameを返すため、
    ここでFetchErrorに変換し、呼び出し側の except でまとめて扱えるようにする。
    """
    df = data_fetcher.fetch_history(ticker, period)
    if df.empty or "Close" not in df.columns:
        raise data_fetcher.FetchError(f"{ticker}: 株価データを取得できませんでした")
    return _naive(df["Close"])


def _clip(v: float) -> float:
    return max(0.0, min(100.0, v))


def _rsi14(close: pd.Series) -> float | None:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    v = rsi.iloc[-1]
    return float(v) if pd.notna(v) else None


def _buy_stars(score: float, trend_up: bool) -> int:
    """スコア(逆張り)と長期トレンドから買い場度(★1〜5)を決める。"""
    if score <= 25:
        return 5 if trend_up else 3
    if score <= 45:
        return 4 if trend_up else 2
    if score < 55:
        return 3
    if score <= 75:
        return 2
    return 1


def _sector_score_parts(c: pd.Series, spx: pd.Series) -> pd.DataFrame:
    """セクター買い場スコアの構成要素(バックテストで選定したS2構成)。

    RSI(14)・対S&P500の20日相対力・52週高値からの下落率の3要素。
    モメンタム(125日線乖離)入りの4要素案より前半/後半分割で頑健だった。
    """
    out = pd.DataFrame(index=c.index)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    out["rsi"] = 100 - 100 / (1 + gain / loss)
    spx_a = spx.reindex(c.index).ffill()
    rel = (c / c.shift(21) - 1) - (spx_a / spx_a.shift(21) - 1)
    out["rel"] = (50 + rel * 500).clip(0, 100)
    dd = (c / c.rolling(252).max() - 1) * 100
    out["dd"] = (100 + dd * 5).clip(0, 100)
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def sector_reliability() -> dict:
    """セクターごとに「逆張りが過去10年機能したか」を実測する。

    戻り値: {シンボル: 恐怖(スコア≤35)後と強欲(≥65)後の3ヶ月平均リターン差(pt)}
    """
    out = {}
    try:
        spx = _close("^GSPC", "10y")
    except data_fetcher.FetchError:
        return out
    for sym in SECTOR_SYMS:
        try:
            c = _close(sym, "10y").dropna()
        except data_fetcher.FetchError:
            continue
        if len(c) < 500:
            continue
        score = _sector_score_parts(c, spx).mean(axis=1)
        fwd = (c.shift(-63) / c - 1) * 100
        pair = pd.concat([score.rename("s"), fwd.rename("f")], axis=1).dropna()
        fear = pair[pair["s"] <= 35]["f"]
        greed = pair[pair["s"] >= 65]["f"]
        if len(fear) >= 30 and len(greed) >= 30:
            out[sym] = float(fear.mean() - greed.mean())
    return out


@st.cache_data(ttl=900, show_spinner=False)
def sector_moods() -> list[dict]:
    """各セクターETFの現在の買い場スコア・判定を返す(スコア昇順=売られた順)。"""
    try:
        spx = _close("^GSPC", "2y")
    except data_fetcher.FetchError:
        return []
    premium = sector_reliability()
    rows = []
    for sym in SECTOR_SYMS:
        try:
            c = _close(sym, "2y").dropna()
        except data_fetcher.FetchError:
            continue
        if len(c) < 260:
            continue
        parts = _sector_score_parts(c, spx)
        last = parts.iloc[-1]
        if last.isna().any():
            continue
        score = float(last.mean())
        zone_label, zone_color = next((lb, cl) for lo, hi, lb, cl in ZONES
                                      if lo <= score < hi)
        trend_up = float(c.iloc[-1]) > float(c.rolling(200).mean().iloc[-1])
        dd_now = (float(c.iloc[-1]) / float(c.rolling(252).max().iloc[-1]) - 1) * 100
        spx_a = spx.reindex(c.index).ffill()
        rel_now = ((float(c.iloc[-1]) / float(c.iloc[-22]) - 1)
                   - (float(spx_a.iloc[-1]) / float(spx_a.iloc[-22]) - 1)) * 100
        prem = premium.get(sym)
        rows.append({
            "sym": sym,
            "label": SECTOR_LABELS.get(sym, sym),
            "score": round(score, 1),
            "zone": zone_label,
            "zone_color": zone_color,
            "stars": _buy_stars(score, trend_up),
            "trend_up": trend_up,
            "dd": dd_now,
            "rel": rel_now,
            "rsi": float(last["rsi"]),
            "premium": prem,
        })
    return sorted(rows, key=lambda r: r["score"])


@st.cache_data(ttl=86400, show_spinner=False)
def zone_stats() -> dict | None:
    """過去10年のスコア履歴を再現し、ゾーン別の先行リターン実績を計算する。

    バックテスト検証済み(2016〜2026): ゾーン別の3ヶ月後平均リターンは
    恐怖側ほど高く単調に並ぶ(等ウェイト構成が前後半分割でも頑健)。
    """
    try:
        spx = _close("^GSPC", "10y")
        vix = _close("^VIX", "10y")
        tlt = _close("TLT", "10y")
    except data_fetcher.FetchError:
        return None
    if len(spx) < 500:
        return None
    vix = vix.reindex(spx.index).ffill()
    tlt = tlt.reindex(spx.index).ffill()

    comp = pd.DataFrame(index=spx.index)
    comp["vix"] = 100 - vix.rolling(252).rank(pct=True) * 100
    comp["mom"] = (50 + (spx / spx.rolling(125).mean() - 1) * 500).clip(0, 100)
    sec = {}
    for sym in SECTOR_SYMS:
        try:
            sec[sym] = _close(sym, "10y").reindex(spx.index)
        except data_fetcher.FetchError:
            continue
    if sec:
        above = pd.DataFrame({s: (c > c.rolling(50).mean()).where(c.notna())
                              for s, c in sec.items()})
        comp["breadth"] = above.mean(axis=1) * 100
    comp["safe"] = (50 + ((spx / spx.shift(21) - 1)
                          - (tlt / tlt.shift(21) - 1)) * 500).clip(0, 100)
    delta = spx.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    comp["rsi"] = 100 - 100 / (1 + gain / loss)

    score = comp.mean(axis=1, skipna=True)
    fwd21 = (spx.shift(-21) / spx - 1) * 100
    fwd63 = (spx.shift(-63) / spx - 1) * 100
    pair = pd.concat([score.rename("s"), fwd21.rename("f21"),
                      fwd63.rename("f63")], axis=1).dropna()
    if len(pair) < 300:
        return None

    rows = []
    for lo, hi, label, _color in ZONES:
        z = pair[(pair["s"] >= lo) & (pair["s"] < hi)]
        if len(z) < 20:
            continue
        rows.append({
            "zone": label, "lo": lo, "hi": hi, "n": int(len(z)),
            "fwd21": float(z["f21"].mean()),
            "fwd63": float(z["f63"].mean()),
            "win63": float((z["f63"] > 0).mean() * 100),
        })
    return {
        "rows": rows,
        "overall21": float(pair["f21"].mean()),
        "overall63": float(pair["f63"].mean()),
        "start": str(pair.index[0].date()),
        "end": str(pair.index[-1].date()),
    }


def compute() -> dict | None:
    """センチメントスコアと買い場判定を計算する。データ不足時はNone。"""
    try:
        vix = data_fetcher.fetch_history("^VIX", "1y")
        spx = data_fetcher.fetch_history("^GSPC", "2y")
    except data_fetcher.FetchError:
        return None
    if vix.empty or len(spx) < 200:
        return None

    close = spx["Close"]
    comps = []

    # 1. VIX水準(1年レンジ内の位置。高VIX=恐怖なのでスコアは反転)
    vix_now = float(vix["Close"].iloc[-1])
    pct = float((vix["Close"] < vix_now).mean() * 100)
    if vix_now >= 30:
        vix_state = "パニック水準"
    elif vix_now >= 20:
        vix_state = "警戒"
    elif vix_now >= 15:
        vix_state = "平常"
    else:
        vix_state = "落ち着き"
    comps.append({"指標": "VIX(恐怖指数)", "現在値": f"{vix_now:.1f}({vix_state})",
                  "読み方": "高いほど市場が恐怖", "スコア": round(_clip(100 - pct))})

    # 2. 株価モメンタム(125日線乖離: -10%→0点、+10%→100点)
    ma125 = close.rolling(125).mean().iloc[-1]
    dist = float(close.iloc[-1] / ma125 - 1)
    comps.append({"指標": "株価モメンタム", "現在値": f"S&P500 125日線比 {dist * 100:+.1f}%",
                  "読み方": "プラスほど強気", "スコア": round(_clip(50 + dist * 500))})

    # 3. 市場の幅(50日線超のセクターETF比率)
    above = total = 0
    for sym in SECTOR_SYMS:
        try:
            h = data_fetcher.fetch_history(sym, "6mo")
        except data_fetcher.FetchError:
            continue
        if len(h) >= 50:
            total += 1
            if float(h["Close"].iloc[-1]) > float(h["Close"].rolling(50).mean().iloc[-1]):
                above += 1
    if total:
        comps.append({"指標": "市場の幅", "現在値": f"{above}/{total}セクターが50日線超え",
                      "読み方": "広いほど健全な上昇", "スコア": round(above / total * 100)})

    # 4. 安全資産需要(株式と長期債の20日リターン差: ±10ptで0/100点)
    try:
        tlt = data_fetcher.fetch_history("TLT", "6mo")
    except data_fetcher.FetchError:
        tlt = pd.DataFrame()
    if len(tlt) > 21 and len(close) > 21:
        r_spx = float(close.iloc[-1] / close.iloc[-21] - 1)
        r_tlt = float(tlt["Close"].iloc[-1] / tlt["Close"].iloc[-21] - 1)
        diff = (r_spx - r_tlt) * 100
        comps.append({"指標": "安全資産需要", "現在値": f"株式−債券(20日) {diff:+.1f}pt",
                      "読み方": "マイナスほど債券へ逃避", "スコア": round(_clip(50 + diff * 5))})

    # 5. 短期過熱感(S&P500のRSI14)
    rsi = _rsi14(close)
    if rsi is not None:
        comps.append({"指標": "短期過熱感", "現在値": f"S&P500 RSI(14) = {rsi:.0f}",
                      "読み方": "70超は過熱・30未満は売られすぎ", "スコア": round(rsi)})

    if not comps:
        return None
    score = sum(c["スコア"] for c in comps) / len(comps)
    zone_label, zone_color = next((lb, cl) for lo, hi, lb, cl in ZONES
                                  if lo <= score < hi)

    # 過去10年のゾーン別実績(バックテストで検証済みの根拠データ)
    hist = zone_stats()
    hist_row = None
    if hist:
        hist_row = next((r for r in hist["rows"] if r["zone"] == zone_label), None)

    # 長期トレンドと調整度(S&P500)
    ma200 = float(close.rolling(200).mean().iloc[-1])
    trend_up = float(close.iloc[-1]) > ma200
    trend_label = ("長期上昇トレンド(200日線の上)" if trend_up
                   else "長期下降トレンド(200日線の下)")
    hi52 = float(close.iloc[-252:].max())
    dd = (float(close.iloc[-1]) / hi52 - 1) * 100
    if dd > -3:
        dd_label = "高値圏"
    elif dd > -10:
        dd_label = "通常の押し"
    elif dd > -20:
        dd_label = "調整局面"
    else:
        dd_label = "弱気相場圏"

    # 買い場判定(逆張りベース+長期トレンドのフィルタ)
    if score <= 25:
        if trend_up:
            stars, title = 5, "🟢 歴史的には絶好の買い場水準"
            text = ("市場は「極度の恐怖」。歴史的にはこの水準での仕込みが報われやすく、"
                    "長期上昇トレンドも維持されています。分割での押し目買いを検討しやすい局面です。")
        else:
            stars, title = 3, "🟡 割安圏だが下降トレンド中"
            text = ("恐怖は極度ですが長期トレンドが下向きです。いわゆる「落ちるナイフ」に"
                    "注意し、エントリーするなら時間分散を意識したい局面です。")
    elif score <= 45:
        if trend_up:
            stars, title = 4, "🟢 押し目買いの検討ゾーン"
            text = ("市場は「恐怖」寄り。上昇トレンド中の悲観は押し目になりやすい、"
                    "というのが歴史的な傾向です。")
        else:
            stars, title = 2, "🟡 慎重ゾーン"
            text = "悲観はまだ極端ではなく、トレンドも下向き。急がず様子見が無難な局面です。"
    elif score < 55:
        stars, title = 3, "⚪ 中立"
        text = "恐怖でも強欲でもない通常運転。個別銘柄の材料や決算に沿った判断が中心になります。"
    elif score <= 75:
        stars, title = 2, "🟠 やや過熱気味"
        text = ("市場は「強欲」寄り。追いかけ買いは高値掴みになりやすい水準です。"
                "新規買いは押し目を待つ選択肢も。")
    else:
        stars, title = 1, "🔴 過熱圏"
        text = ("市場は「極度の強欲」。歴史的には調整が近いことも多い水準で、"
                "新規買いより利益確定・現金比率の見直しが意識されやすい局面です。")

    return {
        "score": round(score, 1),
        "zone_label": zone_label,
        "zone_color": zone_color,
        "components": comps,
        "trend_up": trend_up,
        "trend_label": trend_label,
        "drawdown_pct": dd,
        "dd_label": dd_label,
        "stars": stars,
        "verdict_title": title,
        "verdict_text": text,
        "hist": hist_row,
        "hist_overall": ({"f21": hist["overall21"], "f63": hist["overall63"],
                          "start": hist["start"]} if hist else None),
        "hist_rows": hist["rows"] if hist else [],
    }

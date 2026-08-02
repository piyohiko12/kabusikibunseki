"""銘柄比較ページ: 最大4銘柄の指標とパフォーマンスを並べて比較する。"""

import pandas as pd
import streamlit as st

from lib import charts, data_fetcher, indicators

UP_COLOR = "#006300"
DOWN_COLOR = "#a32d2d"

PERIODS = {"1ヶ月": 30, "6ヶ月": 182, "1年": 365, "3年": 1095}


def fmt(v, suffix="", digits=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:,.{digits}f}{suffix}"


def fmt_cap(v):
    if v is None:
        return "—"
    if v >= 1e12:
        return f"{v / 1e12:,.2f}兆ドル"
    return f"{v / 1e8:,.0f}億ドル"


st.title("⚖️ 銘柄比較")

raw = st.text_input("比較する銘柄(カンマ・スペース区切り、最大4つ)",
                    value="AAPL, MSFT, NVDA")
period_label = st.radio("比較期間", list(PERIODS), index=2, horizontal=True)

tickers = []
for t in raw.replace(",", " ").split():
    t = t.strip().upper()
    if t and t not in tickers:
        tickers.append(t)
tickers = tickers[:4]

if len(tickers) < 2:
    st.info("2つ以上のティッカーを入力してください(例: AAPL, MSFT)。")
    st.stop()

days = PERIODS[period_label]
data = {}
failed = []
for t in tickers:
    try:
        info = data_fetcher.fetch_info(t)
        hist = data_fetcher.fetch_history(t, "5y")
    except data_fetcher.FetchError:
        info, hist = {}, pd.DataFrame()
    if not info.get("price") or hist.empty:
        failed.append(t)
        continue
    data[t] = (info, hist)

if failed:
    st.warning("取得できなかった銘柄: " + ", ".join(failed))
if len(data) < 2:
    st.error("比較できる銘柄が2つ未満です。ティッカーを確認してください。")
    st.stop()

# パフォーマンス比較(期間始点=100)
st.subheader(f"パフォーマンス比較({period_label}、始点=100)")
series = {}
for t, (_info, hist) in data.items():
    view = indicators.slice_display(hist, days)
    series[t] = view["Close"]
st.plotly_chart(charts.comparison_chart(series))

# 指標の比較テーブル(行=指標、列=銘柄)
st.subheader("指標の比較")
rows = {}
for t, (info, hist) in data.items():
    close = hist["Close"]
    view = indicators.slice_display(hist, days)["Close"]
    period_ret = (float(view.iloc[-1]) / float(view.iloc[0]) - 1) * 100 if len(view) > 1 else None
    ret1y = None
    y = indicators.slice_display(hist, 365)["Close"]
    if len(y) > 1:
        ret1y = (float(y.iloc[-1]) / float(y.iloc[0]) - 1) * 100
    daily = close.pct_change().dropna().iloc[-252:]
    vol = float(daily.std() * (252 ** 0.5) * 100) if len(daily) > 20 else None
    dd52 = (float(close.iloc[-1]) / float(close.iloc[-252:].max()) - 1) * 100
    roe = info.get("roe")
    rows[t] = {
        "現在値": f"${info['price']:,.2f}",
        f"リターン({period_label})": fmt(period_ret, "%", 1),
        "リターン(1年)": fmt(ret1y, "%", 1),
        "年率ボラティリティ": fmt(vol, "%", 1),
        "52週高値から": fmt(dd52, "%", 1),
        "PER": fmt(info.get("per"), " 倍"),
        "PBR": fmt(info.get("pbr"), " 倍"),
        "ROE": fmt(roe * 100 if roe is not None else None, " %", 1),
        "配当利回り": fmt(data_fetcher.dividend_yield_percent(info), " %"),
        "時価総額": fmt_cap(info.get("market_cap")),
        "セクター": info.get("sector") or "—",
    }
st.dataframe(pd.DataFrame(rows), use_container_width=True)
st.caption("各銘柄の詳細は「銘柄分析」ページへ: "
           + " / ".join(f"[{t}](/?ticker={t})" for t in data))

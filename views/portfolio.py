"""ポートフォリオ管理ページ。"""

import pandas as pd
import streamlit as st

from lib import charts, data_fetcher, portfolio_store, watchlist_store

UP_COLOR = "#0ca30c"
DOWN_COLOR = "#d03b3b"

VALUE_PERIODS = {"1ヶ月": 30, "3ヶ月": 91, "6ヶ月": 182, "1年": 365}


def _pl_color(v):
    if pd.isna(v) or v == 0:
        return ""
    return f"color: {UP_COLOR}" if v > 0 else f"color: {DOWN_COLOR}"


st.title("💼 ポートフォリオ管理")

st.subheader("保有銘柄の登録・編集")
st.caption("表を直接編集し、「保存」ボタンで確定します。最下行への入力で追加、"
           "行左端のチェック→ゴミ箱アイコン(またはDeleteキー)で削除できます。")

holdings = portfolio_store.load()
edited = st.data_editor(
    holdings,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "ticker": st.column_config.TextColumn("ティッカー", help="例: AAPL", required=True),
        "shares": st.column_config.NumberColumn("株数", min_value=0.0, required=True),
        "avg_cost": st.column_config.NumberColumn("取得単価(ドル)", min_value=0.0,
                                                  format="$%.2f", required=True),
    },
    key="portfolio_editor",
)

if st.button("保存", type="primary"):
    cleaned, errors = portfolio_store.validate(edited)
    if errors:
        for e in errors:
            st.error(e)
    else:
        portfolio_store.save(cleaned)
        st.success("保存しました。")
        holdings = cleaned

st.divider()
st.subheader("評価損益")

if holdings.empty:
    st.info("保有銘柄が登録されていません。上の表にティッカー・株数・取得単価を入力して"
            "「保存」を押してください。")
    st.stop()

rows = []
failed = []
closes = {}
for r in holdings.itertuples():
    try:
        info = data_fetcher.fetch_info(r.ticker)
        hist = data_fetcher.fetch_history(r.ticker, "1y")
    except data_fetcher.FetchError:
        info, hist = {}, pd.DataFrame()
    price = info.get("price")
    if not price or hist.empty:
        failed.append(r.ticker)
        continue
    closes[r.ticker] = hist["Close"]

    prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
    day_chg = (price / prev - 1) * 100 if prev else 0.0

    # RSI(14)を保有銘柄のシグナルとして表示
    delta = hist["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi_series = 100 - 100 / (1 + gain / loss)
    rsi = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else None
    if rsi is None:
        signal = "—"
    elif rsi >= 70:
        signal = "🔴 過熱"
    elif rsi <= 30:
        signal = "🟢 売られすぎ"
    else:
        signal = "中立"

    value = price * r.shares
    cost = r.avg_cost * r.shares
    rows.append({
        "ティッカー": r.ticker,
        "銘柄名": info.get("name", r.ticker),
        "セクター": info.get("sector") or "その他",
        "株数": r.shares,
        "取得単価": r.avg_cost,
        "現在値": price,
        "前日比": day_chg,
        "評価額": value,
        "損益": value - cost,
        "損益率(%)": (value / cost - 1) * 100 if cost else 0.0,
        "RSI": rsi,
        "シグナル": signal,
    })

if failed:
    st.warning("次の銘柄は価格を取得できなかったため、集計から除外しました: "
               + ", ".join(failed))

if not rows:
    st.error("すべての銘柄で価格の取得に失敗しました。ティッカーが正しいか確認し、"
             "しばらく時間をおいてから再試行してください。")
    st.stop()

table = pd.DataFrame(rows)
total_value = float(table["評価額"].sum())
total_cost = float((table["取得単価"] * table["株数"]).sum())
total_pl = total_value - total_cost
total_pl_pct = (total_value / total_cost - 1) * 100 if total_cost else 0.0

usdjpy = None
try:
    fx = data_fetcher.fetch_history("JPY=X", "5d")
    if not fx.empty:
        usdjpy = float(fx["Close"].iloc[-1])
except data_fetcher.FetchError:
    pass

m1, m2, m3, m4 = st.columns(4)
m1.metric("取得額合計", f"${total_cost:,.2f}", border=True)
m2.metric("評価額合計", f"${total_value:,.2f}", border=True)
m3.metric("損益合計", f"${total_pl:,.2f}", f"{total_pl_pct:+.2f}%", border=True)
m4.metric("円換算評価額",
          f"¥{total_value * usdjpy:,.0f}" if usdjpy else "—",
          f"1ドル={usdjpy:.2f}円" if usdjpy else None,
          delta_color="off", border=True)

styled = table.style.format({
    "株数": "{:g}",
    "取得単価": "${:,.2f}",
    "現在値": "${:,.2f}",
    "前日比": "{:+.2f}%",
    "評価額": "${:,.2f}",
    "損益": "${:+,.2f}",
    "損益率(%)": "{:+.2f}%",
    "RSI": lambda v: "—" if pd.isna(v) else f"{v:.0f}",
}).map(_pl_color, subset=["前日比", "損益", "損益率(%)"])
st.dataframe(styled, hide_index=True)
st.caption("シグナル=RSI(14)による過熱/売られすぎの目安。")

col_pl, col_pie = st.columns(2)
with col_pl:
    pl_series = table.set_index("ティッカー")["損益"]
    st.plotly_chart(charts.pl_bar(pl_series))
with col_pie:
    sector_values = table.groupby("セクター")["評価額"].sum()
    st.plotly_chart(charts.sector_pie(sector_values))

# ---------------------------------------------------------------- 評価額推移
st.subheader("📈 評価額の推移")
period_label = st.pills("期間", list(VALUE_PERIODS), default="6ヶ月") or "6ヶ月"
days = VALUE_PERIODS[period_label]

shares_map = {r.ticker: float(r.shares) for r in holdings.itertuples()
              if r.ticker in closes}
close_df = pd.DataFrame(closes).dropna()
if close_df.empty or len(close_df) < 5:
    st.info("推移を計算できるだけの価格履歴がありません。")
else:
    value_series = sum(close_df[t] * s for t, s in shares_map.items())
    cutoff = value_series.index.max() - pd.Timedelta(days=days)
    value_view = value_series.loc[value_series.index >= cutoff]

    ret = value_view.pct_change().dropna()
    period_ret = (value_view.iloc[-1] / value_view.iloc[0] - 1) * 100
    ann_vol = float(ret.std() * (252 ** 0.5) * 100) if len(ret) > 5 else None
    drawdown = float(((value_view / value_view.cummax()) - 1).min() * 100)

    beta = None
    spx_ret_pct = None
    try:
        spx = data_fetcher.fetch_history("^GSPC", "1y")
        spx_view = spx["Close"].loc[spx.index >= cutoff]
        if len(spx_view) > 5:
            spx_ret_pct = (spx_view.iloc[-1] / spx_view.iloc[0] - 1) * 100
            pair = pd.concat([ret, spx_view.pct_change()], axis=1, join="inner").dropna()
            var = float(pair.iloc[:, 1].var())
            if var and len(pair) > 20:
                beta = float(pair.cov().iloc[0, 1]) / var
    except data_fetcher.FetchError:
        pass

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"期間リターン({period_label})", f"{period_ret:+.2f}%",
              (f"S&P500: {spx_ret_pct:+.2f}%" if spx_ret_pct is not None else None),
              delta_color="off", border=True)
    k2.metric("年率ボラティリティ", f"{ann_vol:.1f}%" if ann_vol else "—", border=True,
              help="日次リターンの標準偏差を年率換算した値。大きいほど値動きが荒い")
    k3.metric("最大ドローダウン", f"{drawdown:.1f}%", border=True,
              help="期間中の高値からの最大下落率")
    k4.metric("β(対S&P500)", f"{beta:.2f}" if beta is not None else "—", border=True,
              help="市場全体に対する感応度。1より大きいと市場より値動きが大きい")

    st.plotly_chart(charts.value_chart(value_view, total_cost))
    st.caption("現在の保有数量で過去期間を保有していた場合の推移です(期間中の売買・"
               "配当は考慮しません)。上場から日が浅い銘柄がある場合、共通の期間に"
               "揃えて計算します。")

# ---------------------------------------------------------------- 決算カレンダー
st.subheader("📅 今後の決算予定(保有銘柄+ウォッチリスト)")
cal_syms = list(dict.fromkeys(
    [r.ticker for r in holdings.itertuples()] + watchlist_store.load()))
today = pd.Timestamp.now().normalize()
cal_rows = []
for t in cal_syms:
    an = data_fetcher.fetch_analyst(t)
    ed = an.get("earnings_date")
    if not ed:
        continue
    try:
        d = pd.Timestamp(ed)
    except ValueError:
        continue
    if d < today:
        continue
    eps = an.get("eps_estimate")
    cal_rows.append({"ティッカー": t, "決算日": ed,
                     "あと": f"{(d - today).days}日",
                     "予想EPS": f"${eps:.2f}" if eps else "—",
                     "リンク": f"/?ticker={t}"})
if cal_rows:
    st.dataframe(
        pd.DataFrame(sorted(cal_rows, key=lambda r: r["決算日"])),
        hide_index=True,
        column_config={"リンク": st.column_config.LinkColumn(
            "分析", display_text="開く →")},
    )
    st.caption("決算発表の前後は株価が大きく動きやすいイベントです"
               "(銘柄分析ページの「イベント感応度」で過去の変動幅を確認できます)。")
else:
    st.info("今後の決算予定を取得できる銘柄がありません。")

"""市場概況ページ: 主要指数・セクター・ランキング・ウォッチリスト。"""

import pandas as pd
import streamlit as st

from lib import charts, data_fetcher, market_mood, ui, watchlist_store

UP_COLOR = "#0ca30c"
DOWN_COLOR = "#d03b3b"

# (表示名, シンボル, 値フォーマット, deltaの色反転)
INDICES = [
    ("S&P500", "^GSPC", "{:,.2f}", False),
    ("NASDAQ総合", "^IXIC", "{:,.2f}", False),
    ("ダウ平均", "^DJI", "{:,.2f}", False),
    ("日経平均", "^N225", "{:,.2f}", False),
    ("ドル/円", "JPY=X", "¥{:,.2f}", False),
    ("米10年債利回り", "^TNX", "{:,.2f}%", False),
    ("WTI原油", "CL=F", "${:,.2f}", False),
    ("金(ゴールド)", "GC=F", "${:,.2f}", False),
    ("VIX(恐怖指数)", "^VIX", "{:,.2f}", True),
]

SCREENERS = [
    ("📈 値上がり率", "day_gainers"),
    ("📉 値下がり率", "day_losers"),
    ("🔥 出来高", "most_actives"),
]

SECTOR_ETFS = [
    ("XLK", "テクノロジー"), ("XLC", "通信サービス"), ("XLY", "一般消費財"),
    ("XLF", "金融"), ("XLV", "ヘルスケア"), ("XLI", "資本財"),
    ("XLP", "生活必需品"), ("XLE", "エネルギー"), ("XLB", "素材"),
    ("XLRE", "不動産"), ("XLU", "公益"),
]


def _pct_color(v):
    if pd.isna(v) or v == 0:
        return ""
    return f"color: {UP_COLOR}" if v > 0 else f"color: {DOWN_COLOR}"


st.title("🌐 市場概況")

# ---------------------------------------------------------------- センチメント
st.subheader("🌡️ 市場センチメント・買い場判定")
with st.spinner("市場データを分析中..."):
    mood = market_mood.compute()
if mood is None:
    st.warning("センチメントの計算に必要なデータを取得できませんでした。")
else:
    col_gauge, col_verdict = st.columns([1, 1.4])
    with col_gauge:
        st.plotly_chart(charts.mood_gauge(mood["score"]),
                        config={"displayModeBar": False, "staticPlot": True})
        st.markdown(
            f'<div style="text-align:center;margin-top:-14px;">'
            f'{ui.chip(f"現在: {mood["zone_label"]}", "red" if mood["score"] < 45 else ("green" if mood["score"] > 55 else "gray"))}'
            f'</div>', unsafe_allow_html=True)
        st.caption("0=極度の恐怖 ←→ 100=極度の強欲(Fear & Greed方式の自作指数)")
    with col_verdict:
        with st.container(border=True):
            st.markdown(f"#### {mood['verdict_title']}")
            st.markdown(f"**買い場度: {'★' * mood['stars']}{'☆' * (5 - mood['stars'])}**")
            st.write(mood["verdict_text"])
            if mood["hist"] and mood["hist_overall"]:
                h, o = mood["hist"], mood["hist_overall"]
                st.markdown(
                    f"📊 **過去実績({o['start']}以降)**: このゾーンの後、S&P500は"
                    f"平均で1ヶ月後 **{h['fwd21']:+.2f}%** / 3ヶ月後 **{h['fwd63']:+.2f}%**"
                    f"(全期間平均 {o['f21']:+.2f}% / {o['f63']:+.2f}%、"
                    f"該当{h['n']}営業日)")
            chips = " ".join([
                ui.chip(mood["trend_label"], "green" if mood["trend_up"] else "red"),
                ui.chip(f"S&P500 52週高値から{mood['drawdown_pct']:+.1f}%"
                        f"({mood['dd_label']})", "gray"),
            ])
            st.markdown(chips, unsafe_allow_html=True)

    with st.expander("📊 スコアの内訳と過去実績(バックテスト検証済み)"):
        st.markdown("**スコアの内訳(5要素の平均)**")
        st.dataframe(pd.DataFrame(mood["components"]), hide_index=True)
        if mood["hist_rows"]:
            st.markdown("**ゾーン別の過去実績(その日のスコア → その後のS&P500)**")
            hz = pd.DataFrame([{
                "ゾーン": ("▶ " if r["zone"] == mood["zone_label"] else "") + r["zone"],
                "スコア帯": f"{r['lo']}〜{min(r['hi'] - 1, 100)}",
                "該当日数": r["n"],
                "1ヶ月後平均": f"{r['fwd21']:+.2f}%",
                "3ヶ月後平均": f"{r['fwd63']:+.2f}%",
                "勝率(3ヶ月)": f"{r['win63']:.0f}%",
            } for r in mood["hist_rows"]])
            st.dataframe(hz, hide_index=True)
        st.caption("VIXの位置・S&P500の125日線乖離・50日線超セクター比率・"
                   "株式と長期債の20日リターン差・S&P500のRSI(14)を各0〜100点で平均。"
                   "重み付けやVIX強調などの変種を過去10年のデータで前半/後半に分けて"
                   "検証した結果、この等ウェイト構成が最も頑健でした(恐怖ゾーンほど"
                   "先行リターンが高い単調な関係を確認)。リターンは重複する日次観測の"
                   "平均であり、将来の成果を保証するものではありません。")

st.divider()
cols = st.columns(3)
for i, (label, symbol, fmt, inverse) in enumerate(INDICES):
    with cols[i % 3].container(border=True):
        try:
            hist = data_fetcher.fetch_history(symbol, "1mo")
        except data_fetcher.FetchError:
            hist = pd.DataFrame()
        if hist.empty or len(hist) < 2:
            st.metric(label, "—")
            continue
        last = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2])
        chg_pct = (last / prev - 1) * 100 if prev else 0.0
        st.metric(label, fmt.format(last), f"{chg_pct:+.2f}%",
                  delta_color="inverse" if inverse else "normal")
        st.plotly_chart(charts.sparkline(hist["Close"]),
                        config={"displayModeBar": False, "staticPlot": True},
                        key=f"spark_{symbol}")
st.caption("前日比。ミニチャートは直近1ヶ月の推移(緑=月初比プラス、赤=マイナス)。")

st.divider()
st.subheader("🧭 セクター別の買い場判定(売られている順)")
sec_rows = market_mood.sector_moods()
if not sec_rows:
    st.info("セクター別スコアを計算できませんでした。")
else:
    def _rel_mark(p):
        if p is None:
            return "—"
        if p >= 2:
            return "○ 有効"
        if p > 0:
            return "△ 弱い"
        return "× 不成立"

    sec_df = pd.DataFrame([{
        "セクター": f"{r['label']}({r['sym']})",
        "買い場度": "★" * r["stars"] + "☆" * (5 - r["stars"]),
        "スコア": r["score"],
        "状態": r["zone"],
        "52週高値から": r["dd"],
        "対S&P500(20日)": r["rel"],
        "RSI": r["rsi"],
        "長期トレンド": "上昇" if r["trend_up"] else "下降",
        "逆張り実績": _rel_mark(r["premium"]),
    } for r in sec_rows])
    zone_colors = {"極度の恐怖": "#d03b3b", "恐怖": "#ec835a",
                   "中立": "#52514e", "強欲": "#54a054",
                   "極度の強欲": "#0ca30c"}
    styled_sec = sec_df.style.format({
        "スコア": "{:.1f}",
        "52週高値から": "{:+.1f}%",
        "対S&P500(20日)": "{:+.1f}pt",
        "RSI": "{:.0f}",
    }).map(lambda v: f"color: {zone_colors.get(v, '')}", subset=["状態"]
    ).map(lambda v: "color: #006300" if v == "上昇"
          else "color: #a32d2d", subset=["長期トレンド"])
    st.dataframe(styled_sec, hide_index=True)
    st.caption("スコア=RSI(14)・対S&P500の20日相対力・52週高値からの下落率の合成"
               "(低いほど売られている=逆張り候補)。11セクター×過去10年のプール検証で"
               "スコアが低いほど3ヶ月後リターンが高い単調関係を確認(+8.4%→+1.4%)。"
               "「逆張り実績」はそのセクター単体で恐怖後が強欲後を上回った実績の有無"
               "(通信サービスは歴史的に逆張りが機能していない点に注意)。参考情報です。")

st.divider()
st.subheader("🗂️ セクター別パフォーマンス(前日比)")
sector_changes = {}
for sym, label in SECTOR_ETFS:
    try:
        h = data_fetcher.fetch_history(sym, "5d")
    except data_fetcher.FetchError:
        continue
    if len(h) >= 2:
        sector_changes[label] = (float(h["Close"].iloc[-1])
                                 / float(h["Close"].iloc[-2]) - 1) * 100
if sector_changes:
    st.plotly_chart(charts.perf_bar(pd.Series(sector_changes),
                                    "米国セクターETF(SPDR)の前日比"),
                    config={"displayModeBar": False})
    st.caption("どのセクターに資金が向かっているかの目安になります。")
else:
    st.info("セクターデータを取得できませんでした。")

st.divider()
st.subheader("⭐ ウォッチリスト")
wl = watchlist_store.load()
wl_edited = st.data_editor(
    pd.DataFrame({"ticker": wl if wl else [""]}),
    num_rows="dynamic", hide_index=True,
    column_config={"ticker": st.column_config.TextColumn("ティッカー", help="例: NVDA")},
    key="watchlist_editor",
)
if st.button("ウォッチリストを保存"):
    watchlist_store.save([t for t in wl_edited["ticker"].fillna("").tolist() if str(t).strip()])
    st.success("保存しました。")
    wl = watchlist_store.load()

if wl:
    wl_rows, wl_failed = [], []
    for t in wl:
        try:
            info = data_fetcher.fetch_info(t)
            hist = data_fetcher.fetch_history(t, "3mo")
        except data_fetcher.FetchError:
            info, hist = {}, pd.DataFrame()
        price = info.get("price")
        prev = info.get("previous_close")
        if not price:
            wl_failed.append(t)
            continue
        rsi = None
        if len(hist) > 20:
            delta = hist["Close"].diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14,
                                           adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14,
                                              adjust=False).mean()
            v = (100 - 100 / (1 + gain / loss)).iloc[-1]
            rsi = float(v) if pd.notna(v) else None
        if rsi is None:
            signal = "—"
        elif rsi >= 70:
            signal = "🔴 過熱"
        elif rsi <= 30:
            signal = "🟢 売られすぎ"
        else:
            signal = "中立"
        wl_rows.append({
            "ティッカー": t,
            "銘柄名": info.get("name", t),
            "セクター": info.get("sector") or "—",
            "株価": price,
            "前日比": (price / prev - 1) * 100 if prev else None,
            "RSI": rsi,
            "シグナル": signal,
            "リンク": f"/?ticker={t}",
        })
    if wl_failed:
        st.warning("取得できなかった銘柄: " + ", ".join(wl_failed))
    if wl_rows:
        wl_styled = pd.DataFrame(wl_rows).style.format({
            "株価": "${:,.2f}",
            "前日比": lambda v: "—" if pd.isna(v) else f"{v:+.2f}%",
            "RSI": lambda v: "—" if pd.isna(v) else f"{v:.0f}",
        }).map(_pct_color, subset=["前日比"])
        st.dataframe(wl_styled, hide_index=True,
                     column_config={"リンク": st.column_config.LinkColumn(
                         "分析", display_text="開く →")})
        st.caption("気になる銘柄を登録しておくと、値動きとRSIシグナルをまとめて"
                   "確認できます。「開く →」で銘柄分析ページに移動します。")

st.divider()
st.subheader("🏆 本日のランキング(米国市場)")

tabs = st.tabs([label for label, _ in SCREENERS])
for tab, (label, kind) in zip(tabs, SCREENERS):
    with tab:
        try:
            df = data_fetcher.fetch_screener(kind, count=10)
        except data_fetcher.FetchError:
            st.warning("ランキングの取得に失敗しました。しばらく時間をおいて再試行してください。")
            continue
        if df.empty:
            st.info("データが取得できませんでした。")
            continue
        df = df.copy()
        df["リンク"] = "/?ticker=" + df["ティッカー"].astype(str)
        styled = df.style.format({
            "株価": "${:,.2f}",
            "前日比": "{:+.2f}%",
            "出来高": "{:,.0f}",
        }).map(_pct_color, subset=["前日比"])
        st.dataframe(styled, hide_index=True,
                     column_config={"リンク": st.column_config.LinkColumn(
                         "分析", display_text="開く →")})
st.caption("出典: Yahoo Financeスクリーナー(15分キャッシュ)。気になる銘柄は"
           "「銘柄分析」ページにティッカーを入力すると詳細を確認できます。")

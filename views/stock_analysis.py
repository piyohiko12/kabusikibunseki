"""銘柄分析ページ。"""

import html

import pandas as pd
import streamlit as st

from lib import (charts, data_fetcher, indicators, levels, moomoo_client,
                 news_fetcher, sensitivity, settings_store, ui)

# 表示ラベル → (取得期間, 表示日数)。SMA200を期間の先頭から描くため長めに取得する。
PERIODS = {
    "1ヶ月": ("2y", 30),
    "6ヶ月": ("2y", 182),
    "1年": ("3y", 365),
    "5年": ("10y", 1826),
}

CHART_TYPES = ["ローソク足", "平均足", "ライン"]
INTERVALS = {"1分": "1m", "5分": "5m", "15分": "15m", "1時間": "1h",
             "日足": "1d", "週足": "1wk", "月足": "1mo"}
# 分足はYahooの無料提供範囲に上限がある: (取得期間, 表示できる最大日数)
INTRADAY_LIMITS = {"1m": ("5d", 5), "5m": ("1mo", 30),
                   "15m": ("1mo", 30), "1h": ("1y", 365)}
OVERLAY_OPTIONS = ["SMA20", "SMA50", "SMA200", "EMA20", "EMA50", "VWAP(日中)",
                   "ボリンジャーバンド", "一目均衡表", "サポレジライン",
                   "フィボナッチ", "出来高プロファイル"]
# サポレジの上位足マージ: 表示中の足 → 参照する上位足
HTF_MAP = {"1m": "日足", "5m": "日足", "15m": "日足", "1h": "日足",
           "1d": "週足", "1wk": "月足"}
OSC_OPTIONS = ["出来高", "RSI", "MACD", "ストキャスティクス"]

# ワンクリックで用途別の表示に切り替えるプリセット
PRESETS = {
    "🧭 シンプル": {
        "overlays": ["SMA50", "SMA200"],
        "oscillators": ["出来高"],
    },
    "📐 テクニカル": {
        "overlays": ["SMA20", "SMA50", "SMA200", "サポレジライン"],
        "oscillators": ["出来高", "RSI", "MACD"],
    },
    "🎯 スイング": {
        "overlays": ["SMA50", "SMA200", "サポレジライン", "フィボナッチ",
                     "出来高プロファイル"],
        "oscillators": ["出来高", "RSI"],
    },
    "⚡ デイトレ": {
        "overlays": ["VWAP(日中)", "SMA20", "ボリンジャーバンド", "サポレジライン"],
        "oscillators": ["出来高", "ストキャスティクス"],
    },
}
CUSTOM = "⚙️ カスタム"
BENCHMARKS = {"S&P500": "^GSPC", "NASDAQ総合": "^IXIC", "ダウ平均": "^DJI"}
NEWS_SOURCES = ["Yahoo Finance", "Google News", "🇯🇵 日本語", "SEC開示"]
PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "eraseshape"],
}


def md_escape(text: str) -> str:
    """Markdown/LaTeX として解釈されないように投稿本文をエスケープする。"""
    return text.replace("$", "\\$").replace("#", "\\#")


def fmt(value, suffix="", digits=2):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:,.{digits}f}{suffix}"


def fmt_market_cap(value):
    if value is None:
        return "—"
    if value >= 1e12:
        return f"{value / 1e12:,.2f}兆ドル"
    return f"{value / 1e8:,.0f}億ドル"


st.title("📈 銘柄分析")

_settings = settings_store.load()
_initial = (st.query_params.get("ticker")
            or _settings.get("default_ticker") or "AAPL").strip().upper()

col_ticker, col_star, col_period = st.columns([1, 0.5, 1.7])
with col_ticker:
    ticker = st.text_input(
        "ティッカーシンボル", value=_initial, placeholder="例: AAPL",
    ).strip().upper()
with col_star:
    st.markdown('<div style="height:1.8rem"></div>', unsafe_allow_html=True)
    if st.button("⭐ 既定にする",
                 help="次回からこの銘柄を最初に表示します",
                 disabled=(not ticker
                           or ticker == _settings.get("default_ticker"))):
        settings_store.save(default_ticker=ticker)
        st.toast(f"起動時の銘柄を {ticker} に設定しました", icon="⭐")
with col_period:
    period_label = st.radio("表示期間", list(PERIODS), index=1, horizontal=True)

if not ticker:
    st.info("ティッカーシンボルを入力してください。")
    st.stop()

# URLに反映(ブックマークやランキングからのリンクに使える)
if st.query_params.get("ticker") != ticker:
    st.query_params["ticker"] = ticker

# 最近見た銘柄(セッション内)
_recent = st.session_state.setdefault("recent_tickers", [])
if ticker in _recent:
    _recent.remove(ticker)
_recent.insert(0, ticker)
del _recent[6:]
if len(_recent) > 1:
    links = " ".join(f"[{t}](/?ticker={t})" for t in _recent[1:])
    st.caption(f"最近見た銘柄: {links}")

fetch_period, display_days = PERIODS[period_label]

try:
    hist = data_fetcher.fetch_history(ticker, fetch_period)
except data_fetcher.FetchError:
    st.error("データの取得中にエラーが発生しました。ネットワーク接続を確認し、"
             "しばらく時間をおいてから再試行してください。")
    st.stop()

if hist.empty:
    st.error(f"ティッカー「{ticker}」のデータを取得できませんでした。"
             "ティッカーシンボルが正しいか確認してください(例: AAPL, MSFT, GOOGL)。")
    st.stop()

try:
    info = data_fetcher.fetch_info(ticker)
except data_fetcher.FetchError:
    info = {}

latest = hist["Close"].iloc[-1]
prev = hist["Close"].iloc[-2] if len(hist) > 1 else latest
change = latest - prev
change_pct = (latest / prev - 1) * 100 if prev else 0.0

with_ind = indicators.add_indicators(hist)
view = indicators.slice_display(with_ind, display_days)
year = with_ind.loc[with_ind.index >= with_ind.index.max() - pd.Timedelta(days=365)]
rsi_now = view["RSI"].iloc[-1] if not view.empty else None

st.subheader(f"{info.get('name', ticker)}({ticker})")
badges = [ui.chip(info[k], color) for k, color in
          (("sector", "blue"), ("industry", "violet")) if info.get(k)]
if badges:
    st.markdown(" ".join(badges), unsafe_allow_html=True)
    st.markdown("")

# moomooが使えるときは遅延のない現在値に差し替える(使えなければYahooの終値のまま)
realtime = moomoo_client.snapshot((ticker,)).get(ticker)
if realtime and realtime.get("price"):
    price_now = float(realtime["price"])
    base = realtime.get("previous_close") or prev
    change = price_now - float(base)
    change_pct = (price_now / float(base) - 1) * 100 if base else 0.0
    price_label = "株価(リアルタイム)"
else:
    price_now = latest
    price_label = "株価(直近終値)"

m1, m2, m3, m4 = st.columns(4)
m1.metric(price_label, f"${price_now:,.2f}",
          f"{change:+,.2f}({change_pct:+.2f}%)", border=True)
m2.metric("52週高値", f"${year['High'].max():,.2f}", border=True)
m3.metric("52週安値", f"${year['Low'].min():,.2f}", border=True)
if rsi_now is not None and pd.notna(rsi_now):
    if rsi_now >= 70:
        rsi_delta, rsi_color = "買われすぎ", "inverse"
    elif rsi_now <= 30:
        rsi_delta, rsi_color = "売られすぎ", "normal"
    else:
        rsi_delta, rsi_color = "中立圏", "off"
    m4.metric("RSI(14)", f"{rsi_now:.1f}", rsi_delta,
              delta_color=rsi_color, border=True)
else:
    m4.metric("RSI(14)", "—", border=True)

if realtime:
    st.caption(f"🟢 株価はmoomooのリアルタイム値です(更新: "
               f"{realtime.get('update_time') or '—'})。"
               "チャート・指標はYahoo Financeの日足を使用しています。")
else:
    st.caption("株価はYahoo Financeの値で、15〜20分遅れています。"
               "サイドバーの「moomooリアルタイム連携」を有効にすると即時値になります。")

tab_chart, tab_news, tab_tape, tab_flow = st.tabs(
    ["📊 チャート・指標", "📰 ニュース・ネットの反応", "🔬 板・歩み値", "🏦 需給・IV"])

# ---------------------------------------------------------------- タブ1
with tab_chart:
    # 前回の表示設定を復元する(data/settings.json に保存)
    _saved_chart = _settings.get("chart") or {}
    _preset_names = list(PRESETS) + [CUSTOM]
    _saved_preset = _saved_chart.get("preset")
    if _saved_preset not in _preset_names:
        _saved_preset = "📐 テクニカル"

    # 保存値が壊れていても既定値で開けるようにする
    _def_type = _saved_chart.get("chart_type")
    _def_type = _def_type if _def_type in CHART_TYPES else "ローソク足"
    _def_bar = _saved_chart.get("bar_label")
    _def_bar = _def_bar if _def_bar in INTERVALS else "日足"

    c_type, c_interval = st.columns([2.2, 3])
    with c_type:
        chart_type = st.pills("チャート種別", CHART_TYPES,
                              default=_def_type) or _def_type
    with c_interval:
        bar_label = st.pills("足の間隔", list(INTERVALS),
                             default=_def_bar) or _def_bar

    c_preset, c_cfg = st.columns([3.4, 1.2])
    with c_preset:
        preset = st.pills("表示プリセット", _preset_names,
                          default=_saved_preset) or _saved_preset
    with c_cfg:
        st.markdown('<div style="height:1.8rem"></div>', unsafe_allow_html=True)
        cfg_pop = st.popover("⚙️ 詳細設定", use_container_width=True)

    # プリセットを選んだらその内容、カスタムなら前回の選択を初期値にする
    if preset in PRESETS:
        base_overlays = PRESETS[preset]["overlays"]
        base_oscs = PRESETS[preset]["oscillators"]
    else:
        base_overlays = _saved_chart.get("overlays") or ["SMA50", "サポレジライン"]
        base_oscs = _saved_chart.get("oscillators") or ["出来高", "RSI"]

    with cfg_pop:
        st.caption("プリセットを上書きすると「カスタム」として保存されます。")
        overlays = st.multiselect("オーバーレイ(価格に重ねる指標)",
                                  OVERLAY_OPTIONS,
                                  default=[o for o in base_overlays
                                           if o in OVERLAY_OPTIONS])
        oscillators = st.multiselect("サブチャート", OSC_OPTIONS,
                                     default=[o for o in base_oscs
                                              if o in OSC_OPTIONS])
        col_a, col_b = st.columns(2)
        show_events = col_a.toggle("配当・分割マーカー",
                                   value=_saved_chart.get("events", True))
        log_scale = col_b.toggle("対数スケール",
                                 value=_saved_chart.get("log_scale", False))
        _heights = {360: "低い", 430: "標準", 520: "やや高い",
                    640: "高い", 780: "最大"}
        _def_h = _saved_chart.get("height")
        _def_h = _def_h if _def_h in _heights else 430
        chart_height = st.select_slider(
            "チャートの高さ", options=list(_heights),
            value=_def_h, format_func=lambda v: _heights[v])
        benches = st.multiselect("パフォーマンス比較", list(BENCHMARKS), default=[])

    # 設定が変わったら保存(次回起動時も同じ見た目で開ける)
    _now_chart = {
        "preset": preset if (preset in PRESETS
                             and sorted(overlays) == sorted(PRESETS[preset]["overlays"])
                             and sorted(oscillators) == sorted(PRESETS[preset]["oscillators"]))
        else CUSTOM,
        "chart_type": chart_type, "bar_label": bar_label,
        "overlays": overlays, "oscillators": oscillators,
        "events": show_events, "log_scale": log_scale, "height": chart_height,
    }
    if _now_chart != _saved_chart:
        settings_store.save(chart=_now_chart)

    interval = INTERVALS[bar_label]

    chart_view = view
    chart_period = fetch_period
    chart_days = display_days
    limit_note = ""
    if interval in INTRADAY_LIMITS:
        chart_period, cap = INTRADAY_LIMITS[interval]
        chart_days = min(display_days, cap)
        if display_days > cap:
            limit_note = (f"※ {bar_label}足はYahooの無料提供範囲の都合で"
                          f"直近{cap}日分まで表示します。")
    elif interval != "1d":
        chart_period = "10y" if interval == "1wk" else "max"

    if interval != "1d":
        try:
            chart_hist = data_fetcher.fetch_history(ticker, chart_period, interval)
        except data_fetcher.FetchError:
            chart_hist = pd.DataFrame()
        if chart_hist.empty:
            st.warning("この足の間隔のデータを取得できなかったため、日足で表示しています。")
            interval = "1d"
            chart_view = view
        else:
            chart_view = indicators.slice_display(
                indicators.add_indicators(chart_hist), chart_days)

    lv_list = levels.find_levels(chart_view)

    # 上位足のサポレジをマージ(分足→日足、日足→週足、週足→月足)
    htf_label = HTF_MAP.get(interval)
    if htf_label and lv_list:
        if htf_label == "日足":
            htf_view = view
        else:
            htf_iv = "1wk" if htf_label == "週足" else "1mo"
            try:
                htf_hist = data_fetcher.fetch_history(
                    ticker, "10y" if htf_iv == "1wk" else "max", htf_iv)
            except data_fetcher.FetchError:
                htf_hist = pd.DataFrame()
            htf_view = (indicators.slice_display(
                indicators.add_indicators(htf_hist), max(display_days * 4, 730))
                if not htf_hist.empty else pd.DataFrame())
        if not htf_view.empty:
            lv_list = levels.merge_mtf(lv_list, levels.find_levels(htf_view),
                                       htf_label)

    lv_chart = ([l for l in lv_list if l["type"] == "抵抗線"][:3]
                + [l for l in lv_list if l["type"] == "サポート"][:3])

    opts = {
        "chart_type": chart_type,
        "interval": interval,
        "overlays": overlays,
        "oscillators": oscillators,
        "events": show_events,
        "log_scale": log_scale,
        "levels": lv_chart,
        "height": chart_height,
    }
    st.plotly_chart(charts.price_chart(chart_view, ticker, opts),
                    config=PLOT_CONFIG)
    st.caption("💡 十字カーソルで価格と日付を読めます。ドラッグで拡大、ダブルクリックで戻る。"
               "右上のツールバーからトレンドライン・矩形の描画も可能です。"
               "◆=配当、★=株式分割。赤帯=抵抗ゾーン、緑帯=サポートゾーン"
               "(濃く太いほど強いレベル)。"
               + (f" {limit_note}" if limit_note else ""))

    if benches:
        series = {ticker: chart_view["Close"]}
        for label in benches:
            try:
                bhist = data_fetcher.fetch_history(
                    BENCHMARKS[label], chart_period, interval)
            except data_fetcher.FetchError:
                bhist = pd.DataFrame()
            if not bhist.empty:
                series[label] = indicators.slice_display(bhist, chart_days)["Close"]
        st.subheader("パフォーマンス比較(期間始点=100)")
        st.plotly_chart(charts.comparison_chart(series, interval))

    if lv_list:
        st.subheader("🧱 サポート / レジスタンス(表示期間ベース)")

        for kind, msg in levels.level_alerts(
                lv_list, float(chart_view["Close"].iloc[-1])):
            (st.warning if kind == "testing" else st.info)(msg)

        def _rate_str(lv):
            return (f"{lv['bounces']}/{lv['touches']}回({lv['bounce_rate']:.0f}%)"
                    if lv["touches"] else "—")

        nearest_r, nearest_s = levels.nearest_levels(lv_list)
        n1, n2 = st.columns(2)
        n1.metric("直近の抵抗線",
                  f"${nearest_r['price']:,.2f}" if nearest_r else "—",
                  (f"{nearest_r['distance_pct']:+.1f}%・反発{_rate_str(nearest_r)}"
                   if nearest_r else None),
                  delta_color="inverse", border=True)
        n2.metric("直近のサポート",
                  f"${nearest_s['price']:,.2f}" if nearest_s else "—",
                  (f"{nearest_s['distance_pct']:+.1f}%・反発{_rate_str(nearest_s)}"
                   if nearest_s else None),
                  delta_color="inverse", border=True)

        lv_table = pd.DataFrame([{
            "種別": lv["type"],
            "価格": lv["price"],
            "現在比": lv["distance_pct"],
            "反発実績": _rate_str(lv),
            "ヒゲ拒絶": lv.get("rejects", 0),
            "強さ": "★" * lv["strength"],
            "根拠": lv["basis"] + (" / " + "・".join(lv["confluence"])
                                   if lv["confluence"] else ""),
        } for lv in lv_list])
        styled_lv = lv_table.style.format(
            {"価格": "${:,.2f}", "現在比": "{:+.1f}%"}
        ).map(lambda v: "color: #d03b3b" if v == "抵抗線" else "color: #006300",
              subset=["種別"])
        st.dataframe(styled_lv, hide_index=True)
        st.caption("反発実績=接近時に反転した回数/接近回数。ヒゲ拒絶=実体では入らず"
                   "ヒゲだけが刺さって押し戻された回数(反発の強い証拠)。"
                   "「日足合流」等は上位足でも同じレベルが確認できたもの。"
                   "期間を切り替えると再計算されます。")

        with st.expander("ℹ️ 「強さ★」の意味と、検証でわかった限界"):
            st.markdown(
                "★は **レベル同士の優劣を並べるための相対評価** です。"
                "S&P500構成銘柄の2013〜2018年の日次データで、6ヶ月分から算出した"
                "レベルが40本先までにどうなったかを実測して較正しました。\n\n"
                "較正に使っていない80銘柄での結果:\n\n"
                "| 強さ | 接近後に反発した割合 |\n|---|---|\n"
                "| ★5 | 65.0% |\n| ★4 | 58.0% |\n| ★3 | 60.2% |\n"
                "| ★2 | 57.6% |\n| ★1 | 58.9% |\n\n"
                "**★5は★1より約6pt反発しやすい**、という程度の差です。"
                "★4以下の差はほとんどありません。\n\n"
                "⚠️ さらに重要な限界として、**同じ距離にランダムに引いた線と比べた"
                "反発率の差はほぼゼロ**でした。つまりレベルに触れたあと反発するか"
                "抜けるかは、この手法では予測できていません。"
                "★が高いレベルを「相対的に注目度が高い価格帯」として見る使い方に"
                "留め、売買判断の根拠にはしないでください。")

        with st.expander("📋 詳細データ(ゾーン範囲・テスト履歴)"):
            st.dataframe(pd.DataFrame([{
                "種別": lv["type"],
                "ゾーン": f"${lv['zone_low']:,.2f}〜${lv['zone_high']:,.2f}",
                "中心": f"${lv['price']:,.2f}",
                "スイング数": lv["swings"],
                "反発": lv["bounces"],
                "突破": lv["breaks"],
                "直近テスト": lv["last_touch"],
                "根拠": lv["basis"] + (" / " + "・".join(lv["confluence"])
                                       if lv["confluence"] else ""),
            } for lv in lv_list]), hide_index=True)

        st.subheader("💡 参考指値(自動計算)")
        sugg_risk = st.number_input(
            "1トレードの許容損失額(ドル)", min_value=10.0, value=100.0, step=10.0,
            help="損切りまで逆行した場合に失ってよい金額。推奨株数の計算に使います")
        sugg = levels.suggest_limit_orders(chart_view, lv_list, sugg_risk)
        if sugg:
            s_table = pd.DataFrame([{
                "シナリオ": s["scenario"],
                "指値価格": s["price"],
                "損切り目安": s["stop"],
                "利確目安": s["target"],
                "RR比": s["rr"],
                "推奨株数": s["shares"],
                "想定利益": s["est_profit"],
            } for s in sugg])
            styled_s = s_table.style.format({
                "指値価格": "${:,.2f}",
                "損切り目安": lambda v: "—" if pd.isna(v) else f"${v:,.2f}",
                "利確目安": lambda v: "—" if pd.isna(v) else f"${v:,.2f}",
                "RR比": lambda v: "—" if pd.isna(v) else f"1:{v:.1f}",
                "推奨株数": lambda v: "—" if pd.isna(v) else f"{v:,.0f}株",
                "想定利益": lambda v: "—" if pd.isna(v) else f"${v:,.0f}",
            }).map(lambda v: "color: #006300" if "買い" in str(v)
                   else ("color: #d03b3b" if "売り" in str(v) else ""),
                   subset=["シナリオ"])
            st.dataframe(styled_s, hide_index=True)
            st.caption("⚠️ 買い指値=サポートゾーン上端、損切り=ゾーン下端−0.5ATR、"
                       "利確=直近抵抗ゾーン下端。推奨株数=許容損失額÷1株あたりの損切り幅。"
                       "機械算出の参考値であり投資助言ではありません。")

        piv = levels.pivot_points(chart_view)
        if piv:
            with st.expander("参考: ピボットポイント(クラシック方式、直近1本前ベース)"):
                st.dataframe(pd.DataFrame(
                    [{k: f"${v:,.2f}" for k, v in piv.items()}]), hide_index=True)

    st.subheader("ファンダメンタル指標")
    if not info:
        st.warning("ファンダメンタル情報を取得できませんでした。")
    else:
        roe = info.get("roe")
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("PER", fmt(info.get("per"), " 倍"), border=True,
                  help="株価収益率。利益の何倍まで買われているか")
        f2.metric("PBR", fmt(info.get("pbr"), " 倍"), border=True,
                  help="株価純資産倍率")
        f3.metric("ROE", fmt(roe * 100 if roe is not None else None, " %"),
                  border=True, help="自己資本利益率")
        f4.metric("配当利回り", fmt(data_fetcher.dividend_yield_percent(info), " %"),
                  border=True)
        f5.metric("時価総額", fmt_market_cap(info.get("market_cap")), border=True)

    st.subheader("⚡ イベント感応度")
    sens_rows = sensitivity.evaluate(hist, info,
                                     data_fetcher.fetch_earnings_history(ticker))
    if not sens_rows:
        st.info("感応度を評価するためのデータが不足しています。")
    else:
        st.dataframe(pd.DataFrame(sens_rows), hide_index=True)
        st.caption("決算・FOMCは実際のイベント日直後の変動を平常時(日次変動の中央値)と"
                   "比較した実測値。マクロ要因は過去2年の日次リターンの相関/ベータ。"
                   "★が多いほどそのイベント・要因に反応しやすい銘柄です。")

    st.subheader("業績推移(過去4年)")
    try:
        fin = data_fetcher.fetch_annual_financials(ticker)
    except data_fetcher.FetchError:
        fin = pd.DataFrame()

    if fin.empty:
        st.warning("財務データを取得できませんでした。")
    else:
        col_rev, col_eps = st.columns(2)
        with col_rev:
            st.plotly_chart(charts.revenue_chart(fin))
        with col_eps:
            st.plotly_chart(charts.eps_chart(fin))

# ---------------------------------------------------------------- タブ2
with tab_news:
    analyst = data_fetcher.fetch_analyst(ticker)
    if analyst["targets"] or analyst["ratings"] or analyst["earnings_date"]:
        st.subheader("🎯 アナリスト評価・イベント")
        a1, a2, a3, a4 = st.columns(4)
        tg = analyst["targets"]
        if tg:
            upside = (tg["mean"] / latest - 1) * 100 if latest else 0.0
            a1.metric("平均目標株価", f"${tg['mean']:,.2f}",
                      f"{upside:+.1f}%(現在比)", border=True)
            a2.metric("目標レンジ", f"${tg['low']:,.0f}〜${tg['high']:,.0f}",
                      border=True)
        else:
            a1.metric("平均目標株価", "—", border=True)
            a2.metric("目標レンジ", "—", border=True)
        eps = analyst["eps_estimate"]
        a3.metric("次回決算日", analyst["earnings_date"] or "—",
                  f"予想EPS ${eps:.2f}" if eps else None,
                  delta_color="off", border=True)
        ratings = analyst["ratings"]
        a4.metric("アナリスト数", f"{sum(ratings.values())}名" if ratings else "—",
                  border=True)
        if ratings:
            st.plotly_chart(ui.rating_bar(ratings),
                            config={"displayModeBar": False})
        if analyst["changes"]:
            action_ja = {"up": "⬆️ 引き上げ", "down": "⬇️ 引き下げ",
                         "init": "🆕 新規", "reit": "維持", "main": "維持"}
            with st.expander("直近の格付け変更"):
                st.dataframe(pd.DataFrame([
                    {"日付": c["date"], "会社": c["firm"], "評価": c["grade"],
                     "変更": action_ja.get(c["action"], c["action"] or "—"),
                     "目標株価": f"${c['target']:,.0f}" if c["target"] else "—"}
                    for c in analyst["changes"]
                ]), hide_index=True)
        st.divider()

    col_news, col_social = st.columns([3, 2])

    with col_news:
        st.subheader("📰 最新ニュース")
        news_src = st.pills("ニュースソース", NEWS_SOURCES,
                            default="Yahoo Finance") or "Yahoo Finance"
        try:
            if news_src == "Google News":
                news = news_fetcher.fetch_google_news(ticker, "en")
            elif news_src == "🇯🇵 日本語":
                news = news_fetcher.fetch_google_news(ticker, "ja")
            elif news_src == "SEC開示":
                news = news_fetcher.fetch_sec_filings(ticker)
            else:
                news = news_fetcher.fetch_news(ticker)
        except data_fetcher.FetchError:
            news = []
            st.warning("ニュースの取得に失敗しました。しばらく時間をおいて再試行してください。")
        if not news:
            st.info("このソースでは情報が見つかりませんでした。")
        if news_src == "SEC開示":
            st.caption("米SEC(証券取引委員会)への公式開示書類です。リンク先は英語の原文です。")
        for n in news:
            with st.container(border=True):
                title = f"[{n['title']}]({n['url']})" if n["url"] else n["title"]
                st.markdown(f"**{title}**")
                chip_color = "violet" if n["provider"] == "SEC EDGAR" else "gray"
                meta = ui.chip(html.escape(n["provider"] or "ニュース"), chip_color)
                meta += (f' <span style="color:#898781;font-size:0.8rem;">'
                         f'{ui.relative_time(n["pub_date"])}</span>')
                st.markdown(meta, unsafe_allow_html=True)
                if n["summary"]:
                    summary = n["summary"]
                    st.write(md_escape(summary[:280] + ("…" if len(summary) > 280 else "")))

    with col_social:
        st.subheader("💬 ネットの反応")
        social = news_fetcher.fetch_social(ticker)
        if social["errors"]:
            st.warning("一部ソースの取得に失敗しました: " + ", ".join(social["errors"]))

        if social["total"] == 0:
            st.info("この銘柄への投稿が見つかりませんでした。")
        else:
            s1, s2, s3 = st.columns(3)
            s1.metric("投稿数", f"{social['total']}件", border=True)
            s2.metric("🐂 強気", f"{social['bullish']}件", border=True)
            s3.metric("🐻 弱気", f"{social['bearish']}件", border=True)
            neutral = social["total"] - social["bullish"] - social["bearish"]
            st.plotly_chart(
                ui.sentiment_bar(social["bullish"], social["bearish"], neutral),
                config={"displayModeBar": False})

            terms = news_fetcher.trending_terms(social["posts"], ticker)
            if terms:
                chips = " ".join(ui.chip(f"{t} ×{c}", "blue") for t, c in terms)
                st.markdown("**🔥 一緒に言及されている銘柄・タグ**<br>" + chips,
                            unsafe_allow_html=True)
                st.markdown("")

            selected = st.pills("表示するソース", news_fetcher.SOCIAL_SOURCES,
                                selection_mode="multi",
                                default=news_fetcher.SOCIAL_SOURCES)
            sent_filter = st.pills("センチメントで絞り込み",
                                   ["すべて", "🐂 強気", "🐻 弱気"],
                                   default="すべて") or "すべて"
            st.caption("投稿は原文(主に英語)のまま表示しています。")
            shown = 0
            for p in social["posts"]:
                if p["source"] not in (selected or []):
                    continue
                if sent_filter == "🐂 強気" and p["sentiment"] != "Bullish":
                    continue
                if sent_filter == "🐻 弱気" and p["sentiment"] != "Bearish":
                    continue
                if shown >= 15:
                    break
                shown += 1
                with st.container(border=True):
                    when = ui.relative_time(p["created_at"])
                    when_html = (f'<a href="{p["url"]}" target="_blank" '
                                 f'style="color:#898781;font-size:0.8rem;">{when} ↗</a>'
                                 if p["url"] else
                                 f'<span style="color:#898781;font-size:0.8rem;">{when}</span>')
                    likes = f'&nbsp;<span style="color:#898781;font-size:0.8rem;">♥ {p["likes"]}</span>' if p["likes"] else ""
                    st.markdown(
                        f'{ui.source_chip(p["source"])} {ui.sentiment_chip(p["sentiment"])} '
                        f'<b>{html.escape(p["user"])}</b> · {when_html}{likes}',
                        unsafe_allow_html=True)
                    body = p["body"]
                    st.write(md_escape(body[:300] + ("…" if len(body) > 300 else "")))



# ---------------------------------------------------------------- タブ3
with tab_tape:
    state = moomoo_client.status()
    if state["state"] != "ok":
        st.info(
            "このタブはmoomoo OpenAPI(無料)に接続すると使えます。\n\n"
            f"現在の状態: **{state['message']}**\n\n"
            "1. moomoo証券の口座でログインできる **moomoo OpenD** をPCで起動する\n"
            "2. サイドバーの「moomooリアルタイム連携」を有効にする\n\n"
            "接続しても取得できるのは相場データだけで、このツールは発注を一切行いません。"
        )
    else:
        st.caption(f"{ticker} の板・歩み値(moomoo・自動更新はしません。"
                   "最新にするにはページを再読み込みしてください)")
        col_book, col_tick = st.columns([1, 1])

        with col_book:
            book = moomoo_client.order_book(ticker)
            if not book:
                st.warning("板情報を取得できませんでした。"
                           "米国株の板情報には対応する相場権限が必要です。")
            else:
                st.plotly_chart(charts.depth_chart(book["bids"], book["asks"]),
                                config={"displayModeBar": False})
                if book["bids"] and book["asks"]:
                    spread = book["asks"][0][0] - book["bids"][0][0]
                    mid = (book["asks"][0][0] + book["bids"][0][0]) / 2
                    bid_vol = sum(v for _p, v, _n in book["bids"])
                    ask_vol = sum(v for _p, v, _n in book["asks"])
                    b1, b2 = st.columns(2)
                    b1.metric("スプレッド", f"${spread:,.3f}",
                              f"{spread / mid * 100:.3f}%" if mid else None,
                              delta_color="off", border=True)
                    total = bid_vol + ask_vol
                    b2.metric("買い板の厚み", f"{bid_vol / total * 100:.0f}%" if total else "—",
                              f"買{bid_vol:,} / 売{ask_vol:,}",
                              delta_color="off", border=True)

        with col_tick:
            ticks = moomoo_client.recent_ticks(ticker, num=60)
            if ticks.empty:
                st.warning("歩み値を取得できませんでした。")
            else:
                buy = int((ticks.get("ticker_direction") == "BUY").sum())
                sell = int((ticks.get("ticker_direction") == "SELL").sum())
                if buy or sell:
                    st.plotly_chart(ui.stacked_bar([
                        ("買い約定", buy, "#0ca30c", "#ffffff"),
                        ("売り約定", sell, "#d03b3b", "#ffffff"),
                    ]), config={"displayModeBar": False})
                st.dataframe(
                    ticks.rename(columns={
                        "time": "時刻", "price": "価格", "volume": "数量",
                        "turnover": "代金", "ticker_direction": "方向", "type": "種別"}),
                    hide_index=True, height=420,
                    column_config={
                        "価格": st.column_config.NumberColumn(format="$%.2f"),
                        "数量": st.column_config.NumberColumn(format="%,d"),
                        "代金": st.column_config.NumberColumn(format="$%,.0f"),
                    })

        cap = moomoo_client.capital_distribution(ticker)
        if cap:
            st.divider()
            c_chart, c_note = st.columns([1.4, 1])
            with c_chart:
                st.plotly_chart(charts.capital_bar(cap["tiers"]),
                                config={"displayModeBar": False})
            with c_note:
                st.markdown("#### 資金の出入り")
                st.metric("本日の純流入", f"${cap['net']:+,.0f}",
                          "買い越し" if cap["net"] >= 0 else "売り越し",
                          delta_color="normal" if cap["net"] >= 0 else "inverse",
                          border=True)
                st.caption("大口ほど機関投資家の動きを反映しやすい参考情報です。"
                           "売買代金の内訳であり、将来の値動きを示すものではありません。")


# ---------------------------------------------------------------- タブ4
with tab_flow:
    if moomoo_client.status()["state"] != "ok":
        st.info("このタブはmoomoo OpenAPI(無料)に接続すると使えます。"
                "サイドバーの「moomooリアルタイム連携」から設定してください。")
    else:
        st.caption(f"{ticker} の需給(空売り・機関投資家)とオプションの変動率。"
                   "いずれもmoomooから取得した参考情報です。")

        # --- 空売り残高 ---
        st.markdown("#### 🐻 空売り残高")
        shorts = moomoo_client.short_interest(ticker)
        if shorts.empty:
            st.caption("空売り残高を取得できませんでした(対象外の銘柄か、権限がありません)。")
        else:
            latest_s = shorts.iloc[-1]
            prev_s = shorts.iloc[-2] if len(shorts) > 1 else latest_s
            s1, s2, s3 = st.columns(3)
            delta_shares = latest_s["空売り株数"] - prev_s["空売り株数"]
            s1.metric("空売り株数", f"{latest_s['空売り株数']:,.0f}株",
                      f"{delta_shares:+,.0f}", delta_color="inverse", border=True)
            if pd.notna(latest_s["浮動株比率"]):
                s2.metric("浮動株に対する比率", f"{latest_s['浮動株比率']:.2f}%",
                          f"{latest_s['浮動株比率'] - prev_s['浮動株比率']:+.2f}pt"
                          if pd.notna(prev_s["浮動株比率"]) else None,
                          delta_color="inverse", border=True)
            if pd.notna(latest_s["買い戻し日数"]):
                s3.metric("買い戻しにかかる日数", f"{latest_s['買い戻し日数']:.1f}日",
                          "高いほど踏み上げが起きやすい", delta_color="off", border=True)
            st.plotly_chart(charts.short_interest_chart(shorts),
                            config={"displayModeBar": False})

        # --- 機関投資家の保有推移 ---
        st.divider()
        st.markdown("#### 🏦 機関投資家の保有推移")
        inst = moomoo_client.institutional_holding(ticker)
        if inst.empty:
            st.caption("機関投資家の保有データを取得できませんでした。")
        else:
            latest_i = inst.iloc[-1]
            i1, i2, i3 = st.columns(3)
            i1.metric("保有機関数", f"{latest_i['機関数']:,.0f}",
                      f"{latest_i['機関数の増減']:+,.0f}"
                      if pd.notna(latest_i["機関数の増減"]) else None, border=True)
            if pd.notna(latest_i["保有比率"]):
                i2.metric("機関の保有比率", f"{latest_i['保有比率']:.2f}%",
                          f"{latest_i['保有比率の増減']:+.2f}pt"
                          if pd.notna(latest_i["保有比率の増減"]) else None, border=True)
            i3.metric("最新の報告期", str(latest_i["報告期"]), border=True)
            st.plotly_chart(charts.institution_chart(inst),
                            config={"displayModeBar": False})
            st.caption("四半期ごとの報告(13F等)に基づくため、実際の売買からは遅れます。")

        # --- オプションのIV ---
        st.divider()
        st.markdown("#### 🌪️ オプションの変動率(IV / HV)")
        vol = moomoo_client.option_volatility(ticker)
        if not vol:
            st.caption("オプションの変動率を取得できませんでした"
                       "(オプションが上場していない銘柄の可能性があります)。")
        else:
            v_chart, v_note = st.columns([1.6, 1])
            with v_chart:
                st.plotly_chart(charts.iv_hv_chart(vol["series"]),
                                config={"displayModeBar": False})
            with v_note:
                last_v = vol["series"].iloc[-1]
                if pd.notna(last_v.get("IV")):
                    st.metric("現在のIV", f"{last_v['IV']:.1f}%",
                              f"HVとの差 {last_v['IVプレミアム']:+.1f}pt"
                              if pd.notna(last_v.get("IVプレミアム")) else None,
                              delta_color="off", border=True)
                if vol.get("average_iv") is not None and pd.notna(vol["average_iv"]):
                    st.metric("平均IV", f"{vol['average_iv']:.1f}%", border=True)
                if vol.get("analysis"):
                    st.info(vol["analysis"])
                st.caption("IVが高いほどオプション市場が今後の大きな値動きを"
                           "見込んでいることを示します。HVを大きく上回るときは"
                           "決算などのイベントが控えている場合があります。")

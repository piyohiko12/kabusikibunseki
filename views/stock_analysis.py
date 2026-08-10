"""銘柄分析ページ。"""

import html

import pandas as pd
import streamlit as st

from lib import (charts, data_fetcher, gap, indicators, intraday, levels,
                 moomoo_client, news_fetcher, sensitivity, settings_store, ui)

# 表示ラベル → (取得期間, 表示日数)。SMA200を期間の先頭から描くため長めに取得する。
PERIODS = {
    "1ヶ月": ("2y", 30),
    "6ヶ月": ("2y", 182),
    "1年": ("3y", 365),
    "5年": ("10y", 1826),
}

CHART_TYPES = ["ローソク足", "平均足", "OHLCバー", "ライン", "エリア"]
INTERVALS = {"1分": "1m", "5分": "5m", "15分": "15m", "1時間": "1h",
             "日足": "1d", "週足": "1wk", "月足": "1mo"}
# 取得量とyfinanceフォールバック互換性を保つための範囲: (取得期間, 最大日数)
INTRADAY_LIMITS = {"1m": ("5d", 5), "5m": ("1mo", 30),
                   "15m": ("1mo", 30), "1h": ("1y", 365)}
OVERLAY_OPTIONS = ["移動平均線(SMA)", "指数移動平均線(EMA)", "VWAP(日中)",
                   "ボリンジャーバンド", "一目均衡表", "サポレジライン",
                   "フィボナッチ", "出来高プロファイル"]
# サポレジの上位足マージ: 表示中の足 → 参照する上位足
HTF_MAP = {"1m": "日足", "5m": "日足", "15m": "日足", "1h": "日足",
           "1d": "週足", "1wk": "月足"}
OSC_OPTIONS = ["出来高", "RSI", "MACD", "ストキャスティクス"]

# ワンクリックで用途別の表示に切り替えるプリセット
BENCHMARKS = {"S&P500": "^GSPC", "NASDAQ総合": "^IXIC", "ダウ平均": "^DJI"}
NEWS_SOURCES = ["Yahoo Finance", "Google News", "🇯🇵 日本語", "SEC開示"]
PLOT_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "scrollZoom": True,
    "responsive": True,
    "doubleClick": "reset+autosize",
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawclosedpath",
                            "drawcircle", "drawrect", "eraseshape"],
    "toImageButtonOptions": {"format": "png", "scale": 2,
                             "filename": "stock-chart"},
}

CHART_PRESETS = {
    "標準": {
        "interval": "日足", "chart_type": "ローソク足",
        "overlays": ["移動平均線(SMA)", "サポレジライン"],
        "oscillators": ["出来高", "RSI", "MACD"],
    },
    "デイトレ": {
        "interval": "5分", "chart_type": "ローソク足",
        "overlays": ["指数移動平均線(EMA)", "VWAP(日中)",
                     "ボリンジャーバンド"],
        "oscillators": ["出来高", "MACD"],
        # デイトレは寄り前・引け後の値動きが判断材料になるので既定でオン
        "extended_hours": True,
    },
    "スイング": {
        "interval": "日足", "chart_type": "ローソク足",
        "overlays": ["移動平均線(SMA)", "ボリンジャーバンド",
                     "サポレジライン"],
        "oscillators": ["出来高", "RSI", "MACD"],
    },
    "長期": {
        "interval": "週足", "chart_type": "エリア",
        "overlays": ["移動平均線(SMA)", "サポレジライン"],
        "oscillators": ["出来高", "RSI"],
    },
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
    hist, _base_meta = data_fetcher.fetch_chart_history(
        ticker, fetch_period, "1d")
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

snapshot = data_fetcher.fetch_realtime_snapshot(ticker)
hist_latest = float(hist["Close"].iloc[-1])
hist_prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else hist_latest
latest = snapshot.get("price") or hist_latest
prev = snapshot.get("previous_close") or hist_prev
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
# snapshotはdata_fetcher.fetch_realtime_snapshot経由で1回だけ取得済み。
if snapshot.get("price"):
    price_now = float(snapshot["price"])
    base = snapshot.get("previous_close") or prev
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

if snapshot.get("source") == "moomoo OpenAPI":
    spread = ""
    if snapshot.get("bid") is not None and snapshot.get("ask") is not None:
        spread = f"・Bid ${snapshot['bid']:,.2f} / Ask ${snapshot['ask']:,.2f}"
    updated = f"・更新 {snapshot['update_time']}" if snapshot.get("update_time") else ""
    st.caption(f"⚡ 最新価格: moomoo OpenAPI({snapshot.get('code', ticker)})"
               f"{spread}{updated}。財務・ニュースはYahoo Finance等を併用。")
else:
    reason = snapshot.get("fallback_reason")
    st.caption("データ源: Yahoo Finance"
               + (f"(moomooフォールバック: {reason})" if reason else ""))

tab_chart, tab_day, tab_news, tab_tape, tab_flow = st.tabs(
    ["📊 チャート・指標", "🌅 当日・寄付", "📰 ニュース・ネットの反応",
     "🔬 板・歩み値", "🏦 需給・IV"])

# ---------------------------------------------------------------- タブ1
with tab_chart:
    _saved_adv = _settings.get("advanced_chart") or {}
    _saved_preset = _saved_adv.get("preset")
    if _saved_preset not in CHART_PRESETS:
        _saved_preset = "標準"

    c_preset, c_type, c_interval, c_action, c_cfg, c_refresh = st.columns(
        [1.35, 2.1, 2.15, 1.65, 0.75, 0.45])
    with c_preset:
        chart_preset = st.pills(
            "分析プリセット", list(CHART_PRESETS), default=_saved_preset,
            key="chart_preset") or _saved_preset
    preset_cfg = CHART_PRESETS[chart_preset]
    use_saved = _saved_adv.get("preset") == chart_preset
    default_type = (_saved_adv.get("chart_type") if use_saved
                    else preset_cfg["chart_type"])
    if default_type not in CHART_TYPES:
        default_type = preset_cfg["chart_type"]
    default_interval = (_saved_adv.get("bar_label") if use_saved
                        else preset_cfg["interval"])
    if default_interval not in INTERVALS:
        default_interval = preset_cfg["interval"]
    with c_type:
        chart_type = st.pills("チャート種別", CHART_TYPES,
                              default=default_type,
                              key=f"chart_type_{chart_preset}") or "ローソク足"
    with c_interval:
        bar_label = st.pills(
            "足の間隔", list(INTERVALS), default=default_interval,
            key=f"chart_interval_{chart_preset}") or default_interval
    with c_action:
        default_interaction = _saved_adv.get("interaction", "クロスヘア")
        if default_interaction not in ["クロスヘア", "ズーム", "移動", "ライン描画"]:
            default_interaction = "クロスヘア"
        interaction = st.pills(
            "マウス操作", ["クロスヘア", "ズーム", "移動", "ライン描画"],
            default=default_interaction,
            key="chart_interaction") or default_interaction
    with c_cfg:
        with st.popover("⚙️ 詳細", width="stretch"):
            tab_ind, tab_look = st.tabs(["指標", "表示・パネル"])
            with tab_ind:
                overlays = st.multiselect(
                    "メインチャート指標", OVERLAY_OPTIONS,
                    default=(_saved_adv.get("overlays", preset_cfg["overlays"])
                             if use_saved else preset_cfg["overlays"]),
                    key=f"overlays_{chart_preset}")
                oscillators = st.multiselect(
                    "サブチャート(複数可)", OSC_OPTIONS,
                    default=(_saved_adv.get("oscillators", preset_cfg["oscillators"])
                             if use_saved else preset_cfg["oscillators"]),
                    key=f"oscillators_{chart_preset}")

                st.caption("移動平均線・ボリンジャーバンド")
                saved_params = _saved_adv.get("indicator_params") or {}
                p1, p2, p3 = st.columns(3)
                sma_fast = p1.number_input("短期SMA", 2, 100,
                                           int(saved_params.get("sma_periods", [20, 50, 200])[0]),
                                           key=f"sma_fast_{chart_preset}")
                sma_mid = p2.number_input("中期SMA", 5, 200,
                                          int(saved_params.get("sma_periods", [20, 50, 200])[1]),
                                          key=f"sma_mid_{chart_preset}")
                sma_long = p3.number_input("長期SMA", 20, 500,
                                           int(saved_params.get("sma_periods", [20, 50, 200])[2]),
                                           key=f"sma_long_{chart_preset}")
                p1, p2, p3 = st.columns(3)
                ema_fast = p1.number_input("短期EMA", 2, 100,
                                           int(saved_params.get("ema_periods", [20, 50])[0]),
                                           key=f"ema_fast_{chart_preset}")
                ema_slow = p2.number_input("長期EMA", 3, 200,
                                           int(saved_params.get("ema_periods", [20, 50])[1]),
                                           key=f"ema_slow_{chart_preset}")
                boll_period = p3.number_input("BOLL期間", 5, 100,
                                              int(saved_params.get("boll_period", 20)),
                                              key=f"boll_period_{chart_preset}")
                boll_std = st.slider(
                    "BOLL標準偏差", 0.5, 4.0,
                    float(saved_params.get("boll_std", 2.0)), 0.1,
                    key=f"boll_std_{chart_preset}")

                st.caption("オシレーター")
                p1, p2, p3 = st.columns(3)
                rsi_period = p1.number_input("RSI", 2, 50,
                                             int(saved_params.get("rsi_period", 14)),
                                             key=f"rsi_period_{chart_preset}")
                macd_fast = p2.number_input("MACD短期", 2, 50,
                                             int(saved_params.get("macd_fast", 12)),
                                             key=f"macd_fast_{chart_preset}")
                macd_slow_input = p3.number_input(
                    "MACD長期", 3, 100, int(saved_params.get("macd_slow", 26)),
                    key=f"macd_slow_{chart_preset}")
                p1, p2, p3 = st.columns(3)
                macd_signal = p1.number_input(
                    "MACDシグナル", 2, 50,
                    int(saved_params.get("macd_signal", 9)),
                    key=f"macd_signal_{chart_preset}")
                stoch_period = p2.number_input(
                    "STOCH期間", 3, 50, int(saved_params.get("stoch_period", 14)),
                    key=f"stoch_period_{chart_preset}")
                volume_ma = p3.number_input(
                    "出来高MA", 2, 100, int(saved_params.get("volume_ma", 20)),
                    key=f"volume_ma_{chart_preset}")
                show_signals = st.toggle(
                    "MACD / RSIテクニカルイベント",
                    value=bool(_saved_adv.get("show_signals", False)),
                    key=f"signals_{chart_preset}")

            with tab_look:
                chart_theme = st.segmented_control(
                    "チャートテーマ", ["ダーク", "ライト"],
                    default=_saved_adv.get("theme", "ダーク"),
                    key="chart_theme") or "ダーク"
                color_scheme = st.segmented_control(
                    "値動きの色", ["緑上昇 / 赤下落", "赤上昇 / 緑下落"],
                    default=_saved_adv.get("color_scheme", "緑上昇 / 赤下落"),
                    key="chart_color_scheme") or \
                    "緑上昇 / 赤下落"
                show_events = st.toggle("配当・分割マーカー",
                                        value=bool(_saved_adv.get("events", True)))
                current_price_line = st.toggle(
                    "現在値ライン", value=bool(_saved_adv.get("current_price_line", True)))
                show_grid = st.toggle("グリッド",
                                      value=bool(_saved_adv.get("grid", True)))
                range_selector = st.toggle(
                    "期間ショートカット", value=bool(_saved_adv.get("range_selector", True)))
                range_slider = st.toggle(
                    "期間スライダー", value=bool(_saved_adv.get("range_slider", False)))
                compact_sessions = st.toggle(
                    "休場時間を詰める", value=bool(_saved_adv.get("compact_sessions", True)))
                extended_hours = st.toggle(
                    "時間外も表示(プレ・アフター)",
                    value=bool(_saved_adv.get("extended_hours",
                                              preset_cfg.get("extended_hours", False))),
                    key=f"extended_{chart_preset}",
                    help="分足のとき、プレマーケット(04:00〜)とアフターマーケット"
                         "(〜20:00 ET)のローソク足も表示します。"
                         "時間外は薄い背景色で区別されます。")
                log_scale = st.toggle(
                    "対数スケール(価格軸)", value=bool(_saved_adv.get("log_scale", False)))
                _heights = {360: "低い", 430: "標準", 520: "やや高い",
                            640: "高い", 780: "最大"}
                saved_height = int(_saved_adv.get("height", 520))
                if saved_height not in _heights:
                    saved_height = 520
                chart_height = st.select_slider(
                    "チャートの高さ", options=list(_heights), value=saved_height,
                    format_func=lambda value: _heights[value])
                multi_timeframe = st.toggle(
                    "マルチタイムフレーム(2画面追加)",
                    value=bool(_saved_adv.get("multi_timeframe", False)))
                show_order_book = st.toggle(
                    "moomoo板情報(読み取り専用)",
                    value=bool(_saved_adv.get("show_order_book", False)),
                    help="OpenDと相場権限が利用できる場合のみ表示します。注文は行いません",
                )
                benches = st.multiselect(
                    "パフォーマンス比較", list(BENCHMARKS),
                    default=_saved_adv.get("benchmarks", []))
    with c_refresh:
        st.markdown('<div style="height:1.8rem"></div>', unsafe_allow_html=True)
        if st.button("↻", help="最新データを再取得", key="refresh_chart"):
            data_fetcher.fetch_chart_history.clear()
            data_fetcher.fetch_order_book.clear()
            moomoo_client.snapshot.clear()
            st.rerun()

    macd_slow = max(int(macd_fast) + 1, int(macd_slow_input))
    indicator_params = {
        "sma_periods": (int(sma_fast), int(sma_mid), int(sma_long)),
        "ema_periods": (int(ema_fast), int(ema_slow)),
        "boll_period": int(boll_period), "boll_std": float(boll_std),
        "rsi_period": int(rsi_period), "macd_fast": int(macd_fast),
        "macd_slow": macd_slow, "macd_signal": int(macd_signal),
        "stoch_period": int(stoch_period), "stoch_k": 3, "stoch_d": 3,
        "volume_ma": int(volume_ma),
    }
    _now_adv = {
        "preset": chart_preset, "chart_type": chart_type,
        "bar_label": bar_label, "interaction": interaction,
        "overlays": overlays, "oscillators": oscillators,
        "indicator_params": indicator_params, "show_signals": show_signals,
        "theme": chart_theme, "color_scheme": color_scheme,
        "events": show_events, "current_price_line": current_price_line,
        "grid": show_grid, "range_selector": range_selector,
        "range_slider": range_slider, "compact_sessions": compact_sessions,
        "extended_hours": extended_hours,
        "log_scale": log_scale, "height": chart_height,
        "multi_timeframe": multi_timeframe,
        "show_order_book": show_order_book, "benchmarks": benches,
    }
    if _now_adv != _saved_adv:
        settings_store.save(advanced_chart=_now_adv)
    interval = INTERVALS[bar_label]

    chart_period = fetch_period
    chart_days = display_days
    limit_note = ""
    if interval in INTRADAY_LIMITS:
        chart_period, cap = INTRADAY_LIMITS[interval]
        chart_days = min(display_days, cap)
        if display_days > cap:
            limit_note = (f"※ {bar_label}足は取得量を抑えるため"
                          f"直近{cap}日分まで表示します。")
    elif interval != "1d":
        chart_period = "10y" if interval == "1wk" else "max"

    # 時間外のバーが返るのは分足だけ。日足以上では指定しても意味がない。
    show_extended = bool(extended_hours) and interval in INTRADAY_LIMITS
    try:
        chart_hist, chart_meta = data_fetcher.fetch_chart_history(
            ticker, chart_period, interval, show_extended)
    except data_fetcher.FetchError:
        chart_hist, chart_meta = pd.DataFrame(), {"source": "取得失敗"}
    if chart_hist.empty:
        st.warning("この足の間隔のデータを取得できなかったため、日足で表示しています。")
        interval = "1d"
        show_extended = False
        try:
            chart_hist, chart_meta = data_fetcher.fetch_chart_history(
                ticker, fetch_period, interval)
        except data_fetcher.FetchError:
            chart_hist = hist
            chart_meta = {"source": "Yahoo Finance", "code": ticker,
                          "fallback_reason": "moomoo・再取得とも利用不可"}
        chart_days = display_days
    if chart_hist.empty:
        chart_hist = hist
        chart_meta = {"source": "Yahoo Finance", "code": ticker,
                      "fallback_reason": "moomooからデータを取得できませんでした"}
    chart_view = indicators.slice_display(
        indicators.add_indicators(chart_hist, indicator_params), chart_days)

    lv_list = levels.find_levels(chart_view)

    # 上位足のサポレジをマージ(分足→日足、日足→週足、週足→月足)
    htf_label = HTF_MAP.get(interval)
    if htf_label and lv_list:
        if htf_label == "日足":
            htf_iv, htf_period = "1d", fetch_period
        else:
            htf_iv = "1wk" if htf_label == "週足" else "1mo"
            htf_period = "10y" if htf_iv == "1wk" else "max"
        try:
            htf_hist, _ = data_fetcher.fetch_chart_history(
                ticker, htf_period, htf_iv)
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
        "indicator_params": indicator_params,
        "theme": chart_theme,
        "color_scheme": color_scheme,
        "grid": show_grid,
        "range_slider": range_slider,
        "range_selector": range_selector,
        "current_price_line": current_price_line,
        "signals": show_signals,
        "compact_sessions": compact_sessions,
        "extended_hours": show_extended,
        "interaction": interaction,
        "current_price": snapshot.get("price"),
    }
    st.plotly_chart(charts.price_chart(chart_view, ticker, opts),
                    config=PLOT_CONFIG, key=f"main_chart_{ticker}")
    source_text = (f"データ源: {chart_meta.get('source', '不明')}"
                   f"({chart_meta.get('code', ticker)})")
    if chart_meta.get("fallback_reason"):
        reason = str(chart_meta["fallback_reason"])
        source_text += f" / moomooフォールバック: {reason[:160]}"
    st.caption(source_text)
    st.caption("💡 ホイール=拡大縮小、ダブルクリック=リセット、凡例クリック=線の表示/非表示。"
               "右上のツールバーでトレンドライン・パス・円・矩形を描画できます。"
               " ◆=配当、★=株式分割。赤帯=抵抗ゾーン、緑帯=サポートゾーン"
               "(濃く太いほど強いレベル)。"
               + (f" {limit_note}" if limit_note else ""))

    if multi_timeframe:
        st.subheader("🔲 マルチタイムフレーム")
        if interval in INTRADAY_LIMITS:
            mtf_specs = [("日足・6ヶ月", "1d", "2y", 182),
                         ("週足・5年", "1wk", "10y", 1826)]
        elif interval == "1d":
            mtf_specs = [("1時間足・30日", "1h", "1mo", 30),
                         ("週足・5年", "1wk", "10y", 1826)]
        elif interval == "1wk":
            mtf_specs = [("日足・1年", "1d", "2y", 365),
                         ("月足・10年", "1mo", "max", 3653)]
        else:
            mtf_specs = [("日足・1年", "1d", "2y", 365),
                         ("週足・5年", "1wk", "10y", 1826)]

        mtf_cols = st.columns(2)
        for col, (label, mtf_iv, mtf_period, mtf_days) in zip(mtf_cols, mtf_specs):
            try:
                mtf_hist, mtf_meta = data_fetcher.fetch_chart_history(
                    ticker, mtf_period, mtf_iv)
            except data_fetcher.FetchError:
                mtf_hist, mtf_meta = pd.DataFrame(), {"source": "取得失敗"}
            with col:
                if mtf_hist.empty:
                    st.info(f"{label}を取得できませんでした。")
                else:
                    mtf_view = indicators.slice_display(
                        indicators.add_indicators(mtf_hist), mtf_days)
                    st.plotly_chart(
                        charts.mini_price_chart(
                            mtf_view, label, interval=mtf_iv,
                            theme_name=chart_theme),
                        config={"displaylogo": False, "scrollZoom": True},
                        key=f"mtf_{ticker}_{mtf_iv}",
                    )
                    st.caption(f"{mtf_meta.get('source', '不明')} / SMA20・50")

    if show_order_book:
        st.subheader("📖 moomoo 板情報")
        try:
            order_book = data_fetcher.fetch_order_book(ticker, 10)
        except data_fetcher.FetchError as exc:
            st.info(f"板情報を取得できませんでした: {exc}")
        else:
            if order_book.empty:
                st.info("利用可能な板情報がありません。")
            else:
                styled_book = order_book.style.format({
                    "売数量": lambda v: "—" if pd.isna(v) else f"{v:,.0f}",
                    "売気配値": lambda v: "—" if pd.isna(v) else f"${v:,.3f}",
                    "買気配値": lambda v: "—" if pd.isna(v) else f"${v:,.3f}",
                    "買数量": lambda v: "—" if pd.isna(v) else f"{v:,.0f}",
                })
                st.dataframe(styled_book, hide_index=True)
                st.caption("OpenD経由の読み取り専用データです。相場権限により"
                           "表示段数やリアルタイム性が異なります。")

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
        sens_view = pd.DataFrame([{k: v for k, v in row.items()
                                   if not k.startswith("_")}
                                  for row in sens_rows])
        st.dataframe(sens_view, hide_index=True, width="stretch")
        st.caption("決算・FOMCは実際のイベント日直後の変動を平常時(日次変動の中央値)と"
                   "比較した実測値。マクロ要因は過去2年の日次リターンの相関/ベータ。"
                   "★が多いほど反応しやすく、「方向」は過去に上下どちらへ振れたかの"
                   "実績です(5回以上かつ65%以上のときだけ方向を出します)。")
        biased = [r for r in sens_rows
                  if r.get("_detail", {}).get("event")
                  and r["_detail"].get("biased")]
        if biased:
            lines = []
            for r in biased:
                d = r["_detail"]
                lines.append(f"- **{r['イベント・要因']}**: {d['direction_label']}"
                             f"・平均 {d['avg_signed']:+.1f}%"
                             f"(過去{d['n']}回)")
            st.markdown("**過去に方向の偏りが出ているイベント**\n" + "\n".join(lines))
            st.caption("⚠️ 過去の偏りであって、次回もそうなるという意味ではありません。"
                       "回数が少ないほど偶然の偏りが出やすい点にご注意ください。")

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

# ------------------------------------------------------- タブ2: 当日・寄付
with tab_day:
    TONE_ICON = {"up": "🟢", "down": "🔴", "flat": "⚪"}
    TONE_CHIP = {"up": "green", "down": "red", "flat": "gray"}

    try:
        day_hist, _day_meta = data_fetcher.fetch_chart_history(
            ticker, "5d", "5m", True)
    except data_fetcher.FetchError:
        day_hist = pd.DataFrame()

    prev_daily_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else None

    # ------------------------------------------------------- 当日トレンド
    st.subheader("⚡ 当日のトレンド")
    trend = intraday.analyze(day_hist, prev_daily_close) if not day_hist.empty else None
    if not trend:
        st.info("当日の分足データを取得できませんでした。"
                "市場が開いていない時間帯や、分足が提供されない銘柄では表示されません。")
    else:
        if trend.get("stale"):
            st.caption("※ 新しい取引日のバーがまだ少ないため、直前の取引日を表示しています。")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("当日トレンド",
                  f"{TONE_ICON[trend['tone']]} {trend['short']}",
                  f"{trend['score']:+.0f}点", delta_color="off", border=True)
        t2.metric("寄付からの変化", f"{trend['from_open_pct']:+.2f}%",
                  f"寄付 ${trend['regular_open']:,.2f}", delta_color="off",
                  border=True)
        t3.metric("VWAP", f"${trend['vwap']:,.2f}",
                  f"乖離 {trend['vwap_dev']:+.2f}%",
                  delta_color="normal" if trend["vwap_dev"] >= 0 else "inverse",
                  border=True)
        t4.metric("当日の高安",
                  f"${trend['day_low']:,.2f}〜${trend['day_high']:,.2f}",
                  border=True)
        st.markdown(f"**{trend['advice']}**")

        st.markdown("**判定の内訳**")
        st.dataframe(pd.DataFrame([{
            "観点": p["name"],
            "点数": f"{p['score']:+.1f} / ±{p['max']}",
            "実測値": p["value"],
            "見方": p["note"],
        } for p in trend["parts"]]), hide_index=True, width="stretch")
        st.caption("5つの観点を足し合わせた −100〜+100 のスコアです。"
                   "いま何が起きているかの要約であって、"
                   "この先の値動きの確率ではありません。")

        orb = trend.get("opening_range")
        if orb and orb["complete"]:
            st.markdown(
                f"**オープニングレンジ(寄り{orb['minutes']}分)**: "
                f"\\${orb['low']:,.2f} 〜 \\${orb['high']:,.2f}　"
                + ("🟢 上抜け済み" if orb["broke_up"] and not orb["broke_down"]
                   else "🔴 下抜け済み" if orb["broke_down"] and not orb["broke_up"]
                   else "🟡 上下とも抜けた(ダマシに注意)" if orb["broke_up"]
                   else "⚪ レンジ内"))

        if trend.get("sessions"):
            st.markdown("**セッション別の値動き**")
            st.dataframe(pd.DataFrame([{
                "セッション": s["label"],
                "始値": f"${s['open']:,.2f}",
                "終値": f"${s['close']:,.2f}",
                "高安": f"${s['low']:,.2f}〜${s['high']:,.2f}",
                "変化率": f"{s['change_pct']:+.2f}%",
                "出来高": f"{s['volume']:,.0f}",
                "本数": s["bars"],
            } for s in trend["sessions"]]), hide_index=True, width="stretch")
            st.caption("変化率は1つ前のセッションの終値からの変化です"
                       "(最初のセッションだけ前日の立会終値が基準)。")

    st.divider()

    # ----------------------------------------------------------- 寄付予想
    st.subheader("🌅 寄付の見通し")
    gap_daily = hist if not hist.empty else pd.DataFrame()
    table = gap.gap_table(gap_daily)
    if table.empty:
        st.info("ギャップ統計を出すための日足データが足りません。")
    else:
        ref = gap.extended_reference(day_hist, prev_daily_close)
        manual = None
        cols = st.columns([1.4, 1, 1.6])
        with cols[0]:
            if ref:
                st.markdown(
                    f"**{ref['label']}の最終値**: \\${ref['price']:,.2f}　"
                    f"({ref['bars']}本 / 高安 "
                    f"\\${ref['low']:,.2f}〜\\${ref['high']:,.2f})")
            else:
                st.caption("時間外のバーが無いため、想定ギャップを手入力して"
                           "過去実績を引けます。")
        with cols[1]:
            manual = st.number_input(
                "想定ギャップ(%)",
                value=float(round(ref["gap_pct"], 2)) if ref and ref["gap_pct"]
                else 0.0,
                min_value=-25.0, max_value=25.0, step=0.1, format="%.2f",
                help="時間外の値から自動入力されます。手で変えて試算もできます。")

        gap_now = float(manual)
        cond = gap.conditional(table, gap_now)
        proj = gap.projected_open(prev_daily_close, gap_now, cond)
        base = gap.baseline_for(gap_now)

        if proj:
            g1, g2, g3 = st.columns(3)
            g1.metric("前日終値", f"${proj['prev_close']:,.2f}", border=True)
            g2.metric("想定の寄付値", f"${proj['open_price']:,.2f}",
                      f"{gap_now:+.2f}%({proj['gap_amount']:+,.2f})",
                      delta_color="normal" if gap_now >= 0 else "inverse",
                      border=True)
            if "close_low" in proj:
                g3.metric("過去実績の引け値レンジ",
                          f"${proj['close_low']:,.2f}〜${proj['close_high']:,.2f}",
                          f"中央 ${proj['close_typical']:,.2f}",
                          delta_color="off", border=True)
            else:
                g3.metric("過去実績の引け値レンジ", "—", border=True)

        v = gap.verdict(cond, gap_now)
        if v and cond:
            # follow/fadeは「ギャップと同じ方向か」なので、上窓か下窓かで
            # 実際に上を向くかが変わる。色はその向きに合わせる。
            if v["tone"] == "neutral":
                v_icon = "⚪"
            else:
                v_icon = "🟢" if (v["tone"] == "follow") == (gap_now > 0) else "🔴"
            st.markdown(f"### {v_icon} {v['headline']}")
            st.markdown(f"{v['fill_note']}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("窓埋めした割合", f"{cond['fill_rate']:.0f}%", border=True)
            m2.metric("ギャップ方向に伸びた", f"{cond['follow_rate']:.0f}%", border=True)
            m3.metric("寄り後リターンの中央値", f"{cond['otc_median']:+.2f}%",
                      f"平均 {cond['otc_mean']:+.2f}%", delta_color="off",
                      border=True)
            m4.metric("参照した過去の回数", f"{cond['n']}回", border=True)

            if cond.get("widened_to_side"):
                st.warning("近いギャップ幅の事例が少なかったため、"
                           "同じ方向のギャップすべてを参照しています。"
                           "今回の大きさとは条件が違う点にご注意ください。", icon="⚠️")
            elif cond.get("width"):
                st.caption(f"※ {gap_now:+.2f}% の ±{cond['width']}% 以内の"
                           f"過去のギャップ {cond['n']}件を集計しています。")

            if base:
                st.markdown("**市場平均との比較**")
                st.dataframe(pd.DataFrame([
                    {"項目": "窓埋めした割合",
                     "この銘柄": f"{cond['fill_rate']:.0f}%",
                     "市場平均": f"{base['fill']:.0f}%"},
                    {"項目": "ギャップ方向に伸びた",
                     "この銘柄": f"{cond['follow_rate']:.0f}%",
                     "市場平均": f"{base['follow']:.0f}%"},
                    {"項目": "寄り後リターンの平均",
                     "この銘柄": f"{cond['otc_mean']:+.2f}%",
                     "市場平均": f"{base['otc']:+.2f}%"},
                ]), hide_index=True, width="stretch")
                st.caption(f"市場平均は「{base['label']}」区分の実測値"
                           f"({base['n']:,}件)。{gap.BASELINE_NOTE}")
        else:
            st.info("この銘柄には、今回と近いギャップの過去事例が足りません。")

        with st.expander("この銘柄のギャップ実績(区分ごと)"):
            buckets = gap.bucket_summary(table)
            if buckets:
                st.dataframe(pd.DataFrame([{
                    "ギャップ区分": b["label"],
                    "回数": b["n"],
                    "窓埋め": f"{b['fill_rate']:.0f}%",
                    "順行": f"{b['follow_rate']:.0f}%",
                    "逆行(寄り天/寄り底)": f"{b['fade_rate']:.0f}%",
                    "寄り後の平均": f"{b['otc_mean']:+.2f}%",
                    "当日レンジ": f"{b['range_mean']:.2f}%",
                } for b in buckets]), hide_index=True, width="stretch")
            ov = gap.overall(table)
            if ov:
                st.caption(
                    f"過去{ov['n']}営業日: ギャップの平均絶対値 {ov['mean_abs']:.2f}%・"
                    f"上窓 {ov['up_rate']:.0f}%・"
                    f"±0.3%未満(ほぼ窓なし)が {ov['quiet_rate']:.0f}%。"
                    f"5〜95パーセンタイルは {ov['p05']:+.2f}%〜{ov['p95']:+.2f}%。")

        st.caption("⚠️ ここに出る数字はすべて **過去の頻度** です。"
                   "「窓埋め60%」は過去にそうなった割合であって、"
                   "次回60%の確率で埋まるという意味ではありません。"
                   "決算やニュースがある日は、過去の分布があてになりません。")

# ---------------------------------------------------------------- タブ3
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

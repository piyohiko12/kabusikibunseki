"""銘柄分析ページ。"""

import html

import pandas as pd
import streamlit as st

from lib import (charts, daily_decision, data_fetcher, derivatives_context,
                 event_intelligence, indicators, level_review, levels,
                 moomoo_client, news_fetcher, sensitivity, session_intelligence,
                 settings_store, today_inputs, ui)

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


def fmt_utc_time(value) -> str:
    """API時刻をJSTで表示し、欠損をNaTのまま画面へ出さない。"""
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        return "—"
    return stamp.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d %H:%M JST")


def fmt_et_jst(value) -> str:
    """セッション時刻をET/JSTの両方で表示する。"""
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        return "—"
    et = stamp.tz_convert("America/New_York")
    jst = stamp.tz_convert("Asia/Tokyo")
    return f"{et:%m/%d %H:%M ET} / {jst:%m/%d %H:%M JST}"


def fetch_opening_market_features() -> dict:
    """寄付き診断用の市場特徴を履歴K線枠なしで取得する。

    SPY/QQQはmoomooの読み取り専用snapshotを優先する。先物と不足分は
    Yahoo Financeの5分足（遅延あり）を使い、最初の足からの変化として明示する。
    """
    result = {}
    try:
        live = moomoo_client.snapshot(("SPY", "QQQ"))
    except Exception:
        live = {}
    for ticker_key, feature_key in (("SPY", "spy_pct"), ("QQQ", "qqq_pct")):
        row = live.get(ticker_key) or {}
        value = row.get("change_percent")
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
        if value is not None and pd.notna(value):
            result[feature_key] = {
                "value": value, "timestamp": row.get("update_time"),
                "source": "moomoo OpenAPI snapshot（前日終値比）", "quality": 1.0,
            }

    try:
        intraday = data_fetcher.fetch_intraday_batch(
            ("ES=F", "SPY", "QQQ"), period="1d", interval="5m")
    except data_fetcher.FetchError:
        intraday = {}
    for ticker_key, feature_key in (
        ("ES=F", "futures_pct"), ("SPY", "spy_pct"), ("QQQ", "qqq_pct"),
    ):
        if feature_key in result:
            continue
        frame = intraday.get(ticker_key)
        if frame is None or frame.empty or "Close" not in frame:
            continue
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if len(close) < 2 or float(close.iloc[0]) <= 0:
            continue
        result[feature_key] = {
            "value": (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100,
            "timestamp": close.index[-1],
            "source": "Yahoo Finance 5分足（当日最初の足比・遅延あり）",
            "quality": 0.65,
        }
    return result


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
        ticker, fetch_period, "1d", allow_new_quota=True)
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

tab_today, tab_chart, tab_news, tab_tape, tab_flow, tab_derivatives = st.tabs([
    "🧭 今日の判断", "📊 チャート・指標", "📰 ニュース・ネットの反応",
    "🔬 板・歩み値", "🏦 需給・IV", "🌐 先物・PERP",
])

# ---------------------------------------------------------------- 今日の判断
with tab_today:
    st.caption("現在セッション → 当日の方向 → 支持抵抗 → 次回寄付き → イベントの順に"
               "確認します。ここでの数値は説明可能な参考診断で、注文や利益を保証しません。")

    eligibility = today_inputs.overnight_eligibility(snapshot)
    market_state = data_fetcher.fetch_market_state(ticker)
    session_state = session_intelligence.detect_current_session(
        market_state=market_state.get("market_state"),
        overnight_eligible=eligibility,
    )
    session_rows = today_inputs.session_prices(snapshot, hist.iloc[-1])
    session_changes = session_intelligence.compute_session_changes(
        session_rows, previous_close=prev)
    session_labels = {
        "premarket": "プレ", "regular": "立会", "afterhours": "アフター",
        "overnight": "夜間・24h",
    }
    session_names = {
        "premarket": "プレマーケット", "regular": "立会時間",
        "afterhours": "アフターマーケット", "overnight": "夜間・24時間帯",
        "closed": "セッション外・休場", "unknown": "取引可否を確認",
    }

    current = session_state["session"]
    session_color = ("green" if current == "regular" else
                     "blue" if current in {"premarket", "afterhours", "overnight"}
                     else "orange")
    st.markdown(
        ui.chip(f"現在: {session_names.get(current, current)}", session_color)
        + " " + ui.chip(
            session_state["as_of"].strftime("%m/%d %H:%M ET"), "gray"),
        unsafe_allow_html=True,
    )

    session_cols = st.columns(4)
    for column, key in zip(session_cols,
                           ("premarket", "regular", "afterhours", "overnight")):
        row = session_changes["sessions"][key]
        price = row.get("price")
        change_vs_close = row.get("change_vs_previous_close_pct")
        value = "—" if price is None else f"${price:,.2f}"
        delta = ("データなし" if change_vs_close is None
                 else f"前日終値比 {change_vs_close:+.2f}%")
        column.metric(
            session_labels[key] + (" ●" if session_state.get("calendar_session") == key else ""),
            value, delta, delta_color="off", border=True)
        if row.get("source"):
            column.caption(str(row["source"]))
    if snapshot.get("source") != "moomoo OpenAPI":
        st.info("moomoo snapshotを取得できないため、確定日足だけを立会欄に表示しています。"
                "欠損したプレ・アフター・夜間価格を終値で推測していません。")
    elif eligibility is None:
        st.caption("夜間値がないだけでは24時間取引の対象外と断定しません。"
                   "対象可否はmoomooの銘柄詳細でも確認してください。")

    st.markdown("#### 当日の方向と重要価格帯")
    trend = daily_decision.intraday_trend(snapshot, hist.iloc[-1])
    level_frame = with_ind.tail(252)
    today_levels = levels.find_levels(level_frame)
    nearby = daily_decision.nearest_levels(today_levels, price_now, min_strength=3)
    support, resistance = nearby["support"], nearby["resistance"]
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("当日の方向", trend["label"],
              ("強さ —" if trend["strength"] is None
               else f"観測一致度 {trend['strength']:.0f}%"),
              delta_color="off", border=True)
    t2.metric("最寄り支持帯（★3以上）",
              "—" if support is None else f"${support['edge_price']:,.2f}",
              "データ不足" if support is None else f"現在値から {support['distance_pct']:+.2f}%",
              delta_color="off", border=True)
    t3.metric("最寄り抵抗帯（★3以上）",
              "—" if resistance is None else f"${resistance['edge_price']:,.2f}",
              "データ不足" if resistance is None else f"現在値から {resistance['distance_pct']:+.2f}%",
              delta_color="off", border=True)
    rr = nearby.get("reward_risk")
    t4.metric("支持帯までの下方余地 : 抵抗帯までの上方余地",
              "—" if rr is None else f"1 : {rr:.2f}",
              "両側の強い帯が必要", delta_color="off", border=True)

    check_df = pd.DataFrame([{
        "観測": row["label"],
        "方向": {"up": "上向き", "down": "下向き", "flat": "横ばい",
                "unknown": "未取得"}.get(row["status"], row["status"]),
        "実測": row["actual"],
    } for row in trend["checks"]])
    st.dataframe(check_df, hide_index=True, use_container_width=True)
    quality_label = ("リアルタイムsnapshot" if trend["data_quality"] == "realtime"
                     else "直近確定日足" if trend["data_quality"] == "close_only"
                     else "取得不能")
    st.caption(f"方向のデータ品質: {quality_label}。支持抵抗は既存の検証済み検出器から"
               "★3以上だけを抜粋し、反発保証ではなく損益幅の確認に使います。")

    st.markdown("#### 次回寄付き・セッション開始の方向診断")
    target_labels = {
        "regular": "次の立会寄付き", "premarket": "次のプレ開始",
        "afterhours": "次のアフター開始", "overnight": "次の夜間開始",
    }
    target_session = st.selectbox(
        "予想する開始時点", list(target_labels),
        format_func=lambda key: target_labels[key], key=f"today_target_{ticker}")
    load_today = st.button(
        "寄付き・イベント診断を読み込む", type="primary",
        key=f"load_today_intelligence_{ticker}_{target_session}",
        help="必要なYahooデータとニュースをこの操作時だけ取得します。moomoo過去K線枠は使いません。",
    )

    result_store = st.session_state.setdefault("today_intelligence_results", {})
    result_key = (ticker, target_session)
    if load_today:
        try:
            with st.spinner("市場・寄付き・イベント影響を整理中..."):
                event_report = event_intelligence.fetch_event_intelligence(
                    ticker, include_news=True, horizon_days=120)
                market_features = fetch_opening_market_features()
                features = today_inputs.opening_features(
                    hist, snapshot, market_features, event_report)
                session_report = session_intelligence.analyze_session_intelligence(
                    market_state=market_state.get("market_state"),
                    overnight_eligible=eligibility,
                    session_prices=session_rows,
                    previous_close=prev,
                    target_session=target_session,
                    features=features,
                )
            result_store[result_key] = {
                "loaded_at": pd.Timestamp.now(tz="UTC"),
                "session": session_report, "events": event_report,
            }
            while len(result_store) > 8:
                result_store.pop(next(iter(result_store)))
        except Exception as exc:
            st.warning(f"寄付き・イベント診断を取得できませんでした: {exc}")

    result = result_store.get(result_key)
    if result:
        loaded_at = pd.to_datetime(result.get("loaded_at"), utc=True, errors="coerce")
        if (pd.isna(loaded_at)
                or pd.Timestamp.now(tz="UTC") - loaded_at > pd.Timedelta(minutes=15)):
            result_store.pop(result_key, None)
            result = None

    if result:
        diagnosis = result["session"]["next_open_diagnosis"]
        direction_labels = {
            "up": "上向き", "down": "下向き", "neutral": "方向拮抗",
            "unknown": "入力不足で判定しない",
        }
        up_probability = diagnosis.get("probability_up")
        down_probability = diagnosis.get("probability_down")
        quality = diagnosis["data_quality"]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("参考方向", direction_labels.get(diagnosis["direction"], "—"),
                  "未校正ヒューリスティック", delta_color="off", border=True)
        d2.metric("上向き参考確率（未校正）",
                  "—" if up_probability is None else f"{up_probability * 100:.0f}%",
                  border=True)
        d3.metric("下向き参考確率（未校正）",
                  "—" if down_probability is None else f"{down_probability * 100:.0f}%",
                  border=True)
        d4.metric("データ品質", f"{quality['score'] * 100:.0f}%",
                  f"方向特徴 {quality['directional_feature_count']}件",
                  delta_color="off", border=True)
        st.caption("対象開始: " + fmt_et_jst(
            diagnosis.get("target_open", {}).get("open_time")))
        if up_probability is None:
            st.warning(diagnosis["reason"])
        else:
            st.info(diagnosis["reason"])

        feature_df = pd.DataFrame([{
            "特徴": row["label"],
            "値": "—" if row["value"] is None else f"{row['value']:.3f}",
            "状態": row["status"], "固定重み": row["weight"],
            "寄与": row["contribution"], "取得元": row.get("source") or "—",
        } for row in diagnosis["features"]])
        with st.expander("参考確率の内訳（固定重み・データ鮮度）"):
            st.dataframe(feature_df.style.format({
                "固定重み": "{:.2f}", "寄与": "{:+.3f}",
            }), hide_index=True, use_container_width=True)
            st.caption(diagnosis["disclaimer"])

        event_report = result["events"]
        st.markdown("#### イベントの影響度と上下シナリオ")
        visible_events = list(event_report.get("events") or [])[:8]
        if not visible_events:
            st.info("表示期間内にイベント候補を確認できませんでした。")
        else:
            event_df = pd.DataFrame([{
                "日付": event.get("event_date") or "—",
                "時刻(ET)": event.get("event_time_et") or "—",
                "イベント": event["name"], "影響セッション": event["session_label"],
                "影響度": f"{event['impact_level']} ({event['impact_score']})",
                "過去方向": event["directional_bias"],
                "根拠信頼度": f"{event['confidence']}%",
            } for event in visible_events])
            st.dataframe(event_df, hide_index=True, use_container_width=True)
            for event in visible_events[:4]:
                with st.expander(
                    f"{event.get('event_date') or '日付不明'} — {event['name']} "
                    f"({event['impact_level']})"):
                    scenarios = event["scenarios"]
                    st.markdown(f"- 上向き: {scenarios['up']}\n"
                                f"- 下向き: {scenarios['down']}\n"
                                f"- 両方向: {scenarios['two_sided']}")
                    if event.get("evidence"):
                        st.caption("根拠: " + " / ".join(event["evidence"][:3]))
            st.caption("イベントは不確実性の確認材料で、売買スコアへ自動加点していません。"
                       "過去方向は将来の上下を保証せず、発表値と市場予想の差を確認してください。")
        for warning in event_report.get("warnings") or []:
            st.warning(warning)
    else:
        st.info("ボタンを押すと、市場の直近5分足・イベント日程・ニュースを必要時だけ"
                "読み込みます。データが不足・古い場合は確率を表示しません。")

    st.link_button("🎯 詳細な売買判定・アラートへ", f"/signals?ticker={ticker}")

# ---------------------------------------------------------------- チャート・指標
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

    try:
        chart_hist, chart_meta = data_fetcher.fetch_chart_history(
            ticker, chart_period, interval, allow_new_quota=True)
    except data_fetcher.FetchError:
        chart_hist, chart_meta = pd.DataFrame(), {"source": "取得失敗"}
    if chart_hist.empty:
        st.warning("この足の間隔のデータを取得できなかったため、日足で表示しています。")
        interval = "1d"
        try:
            chart_hist, chart_meta = data_fetcher.fetch_chart_history(
                ticker, fetch_period, interval, allow_new_quota=True)
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
                ticker, htf_period, htf_iv, allow_new_quota=True)
        except data_fetcher.FetchError:
            htf_hist = pd.DataFrame()
        htf_view = (indicators.slice_display(
            indicators.add_indicators(htf_hist), max(display_days * 4, 730))
            if not htf_hist.empty else pd.DataFrame())
        if not htf_view.empty:
            lv_list = levels.merge_mtf(lv_list, levels.find_levels(htf_view),
                                       htf_label)

    # 再評価の適用状態は銘柄・足・期間・最新バーが一致する間だけ有効。
    # 時間軸を切り替えた際に古い板情報を持ち越さない。
    base_lv_list = [dict(level) for level in lv_list]
    review_context = (f"{ticker}|{interval}|{chart_period}|{len(chart_view)}|"
                      f"{chart_view.index[-1] if not chart_view.empty else 'empty'}")
    applied_review = st.session_state.get("moomoo_level_review_applied")
    review_is_applied = bool(
        applied_review
        and applied_review.get("context") == review_context
        and applied_review.get("levels")
    )
    if review_is_applied:
        lv_list = [dict(level) for level in applied_review["levels"]]

    lv_chart = (sorted(
        [level for level in lv_list if level["type"] == "抵抗線"],
        key=lambda level: abs(level["distance_pct"]))[:3]
        + sorted([level for level in lv_list if level["type"] == "サポート"],
                 key=lambda level: abs(level["distance_pct"]))[:3])

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
                    ticker, mtf_period, mtf_iv, allow_new_quota=True)
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

        st.markdown("#### 🤖 moomooデータによるAI再評価")
        st.caption(
            "OHLCVで算出した基礎レベルに、moomoo OpenAPIの板・歩み値・"
            "当日資金フローを重ねて更新案を作ります。moomooアプリ内AIの"
            "非公開チャットAPIではなく、根拠を確認できる読み取り専用の再評価です。")
        moomoo_state = data_fetcher.moomoo_status()
        if review_is_applied:
            st.success(
                f"再評価を適用中: {applied_review.get('summary', '—')} "
                f"({applied_review.get('reviewed_at', '時刻不明')})",
                icon="✅",
            )
            if st.button("基礎レベルに戻す", key=f"reset_level_review_{review_context}"):
                st.session_state.pop("moomoo_level_review_applied", None)
                st.rerun()

        if st.button(
            "moomooデータで再評価",
            key=f"run_level_review_{review_context}",
            type="primary" if not review_is_applied else "secondary",
            disabled=not moomoo_state.get("available", False),
            help="OpenDから板・歩み値・資金フローを取得して更新候補を作ります",
        ):
            warnings = []
            book = pd.DataFrame()
            ticks = pd.DataFrame()
            capital = None
            with st.spinner("moomooの需給データを確認しています..."):
                try:
                    book = data_fetcher.fetch_order_book(ticker, 10)
                except data_fetcher.FetchError as exc:
                    warnings.append(f"板情報: {exc}")
                try:
                    ticks = data_fetcher.fetch_recent_ticks(ticker, 60)
                except data_fetcher.FetchError as exc:
                    warnings.append(f"歩み値: {exc}")
                try:
                    capital = data_fetcher.fetch_capital_distribution(ticker)
                except data_fetcher.FetchError as exc:
                    warnings.append(f"資金フロー: {exc}")
                proposal = level_review.review_levels(
                    chart_view, base_lv_list, order_book=book,
                    capital=capital, ticks=ticks,
                )
            proposal.update({"context": review_context, "warnings": warnings})
            if proposal.get("sources"):
                st.session_state["moomoo_level_review_preview"] = proposal
            else:
                st.session_state.pop("moomoo_level_review_preview", None)
                st.warning("再評価に使える板・歩み値・資金フローを取得できませんでした。"
                           "OpenDのログイン状態と相場権限を確認してください。")

        if not moomoo_state.get("available", False):
            st.info(f"再評価を使うにはOpenDを起動してログインしてください。"
                    f"現在: {moomoo_state.get('message', '接続できません')}")

        preview = st.session_state.get("moomoo_level_review_preview")
        if preview and preview.get("context") == review_context:
            st.info(f"更新案: {preview['summary']} / 使用データ: "
                    f"{', '.join(preview['sources'])}", icon="💡")
            metrics = preview.get("metrics", {})
            m1, m2, m3 = st.columns(3)
            m1.metric("板の買い優勢度",
                      f"{metrics.get('book_imbalance', 0):+.0%}", border=True,
                      help="買い板数量−売り板数量を合計板数量で割った値")
            m2.metric("直近約定の買い優勢度",
                      f"{metrics.get('tick_imbalance', 0):+.0%}", border=True)
            m3.metric("当日資金純流入",
                      f"{metrics.get('capital_net', 0):+,.0f}", border=True)

            proposal_table = pd.DataFrame([{
                "種別": level["type"],
                "価格": level["price"],
                "基礎★": ("—" if level.get("base_strength", 0) == 0 else
                          "★" * level.get("base_strength", 1)),
                "再評価★": "★" * level["strength"],
                "スコア": level.get("moomoo_score"),
                "信頼度": level.get("moomoo_confidence", "低"),
                "根拠": level.get("moomoo_reason", "—"),
            } for level in preview["levels"]])
            st.dataframe(
                proposal_table.style.format({"価格": "${:,.2f}", "スコア": "{:.0f}"}),
                hide_index=True,
            )
            for warning in preview.get("warnings", []):
                st.caption(f"一部データ未取得: {warning}")
            apply_col, dismiss_col, _ = st.columns([1, 1, 3])
            if apply_col.button("更新案を適用", type="primary",
                                key=f"apply_level_review_{review_context}"):
                st.session_state["moomoo_level_review_applied"] = preview
                st.session_state.pop("moomoo_level_review_preview", None)
                st.rerun()
            if dismiss_col.button("却下", key=f"dismiss_level_review_{review_context}"):
                st.session_state.pop("moomoo_level_review_preview", None)
                st.rerun()

        st.caption("再評価★はライブ需給を加えた表示用評価です。基礎★の"
                   "ウォークフォワード検証結果とは別物で、板候補は時間とともに変化します。")

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
            "基礎★": ("—" if lv.get("base_strength", lv["strength"]) == 0 else
                      "★" * lv.get("base_strength", lv["strength"])),
            "再評価": (f"{lv.get('moomoo_score', 0):.0f}点・"
                      f"信頼度{lv.get('moomoo_confidence', '—')}"
                      if lv.get("moomoo_reviewed") else "—"),
            "根拠": (lv["basis"]
                    + (" / " + "・".join(lv["confluence"])
                       if lv["confluence"] else "")
                    + (" / moomoo: " + lv.get("moomoo_reason", "")
                       if lv.get("moomoo_reviewed") else "")),
        } for lv in lv_list])
        styled_lv = lv_table.style.format(
            {"価格": "${:,.2f}", "現在比": "{:+.1f}%"}
        ).map(lambda v: "color: #d03b3b" if v == "抵抗線" else "color: #006300",
              subset=["種別"])
        st.dataframe(styled_lv, hide_index=True)
        st.caption("反発実績=接近時に反転した回数/接近回数。ヒゲ拒絶=実体では入らず"
                   "ヒゲだけが刺さって押し戻された回数(反発の強い証拠)。"
                   "「日足合流」等は上位足でも同じレベルが確認できたもの。"
                   "適用中の強さは再評価★、基礎★は履歴データだけの評価です。"
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


# ---------------------------------------------------------------- タブ5
with tab_derivatives:
    st.caption(
        "関連市場の値動きを独立したコンテキストとして表示します。"
        "このタブのデータは売買判定の点数・アラート・注文には使用しません。")

    product = st.pills(
        "表示する商品", ["先物", "PERP"], default="先物",
        key=f"derivatives_product_{ticker}",
    ) or "先物"
    result_store = st.session_state.setdefault("derivatives_context_results", {})

    def remember_result(key, value):
        result_store[key] = value
        # 長時間に多数の組合せを試してもDataFrameを無制限に保持しない。
        while len(result_store) > 20:
            result_store.pop(next(iter(result_store)))

    def fresh_result(key, ttl_seconds):
        result = result_store.get(key)
        if not result:
            return None
        fetched_at = pd.to_datetime(
            (result.get("meta") or {}).get("fetched_at"), utc=True, errors="coerce")
        if pd.isna(fetched_at):
            result_store.pop(key, None)
            return None
        age = (pd.Timestamp.now(tz="UTC") - fetched_at).total_seconds()
        if age < -30 or age > ttl_seconds:
            result_store.pop(key, None)
            return None
        return result

    if product == "先物":
        defaults = list(derivatives_context.suggested_future_symbols(
            info.get("sector", ""), info.get("industry", ""), ticker))
        catalog = derivatives_context.FUTURES_CATALOG
        period_options = {"1ヶ月": "1mo", "3ヶ月": "3mo", "6ヶ月": "6mo"}
        c_symbols, c_period, c_load = st.columns([3.4, 1.2, 1.1])
        with c_symbols:
            selected_futures = st.multiselect(
                "関連先物（最大3本）", list(catalog), default=defaults,
                max_selections=3,
                format_func=lambda symbol: (
                    f"{symbol}  {catalog[symbol].get('name', symbol)}"),
                key=f"future_symbols_{ticker}",
            )
        with c_period:
            future_period_label = st.selectbox(
                "比較期間", list(period_options), index=1,
                key=f"future_period_{ticker}")
        future_period = period_options[future_period_label]
        future_key = (ticker, "futures", tuple(selected_futures), future_period)
        with c_load:
            st.markdown('<div style="height:1.8rem"></div>', unsafe_allow_html=True)
            future_load = st.button(
                "読み込む / 更新",
                type="primary", disabled=not selected_futures,
                key=f"load_futures_{ticker}")

        if future_load:
            if future_key in result_store:
                clear = getattr(derivatives_context.fetch_yahoo_futures, "clear", None)
                if clear:
                    clear(tuple(selected_futures), period=future_period)
            with st.spinner("関連先物を取得しています…"):
                try:
                    future_summary, future_normalized, future_meta = (
                        derivatives_context.fetch_yahoo_futures(
                            tuple(selected_futures), period=future_period))
                except Exception as exc:
                    future_summary = pd.DataFrame(
                        columns=derivatives_context.FUTURES_SUMMARY_COLUMNS)
                    future_normalized = pd.DataFrame()
                    future_meta = {
                        "status": "unavailable",
                        "errors": {"取得処理": str(exc)},
                        "fetched_at": pd.Timestamp.now(tz="UTC"),
                    }
            remember_result(future_key, {
                "summary": future_summary,
                "normalized": future_normalized,
                "meta": future_meta,
            })

        future_result = fresh_result(future_key, 300)
        if future_result is None:
            st.info("先物を選び「読み込む / 更新」を押してください。"
                    "この操作はmoomooの過去K線利用枠を消費しません。")
        else:
            future_summary = future_result["summary"]
            future_normalized = future_result["normalized"]
            future_meta = future_result.get("meta") or {}
            if future_summary.empty:
                st.warning("選択した先物を取得できませんでした。"
                           "時間をおいて再読み込みしてください。")
                for failed_symbol, reason in future_meta.get("errors", {}).items():
                    st.caption(f"{failed_symbol}: {reason}")
            else:
                metric_columns = st.columns(min(3, len(future_summary)))
                for column, (_, row) in zip(metric_columns, future_summary.iterrows()):
                    daily = row.get("change_1d_pct")
                    delta = (f"{daily:+.2f}%（前営業日比）"
                             if pd.notna(daily) else None)
                    price_date = (pd.Timestamp(row["as_of"]).date().isoformat()
                                  if pd.notna(row.get("as_of")) else "—")
                    digits = int(catalog.get(row.get("symbol"), {}).get("digits", 2))
                    column.metric(
                        f"{row.get('symbol')}  {row.get('name')}",
                        f"{fmt(row.get('price'), digits=digits)} {row.get('unit', '')}",
                        delta, border=True)
                    column.caption(
                        f"{row.get('relation', '市場コンテキスト')}｜"
                        f"価格日 {price_date}")

                compare_frame = future_normalized.copy()
                compare_frame.index = (
                    pd.to_datetime(compare_frame.index).tz_localize(None).normalize())
                stock_for_compare = hist["Close"].copy()
                stock_for_compare.index = (
                    pd.to_datetime(stock_for_compare.index).tz_localize(None).normalize())
                compare_frame[ticker] = stock_for_compare
                compare_frame = compare_frame.dropna(
                    subset=[ticker, *future_normalized.columns])
                series_map = {ticker: compare_frame[ticker]}
                for symbol in future_normalized.columns:
                    series_map[symbol] = compare_frame[symbol]
                if not compare_frame.empty and len(series_map) > 1:
                    st.plotly_chart(
                        charts.comparison_chart(series_map),
                        config={"displaylogo": False, "responsive": True})

                stock_series = hist["Close"].copy()
                stock_series.index = pd.to_datetime(stock_series.index).tz_localize(None).normalize()
                relation_rows = []
                for symbol in future_normalized.columns:
                    future_series = future_normalized[symbol].copy()
                    future_series.index = (
                        pd.to_datetime(future_series.index).tz_localize(None).normalize())
                    aligned = pd.concat(
                        [stock_series.rename("stock"), future_series.rename("future")],
                        axis=1, join="inner").dropna().pct_change(fill_method=None).dropna()
                    corr_20 = aligned.tail(20).corr().iloc[0, 1] if len(aligned) >= 20 else None
                    corr_60 = aligned.tail(60).corr().iloc[0, 1] if len(aligned) >= 60 else None
                    variance = aligned.tail(60)["future"].var() if len(aligned) >= 60 else None
                    beta = (aligned.tail(60).cov().loc["stock", "future"] / variance
                            if variance is not None and pd.notna(variance) and variance > 0
                            else None)
                    relation_rows.append({
                        "先物": symbol,
                        "20日相関": corr_20,
                        "60日相関": corr_60,
                        "60日β": beta,
                        "共通日数": len(aligned),
                        "相関符号": (
                            "不足" if corr_20 is None or corr_60 is None
                            or pd.isna(corr_20) or pd.isna(corr_60)
                            else "一致" if corr_20 * corr_60 > 0 else "不一致"),
                    })
                if relation_rows:
                    with st.expander(f"{ticker}との実測関連性"):
                        relation_table = pd.DataFrame(relation_rows)
                        st.dataframe(
                            relation_table.style.format({
                                "20日相関": lambda value: "—" if pd.isna(value) else f"{value:+.2f}",
                                "60日相関": lambda value: "—" if pd.isna(value) else f"{value:+.2f}",
                                "60日β": lambda value: "—" if pd.isna(value) else f"{value:+.2f}",
                            }), hide_index=True)
                        st.caption("日次リターンの同時点比較です。相関・βは変化し、"
                                   "因果関係や将来の方向を示しません。")

                display_table = future_summary.rename(columns={
                    "symbol": "コード", "name": "名称", "category": "分類",
                    "price": "価格", "unit": "単位", "change_1d_pct": "1日%",
                    "change_5d_pct": "5日%", "change_1m_pct": "1ヶ月%",
                    "as_of": "価格日", "source": "データ源", "relation": "関連理由",
                })
                display_table["価格日"] = pd.to_datetime(
                    display_table["価格日"], utc=True, errors="coerce").dt.date
                st.dataframe(display_table, hide_index=True)
                for failed_symbol, reason in future_meta.get("errors", {}).items():
                    st.warning(f"{failed_symbol}: {reason}")
                st.caption(
                    "先物はYahoo Financeの検証済み連続先物コードです。実限月ではなく、"
                    "ロール時に価格差の影響を受ける場合があります。遅延データであり、"
                    "moomooの履歴K線利用枠は使いません。"
                    f"取得状態: {future_meta.get('status', '—')}")

    else:
        assets = list(derivatives_context.PERP_ASSETS)
        related_assets = derivatives_context.related_perp_assets(ticker)
        direct_relation = bool(related_assets)
        default_candidates = related_assets or ("BTC", "ETH")
        default_assets = [asset for asset in default_candidates if asset in assets]
        c_assets, c_load = st.columns([4.6, 1.1])
        with c_assets:
            selected_assets = st.multiselect(
                "PERP（OKX・USDT建て）", assets, default=default_assets,
                max_selections=3, key=f"perp_assets_{ticker}")
        perp_key = (ticker, "perp", tuple(selected_assets), "snapshot")
        with c_load:
            st.markdown('<div style="height:1.8rem"></div>', unsafe_allow_html=True)
            perp_load = st.button(
                "読み込む / 更新",
                type="primary", disabled=not selected_assets,
                key=f"load_perp_{ticker}")

        if direct_relation:
            st.info(f"{ticker}は暗号資産関連株のホワイトリスト対象です。"
                    "それでもPERPは株式そのものの先物ではありません。")
        else:
            st.info(f"{ticker}との直接対応は設定していません。"
                    "暗号資産市場全体のリスク選好をみる参考欄です。")
        st.warning("表示するPERPは、この株式の先物・直接ヘッジ・株価連動商品では"
                   "ありません。相関や因果関係を示すものでもありません。")

        if perp_load:
            if perp_key in result_store:
                clear = getattr(derivatives_context.fetch_okx_perpetuals, "clear", None)
                if clear:
                    clear(tuple(selected_assets))
            with st.spinner("PERPの公開市場データを取得しています…"):
                try:
                    perp_frame, perp_meta = derivatives_context.fetch_okx_perpetuals(
                        tuple(selected_assets))
                except Exception as exc:
                    perp_frame = pd.DataFrame(columns=derivatives_context.PERP_COLUMNS)
                    perp_meta = {
                        "status": "unavailable",
                        "errors": {"取得処理": str(exc)},
                        "fetched_at": pd.Timestamp.now(tz="UTC"),
                    }
            remember_result(perp_key, {"frame": perp_frame, "meta": perp_meta})

        perp_result = fresh_result(perp_key, 60)
        if perp_result is None:
            st.info("PERPを選び「読み込む / 更新」を押してください。"
                    "APIキーやログインは不要です。")
        else:
            perp_frame = perp_result["frame"]
            perp_meta = perp_result.get("meta") or {}
            if perp_frame.empty or perp_meta.get("status") == "unavailable":
                st.warning("PERP情報を取得できませんでした。"
                           "OKXの公開API状態を確認し、時間をおいて再読み込みしてください。")
                for failed_asset, reason in perp_meta.get("errors", {}).items():
                    st.caption(f"{failed_asset}: {reason}")
            else:
                for start in range(0, len(perp_frame), 3):
                    card_columns = st.columns(min(3, len(perp_frame) - start))
                    for column, (_, row) in zip(
                            card_columns, perp_frame.iloc[start:start + 3].iterrows()):
                        with column.container(border=True):
                            st.markdown(f"#### {row.get('instrument', row.get('asset', 'PERP'))}")
                            change_24h = row.get("change_24h_pct")
                            price_digits = derivatives_context.PERP_PRICE_DIGITS.get(
                                row.get("asset"), 4)
                            st.metric(
                                "Mark価格",
                                f"{row['mark_price']:,.{price_digits}f} USDT"
                                if pd.notna(row.get("mark_price")) else "—",
                                f"{change_24h:+.2f}%（24時間）"
                                if pd.notna(change_24h) else None)
                            st.caption(
                                f"Last {fmt(row.get('last'), digits=price_digits)} USDT / "
                                f"Funding premium "
                                f"{fmt(row.get('funding_premium_pct'), '%', 4)}")
                            st.metric(
                                f"Funding / {fmt(row.get('funding_interval_hours'), 'h', 0)}",
                                fmt(row.get("funding_rate_pct"), "%", 4),
                                f"単純年率 {fmt(row.get('funding_annualized_pct'), '%', 2)}",
                                delta_color="off")
                            st.caption(
                                f"OI {fmt(row.get('open_interest_usd'), ' USD', 0)}｜"
                                f"24h出来高 {fmt(row.get('volume_base_24h'), '', 0)} "
                                f"{row.get('asset', '')}\n\n"
                                f"次回Funding決済 {fmt_utc_time(row.get('funding_time'))}｜"
                                f"更新 {fmt_utc_time(row.get('as_of'))}")
                            if row.get("error"):
                                st.warning(str(row["error"]))

                st.caption(
                    "データ源: OKX公開・読み取り専用API。Mark価格を主表示し、"
                    "Funding premiumはFunding計算に使われる参考値です（現在の"
                    "Mark-Index乖離そのものではありません）。正のFundingは通常ロング側からショート側への"
                    "支払いを示します。単純年率は現在レートの機械換算で、予測ではありません。"
                    f"取得状態: {perp_meta.get('status', '—')}")

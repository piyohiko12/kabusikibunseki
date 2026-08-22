"""銘柄分析ページ。"""

import html
import re

import pandas as pd
import streamlit as st

from lib import (alerts as alerts_lib, board_ui, charts, daily_decision, data_fetcher,
                 derivatives_context,
                 event_intelligence, indicators, level_review, levels,
                 information_board, moomoo_client, news_fetcher, sensitivity,
                 realtime_signal,
                 session_intelligence,
                 rules as rules_lib, settings_store, today_inputs,
                 trade_summary, trade_visuals, trading_context, ui)

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

REALTIME_REFRESH_SECONDS = (5, 10, 30)
REALTIME_POSITION_MODES = {
    "これから買う": "entry",
    "すでに保有中": "holding",
}


def fetch_realtime_klines_for_ui(ticker: str, session_code: str) -> dict:
    """表示中の1銘柄とSPYの当日足を取得する薄いUIアダプター。

    OpenAPI層の公開名や返却形式に関する知識をここだけに閉じ込める。失敗を
    Yahooデータで埋めず、呼び出し側が安全に「判定不能」と表示できる形で返す。
    """
    loader = getattr(data_fetcher, "fetch_current_klines", None)
    if not callable(loader):
        return {
            "bars": pd.DataFrame(), "benchmark_bars": pd.DataFrame(),
            "meta": {"available": False, "uses_history_quota": False},
            "error": "リアルタイム足の取得機能を準備中です",
        }
    # 画面の正式名とOpenAPIアダプターの短い引数名の差をここだけで吸収する。
    api_session = {
        "premarket": "pre",
        "afterhours": "after",
    }.get(str(session_code or "").strip().lower(), session_code)
    try:
        payload = loader((ticker,), num=120, session=api_session)
    except Exception as exc:
        return {
            "bars": pd.DataFrame(), "benchmark_bars": pd.DataFrame(),
            "meta": {"available": False, "uses_history_quota": False},
            "error": f"リアルタイム足を取得できませんでした（{type(exc).__name__}）",
        }
    if not isinstance(payload, dict):
        payload = {}
    frames = payload.get("frames") if isinstance(payload.get("frames"), dict) else {}
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}

    def frame_for(symbol: str) -> pd.DataFrame:
        expected = symbol.strip().upper()
        for key, value in frames.items():
            normalized = str(key or "").strip().upper().split(".", 1)[-1]
            if normalized == expected and isinstance(value, pd.DataFrame):
                return value
        return pd.DataFrame()

    bars = frame_for(ticker)
    benchmark = frame_for("SPY")
    error = None
    expected_session = str(api_session or "").strip().lower()
    received_session = str(meta.get("session") or "").strip().lower()
    source = str(meta.get("source") or "")
    if meta.get("uses_history_quota") is not False:
        error = "過去K線枠を使う取得結果はリアルタイム判定に使用しません"
    elif (meta.get("available") is not True
          or meta.get("partial") is not False):
        details = meta.get("errors")
        error = "対象銘柄とSPYのリアルタイム足を完全には受信できません"
        if details:
            error += f"（{details}）"
    elif not received_session or received_session != expected_session:
        error = "現在の取引セッションに対応する1分足か確認できません"
    elif "moomoo" not in source.lower():
        error = "moomooのリアルタイム1分足か確認できません"
    elif bars.empty or benchmark.empty:
        error = "対象銘柄またはSPYのセッション別1分足が不足しています"
    return {
        "bars": bars, "benchmark_bars": benchmark,
        "meta": meta, "error": error,
    }


def realtime_action_visual(result: dict) -> dict:
    """エンジンの状態を、色だけに頼らない初心者向け表示へそろえる。"""
    action = result.get("action") if isinstance(result.get("action"), dict) else {}
    state = str(result.get("state") or result.get("status") or "DATA_WAIT").upper()
    defaults = {
        "BUY_READY": ("🟢", "買いを検討", "success"),
        "BUY_SETUP": ("🟡", "買い条件あり・今は待つ", "warning"),
        "RISK_EXIT": ("🔴", "保有株を売る候補（損失を抑える）", "error"),
        "TAKE_PROFIT": ("🔵", "保有株を売る候補（利益を確定）", "info"),
        "HOLD": ("🟢", "保有を続けて監視", "info"),
        "NEUTRAL": ("⚪", "今は売買しない", "info"),
        "WAIT": ("🟡", "今は待つ", "warning"),
        "DATA_WAIT": ("⚫", "判断できない", "warning"),
    }
    icon, label, severity = defaults.get(state, defaults["DATA_WAIT"])
    description = (action.get("description_ja")
                   or result.get("reason_ja")
                   or "必要な情報がそろうまで売買せずに待ちます。")
    nonactionable_exit = (
        state in {"RISK_EXIT", "TAKE_PROFIT"}
        and result.get("actionable") is not True
    )
    if nonactionable_exit:
        icon, label, severity = (
            "🟡", "価格目安に到達（今は取引できません）", "warning")
        blocked = next((
            row for row in result.get("gates") or []
            if isinstance(row, dict)
            and row.get("key") in {"session", "suspension", "quote_freshness"}
            and row.get("passed") is not True
        ), None)
        if blocked:
            description = (
                f"{blocked.get('reason_ja') or '取引可否を確認できません'}。"
                f"{description}")
    if not nonactionable_exit:
        tone = str(action.get("tone") or "").lower()
        severity = {
            "positive": "success", "success": "success", "buy": "success",
            "negative": "error", "danger": "error", "error": "error",
            "warning": "warning", "caution": "warning", "info": "info",
            "neutral": "info",
        }.get(tone, severity)
    return {
        "icon": icon,
        # 主見出しは内部の専門語より、固定した初心者向けの行動語を優先する。
        "label_ja": label,
        "description_ja": description,
        "severity": severity,
    }


def realtime_gate_summary(gates) -> tuple[str, list[str]]:
    """安全ゲートを短い件数と、利用者が直せる未確認理由へまとめる。"""
    rows = [row for row in (gates or []) if isinstance(row, dict)]
    passed = [row for row in rows if row.get("passed") is True]
    blocked = [row for row in rows if row.get("passed") is not True]
    reasons = []
    for row in blocked:
        label = row.get("label_ja") or row.get("label") or "安全項目"
        detail = row.get("reason_ja") or row.get("reason") or "確認が必要です"
        reasons.append(f"{label}: {detail}")
    if not rows:
        return "安全情報を確認できません", ["売買前の安全情報がありません"]
    return f"{len(passed)} / {len(rows)}項目を確認", reasons


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


EVENT_SOURCE_LABELS_JA = {
    "Federal Reserve": "米連邦準備制度理事会（FRB）",
    "U.S. Bureau of Labor Statistics": "米国労働統計局（BLS）",
    "Yahoo Finance calendar": "Yahoo Finance 決算カレンダー",
    "Yahoo Finance corporate actions": "Yahoo Finance 企業アクション",
    "Yahoo Finance": "Yahoo Finance",
    "SEC EDGAR": "米国証券取引委員会（SEC）",
}

TRADE_REGIME_LABELS = {
    "UPTREND": "上昇トレンド",
    "DOWNTREND": "下降トレンド",
    "RANGE": "レンジ",
    "HIGH_VOL": "高ボラティリティ",
}

SUMMARY_SESSION_LABELS = {
    "premarket": "プレマーケット",
    "regular": "立会時間",
    "afterhours": "アフター",
    "overnight": "夜間・24h",
    "closed": "セッション外",
    "unknown": "取引可否不明",
}


def verdict_score_text(result: dict, side_key: str) -> str:
    """売買判定の点数を、合格点と混同しない短い表示へ整える。"""
    side = result.get(side_key) or {}
    score, total, threshold = side.get("score"), side.get("total"), side.get("threshold")
    if score is None or total in (None, 0) or threshold is None:
        return "点数を確認できません"
    return f"{score:g} / {total:g}点（合格 {threshold:g}点）"


def price_plan_delta(item: dict, entry_price) -> str:
    """確定終値基準の価格計画を、現在値とは混ぜずに差分表示する。"""
    price = item.get("price") if isinstance(item, dict) else None
    try:
        price, entry_price = float(price), float(entry_price)
    except (TypeError, ValueError):
        return "算出できず"
    if entry_price <= 0:
        return "算出できず"
    return f"判定価格比 {(price / entry_price - 1) * 100:+.2f}%"


def level_summary_text(item: dict) -> str:
    """支持抵抗の帯と現在地を、中心値・現在値と混同せず表示する。"""
    if not isinstance(item, dict) or not item.get("available"):
        return "—"
    stars = item.get("stars") or ""
    if item.get("state") == "ゾーン内":
        return (f"\\${item['zone_low']:,.2f}〜\\${item['zone_high']:,.2f} "
                f"（帯の中で攻防中） {stars}")
    return (f"\\${item['edge_price']:,.2f} "
            f"({item['distance_pct']:+.2f}%) {stars}")


def event_source_label_ja(source) -> str:
    """取得元を日本語中心で表示し、未知の固有名は詳細欄だけに残す。"""
    text = str(source or "").strip()
    if not text:
        return "取得元不明"
    if text in EVENT_SOURCE_LABELS_JA:
        return EVENT_SOURCE_LABELS_JA[text]
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text):
        return text
    return f"外部ニュース配信（{text}）"


def localize_event_warning(message) -> str:
    """取得失敗の内部キーを利用者向け日本語へ置換する。"""
    text = str(message or "")
    replacements = {
        "company_info": "企業基本情報", "calendar": "決算予定",
        "earnings_history": "過去の決算日", "price_history": "株価履歴",
        "yahoo_news": "Yahooニュース", "sec_filings": "米SEC開示",
    }
    for internal, label in replacements.items():
        text = text.replace(internal, label)
    return text


def japanese_evidence(event: dict) -> list[str]:
    """主表示には日本語の根拠だけを残し、英語原文は詳細欄へ分離する。"""
    rows = []
    original = str(event.get("original_name") or "").strip()
    for item in event.get("evidence") or []:
        text = str(item or "").strip()
        if not text or text == original:
            continue
        if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text):
            rows.append(text)
    return rows


JP_WEEKDAYS = ("月", "火", "水", "木", "金", "土", "日")


def event_datetime_labels(event: dict) -> tuple[str, str, str]:
    """米東部基準のイベント日時を、日本時間優先の短い表示へ整える。"""
    day = pd.to_datetime(event.get("event_date"), errors="coerce")
    if pd.isna(day):
        return "日付未定", "発表時刻未定", "—"
    date_label = f"{day.month}月{day.day}日（{JP_WEEKDAYS[day.weekday()]}）"
    raw_time = str(event.get("event_time_et") or "").strip()
    if not re.fullmatch(r"\d{1,2}:\d{2}", raw_time):
        return date_label, "発表時刻未定", "—"
    try:
        event_et = pd.Timestamp(
            f"{day.strftime('%Y-%m-%d')} {raw_time}",
            tz="America/New_York",
        )
        event_jst = event_et.tz_convert("Asia/Tokyo")
    except (TypeError, ValueError):
        return date_label, "発表時刻未定", "—"
    date_label = (
        f"{event_jst.month}月{event_jst.day}日"
        f"（{JP_WEEKDAYS[event_jst.weekday()]}）"
    )
    primary = f"{event_jst:%H:%M} JST"
    secondary = f"{event_et.month}/{event_et.day} {event_et:%H:%M} 米東部"
    return date_label, primary, secondary


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


def render_realtime_timing_card(ticker: str) -> None:
    """日足履歴と独立して、現在1分足の売買タイミングカードを描画する。"""
    # 明示的にONにした間だけ現在1分足を購読し、このカードだけを部分更新する。
    realtime_monitor_key = f"realtime_monitor_{ticker}"
    realtime_mode_key = f"realtime_position_mode_{ticker}"
    realtime_interval_key = f"realtime_refresh_seconds_{ticker}"
    if realtime_monitor_key not in st.session_state:
        st.session_state[realtime_monitor_key] = False
    if realtime_interval_key not in st.session_state:
        st.session_state[realtime_interval_key] = 5

    with st.container(border=True):
        st.markdown("### ⚡ リアルタイム売買タイミング")
        st.caption(
            "確定1分足の短期条件、今の値動き、売買前の安全確認を順番に見ます。"
            "短い値動きだけで売買を決めません。")
        realtime_controls = st.columns([1.45, 1, 1.25])
        with realtime_controls[0]:
            realtime_mode_label = st.radio(
                "現在の状況",
                list(REALTIME_POSITION_MODES),
                horizontal=True,
                key=realtime_mode_key,
            )
        with realtime_controls[1]:
            realtime_refresh_seconds = st.selectbox(
                "更新間隔",
                REALTIME_REFRESH_SECONDS,
                format_func=lambda seconds: f"{seconds}秒ごと",
                key=realtime_interval_key,
                help="表示中の銘柄だけを部分更新します。",
            )
        with realtime_controls[2]:
            st.markdown('<div style="height:1.8rem"></div>', unsafe_allow_html=True)
            realtime_monitoring = st.toggle(
                "リアルタイム監視（ONで開始 / OFFで停止）",
                key=realtime_monitor_key,
                help="既定はOFFです。ONの間だけmoomooの当日足と最新価格を確認します。",
            )
            st.caption("⏸ 監視中" if realtime_monitoring else "▶ 停止中")

        realtime_mode = REALTIME_POSITION_MODES.get(realtime_mode_label, "entry")
        holding_entry_value = holding_stop_value = holding_target_value = None
        if realtime_mode == "holding":
            st.info("ここでの『売却候補』は、すでに保有している株の売却だけを意味します。"
                    "新しい空売りではありません。")
            holding_entry_key = f"realtime_holding_entry_{ticker}"
            holding_stop_key = f"realtime_holding_stop_{ticker}"
            holding_target_key = f"realtime_holding_target_{ticker}"
            for key in (holding_entry_key, holding_stop_key, holding_target_key):
                if key not in st.session_state:
                    st.session_state[key] = 0.0
            holding_inputs = st.columns(3)
            holding_entry_value = holding_inputs[0].number_input(
                "取得価格（任意）", min_value=0.0, step=0.50,
                key=holding_entry_key,
                help="0のままでも監視できます。価格関係の確認にだけ使います。") or None
            holding_stop_value = holding_inputs[1].number_input(
                "監視する損切り価格", min_value=0.0, step=0.50,
                key=holding_stop_key,
                help="未設定は0。損失を抑える売却候補を出すには入力してください。") or None
            holding_target_value = holding_inputs[2].number_input(
                "監視する利益確定価格", min_value=0.0, step=0.50,
                key=holding_target_key,
                help="未設定は0。利益確定の売却候補を出すには入力してください。") or None
            st.caption("入力値はこのブラウザー内の判定だけに使い、注文や口座情報には接続しません。")

        realtime_memories = st.session_state.setdefault("realtime_signal_memories", {})
        if not realtime_monitoring:
            # 停止後の古い連続成立を、再開時に引き継がない。
            for memory_key in list(realtime_memories):
                if (isinstance(memory_key, tuple) and memory_key
                        and memory_key[0] == ticker):
                    realtime_memories.pop(memory_key, None)
            active_identity = st.session_state.get("realtime_signal_active_identity")
            if (isinstance(active_identity, tuple) and active_identity
                    and active_identity[0] == ticker):
                st.session_state.pop("realtime_signal_active_identity", None)

        @st.fragment(run_every=(realtime_refresh_seconds
                                if realtime_monitoring else None))
        def render_realtime_timing_panel():
            """最新データの取得・判定・表示を、このカード内だけで更新する。"""
            if not realtime_monitoring:
                st.info("監視は停止中です。必要なときに上の『リアルタイム監視』をONにしてください。")
                st.caption("停止中はリアルタイム足を取得せず、連続成立回数もリセットします。")
                return

            def reset_ticker_memory() -> None:
                for key in list(realtime_memories):
                    if isinstance(key, tuple) and key and key[0] == ticker:
                        realtime_memories.pop(key, None)

            now_utc = pd.Timestamp.now(tz="UTC")
            try:
                live_snapshot = data_fetcher.fetch_realtime_snapshot(ticker)
            except Exception as exc:
                reset_ticker_memory()
                st.warning(f"### ⚫ 判断できない\n\n最新価格を受信できませんでした（{type(exc).__name__}）。")
                return

            if live_snapshot.get("source") != "moomoo OpenAPI":
                reset_ticker_memory()
                st.warning("### ⚫ 判断できない\n\nmoomooの最新価格を確認できません。"
                           "Yahoo Financeの代替値はリアルタイム判定に使用しません。")
                reason = live_snapshot.get("fallback_reason")
                if reason:
                    st.caption(f"取得状態: {reason}")
                return

            live_eligibility = today_inputs.overnight_eligibility(live_snapshot)
            try:
                live_market_state = data_fetcher.fetch_market_state(ticker)
            except Exception:
                live_market_state = {}
            live_session = session_intelligence.detect_current_session(
                now=now_utc,
                market_state=live_market_state.get("market_state"),
                overnight_eligible=live_eligibility,
            )
            calendar_session = session_intelligence.detect_current_session(
                now=now_utc,
                market_state=None,
                overnight_eligible=live_eligibility,
            )
            live_session = {
                **live_session,
                "calendar_session": calendar_session.get("session"),
                "calendar_reason": calendar_session.get("reason"),
            }
            live_session_code = str(live_session.get("session") or "unknown")
            identity = (ticker, realtime_mode, live_session_code)
            previous_identity = st.session_state.get("realtime_signal_active_identity")
            if previous_identity != identity:
                # 銘柄・保有状況・セッションが変わったら確認回数をゼロから始める。
                realtime_memories.pop(identity, None)
                st.session_state["realtime_signal_active_identity"] = identity

            if live_session_code not in session_intelligence.SESSION_NAMES:
                realtime_memories.pop(identity, None)
                label = SUMMARY_SESSION_LABELS.get(live_session_code, "取引可否を確認")
                st.warning(f"### ⚫ 判断できない\n\n現在は{label}です。取引可能なセッションで再確認してください。")
                st.caption(f"判定時刻: {fmt_et_jst(live_session.get('as_of'))}")
                return

            # 買い・保有中のどちらも、現在セッションの確定1分足とSPYを同じ条件で使う。
            # 日足や別カードの購入プランは、このリアルタイム判定へ渡さない。
            live_klines = fetch_realtime_klines_for_ui(ticker, live_session_code)
            live_meta = live_klines.get("meta") or {}
            live_source = str(live_meta.get("source") or "")
            if live_klines.get("error"):
                realtime_memories.pop(identity, None)
                st.warning(f"### ⚫ 判断できない\n\n{live_klines['error']}。売買せずに待ちます。")
                return
            if "moomoo" not in live_source.lower():
                realtime_memories.pop(identity, None)
                st.warning("### ⚫ 判断できない\n\nmoomoo以外の代替足は、"
                           "リアルタイム売買タイミングに使用しません。")
                return

            decision_quotes = (live_meta.get("decision_quotes")
                               if isinstance(live_meta.get("decision_quotes"), dict)
                               else {})
            decision_quote = (decision_quotes.get(ticker)
                              or decision_quotes.get(str(ticker).upper()) or {})
            live_decision_snapshot = data_fetcher.build_session_snapshot(
                live_snapshot, decision_quote, live_session_code)
            if live_decision_snapshot.get("decision_ready") is not True:
                realtime_memories.pop(identity, None)
                st.warning("### ⚫ 判断できない\n\n現在の取引時間に対応した価格・気配・"
                           "1分足時刻を照合できません。推測せず待ちます。")
                decision_errors = live_decision_snapshot.get("decision_errors") or []
                if decision_errors:
                    st.caption("確認状態: " + " ／ ".join(
                        str(item) for item in decision_errors[:2]))
                return

            position = None
            if realtime_mode == "holding":
                position = {
                    "held": True,
                    "entry_price": holding_entry_value,
                    "stop": holding_stop_value,
                    "target": holding_target_value,
                    "source": "利用者が入力した保有株の監視価格",
                }

            try:
                realtime_result = realtime_signal.evaluate_realtime_signal(
                    ticker,
                    live_klines["bars"],
                    benchmark_bars=live_klines.get("benchmark_bars"),
                    snapshot=live_decision_snapshot,
                    session=live_session,
                    now=now_utc,
                    position=position,
                    previous_memory=realtime_memories.get(identity),
                    config={
                        "actionable_sessions": (
                            "regular", "premarket", "afterhours", "overnight"),
                    },
                )
            except Exception as exc:
                realtime_memories.pop(identity, None)
                st.warning(f"### ⚫ 判断できない\n\nリアルタイム判定を更新できませんでした"
                           f"（{type(exc).__name__}）。売買せずに待ちます。")
                return
            if not isinstance(realtime_result, dict):
                realtime_memories.pop(identity, None)
                st.warning("### ⚫ 判断できない\n\n判定結果を確認できません。売買せずに待ちます。")
                return

            memory = realtime_result.get("memory")
            if isinstance(memory, dict):
                # エンジンはsignal_bar_timeをmemoryへ保持し、同じ確定足を連続確認へ加えない。
                realtime_memories[identity] = memory
            else:
                realtime_memories.pop(identity, None)

            st.markdown("**① 確定1分足の短期条件**")
            components = (realtime_result.get("components")
                          if isinstance(realtime_result.get("components"), dict)
                          else {})
            technical = (components.get("technical")
                         if isinstance(components.get("technical"), dict) else {})
            score = realtime_result.get("score")
            score_text = "確認できません" if score is None else f"{float(score):.0f} / 100点"
            required_text = ("必須条件を確認済み" if technical.get("required_passed") is True
                             else "必須条件に未確認または未成立があります")
            st.caption(
                f"確定した現在セッションの1分足: {score_text} ／ {required_text}。"
                "形成途中の足、日足、購入プランの判定は混ぜません。")

            st.markdown("**② 今のタイミング**")
            visual = realtime_action_visual(realtime_result)
            getattr(st, visual["severity"])(
                f"### {visual['icon']} {visual['label_ja']}\n\n"
                f"{visual['description_ja']}")
            confirmation = (realtime_result.get("confirmation")
                            if isinstance(realtime_result.get("confirmation"), dict)
                            else {})
            confirmed_count = (confirmation.get("streak")
                               if confirmation.get("streak") is not None
                               else confirmation.get("count"))
            required_count = (confirmation.get("required")
                              if confirmation.get("required") is not None
                              else confirmation.get("required_count"))
            if (realtime_mode == "entry" and confirmed_count is not None
                    and required_count is not None):
                st.caption(
                    f"確定したリアルタイム足で連続確認 {confirmed_count} / {required_count}回。"
                    "同じ足を何度更新しても確認回数には加えません。")

            st.markdown("**③ 売買前の安全確認**")
            gate_text, gate_reasons = realtime_gate_summary(
                realtime_result.get("gates"))
            if gate_reasons:
                st.warning("🛡️ " + gate_text + "。" + " ／ ".join(gate_reasons[:2]))
            else:
                st.success("🛡️ " + gate_text)

            checks = [row for row in realtime_result.get("checks") or []
                      if isinstance(row, dict)]
            with st.expander("リアルタイム判定の根拠を見る"):
                if not checks:
                    st.caption("短期の判定根拠を確認できませんでした。")
                for check in checks:
                    passed = check.get("passed")
                    status = str(check.get("status") or "").lower()
                    icon = ("✅" if passed is True or status in {"passed", "up", "buy"}
                            else "❌" if passed is False or status in {"failed", "down", "sell"}
                            else "—")
                    label = check.get("label_ja") or check.get("label") or "確認項目"
                    detail = (check.get("actual_ja") or check.get("actual")
                              or check.get("reason_ja") or check.get("reason") or "—")
                    st.caption(f"{icon} {label}: {detail}")

            source_label = live_meta.get("source") or "moomoo OpenAPI 当日足"
            fetched_at = live_meta.get("fetched_at") or now_utc
            session_label = SUMMARY_SESSION_LABELS.get(
                live_session_code, live_session_code)
            st.caption(
                f"{session_label} ｜ {source_label} ｜ 判定更新 {fmt_utc_time(fetched_at)} ｜ "
                f"{realtime_refresh_seconds}秒ごとに確認")
            if live_session_code != "regular":
                st.caption("プレ・アフター・夜間も取引対象です。"
                           "そのセッションの1分足・気配値・データ品質を確認できない場合は判断を保留します。")

        render_realtime_timing_panel()
        st.caption(
            "このカードは判定表示だけで、注文APIを呼びません。"
            "現在足・snapshotだけを使うため、このリアルタイム機能による"
            "moomoo過去K線枠の追加使用は0です。"
            "『買いを検討』は値上がりや利益の保証ではなく、1分足や気配値は短時間で反転します。"
            "監視を止めると取得更新は止まりますが、再利用のため購読枠が保持される場合があります。")
        st.caption(
            "※ 同じページの日足・チャートは別機能です。サイドバーで"
            "『過去K線もmoomooを使用』を有効にした場合は、そちらが別途"
            "過去K線枠を使うことがあります。")


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

# 現在1分足の監視は日足履歴と独立している。履歴取得が失敗しても、このカードは
# すでに描画・起動済みのため利用を続けられる。
render_realtime_timing_card(ticker)

fetch_period, display_days = PERIODS[period_label]

try:
    hist, _base_meta = data_fetcher.fetch_chart_history(
        ticker, fetch_period, "1d", allow_new_quota=True)
except data_fetcher.FetchError:
    st.error("日足データを取得できませんでした。ネットワーク接続を確認し、"
             "しばらく時間をおいてから再試行してください。"
             "上のリアルタイム売買タイミングは、moomooの現在データがあれば利用できます。")
    st.stop()

if hist.empty:
    st.error(f"ティッカー「{ticker}」のデータを取得できませんでした。"
             "ティッカーシンボルが正しいか確認してください(例: AAPL, MSFT, GOOGL)。"
             "上のリアルタイム売買タイミングは、moomooの現在データがあれば利用できます。")
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

tab_today, tab_chart, tab_news, tab_board, tab_tape, tab_flow, tab_derivatives = st.tabs([
    "🧭 今日の判断", "📊 チャート・指標", "📰 ニュース・ネットの反応",
    "📋 情報掲示板", "🔬 板・歩み値", "🏦 需給・IV", "🌐 先物・PERP",
])

# ----------------------------------------------------------- 売買情報サマリー
# 既に取得したhistを再利用する。ここから過去K線APIを追加では呼ばない。
eligibility = today_inputs.overnight_eligibility(snapshot)
market_state = data_fetcher.fetch_market_state(ticker)
session_state = session_intelligence.detect_current_session(
    market_state=market_state.get("market_state"),
    overnight_eligible=eligibility,
)
# 決算日だけはsignals画面と同じ6時間キャッシュを使う。moomoo履歴枠は使わない。
try:
    summary_analyst = data_fetcher.fetch_analyst(ticker)
except Exception:
    summary_analyst = {}
earnings_date = summary_analyst.get("earnings_date")

decision_context = trading_context.prepare_from_history(
    hist,
    source_meta=_base_meta,
    market_meta=market_state,
    snapshot=snapshot,
    earnings_date=earnings_date,
)
rule_store = rules_lib.load()
active_rule_name = rule_store["active"]
active_rule = rule_store["rules"][active_rule_name]
rule_warnings = rules_lib.validate(active_rule)

if decision_context is None:
    external_gates = []
    entry_evaluation = {
        "verdict": "WAIT", "summary": "確定日足を準備できないため判定を待機します",
        "position_mode": "entry", "regime": None, "risk_plan": {},
    }
    holding_evaluation = {
        "verdict": "WAIT", "summary": "確定日足を準備できないため判定を待機します",
        "position_mode": "holding", "regime": None, "risk_plan": {},
    }
    decision_levels = []
    summary_levels = []
else:
    external_gates = trading_context.external_gates(decision_context, active_rule)
    entry_evaluation = rules_lib.evaluate(
        decision_context["df"], active_rule, decision_context["levels"],
        position_mode="entry", external_gates=external_gates)
    holding_evaluation = rules_lib.evaluate(
        decision_context["df"], active_rule, decision_context["levels"],
        position_mode="holding", external_gates=external_gates)
    decision_levels = list(decision_context["levels"])
    minimum_strength = int(
        (active_rule.get("risk") or {}).get("min_level_strength", 2))
    summary_levels = [
        row for row in decision_levels
        if int(row.get("strength", row.get("base_strength", 0)) or 0)
        >= minimum_strength
    ]

fallback_daily_bar = (None if decision_context is None
                      else decision_context["df"].iloc[-1])
session_rows = today_inputs.session_prices(snapshot, fallback_daily_bar)
session_changes = session_intelligence.compute_session_changes(
    session_rows, previous_close=prev)
trend = daily_decision.intraday_trend(snapshot, fallback_daily_bar)

target_labels = {
    "regular": "次の立会寄付き", "premarket": "次のプレ開始",
    "afterhours": "次のアフター開始", "overnight": "次の夜間開始",
}
summary_box = tab_today.container(border=True)
with summary_box:
    summary_title, summary_target, summary_action = st.columns([2.4, 1.4, 1.2])
    with summary_title:
        st.markdown("### 🛒 購入プラン")
        last_bar = (decision_context or {}).get("bar_meta", {}).get("last_bar")
        st.caption("買う・待つ・見送るを、価格と許容損失まで含めて確認します。")
    with summary_target:
        target_session = st.selectbox(
            "寄付き診断の対象", list(target_labels),
            format_func=lambda key: target_labels[key],
            key=f"today_target_{ticker}")
    with summary_action:
        st.markdown('<div style="height:1.8rem"></div>', unsafe_allow_html=True)
        load_today = st.button(
            "寄付き・イベントを更新", type="primary", width="stretch",
            key=f"load_today_intelligence_{ticker}_{target_session}",
            help="Yahooデータとニュースを必要時だけ取得します。moomoo過去K線枠は使いません。",
        )

result_store = st.session_state.setdefault("today_intelligence_results", {})
result_key = (ticker, target_session)
if load_today:
    try:
        with summary_box:
            with st.spinner("市場・寄付き・イベント影響を整理中..."):
                event_report = event_intelligence.fetch_event_intelligence(
                    ticker, include_news=True, horizon_days=120)
                market_features = fetch_opening_market_features()
                features = today_inputs.opening_features(
                    None if decision_context is None else decision_context["df"],
                    snapshot, market_features, event_report)
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
        with summary_box:
            st.warning(f"寄付き・イベント診断を取得できませんでした: {exc}")

result = result_store.get(result_key)
result_expired = False
if result:
    loaded_at = pd.to_datetime(result.get("loaded_at"), utc=True, errors="coerce")
    if (pd.isna(loaded_at)
            or pd.Timestamp.now(tz="UTC") - loaded_at > pd.Timedelta(minutes=15)):
        result_store.pop(result_key, None)
        result = None
        result_expired = True

event_payload = None
if result:
    event_payload = dict(result["events"])
    event_payload["events"] = [
        event_intelligence.localize_event_for_display(event)
        for event in result["events"].get("events") or []
    ]

summary_common = {
    "current_price": price_now,
    "current_price_source": (snapshot.get("source") or _base_meta.get("source")),
    "current_price_as_of": (snapshot.get("update_time") or _base_meta.get("fetched_at")),
    "snapshot": snapshot,
    "history": None if decision_context is None else decision_context["df"],
    "levels": summary_levels,
    "trend": trend,
    "session": {
        "current_session": session_state,
        "session_changes": session_changes,
    },
}
if event_payload is not None:
    summary_common["event"] = event_payload

entry_summary = trade_summary.build_trade_summary({
    **summary_common, "position_mode": "entry",
    "rule_evaluation": entry_evaluation,
})
holding_summary = trade_summary.build_trade_summary({
    **summary_common, "position_mode": "holding",
    "rule_evaluation": holding_evaluation,
})

# 保存済みアラートを、追加取得なしで現在の画面データに対してだけ照合する。
active_ticker_alerts = [
    alert for alert in alerts_lib.load()
    if alert.get("enabled") and str(alert.get("ticker") or "").upper() == ticker
]
triggered_alerts = []
unavailable_alerts = []
alert_check_errors = 0
if decision_context is not None:
    alert_context = {
        "df": decision_context["df"],
        "levels": decision_context["levels"],
        "price": price_now,
        "previous_close": prev,
        "entry_verdict": entry_evaluation.get("verdict"),
        "entry_blocked": any(
            action.get("blocking") is True
            for action in entry_summary.get("action_priorities", [])
        ),
        "holding_verdict": holding_evaluation.get("verdict"),
    }
    for alert in active_ticker_alerts:
        try:
            checked_alert = alerts_lib.check(alert, alert_context)
        except Exception:
            alert_check_errors += 1
            continue
        if checked_alert.get("reason"):
            unavailable_alerts.append((alert, checked_alert))
        elif checked_alert.get("triggered"):
            triggered_alerts.append((alert, checked_alert))
else:
    alert_check_errors = len(active_ticker_alerts)
alert_checked_at = pd.Timestamp.now(tz="America/New_York")

# 購入プランは取得済みの確定日足・snapshot・イベントだけで計算する。
# 予算と許容損失はこのブラウザーセッション内だけに保持し、注文には使用しない。
purchase_budget_key = f"purchase_budget_{ticker}"
purchase_loss_key = f"purchase_loss_limit_{ticker}"
if purchase_budget_key not in st.session_state:
    st.session_state[purchase_budget_key] = 1_000.0
if purchase_loss_key not in st.session_state:
    st.session_state[purchase_loss_key] = 100.0
purchase_safety = trading_context.safety_config(active_rule)
purchase_plan = trade_summary.build_purchase_plan(
    {
        **summary_common,
        "position_mode": "entry",
        "rule_evaluation": entry_evaluation,
        "snapshot": snapshot,
    },
    max_loss=st.session_state[purchase_loss_key],
    max_investment=st.session_state[purchase_budget_key],
    max_spread_pct=purchase_safety.get("max_spread_pct"),
    now=pd.Timestamp.now(tz="UTC"),
)

with summary_box:
    st.markdown("#### 1. 今、購入を検討できるか")
    daily_signal = purchase_plan.get("daily_signal") or {}
    daily_icon = ("✅" if daily_signal.get("is_buy_candidate") else
                  "⏳" if daily_signal.get("verdict") == "WAIT" else "—")
    st.caption(
        f"確定日足の分析: {daily_icon} "
        f"{daily_signal.get('label_ja') or '確認できません'}。"
        "これは『現在の価格ですぐ買える』という意味ではありません。")

    purchase_status = purchase_plan.get("status")
    purchase_icon = ("🟢" if purchase_status == "READY" else
                     "⚪" if purchase_status == "NOT_CANDIDATE" else "🟠")
    purchase_message = (
        f"### {purchase_icon} {purchase_plan['label_ja']}\n\n"
        f"{purchase_plan['description_ja']}"
    )
    if purchase_status == "READY":
        st.success(purchase_message)
    elif purchase_status == "NOT_CANDIDATE":
        st.info(purchase_message)
    else:
        st.warning(purchase_message)

    wait_reasons = purchase_plan.get("wait_reasons") or []
    if wait_reasons:
        for reason in wait_reasons[:2]:
            st.markdown(f"- **{reason['label_ja']}** — {reason['detail_ja']}")
        if len(wait_reasons) > 2:
            st.caption(f"ほか {len(wait_reasons) - 2}件は「購入を中止する条件」で確認できます。")

    st.markdown("#### 2. 買う価格・損切り・利益確定")
    quote = purchase_plan.get("quote") or {}
    buy_zone = purchase_plan.get("buy_zone") or {}
    risk = purchase_plan.get("risk") or {}
    ask_price = quote.get("ask")
    buy_limit = buy_zone.get("high")
    ask_delta = (
        "気配値を取得できず" if ask_price is None
        else "上限を算出できず" if buy_limit is None
        else "上限内" if ask_price <= buy_limit
        else "上限を超過"
    )
    price_columns = st.columns(4)
    price_columns[0].metric(
        "現在の売り気配（Ask）",
        "—" if ask_price is None else f"${ask_price:,.2f}",
        ask_delta,
        delta_color="off", border=True)
    price_columns[1].metric(
        "買う価格の上限",
        "—" if buy_limit is None else f"${buy_limit:,.2f}",
        "必要な損失・利益比から逆算", delta_color="off", border=True)
    price_columns[2].metric(
        "損切りの目安",
        "—" if risk.get("stop") is None else f"${risk['stop']:,.2f}",
        "約定価格の保証ではありません", delta_color="off", border=True)
    price_columns[3].metric(
        "利益確定の目安",
        "—" if risk.get("target") is None else f"${risk['target']:,.2f}",
        "目標であり保証ではありません", delta_color="off", border=True)
    st.caption("Ask（売り気配）は、購入時に相手が提示している参考価格です。")

    chase = purchase_plan.get("chase_warning") or {}
    if ask_price is not None and buy_limit is not None:
        gap = ask_price - buy_limit
        if gap > 0:
            st.warning(f"売り気配（Ask）は上限より ${gap:,.2f} 高いため、追いかけず待ちます。")
        else:
            st.caption(f"売り気配（Ask）は上限まで ${abs(gap):,.2f} の範囲内です。")
    below_reference = next(
        (item for item in purchase_plan.get("cautions") or []
         if item.get("code") == "below_reference"),
        None,
    )
    if below_reference:
        st.warning(
            f"{below_reference.get('label_ja', '判定時より下落しています')}。"
            f"{below_reference.get('detail_ja', '下落が続いていないかチャートを再確認してください。')}"
        )
    if chase.get("current_rr") is not None and chase.get("minimum_rr") is not None:
        st.caption(
            f"現在の売り気配（Ask）での利益÷損失: {chase['current_rr']:.2f}倍 ／ "
            f"利用中ルールの最低基準: {chase['minimum_rr']:.2f}倍")

    st.markdown("#### 3. 購入株数の上限を計算")
    input_budget, input_loss = st.columns(2)
    input_budget.number_input(
        "この銘柄に使える金額の上限（ドル）",
        min_value=1.0, step=100.0, key=purchase_budget_key,
        help="この画面だけで使う計算値です。口座残高や注文には接続しません。")
    input_loss.number_input(
        "この1回で許容する損失（ドル）",
        min_value=1.0, step=10.0, key=purchase_loss_key,
        help="損切り目安まで通常どおり約定した場合の損失上限です。")

    position = purchase_plan.get("position_size") or {}
    shares = position.get("shares") if position.get("available") else None
    planned_price = position.get("purchase_price")
    expected_profit = position.get("estimated_target_profit")
    if expected_profit is None and None not in (shares, planned_price, risk.get("target")):
        expected_profit = max(float(shares) * (float(risk["target"]) - float(planned_price)), 0.0)
    size_columns = st.columns(4)
    size_columns[0].metric(
        "上限株数（推奨ではない）",
        "—" if shares is None else f"{shares:,}株",
        position.get("limiting_factor") or position.get("reason_ja") or "計算不能",
        delta_color="off", border=True)
    size_columns[1].metric(
        "使用額の目安",
        "—" if position.get("estimated_cost") is None
        else f"${position['estimated_cost']:,.2f}",
        "買う価格の上限で計算", delta_color="off", border=True)
    size_columns[2].metric(
        "通常時の想定損失",
        "—" if position.get("estimated_max_loss") is None
        else f"${position['estimated_max_loss']:,.2f}",
        "価格飛び・滑りは未反映", delta_color="off", border=True)
    size_columns[3].metric(
        "目標到達時の想定利益",
        "—" if expected_profit is None else f"${expected_profit:,.2f}",
        "手数料・税・為替は未反映", delta_color="off", border=True)
    if position.get("available") and not position.get("can_buy_one_share"):
        st.warning("1株でも購入予算または許容損失を超えるため、計算上の上限は0株です。")
    st.caption(
        "上限株数は推奨株数ではありません。損切り注文の価格は約定を保証せず、"
        "相場急変・価格飛び・滑りによって実際の損失が計算値を超えることがあります。")

    if entry_summary["session"].get("code") in {"premarket", "afterhours", "overnight"}:
        st.warning("立会時間外は価格差と値動きが大きくなりやすいため、成行へ切り替えず、"
                   "買う価格の上限と注文条件を証券会社の画面で再確認してください。")

    plan = entry_summary
    plan_entry = plan["entry"].get("price")
    support_item, resistance_item = plan["support"], plan["resistance"]
    support_text = level_summary_text(support_item)
    resistance_text = level_summary_text(resistance_item)
    required_gates = [gate for gate in entry_evaluation.get("gates", [])
                      if gate.get("required")]
    passed_gates = [gate for gate in required_gates if gate.get("passed") is True]
    blocked_gates = [gate for gate in required_gates if gate.get("passed") is False]
    unknown_gates = [gate for gate in entry_evaluation.get("gates", [])
                     if gate.get("passed") is None]

    with st.expander("購入を中止する条件"):
        st.markdown(
            "- 買い判定が成立しなくなった\n"
            "- 売り気配（Ask）が上限価格を超えた\n"
            "- 気配値が古い、価格差が広い、または取引停止になった\n"
            "- 損切り・利益確定の価格関係が崩れた\n"
            "- 決算禁止期間または直近の高影響イベントに入った\n"
            "- 購入株数の上限が0株になった")
        if wait_reasons:
            st.markdown("**現在該当している項目**")
            for reason in wait_reasons:
                st.markdown(f"- {reason['label_ja']}: {reason['detail_ja']}")
        else:
            st.success("現在、上記の中止条件は検出されていません。")

    with st.expander("すでに株を持っている場合"):
        holding_blocking = [
            action for action in holding_summary.get("action_priorities", [])
            if action.get("blocking")]
        holding_visual = trade_visuals.evaluation_visual({
            "verdict": holding_summary["verdict"]["code"],
            "risk_plan": holding_evaluation.get("risk_plan"),
            "visual_blocked": bool(holding_blocking),
        }, position_mode="holding")
        getattr(st, holding_visual["severity"])(
            f"### {holding_visual['icon']} {holding_visual['action_label_ja']}\n\n"
            f"{holding_visual['description_ja']}")
        st.caption(
            "損失を抑えて売る条件: "
            + verdict_score_text(holding_evaluation, "risk_exit")
            + " ／ 利益を確定して売る条件: "
            + verdict_score_text(holding_evaluation, "take_profit"))

    with st.expander("判定の理由・相場の状態・支持抵抗"):
        st.write(str(entry_summary["verdict"]["reason"]))
        st.caption("買う条件の点数: " + verdict_score_text(entry_evaluation, "buy"))
        st.caption(
            f"使用ルール: {active_rule_name} ／ 判定に使った確定日足: "
            f"{last_bar or '取得不能'}")
        detail_columns = st.columns(4)
        detail_columns[0].metric(
            "現在の取引時間",
            SUMMARY_SESSION_LABELS.get(
                entry_summary["session"]["code"], entry_summary["session"]["label_ja"]),
            "取引可" if entry_summary["session"]["tradable"] is True
            else "取引時間外" if entry_summary["session"]["tradable"] is False
            else "取引可否を確認", delta_color="off", border=True)
        detail_columns[1].metric(
            "相場の状態（日足）",
            TRADE_REGIME_LABELS.get(entry_evaluation.get("regime"), "判定不能"),
            "売買ルールの前提", delta_color="off", border=True)
        detail_columns[2].metric(
            "判定に使った株価",
            "—" if plan_entry is None else f"${plan_entry:,.2f}",
            "確定終値・参考現在値とは別", delta_color="off", border=True)
        detail_columns[3].metric(
            "判定時の利益÷損失", plan["rr"]["label_ja"],
            "現在のAskでは上で再計算", delta_color="off", border=True)
        st.markdown(
            f"**下値の目安（支持帯）** {support_text}　←　"
            f"**参考現在値 \\${price_now:,.2f}**　→　"
            f"**上値の目安（抵抗帯）** {resistance_text}")
        st.caption("支持・抵抗は反発予測ではなく、価格幅を確認する参考情報です。")
        if blocked_gates:
            st.warning("確認できていない項目: " + " ／ ".join(
                f"{gate['label']}（{gate.get('reason') or '要確認'}）"
                for gate in blocked_gates))
        elif required_gates:
            st.success(
                f"必要な項目 {len(passed_gates)} / {len(required_gates)} を確認済み")
        if unknown_gates:
            st.caption("⚠ 情報を取得できず確認が必要: " + "、".join(
                str(gate.get("label")) for gate in unknown_gates))

    if active_ticker_alerts:
        unavailable_count = len(unavailable_alerts) + alert_check_errors
        not_triggered_count = max(
            len(active_ticker_alerts) - len(triggered_alerts) - unavailable_count, 0)
        alert_line = (
            f"🔔 この銘柄のアラート: 有効 {len(active_ticker_alerts)}件 ／ "
            f"現在成立 {len(triggered_alerts)}件 ／ 未成立 {not_triggered_count}件 ／ "
            f"判定不能 {unavailable_count}件 ／ "
            f"確認 {alert_checked_at:%H:%M:%S} ET")
        if triggered_alerts:
            st.warning(alert_line + "\n\n" + " ／ ".join(
                f"{alerts_lib.describe(alert)}（実測 "
                f"{alerts_lib.format_actual(alert, checked.get('actual', '—'))}）"
                for alert, checked in triggered_alerts[:3]))
        elif unavailable_count:
            st.warning(alert_line)
        else:
            st.info(alert_line)
        if unavailable_count:
            reasons = [
                f"{alerts_lib.describe(alert)}（{checked.get('reason') or 'データ不足'}）"
                for alert, checked in unavailable_alerts[:2]
            ]
            if alert_check_errors:
                reasons.append(f"{alert_check_errors}件は保存値を評価できませんでした")
            st.caption("⚠ 判定不能: " + " ／ ".join(reasons))
    else:
        st.caption("🔔 この銘柄の有効アラートはありません。"
                   "必要なら下の詳細画面から追加できます。")
    st.caption("アラートはこの画面の更新時点だけを照合し、アプリを閉じている間は監視しません。")

    event_risk = entry_summary["event_risk"]
    if event_risk["available"] and event_risk.get("event_name"):
        days = event_risk.get("days_until")
        day_text = "日程差不明" if days is None else "本日" if days == 0 else f"{days}日後"
        st.info(
            f"📅 次の予定: {event_risk['event_name']} "
            f"{event_risk['impact_stars_text']}（{day_text}）・"
            f"{event_risk.get('session') or '発表時間未定'}")
        if event_risk.get("warnings"):
            st.caption("⚠ 一部取得できないイベント情報があります: " + " ／ ".join(
                localize_event_warning(item) for item in event_risk["warnings"]))
    elif result is not None and event_risk.get("report_status") in {
            "partial", "unavailable"}:
        st.warning("📅 " + event_risk["reason"] + "。予定なしとは判定しません。")
    elif result_expired:
        st.warning("📅 前回のイベント診断は15分を超えたため失効しました。再更新してください。")
    elif result is None and earnings_date:
        st.info(f"📅 次回決算予定: {str(earnings_date)[:10]}。"
                "影響度★と他イベントは「寄付き・イベントを更新」で確認できます。")
    elif result is None:
        st.warning("イベント情報は未確認です。「寄付き・イベントを更新」で確認してください。")
    else:
        st.info("診断期間内に今後のイベントを確認できませんでした。")

    if result:
        opening = result["session"]["next_open_diagnosis"]
        opening_labels = {"up": "上向き", "down": "下向き", "neutral": "方向拮抗",
                          "unknown": "入力不足"}
        probability = opening.get("probability_up")
        st.caption(
            f"次回開始の参考方向: {opening_labels.get(opening.get('direction'), '—')}"
            + ("" if probability is None else f"（上向き参考確率 {probability * 100:.0f}%・未校正）")
            + f" ／ データ品質 {opening['data_quality']['score'] * 100:.0f}%"
            + f" ／ 最終更新 {fmt_utc_time(result.get('loaded_at'))}")

    if rule_warnings:
        st.warning("使用ルールの設定確認: " + " ／ ".join(rule_warnings))
    st.caption(
        "このまとめは既存の確定日足ルールを整理した参考情報です。"
        "現在値・イベント・当日方向は売買スコアへ自動加点せず、注文や空売りは実行しません。")
    st.link_button("詳しい判定基準・内訳・株価アラートを開く",
                   f"/signals?ticker={ticker}")

# ---------------------------------------------------------------- 今日の判断
with tab_today:
    st.caption("現在セッション → 当日の方向 → 支持抵抗 → 次回寄付き → イベントの順に"
               "確認します。ここでの数値は説明可能な参考診断で、注文や利益を保証しません。")

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
    nearby = daily_decision.nearest_levels(decision_levels, price_now, min_strength=3)
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
    t4.metric("支持・抵抗の単純な上下距離比",
              "—" if rr is None else f"1 : {rr:.2f}",
              "ストップ未反映・売買計画のR:Rとは別",
              delta_color="off", border=True)

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

        feature_status_labels = {
            "ok": "利用中", "missing": "未取得", "stale": "鮮度不足",
            "invalid_time": "時刻不正",
        }
        feature_df = pd.DataFrame([{
            "特徴": row["label"],
            "値": "—" if row["value"] is None else f"{row['value']:.3f}",
            "状態": feature_status_labels.get(row["status"], "利用対象外"),
            "固定重み": row["weight"],
            "寄与": row["contribution"], "取得元": row.get("source") or "—",
        } for row in diagnosis["features"]])
        with st.expander("参考確率の内訳（固定重み・データ鮮度）"):
            st.dataframe(feature_df.style.format({
                "固定重み": "{:.2f}", "寄与": "{:+.3f}",
            }), hide_index=True, use_container_width=True)
            st.caption(diagnosis["disclaimer"])

        event_report = result["events"]
        st.markdown("#### イベント影響カレンダー")
        st.caption("★は株価が動く方向ではなく、**影響を受けやすい大きさ**です。"
                   "過去の実測が十分なら平常時との比較を優先し、不足時はイベント種類別の"
                   "目安を表示します。")
        localized_events = event_payload["events"]
        scope_column, sort_column = st.columns([2.2, 1.2])
        with scope_column:
            event_scope = st.segmented_control(
                "表示範囲", ["今後の予定", "最近の材料も含む"],
                default="今後の予定",
                key=f"event_scope_{ticker}_{target_session}")
        with sort_column:
            event_sort = st.pills(
                "並び順", ["日付順", "影響度順"], default="日付順",
                key=f"event_sort_{ticker}_{target_session}") or "日付順"
        upcoming_events = [
            event for event in localized_events if event.get("status") == "UPCOMING"]
        visible_events = list(upcoming_events if event_scope == "今後の予定"
                              else localized_events)
        if event_sort == "影響度順":
            visible_events.sort(
                key=lambda event: (
                    -int(event.get("impact_stars") or 0),
                    str(event.get("event_date") or "9999-12-31"),
                ))
        visible_events = visible_events[:8]
        if not visible_events:
            st.info("表示期間内に今後のイベントを確認できませんでした。"
                    "「最近の材料も含む」に切り替えると直近ニュースも確認できます。")
        else:
            scored_events = [
                event for event in visible_events if event.get("impact_available")]
            largest = (max(scored_events, key=lambda event: event["impact_stars"])
                       if scored_events else None)
            next_event = upcoming_events[0] if upcoming_events else None
            if next_event is None:
                next_event_time = "—"
            else:
                next_date, next_time, _ = event_datetime_labels(next_event)
                next_event_time = f"{next_date} {next_time}"
            high_count = sum(
                int(event.get("impact_stars") or 0) >= 4 for event in upcoming_events)
            e1, e2, e3 = st.columns(3)
            e1.metric(
                "次の予定イベント",
                next_event_time,
                "予定なし" if next_event is None else next_event["display_name_ja"],
                delta_color="off", border=True)
            e2.metric(
                "表示中の最大影響度",
                "—" if largest is None else largest["impact_stars_text"],
                "判定材料なし" if largest is None else largest["display_name_ja"],
                delta_color="off", border=True)
            e3.metric(
                "今後の予定",
                f"{len(upcoming_events)}件",
                f"★★★★以上 {high_count}件", delta_color="off", border=True)

            event_rows = []
            for event in visible_events:
                date_label, primary_time, _ = event_datetime_labels(event)
                event_rows.append({
                    "日本時間": f"{date_label} {primary_time}",
                    "イベント": event["display_name_ja"],
                    "影響度": event["impact_stars_accessible_ja"],
                    "対象時間": event["session_label_ja"],
                    "過去の反応": event["directional_bias_label_ja"],
                    "評価根拠": event.get("impact_star_source_ja") or "判定材料不足",
                })
            event_df = pd.DataFrame(event_rows)
            with st.expander("イベントを一覧で比較", expanded=False):
                st.dataframe(event_df, hide_index=True, width="stretch")
                st.caption("日本時間を優先表示しています。詳しい根拠と米東部時間は"
                           "下の各カードで確認できます。")

            for index, event in enumerate(visible_events):
                with st.container(border=True):
                    date_label, primary_time, secondary_time = event_datetime_labels(event)
                    h1, h2, h3 = st.columns([1.15, 4.6, 1.45])
                    h1.markdown(f"**{date_label}**")
                    h1.caption(primary_time)
                    if secondary_time != "—":
                        h1.caption(secondary_time)
                    h2.markdown(f"**{event['display_name_ja']}**")
                    h2.markdown(
                        ui.chip(event["status_label_ja"],
                                "blue" if event.get("status") == "UPCOMING" else "gray")
                        + " " + ui.chip(event["session_label_ja"], "violet"),
                        unsafe_allow_html=True)
                    h3.markdown(f"### {event['impact_stars_text']}"
                                if event["impact_available"] else "### —")
                    h3.caption(
                        "判定材料なし" if not event["impact_available"]
                        else f"{event['impact_stars']} / 5・{event['impact_label_ja']}")
                    h3.caption(event.get("impact_star_source_ja") or "判定材料不足")

                    study = event.get("historical_sensitivity") or {}
                    sample_size = int(study.get("sample_size") or 0)
                    median_move = study.get("median_abs_move_pct")
                    sensitivity_ratio = study.get("sensitivity_ratio")
                    up_rate = study.get("up_rate_pct")
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("過去の実測", f"{sample_size}回" if sample_size else "—",
                              border=True)
                    s2.metric("中央値の変動幅",
                              "—" if median_move is None else f"±{median_move:.2f}%",
                              border=True)
                    s3.metric("平常時との比較",
                              "—" if sensitivity_ratio is None else f"{sensitivity_ratio:.2f}倍",
                              border=True)
                    s4.metric("過去の上昇割合",
                              "—" if up_rate is None else f"{up_rate:.0f}%",
                              event["directional_bias_label_ja"],
                              delta_color="off", border=True)

                    with st.expander("上下シナリオ・根拠・原文を確認"):
                        scenario_columns = st.columns(3)
                        scenario_icons = {"up": "🟢", "down": "🔴", "two_sided": "🟡"}
                        for column, scenario in zip(
                                scenario_columns, event["scenario_rows_ja"]):
                            with column.container(border=True):
                                st.markdown(
                                    f"**{scenario_icons[scenario['key']]} "
                                    f"{scenario['label_ja']}**")
                                st.write(scenario["description"])

                        evidence = japanese_evidence(event)
                        if evidence:
                            st.markdown("**日本語で確認できる根拠**")
                            for item in evidence[:4]:
                                st.markdown(f"- {md_escape(item)}")
                        original_name = event.get("original_name")
                        english_evidence = [
                            str(item) for item in event.get("evidence") or []
                            if item and not re.search(
                                r"[\u3040-\u30ff\u3400-\u9fff]", str(item))
                        ]
                        if original_name or english_evidence:
                            st.markdown("**英語原文（必要な場合のみ）**")
                            if original_name:
                                st.text(str(original_name))
                            for item in english_evidence[:2]:
                                if str(item) != str(original_name):
                                    st.text(str(item))
                        source_text = event_source_label_ja(event.get("source"))
                        st.caption(
                            f"取得元: {source_text} ・ 根拠の充足度: "
                            f"{event['confidence_label_ja']}")
                        st.caption(
                            "根拠の充足度は、過去標本数・情報源・日程の確かさを"
                            "まとめた説明用の目安で、統計的な信頼区間ではありません。")
                        if event.get("url"):
                            st.link_button(
                                "原文の情報源を開く", event["url"],
                                key=f"event_source_{ticker}_{target_session}_{index}")
            st.caption("イベントは不確実性の確認材料で、売買スコアへ自動加点していません。"
                       "過去の方向や★の数は将来を保証せず、発表値と市場予想の差を確認してください。")
        for warning in event_report.get("warnings") or []:
            st.warning(localize_event_warning(warning))
    else:
        st.info("上のサマリーで更新すると、市場の直近5分足・イベント日程・ニュースを必要時だけ"
                "読み込みます。データが不足・古い場合は確率を表示しません。")

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

        with st.expander("上級者向け: 支持抵抗からの参考シナリオ"):
            st.caption(
                "上部の購入プランとは別の、選択中チャートを使った参考計算です。"
                "支持帯に来ただけで買う判断には使いません。")
            sugg_risk = st.number_input(
                "1回で許容する損失額(ドル)", min_value=10.0, value=100.0, step=10.0,
                help="損切りまで通常どおり約定した場合の損失額で株数上限を計算します")
            sugg = levels.suggest_limit_orders(chart_view, lv_list, sugg_risk)
            if sugg:
                s_table = pd.DataFrame([{
                    "シナリオ": s["scenario"],
                    "指値価格": s["price"],
                    "損切り目安": s["stop"],
                    "利確目安": s["target"],
                    "利益÷損失": s["rr"],
                    "株数上限": s["shares"],
                    "想定利益": s["est_profit"],
                } for s in sugg])
                styled_s = s_table.style.format({
                    "指値価格": "${:,.2f}",
                    "損切り目安": lambda v: "—" if pd.isna(v) else f"${v:,.2f}",
                    "利確目安": lambda v: "—" if pd.isna(v) else f"${v:,.2f}",
                    "利益÷損失": lambda v: "—" if pd.isna(v) else f"{v:.1f}倍",
                    "株数上限": lambda v: "—" if pd.isna(v) else f"{v:,.0f}株",
                    "想定利益": lambda v: "—" if pd.isna(v) else f"${v:,.0f}",
                }).map(lambda v: "color: #006300" if "買い" in str(v)
                       else ("color: #d03b3b" if "売り" in str(v) else ""),
                       subset=["シナリオ"])
                st.dataframe(styled_s, hide_index=True)
                st.caption(
                    "買い指値=サポート帯上端、損切り=帯下端−0.5ATR、"
                    "利確=直近抵抗帯下端。株数は上限であり推奨ではありません。"
                    "価格飛び・滑り・手数料・税・為替は未反映です。")

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


# -------------------------------------------------------------- 情報掲示板
# 既にこのページで取得・計算した値だけを共通形式へ投影する。追加のK線取得は行わない。
stock_board_alerts = []
for alert in active_ticker_alerts:
    try:
        checked = (alerts_lib.check(alert, alert_context)
                   if decision_context is not None else
                   {"triggered": False, "actual": "—",
                    "reason": "確定日足が不足しています"})
    except Exception as exc:
        checked = {"triggered": False, "actual": "—", "reason": str(exc)}
    stock_board_alerts.append({
        "alert": alert,
        "result": {**checked, "checked_at": alert_checked_at},
    })

# ニュースタブの選択状態に左右されないYahooの一覧を使う。これはキャッシュ済みの
# 読み取り専用ニュース取得であり、moomooの過去K線枠には触れない。
try:
    stock_board_news = (news if news_src == "Yahoo Finance"
                        else news_fetcher.fetch_news(ticker))
except Exception:
    stock_board_news = []

if event_payload is not None:
    stock_board_events = event_payload
else:
    try:
        stock_board_events = event_intelligence.build_event_intelligence(
            ticker, hist, info, analyst, [], stock_board_news,
            fetched_at=pd.Timestamp.now(tz="UTC"), horizon_days=120)
        stock_board_events["events"] = [
            event_intelligence.localize_event_for_display(event)
            for event in stock_board_events.get("events") or []
        ]
    except Exception as exc:
        stock_board_events = {
            "status": "unavailable", "events": [], "warnings": [str(exc)]}

stock_board_snapshot = dict(snapshot or {})
if not stock_board_snapshot.get("price"):
    completed = None if decision_context is None else decision_context["df"]
    completed_price = (float(completed["Close"].iloc[-1])
                       if completed is not None and not completed.empty else None)
    completed_previous = (float(completed["Close"].iloc[-2])
                          if completed is not None and len(completed) > 1 else None)
    stock_board_snapshot.update({
        "price": completed_price,
        "previous_close": completed_previous,
        "source": "Yahoo Finance（直近確定日足）",
        "update_time": (completed.index[-1]
                        if completed is not None and not completed.empty else None),
        "is_realtime": False,
    })
if (stock_board_snapshot.get("price") is not None
        and stock_board_snapshot.get("previous_close")):
    stock_board_snapshot["change_pct"] = (
        float(stock_board_snapshot["price"])
        / float(stock_board_snapshot["previous_close"]) - 1
    ) * 100

stock_board_report = information_board.build_information_board({
    "ticker": ticker,
    "as_of": alert_checked_at,
    "snapshot": stock_board_snapshot,
    "entry_evaluation": {
        **entry_evaluation,
        "evaluated_at": alert_checked_at,
        "source": f"保存ルール: {active_rule_name}（確定日足）",
        "visual_blocked": any(
            action.get("blocking")
            for action in entry_summary.get("action_priorities", [])
        ),
    },
    "holding_evaluation": {
        **holding_evaluation,
        "evaluated_at": alert_checked_at,
        "source": f"保存ルール: {active_rule_name}（確定日足）",
    },
    "levels": summary_levels,
    "checked_alerts": stock_board_alerts,
    "alert_checked_at": alert_checked_at,
    "analyst": analyst,
    "event_report": stock_board_events,
    "news": stock_board_news,
})

with tab_board:
    board_ui.render_information_board(
        stock_board_report, show_heading=True,
        key_prefix=f"stock_board_{ticker}")



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

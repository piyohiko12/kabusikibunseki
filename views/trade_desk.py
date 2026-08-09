"""米国株の一日の確認順序をまとめたトレードデスク（読み取り専用）。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from lib import (event_intelligence, market_intelligence, moomoo_client,
                 session_intelligence, settings_store, ui)


SESSION_LABELS = {
    "overnight": "🌙 夜間・24時間帯",
    "premarket": "🌅 プレマーケット",
    "regular": "🔔 立会時間",
    "afterhours": "🌆 アフターマーケット",
    "closed": "⏸ 休場・セッション外",
    "unknown": "❓ 対象可否を確認",
}
SESSION_RANK_KEYS = {
    "プレマーケット": "pre",
    "アフターマーケット": "after",
    "夜間取引": "overnight",
}
TREND_COLORS = {"up": "green", "down": "red", "neutral": "gray",
                "unavailable": "orange"}
IMPACT_LABELS = {"HIGH": "高", "MEDIUM": "中", "LOW": "低"}


def _num(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _fmt_pct(value) -> str:
    number = _num(value)
    return "—" if number is None else f"{number:+.2f}%"


def _fmt_score(value) -> str:
    number = _num(value)
    return "—" if number is None else f"{number:.0f}/100"


def _pct_style(value):
    number = _num(value)
    if number is None or number == 0:
        return ""
    return "color: #006300" if number > 0 else "color: #a32d2d"


def _cached_session_result(key: str, ttl_seconds: int):
    record = st.session_state.get(key)
    if not isinstance(record, dict) or "loaded_at" not in record:
        return None
    loaded = pd.to_datetime(record["loaded_at"], utc=True, errors="coerce")
    if pd.isna(loaded) or pd.Timestamp.now(tz="UTC") - loaded > pd.Timedelta(seconds=ttl_seconds):
        st.session_state.pop(key, None)
        return None
    return record.get("value")


st.title("🧭 今日のトレードデスク")
st.caption("米国市場 → セクター → 候補銘柄 → 個別タイミングの順に確認します。"
           "表示・分析のみで、注文は行いません。")

# ---------------------------------------------------------------- セッション時計
session = session_intelligence.detect_current_session(overnight_eligible=True)
now_et = session["as_of"]
now_jst = now_et.tz_convert("Asia/Tokyo")
next_regular = session_intelligence.next_session_open(
    "regular", now=now_et, overnight_eligible=True)
current_name = SESSION_LABELS.get(session["session"], session["session"])

with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.25, 1, 1.2, 1.1])
    c1.metric("現在の米国株セッション", current_name, border=True)
    c2.metric("米東部時間", now_et.strftime("%m/%d %H:%M ET"), border=True)
    c3.metric("日本時間", now_jst.strftime("%m/%d %H:%M JST"), border=True)
    next_open = next_regular.get("open_time")
    c4.metric("次の立会寄付き",
              next_open.strftime("%m/%d %H:%M ET") if next_open is not None else "—",
              border=True)
    active = session["calendar_session"]
    chips = []
    for key, label, hours in (
        ("overnight", "夜間", "20:00–04:00"),
        ("premarket", "プレ", "04:00–09:30"),
        ("regular", "立会", "09:30–16:00"),
        ("afterhours", "アフター", "16:00–20:00"),
    ):
        chips.append(ui.chip(f"{label} {hours} ET", "blue" if key == active else "gray"))
    st.markdown(" ".join(chips), unsafe_allow_html=True)
    st.caption("夜間・24時間取引は対応銘柄のみ。休場日・短縮日は取引所日程を優先し、"
               "時間外は流動性低下とスプレッド拡大に注意してください。")

# ---------------------------------------------------------------- 市場・セクター
head1, head2 = st.columns([5, 1])
with head1:
    st.subheader("1. 市場全体とセクターの方向")
with head2:
    if st.button("↻ 更新", help="市場・セクター専用の15分キャッシュを更新します"):
        market_intelligence.clear_market_intelligence_cache()
        st.session_state.pop("trade_desk_candidate_result", None)
        st.rerun()

moomoo_ready = moomoo_client.status()["state"] == "ok"
try:
    with st.spinner("米国市場と11セクターを確認中..."):
        overview = market_intelligence.get_us_market_overview(
            include_realtime=moomoo_ready)
except Exception as exc:
    overview = None
    st.warning(f"市場分析を取得できませんでした: {exc}")

if overview:
    market = overview["market"]
    metrics = market["metrics"]
    breadth = overview["breadth"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("米国市場の日足トレンド", market["trend_label"],
              _fmt_score(market["score"]), delta_color="off", border=True)
    m2.metric("SPY 20営業日", _fmt_pct(metrics["return_20d_pct"]), border=True)
    m3.metric("上昇セクター比率",
              "—" if breadth["up_ratio_pct"] is None else f"{breadth['up_ratio_pct']:.0f}%",
              f"上昇 {breadth['up']} / 下降 {breadth['down']}",
              delta_color="off", border=True)
    m4.metric("判定カバレッジ", f"{market['confidence_pct']:.0f}%",
              overview["meta"]["history_source"], delta_color="off", border=True)
    st.markdown(ui.chip(
        f"市場: {market['trend_label']}", TREND_COLORS.get(market["trend"], "gray")),
        unsafe_allow_html=True)

    sectors = sorted(overview["sectors"], key=lambda row: (
        row.get("rank") is None, row.get("rank") or 999))
    sector_df = pd.DataFrame([{
        "順位": row.get("rank"),
        "セクター": row["sector_name_ja"],
        "ETF": row["symbol"],
        "方向": row["trend_label"],
        "スコア": row["score"],
        "20営業日": row["metrics"]["return_20d_pct"],
        "SPY比20日": row["metrics"]["relative_20d_pct"],
        "リアルタイム前日比": ((row.get("realtime_snapshot") or {}).get("change_percent")),
        "信頼度": row["confidence_pct"],
    } for row in sectors])
    styled = sector_df.style.format({
        "順位": lambda v: "—" if pd.isna(v) else f"{int(v)}",
        "スコア": lambda v: "—" if pd.isna(v) else f"{v:.0f}",
        "20営業日": lambda v: "—" if pd.isna(v) else f"{v:+.2f}%",
        "SPY比20日": lambda v: "—" if pd.isna(v) else f"{v:+.2f}%",
        "リアルタイム前日比": lambda v: "—" if pd.isna(v) else f"{v:+.2f}%",
        "信頼度": lambda v: "—" if pd.isna(v) else f"{v:.0f}%",
    }).map(_pct_style, subset=["20営業日", "SPY比20日", "リアルタイム前日比"])
    st.dataframe(styled, hide_index=True, use_container_width=True)
    st.caption("上昇しやすさの保証ではなく、SMA構造・20/63日リターン・SPY相対力を"
               "説明可能な日足スコアへ整理したものです。")

    st.subheader("2. 強いセクターから代表銘柄を絞る")
    catalog = {item["key"]: item for item in market_intelligence.SECTOR_CATALOG}
    sector_key = st.selectbox(
        "確認するセクター",
        [row["sector_key"] for row in sectors],
        format_func=lambda key: f"{catalog[key]['name_ja']} ({catalog[key]['etf']})",
        key="trade_desk_sector")
    load_col, note_col = st.columns([1.5, 4])
    with load_col:
        load_candidates = st.button("候補銘柄を読み込む", type="primary",
                                    use_container_width=True)
    with note_col:
        st.caption("選択したセクターの代表4銘柄だけを遅延取得します。全銘柄を一括取得しません。")
    if load_candidates:
        try:
            with st.spinner("候補銘柄を採点中..."):
                candidate_value = market_intelligence.get_sector_candidates(
                    sector_key, top_n=4, include_realtime=moomoo_ready)
            st.session_state["trade_desk_candidate_result"] = {
                "sector": sector_key,
                "loaded_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "value": candidate_value,
            }
        except Exception as exc:
            st.warning(f"候補銘柄を取得できませんでした: {exc}")

    candidate_record = st.session_state.get("trade_desk_candidate_result")
    candidate_result = None
    if isinstance(candidate_record, dict) and candidate_record.get("sector") == sector_key:
        loaded = pd.to_datetime(candidate_record.get("loaded_at"), utc=True, errors="coerce")
        if pd.notna(loaded) and pd.Timestamp.now(tz="UTC") - loaded <= pd.Timedelta(minutes=15):
            candidate_result = candidate_record.get("value")
    if candidate_result:
        ranking = candidate_result["ranking"]
        candidate_df = pd.DataFrame([{
            "順位": row["rank"], "ティッカー": row["symbol"],
            "状態": row["candidate_status"], "候補スコア": row["recommendation_score"],
            "日足方向": row["trend"]["trend_label"],
            "20日": row["trend"]["metrics"]["return_20d_pct"],
            "SPY比20日": row["trend"]["metrics"]["relative_20d_pct"],
            "データ充足": row["coverage_pct"],
            "分析": f"/?ticker={row['symbol']}",
        } for row in ranking])
        st.dataframe(candidate_df.style.format({
            "候補スコア": lambda v: "—" if pd.isna(v) else f"{v:.0f}",
            "20日": lambda v: "—" if pd.isna(v) else f"{v:+.2f}%",
            "SPY比20日": lambda v: "—" if pd.isna(v) else f"{v:+.2f}%",
            "データ充足": lambda v: "—" if pd.isna(v) else f"{v:.0f}%",
        }).map(_pct_style, subset=["20日", "SPY比20日"]),
            hide_index=True, use_container_width=True,
            column_config={"分析": st.column_config.LinkColumn(
                "個別分析", display_text="開く →")})
        if ranking:
            with st.expander("採点根拠を確認"):
                for row in ranking:
                    st.markdown(f"**{row['symbol']} — {row['candidate_status']}**")
                    for reason in row["reasons"]:
                        st.caption("・" + reason)
        st.caption(candidate_result["disclaimer"])
else:
    st.info("市場・セクターデータを取得できませんでした。")

# ---------------------------------------------------------------- 時間外ランキング
st.divider()
st.subheader("3. セッション別に今動いている銘柄")
if not moomoo_ready:
    st.info("moomoo OpenDを有効にすると、プレ・アフター・夜間の値上がり／値下がり上位を表示します。")
else:
    rank_label = st.pills("セッション", list(SESSION_RANK_KEYS),
                          default=("プレマーケット" if session["calendar_session"] == "premarket"
                                   else "アフターマーケット" if session["calendar_session"] == "afterhours"
                                   else "夜間取引" if session["calendar_session"] == "overnight"
                                   else "プレマーケット"),
                          key="trade_desk_rank_session")
    rank_key = SESSION_RANK_KEYS[rank_label or "プレマーケット"]
    up_col, down_col = st.columns(2)
    for column, losers, title in ((up_col, False, "📈 値上がり"),
                                  (down_col, True, "📉 値下がり")):
        with column:
            st.markdown(f"**{title}**")
            rank = moomoo_client.session_rank(rank_key, count=10, losers=losers)
            if rank.empty:
                st.caption("現在データがありません（時間帯・権限・対象銘柄を確認してください）。")
            else:
                rank = rank.copy()
                rank["分析"] = "/?ticker=" + rank["ティッカー"].astype(str)
                st.dataframe(rank.style.format({
                    "時間外価格": "${:,.2f}", "終値": "${:,.2f}",
                    "時間外変化率": "{:+.2f}%", "出来高": "{:,.0f}",
                }, na_rep="—").map(_pct_style, subset=["時間外変化率"]),
                    hide_index=True, use_container_width=True,
                    column_config={"分析": st.column_config.LinkColumn(
                        "個別分析", display_text="開く →")})

# ---------------------------------------------------------------- イベントカレンダー
st.divider()
st.subheader("4. 今日から先の高影響イベント")
upcoming = [row for row in event_intelligence.default_macro_events()
            if str(row.get("event_date") or "") >= date.today().isoformat()][:8]
if upcoming:
    macro_df = pd.DataFrame([{
        "日付": row["event_date"], "時刻(ET)": row.get("event_time_et") or "—",
        "イベント": row["name"], "影響セッション": row["session"],
        "出典": row["source"],
    } for row in upcoming])
    st.dataframe(macro_df, hide_index=True, use_container_width=True)
    st.caption("FOMC・CPI・雇用統計の公式公表日程。結果が市場予想を上回るか下回るかで"
               "方向は変わるため、日程だけで上昇・下降を決めません。")

if moomoo_ready:
    if st.button("今週の経済指標・主要決算を読み込む"):
        st.session_state["trade_desk_calendars"] = {
            "loaded_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "value": {"economic": moomoo_client.economic_calendar(7),
                      "earnings": moomoo_client.earnings_calendar(7, 50)},
        }
    calendars = _cached_session_result("trade_desk_calendars", 900)
    if calendars:
        cal1, cal2 = st.tabs(["重要経済指標", "主要決算"])
        with cal1:
            economic = calendars["economic"]
            if economic.empty:
                st.info("重要経済指標を取得できませんでした。")
            else:
                st.dataframe(economic, hide_index=True, use_container_width=True)
        with cal2:
            earnings = calendars["earnings"]
            if earnings.empty:
                st.info("主要決算を取得できませんでした。")
            else:
                view = earnings.copy()
                if "ticker" in view:
                    view["分析"] = "/?ticker=" + view["ticker"].astype(str)
                st.dataframe(view, hide_index=True, use_container_width=True,
                             column_config={"分析": st.column_config.LinkColumn(
                                 "個別分析", display_text="開く →")})

# ---------------------------------------------------------------- 個別銘柄へ
st.divider()
st.subheader("5. 銘柄ごとのタイミングを確認")
default_ticker = str(settings_store.load().get("default_ticker") or "AAPL").upper()
target = st.text_input("ティッカー", value=default_ticker,
                       placeholder="例: NVDA").strip().upper()
if target:
    b1, b2, b3 = st.columns(3)
    b1.link_button("📈 今日の判断・チャート", f"/?ticker={target}",
                   use_container_width=True, type="primary")
    b2.link_button("🎯 売買判定", f"/signals?ticker={target}",
                   use_container_width=True)
    b3.link_button("⚖️ 銘柄比較", "/compare", use_container_width=True)
st.caption("個別画面ではセッション別価格、寄付き方向の未校正参考確率、当日トレンド、"
           "支持抵抗、イベント影響を同じタブで確認できます。最終判断は売買判定の"
           "安全ゲートとチャートの両方で確認してください。")

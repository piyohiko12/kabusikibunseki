"""価格・判定・イベント・ニュースをまとめる読み取り専用情報掲示板。"""

import re
from urllib.parse import quote

import pandas as pd
import streamlit as st

from lib import (alerts as alerts_lib, board_ui, data_fetcher,
                 event_intelligence, information_board, news_fetcher,
                 rules as rules_lib, session_intelligence, settings_store,
                 today_inputs, trade_summary, trading_context)


def _wait_evaluation(mode: str, message: str) -> dict:
    return {
        "verdict": "WAIT",
        "summary": message,
        "position_mode": mode,
        "regime": None,
        "risk_plan": {},
        "gates": [],
    }


st.title("📋 情報掲示板")
st.caption("価格・売買判定・アラート・イベント・ニュースを、重要度付きの一覧で確認します。"
           "投稿・返信・外部への情報送信は行いません。")

settings = settings_store.load()
initial = (st.query_params.get("ticker")
           or st.session_state.get("information_board_current_ticker")
           or settings.get("default_ticker")
           or "AAPL")

head1, head2 = st.columns([1.2, 2.8])
with head1:
    ticker = st.text_input(
        "ティッカー", value=str(initial).strip().upper(),
        placeholder="例: AAPL、NVDA、BRK.B").strip().upper()
with head2:
    st.markdown('<div style="height:1.8rem"></div>', unsafe_allow_html=True)
    st.caption("過去日足はYahoo Financeを使います。moomooの過去K線利用枠は消費しません。")

if not ticker:
    st.info("ティッカーを入力してください。")
    st.stop()
if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=_/-]{0,31}", ticker):
    st.error("ティッカーの形式を確認してください（例: AAPL、BRK.B）。")
    st.stop()
if st.query_params.get("ticker") != ticker:
    st.query_params["ticker"] = ticker
st.session_state["information_board_current_ticker"] = ticker

st.link_button(f"📈 {ticker}の銘柄分析へ", f"/?ticker={quote(ticker)}")

warnings = []
with st.spinner(f"{ticker}の情報を整理中..."):
    try:
        history = data_fetcher.fetch_history(ticker, "2y", "1d")
    except Exception as exc:
        history = pd.DataFrame()
        warnings.append(f"Yahoo Financeの日足を取得できませんでした: {exc}")

    try:
        info = data_fetcher.fetch_info(ticker)
    except Exception as exc:
        info = {}
        warnings.append(f"企業情報を取得できませんでした: {exc}")
    try:
        analyst = data_fetcher.fetch_analyst(ticker)
    except Exception as exc:
        analyst = {}
        warnings.append(f"決算・アナリスト情報を取得できませんでした: {exc}")
    try:
        snapshot = data_fetcher.fetch_realtime_snapshot(ticker)
    except Exception as exc:
        snapshot = {}
        warnings.append(f"最新価格を取得できませんでした: {exc}")
    try:
        market_state = data_fetcher.fetch_market_state(ticker)
    except Exception as exc:
        market_state = {}
        warnings.append(f"市場状態を取得できませんでした: {exc}")

    # 判定作成時は取得できたsnapshotだけを渡す。空の場合は共通前処理が
    # 形成途中足を除外した確定日足へ安全にフォールバックする。
    snapshot_for_board = dict(snapshot or {})

    decision_context = trading_context.prepare_from_history(
        history,
        source_meta={
            "source": "Yahoo Finance",
            "code": (market_state.get("code") or f"US.{ticker}"),
            "fetched_at": pd.Timestamp.now(tz="UTC"),
            "cache_status": "cached",
        },
        market_meta=market_state,
        snapshot=snapshot_for_board,
        earnings_date=analyst.get("earnings_date"),
    ) if not history.empty else None

    if not snapshot_for_board.get("price") and decision_context is not None:
        completed = decision_context["df"]
        snapshot_for_board.update({
            "price": float(completed["Close"].iloc[-1]),
            "previous_close": (float(completed["Close"].iloc[-2])
                               if len(completed) > 1 else None),
            "source": "Yahoo Finance（直近確定日足）",
            "update_time": completed.index[-1],
            "is_realtime": False,
        })
    if (snapshot_for_board.get("price") is not None
            and snapshot_for_board.get("previous_close")):
        snapshot_for_board["change_pct"] = (
            float(snapshot_for_board["price"])
            / float(snapshot_for_board["previous_close"]) - 1
        ) * 100

    rule_store = rules_lib.load()
    active_rule_name = rule_store["active"]
    active_rule = rule_store["rules"][active_rule_name]
    if decision_context is None:
        entry_evaluation = _wait_evaluation("entry", "確定日足が不足しています")
        holding_evaluation = _wait_evaluation("holding", "確定日足が不足しています")
        decision_levels = []
    else:
        external_gates = trading_context.external_gates(decision_context, active_rule)
        entry_evaluation = rules_lib.evaluate(
            decision_context["df"], active_rule, decision_context["levels"],
            position_mode="entry", external_gates=external_gates)
        holding_evaluation = rules_lib.evaluate(
            decision_context["df"], active_rule, decision_context["levels"],
            position_mode="holding", external_gates=external_gates)
        decision_levels = decision_context["levels"]

    board_session = session_intelligence.detect_current_session(
        market_state=market_state.get("market_state"),
        overnight_eligible=today_inputs.overnight_eligibility(snapshot_for_board),
    )
    board_entry_summary = trade_summary.build_trade_summary({
        "current_price": snapshot_for_board.get("price"),
        "current_price_source": snapshot_for_board.get("source"),
        "snapshot": snapshot_for_board,
        "history": None if decision_context is None else decision_context["df"],
        "levels": decision_levels,
        "session": {"current_session": board_session},
        "position_mode": "entry",
        "rule_evaluation": entry_evaluation,
    })
    board_entry_blocked = any(
        action.get("blocking") is True
        for action in board_entry_summary.get("action_priorities", [])
    )
    board_checked_at = pd.Timestamp.now(tz="UTC")
    board_entry_evaluation = {
        **entry_evaluation,
        "evaluated_at": board_checked_at,
        "source": f"保存ルール: {active_rule_name}（確定日足）",
        "visual_blocked": (
            entry_evaluation.get("verdict") == "BUY" and board_entry_blocked
        ),
    }
    board_holding_evaluation = {
        **holding_evaluation,
        "evaluated_at": board_checked_at,
        "source": f"保存ルール: {active_rule_name}（確定日足）",
    }

    checked_alerts = []
    active_alerts = [
        alert for alert in alerts_lib.load()
        if alert.get("enabled")
        and str(alert.get("ticker") or "").strip().upper() == ticker
    ]
    alert_context = {
        "df": None if decision_context is None else decision_context["df"],
        "levels": decision_levels,
        "price": snapshot_for_board.get("price"),
        "previous_close": snapshot_for_board.get("previous_close"),
        "entry_verdict": entry_evaluation.get("verdict"),
        "entry_blocked": board_entry_blocked,
        "holding_verdict": holding_evaluation.get("verdict"),
    }
    for alert in active_alerts:
        try:
            result = alerts_lib.check(alert, alert_context)
        except Exception as exc:
            result = {"triggered": False, "actual": "—", "reason": str(exc)}
        checked_alerts.append({
            "alert": alert,
            "result": {**result, "checked_at": board_checked_at},
        })

    try:
        news = news_fetcher.fetch_news(ticker, info.get("name"))
    except Exception as exc:
        news = []
        warnings.append(f"ニュースを取得できませんでした: {exc}")

    try:
        event_report = event_intelligence.build_event_intelligence(
            ticker, history, info, analyst, [], news,
            fetched_at=pd.Timestamp.now(tz="UTC"), horizon_days=120)
        event_report["events"] = [
            event_intelligence.localize_event_for_display(event)
            for event in event_report.get("events") or []
        ]
    except Exception as exc:
        event_report = {"status": "unavailable", "events": [],
                        "warnings": [str(exc)]}
        warnings.append(f"イベント情報を整理できませんでした: {exc}")

report = information_board.build_information_board({
    "ticker": ticker,
    "as_of": board_checked_at,
    "snapshot": snapshot_for_board,
    "entry_evaluation": board_entry_evaluation,
    "holding_evaluation": board_holding_evaluation,
    "levels": decision_levels,
    "checked_alerts": checked_alerts,
    "alert_checked_at": board_checked_at,
    "analyst": analyst,
    "event_report": event_report,
    "news": news,
    "warnings": warnings,
})

board_ui.render_information_board(
    report, show_heading=False, key_prefix=f"standalone_board_{ticker}")

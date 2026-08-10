"""売買判定、時系列検証、株価アラートのページ（注文機能なし）。"""

from __future__ import annotations

import copy
import json
import math

import pandas as pd
import streamlit as st

from lib import (alerts as alerts_lib, data_fetcher, rule_backtest,
                 rules as rules_lib, trading_context, watchlist_store)


VERDICT_STYLE = {
    "BUY": ("🟢", "買い候補", "success"),
    "TAKE_PROFIT": ("🔵", "利益確定候補", "info"),
    "RISK_EXIT": ("🔴", "リスク退出候補", "error"),
    "HOLD": ("⚪", "保有継続", "info"),
    "WAIT": ("🟠", "判定待機", "warning"),
    "NEUTRAL": ("⚪", "新規エントリー見送り", "info"),
}
REGIME_LABELS = {
    "UPTREND": "上昇トレンド",
    "DOWNTREND": "下降トレンド",
    "RANGE": "レンジ",
    "HIGH_VOL": "高ボラティリティ",
}
SIDE_SPECS = (
    ("buy", "🟢 新規ロング条件"),
    ("take_profit", "🔵 利益確定条件"),
    ("risk_exit", "🔴 リスク退出条件"),
)
POSITION_MODES = {
    "新規ロングを検討": "entry",
    "ロング保有中": "holding",
}
REFRESH_CHOICES = {
    "自動更新しない": 0,
    "1分ごと": 60,
    "3分ごと": 180,
    "5分ごと": 300,
}
SAFETY_DEFAULTS = trading_context.SAFETY_DEFAULTS


def _query_ticker(default: str = "AAPL") -> str:
    try:
        value = st.query_params.get("ticker", default)
    except Exception:
        value = default
    if isinstance(value, list):
        value = value[0] if value else default
    return str(value or default).strip().upper()


@st.cache_data(ttl=300, show_spinner=False)
def _context(ticker: str, allow_new_quota: bool = True,
             include_events: bool = True) -> dict | None:
    """確定日足、データメタ情報、支持抵抗、イベントをまとめる。"""
    try:
        history, source_meta = data_fetcher.fetch_chart_history(
            ticker, "2y", "1d", allow_new_quota=allow_new_quota)
    except data_fetcher.FetchError:
        return None
    if history.empty:
        return None

    market_meta = data_fetcher.fetch_market_state(ticker)
    snapshot = data_fetcher.fetch_realtime_snapshot(ticker)

    earnings_date = None
    if include_events:
        try:
            earnings_date = data_fetcher.fetch_analyst(ticker).get("earnings_date")
        except Exception:
            earnings_date = None

    return trading_context.prepare_from_history(
        history,
        source_meta=source_meta,
        market_meta=market_meta,
        snapshot=snapshot,
        earnings_date=earnings_date,
    )


def _safety(rule: dict) -> dict:
    return trading_context.safety_config(rule)


def _external_gates(ctx: dict, rule: dict) -> list[dict]:
    return trading_context.external_gates(ctx, rule)


def _with_latest_snapshot(ctx: dict, ticker: str) -> dict:
    """履歴/指標キャッシュを保ったまま、30秒TTLのスナップショットだけ更新する。"""
    item = dict(ctx)
    latest = data_fetcher.fetch_realtime_snapshot(ticker)
    item["snapshot"] = latest
    if latest.get("price") is not None:
        item["price"] = float(latest["price"])
        if latest.get("previous_close") is not None:
            item["previous_close"] = float(latest["previous_close"])
    else:
        frame = item.get("df")
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            item["price"] = float(frame["Close"].iloc[-1])
            item["previous_close"] = (float(frame["Close"].iloc[-2])
                                      if len(frame) > 1 else item["price"])
    return item


def _score(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.0f}" if number.is_integer() else f"{number:.1f}"


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _verdict_banner(result: dict, ticker: str, rule_name: str):
    verdict = result["verdict"]
    icon, label, kind = VERDICT_STYLE.get(verdict, ("⚪", verdict, "info"))
    body = f"### {icon} {ticker}: {label}"
    getattr(st, kind)(body)
    st.caption(
        f"ルール「{rule_name}」・{REGIME_LABELS.get(result.get('regime'), result.get('regime'))}。"
        f"{result.get('summary', '')} この画面は注文を出しません。"
    )


def _data_source_panel(ctx: dict):
    meta = ctx["source_meta"]
    cache_labels = {
        "refreshed": "API更新",
        "fresh": "足種別キャッシュ",
        "stale": "期限切れキャッシュ",
        "fallback": "フォールバック",
        "miss": "データなし",
    }
    source = meta.get("source", "不明")
    cache = cache_labels.get(meta.get("cache_status"), meta.get("cache_status") or "—")
    state = ctx["market_meta"].get("market_state") or "取得不能"
    bar = ctx["bar_meta"].get("last_bar") or "—"
    st.caption(f"取得元: **{source}** / {cache} ｜ 市場状態: **{state}** ｜ "
               f"判定に使う最終確定足: **{bar}**")
    if meta.get("remain") is not None:
        st.caption(f"moomoo過去K線: 残り {meta['remain']} / 総枠 {meta.get('quota', '—')}。"
                   "同一銘柄の30日以内再取得は新しい枠を消費しません。")
    if meta.get("cache_status") == "stale":
        st.warning("利用枠または接続を保護するため期限切れキャッシュを使っています。"
                   f" {meta.get('fallback_reason') or ''}", icon="🛡️")
    elif meta.get("fallback_reason"):
        st.caption(f"moomoo未使用理由: {meta['fallback_reason']}")


def _gate_table(gates: list[dict]):
    st.markdown("#### 判定前の安全ゲート")
    rows = []
    for gate in gates:
        passed = gate.get("passed")
        rows.append({
            "項目": gate.get("label"),
            "必須": "必須" if gate.get("required") else "警告のみ",
            "状態": "✅ 成立" if passed is True else "⛔ 不成立" if passed is False else "⚠️ 不明",
            "根拠": gate.get("reason", ""),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _side_table(side: dict, title: str):
    st.markdown(
        f"**{title}** — {_score(side.get('score'))} / {_score(side.get('total'))}点 "
        f"（合格 {_score(side.get('threshold'))}点）"
    )
    threshold = max(float(side.get("threshold") or 1), 1.0)
    st.progress(min(float(side.get("score") or 0) / threshold, 1.0))
    if side.get("invalid_reasons"):
        st.warning(" / ".join(side["invalid_reasons"]))
    if not side.get("checks"):
        st.caption("条件が設定されていません。")
        return
    rows = []
    for check in side["checks"]:
        rows.append({
            "役割": "必須" if check.get("required") else "加点",
            "分類": rules_lib.GROUP_LABELS.get(check.get("group"), check.get("group")),
            "条件": check.get("label"),
            "必要": check.get("requirement"),
            "実測": check.get("actual_text"),
            "判定": ("✅" if check.get("status") == "passed" else
                     "❌" if check.get("status") == "failed" else "—"),
            "配点": check.get("points"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    group_scores = side.get("group_scores") or {}
    if group_scores:
        st.caption("相関グループごとの上限適用後")
        st.dataframe(pd.DataFrame([{
            "分類": rules_lib.GROUP_LABELS.get(group, group),
            "獲得": _score(score),
            "上限": _score((side.get("group_caps") or {}).get(group)),
            "上限前": _score((side.get("group_raw_scores") or {}).get(group, 0)),
        } for group, score in group_scores.items()]), hide_index=True,
            use_container_width=True)


def _risk_plan(plan: dict):
    st.markdown("#### ATR・支持抵抗による参考リスク計画")
    if not plan.get("valid"):
        st.warning("ATRまたは支持抵抗が不足し、参考価格を計算できません。")
        return
    entry, stop, target = plan["entry"], plan["stop"], plan["target"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("判定終値", f"{entry:,.2f}", border=True)
    c2.metric("参考ストップ", f"{stop:,.2f}",
              f"{(stop / entry - 1) * 100:.2f}%", delta_color="inverse", border=True)
    c3.metric("参考目標", f"{target:,.2f}",
              f"+{(target / entry - 1) * 100:.2f}%", border=True)
    c4.metric("R:R", f"{plan['rr']:.2f}倍", border=True)
    st.caption(f"ストップ根拠: {plan.get('stop_source')} / "
               f"目標根拠: {plan.get('target_source')}。発注価格ではありません。")


def _verdict_label(verdict: str) -> str:
    icon, label, _ = VERDICT_STYLE.get(verdict, ("⚪", verdict, "info"))
    return f"{icon} {label}"


st.title("🎯 売買判定・検証・アラート")
st.caption("確定日足と必須ゲートを使い、新規ロングと保有中の退出を分けて判定します。"
           "**空売り判定や注文実行は行いません。**")

store = rules_lib.load()
rule_names = list(store["rules"])
tab_judge, tab_rules, tab_test, tab_alerts = st.tabs(
    ["📊 判定", "⚙️ 判定基準", "🧪 時系列検証", "🔔 アラート"])


# ============================================================== 判定タブ
with tab_judge:
    c_ticker, c_rule, c_position = st.columns([1, 1.35, 1.35])
    ticker = c_ticker.text_input(
        "ティッカーシンボル", value=_query_ticker(), key="judge_ticker").strip().upper()
    active = c_rule.selectbox(
        "使うルール", rule_names,
        index=rule_names.index(store["active"]) if store["active"] in rule_names else 0)
    position_label = c_position.selectbox("判定する状況", list(POSITION_MODES))
    position_mode = POSITION_MODES[position_label]
    if active != store["active"]:
        rules_lib.save(store["rules"], active)
        store["active"] = active

    if not ticker:
        st.info("ティッカーシンボルを入力してください。")
    else:
        ctx = _context(ticker, allow_new_quota=True, include_events=True)
        if ctx is None:
            st.error(f"「{ticker}」のデータを取得できませんでした。")
        else:
            ctx = _with_latest_snapshot(ctx, ticker)
            rule = store["rules"][active]
            gates = _external_gates(ctx, rule)
            result = rules_lib.evaluate(
                ctx["df"], rule, ctx["levels"], position_mode=position_mode,
                external_gates=gates)
            _verdict_banner(result, ticker, active)
            _data_source_panel(ctx)

            confirmed = float(ctx["df"]["Close"].iloc[-1])
            previous = float(ctx["df"]["Close"].iloc[-2]) if len(ctx["df"]) > 1 else confirmed
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("判定価格（確定終値）", f"{confirmed:,.2f}",
                      f"{(confirmed / previous - 1) * 100:+.2f}%", border=True)
            is_realtime = (ctx["snapshot"].get("source") == "moomoo OpenAPI"
                           and ctx["snapshot"].get("price") is not None)
            c2.metric("参考リアルタイム値" if is_realtime else "参考値（確定終値）",
                      f"{ctx['price']:,.2f}",
                      f"{(ctx['price'] / ctx['previous_close'] - 1) * 100:+.2f}%"
                      if ctx["previous_close"] else None, border=True)
            c3.metric("相場レジーム", REGIME_LABELS.get(result["regime"], result["regime"]),
                      border=True)
            if position_mode == "entry":
                shown_side, score_label = result["buy"], "新規買いスコア"
            elif result["verdict"] == "TAKE_PROFIT":
                shown_side, score_label = result["take_profit"], "利益確定スコア"
            else:
                shown_side, score_label = result["risk_exit"], "リスク退出スコア"
            c4.metric(score_label, f"{_score(shown_side['score'])} / {_score(shown_side['total'])}",
                      f"合格 {_score(shown_side['threshold'])}点", delta_color="off", border=True)

            _gate_table(result["gates"])
            if position_mode == "entry":
                _side_table(result["buy"], "🟢 新規ロング")
            else:
                col_exit, col_profit = st.columns(2)
                with col_exit:
                    _side_table(result["risk_exit"], "🔴 リスク退出（優先）")
                with col_profit:
                    _side_table(result["take_profit"], "🔵 利益確定")
            _risk_plan(result["risk_plan"])

            st.divider()
            st.markdown("#### ウォッチリストを一括判定")
            st.caption("一括判定は、未取得銘柄のmoomoo過去K線枠を新しく消費しません。"
                       "キャッシュまたはYahoo Financeを使います。")
            watchlist = watchlist_store.load()
            if not watchlist:
                st.caption("ウォッチリストが空です。「市場概況」ページで登録できます。")
            elif st.button("新規ロング候補を一括判定", key="judge_watchlist"):
                rows = []
                for symbol in watchlist:
                    item = _context(symbol, allow_new_quota=False, include_events=True)
                    if item is None:
                        rows.append({"銘柄": symbol, "判定": "取得失敗", "スコア": None,
                                     "確定終値": None, "取得元": None})
                        continue
                    evaluated = rules_lib.evaluate(
                        item["df"], rule, item["levels"], position_mode="entry",
                        external_gates=_external_gates(item, rule))
                    rows.append({
                        "銘柄": symbol,
                        "判定": _verdict_label(evaluated["verdict"]),
                        "スコア": evaluated["buy"]["score"],
                        "確定終値": float(item["df"]["Close"].iloc[-1]),
                        "取得元": item["source_meta"].get("source"),
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ====================================================== 判定基準の設定タブ
with tab_rules:
    st.caption("必須条件、相場レジーム、相関グループ上限、安全ゲート、ATRリスク幅を"
               "詳細に設定できます。既定値は暫定仮説で、検証結果による確認が必要です。")
    c_select, c_new = st.columns([2, 1])
    edit_name = c_select.selectbox("編集するルール", rule_names, key="edit_rule")
    with c_new.popover("➕ 新規作成", use_container_width=True):
        base = st.selectbox("複製元", rule_names, key="clone_from")
        new_name = st.text_input("新しい名前", key="new_rule_name")
        if st.button("作成", key="create_rule"):
            name = new_name.strip()
            if not name:
                st.warning("名前を入力してください。")
            elif name in store["rules"]:
                st.warning("同じ名前のルールがあります。")
            else:
                store["rules"][name] = copy.deepcopy(store["rules"][base])
                rules_lib.save(store["rules"], name)
                st.rerun()

    original = rules_lib.upgrade_rule(store["rules"][edit_name])
    edited = {
        "schema_version": rules_lib.SCHEMA_VERSION,
        "allowed_regimes": [],
        "group_caps": {},
        "risk": {},
        "safety": {},
        "buy": {}, "take_profit": {}, "risk_exit": {},
    }

    st.markdown("#### 1. 相場状態と安全ゲート")
    edited["allowed_regimes"] = st.multiselect(
        "新規ロングを許可する相場レジーム",
        list(rules_lib.REGIMES),
        default=[item for item in original.get("allowed_regimes", [])
                 if item in rules_lib.REGIMES],
        format_func=lambda item: REGIME_LABELS.get(item, item),
        key=f"regimes_{edit_name}")
    saved_safety = _safety(original)
    s1, s2, s3, s4 = st.columns(4)
    edited["safety"] = {
        "min_history": int(s1.number_input(
            "必要な確定日足", 50, 1000, int(saved_safety["min_history"]), 10,
            key=f"history_{edit_name}")),
        "min_median_dollar_volume": float(s2.number_input(
            "20日売買代金の下限", 0.0, 1_000_000_000.0,
            float(saved_safety["min_median_dollar_volume"]), 1_000_000.0,
            format="%.0f", key=f"liquidity_{edit_name}")),
        "max_spread_pct": float(s3.number_input(
            "許容スプレッド(%)", 0.01, 10.0,
            float(saved_safety["max_spread_pct"]), 0.05,
            key=f"spread_{edit_name}")),
        "earnings_blackout_days": int(s4.number_input(
            "決算前後の待機(営業日)", 0, 20,
            int(saved_safety["earnings_blackout_days"]), 1,
            key=f"earnings_{edit_name}")),
        "max_stale_business_days": int(saved_safety["max_stale_business_days"]),
    }

    st.markdown("#### 2. 相関グループの配点上限")
    cap_columns = st.columns(len(rules_lib.GROUPS))
    for column, group in zip(cap_columns, rules_lib.GROUPS):
        edited["group_caps"][group] = int(column.number_input(
            rules_lib.GROUP_LABELS[group], 1, 100,
            int((original.get("group_caps") or {}).get(group, 25)), 5,
            key=f"cap_{edit_name}_{group}"))

    st.markdown("#### 3. ATRリスク設定")
    saved_risk = original.get("risk") or {}
    r1, r2, r3, r4 = st.columns(4)
    edited["risk"] = {
        "support_buffer_atr": float(r1.number_input(
            "支持帯の外側(ATR)", 0.0, 5.0,
            float(saved_risk.get("support_buffer_atr", 0.5)), 0.1,
            key=f"support_buffer_{edit_name}")),
        "fallback_stop_atr": float(r2.number_input(
            "代替ストップ(ATR)", 0.1, 10.0,
            float(saved_risk.get("fallback_stop_atr", 1.5)), 0.1,
            key=f"stop_atr_{edit_name}")),
        "fallback_target_atr": float(r3.number_input(
            "代替目標(ATR)", 0.1, 20.0,
            float(saved_risk.get("fallback_target_atr", 2.0)), 0.1,
            key=f"target_atr_{edit_name}")),
        "min_level_strength": int(r4.number_input(
            "利用する支持抵抗の最低★", 1, 5,
            int(saved_risk.get("min_level_strength", 3)), 1,
            key=f"level_strength_{edit_name}")),
    }

    st.markdown("#### 4. 条件・トリガー・配点")
    metric_ids = list(rules_lib.METRICS)
    metric_labels = {key: value["label"] for key, value in rules_lib.METRICS.items()}
    for side_key, side_title in SIDE_SPECS:
        with st.expander(side_title, expanded=side_key == "buy"):
            side = original.get(side_key, {})
            table = pd.DataFrame([{
                "必須": bool(condition.get("required", False)),
                "指標": metric_labels.get(condition.get("metric"), condition.get("metric")),
                "条件": condition.get("op", ">="),
                "しきい値": float(condition.get("value", 0)),
                "配点": int(condition.get("points", 0)),
            } for condition in side.get("conditions", [])])
            if table.empty:
                table = pd.DataFrame({
                    "必須": pd.Series(dtype=bool), "指標": pd.Series(dtype=str),
                    "条件": pd.Series(dtype=str), "しきい値": pd.Series(dtype=float),
                    "配点": pd.Series(dtype=int),
                })
            output = st.data_editor(
                table, num_rows="dynamic", hide_index=True,
                key=f"editor_{edit_name}_{side_key}", use_container_width=True,
                column_config={
                    "必須": st.column_config.CheckboxColumn(
                        help="未取得ならWAIT、不成立ならこの側の判定を止めます"),
                    "指標": st.column_config.SelectboxColumn(
                        options=[metric_labels[key] for key in metric_ids], required=True),
                    "条件": st.column_config.SelectboxColumn(
                        options=[">=", "<="], required=True),
                    "しきい値": st.column_config.NumberColumn(format="%.3f", required=True),
                    "配点": st.column_config.NumberColumn(
                        min_value=0, max_value=100, step=5, required=True),
                })
            conditions = rules_lib.conditions_from_table(output)
            totals = {}
            for condition in conditions:
                group = rules_lib.METRICS[condition["metric"]]["group"]
                totals[group] = totals.get(group, 0) + int(condition["points"])
            side_caps = side.get("group_caps") if isinstance(side.get("group_caps"), dict) else {}
            merged_caps = {**edited["group_caps"], **side_caps}
            maximum = int(sum(min(total, int(merged_caps.get(group, total)))
                              for group, total in totals.items()))
            current_threshold = max(1, min(int(side.get("threshold", 1)), max(maximum, 1)))
            threshold = int(st.number_input(
                f"{side_title}の合格点（上限適用後の満点 {maximum}）",
                min_value=1, max_value=max(maximum, 1), value=current_threshold, step=5,
                key=f"threshold_{edit_name}_{side_key}"))
            edited[side_key] = {
                "conditions": conditions,
                "threshold": threshold,
                **({"group_caps": copy.deepcopy(side_caps)} if side_caps else {}),
            }

    with st.expander("指標一覧と相関分類"):
        st.dataframe(pd.DataFrame([{
            "指標": value["label"],
            "分類": rules_lib.GROUP_LABELS[value["group"]],
            "単位": value["unit"] or "—",
            "説明": value["help"] or "—",
        } for value in rules_lib.METRICS.values()]), hide_index=True,
            use_container_width=True)

    problems = rules_lib.validate(edited)
    for problem in problems:
        st.warning(problem, icon="⚠️")
    c_save, c_delete = st.columns(2)
    if c_save.button(
            "💾 このルールを保存", type="primary", key="save_rule",
            disabled=bool(problems), use_container_width=True):
        store["rules"][edit_name] = edited
        rules_lib.save(store["rules"], store["active"])
        st.success(f"「{edit_name}」を保存しました。")
    if c_delete.button(
            "🗑️ このルールを削除", key="delete_rule",
            disabled=len(rule_names) <= 1, use_container_width=True):
        del store["rules"][edit_name]
        rules_lib.save(store["rules"], next(iter(store["rules"])))
        st.rerun()


# ============================================================ 時系列検証タブ
with tab_test:
    st.caption("取得した確定日足だけを使い、各日で過去データだけから支持抵抗を"
               "再計算します。検証用の銘柄取得で新しいmoomoo過去K線枠は消費しません。")
    st.caption("決算日、当時のスプレッド・板・資金フローは履歴検証に含みません。"
               "これらは現在時点の判定ゲートとしてのみ使います。")
    v1, v2 = st.columns(2)
    test_ticker = v1.text_input("検証ティッカー", value=_query_ticker(),
                                key="backtest_ticker").strip().upper()
    test_rule_name = v2.selectbox("検証ルール", rule_names, key="backtest_rule")
    p1, p2, p3, p4 = st.columns(4)
    train_bars = int(p1.number_input("学習/ウォームアップ本数", 220, 1000, 252, 20))
    test_bars = int(p2.number_input("1区間の検証本数", 20, 252, 63, 5))
    one_way_bps = float(p3.number_input("片道コスト(bp)", 0.0, 100.0, 10.0, 1.0))
    slippage_bps = float(p4.number_input("片道スリッページ(bp)", 0.0, 100.0, 5.0, 1.0))
    h1, h2 = st.columns(2)
    max_holding = int(h1.number_input("最大保有日数", 1, 252, 20, 1))
    step_bars = int(h2.number_input("検証区間の移動幅", 10, 252, 63, 5))
    overlaps = step_bars < test_bars
    if overlaps:
        st.error("検証区間の移動幅は、1区間の検証本数以上にしてください。"
                 "重複区間を集計すると同じ期間の成績を二重計上します。")

    rule_fingerprint = json.dumps(
        store["rules"][test_rule_name], ensure_ascii=False, sort_keys=True)
    diagnostic_key = (
        test_ticker, test_rule_name, rule_fingerprint, train_bars, test_bars,
        step_bars, one_way_bps, slippage_bps, max_holding)

    if st.button("ウォークフォワード診断を実行", type="primary", key="run_walk_forward",
                 disabled=overlaps or not test_ticker):
        # 診断はキャッシュ/既取得銘柄またはYahooを使い、新規moomoo履歴枠を消費しない。
        context = _context(test_ticker, allow_new_quota=False, include_events=False)
        if context is None:
            st.session_state["walk_forward_result"] = {
                "error": f"{test_ticker}のデータを取得できませんでした"}
        else:
            with st.spinner("確定日足を時系列順に検証しています..."):
                result = rule_backtest.run_walk_forward(
                    context["df"], store["rules"][test_rule_name],
                    train_bars=train_bars, test_bars=test_bars,
                    step_bars=step_bars, one_way_bps=one_way_bps,
                    slippage_bps=slippage_bps,
                    max_holding_days=max_holding)
            st.session_state["walk_forward_result"] = result
        st.session_state["walk_forward_key"] = diagnostic_key

    saved_diagnostic = st.session_state.get("walk_forward_result")
    diagnostic = (saved_diagnostic
                  if st.session_state.get("walk_forward_key") == diagnostic_key else None)
    if saved_diagnostic is not None and diagnostic is None:
        st.info("銘柄・ルール・検証条件が変わりました。現在の条件で診断を再実行してください。")
    if diagnostic:
        if diagnostic.get("error"):
            st.error(diagnostic["error"])
        else:
            summary = diagnostic.get("summary", {})
            if summary.get("passed"):
                st.success("暫定診断基準を満たしました。ただし将来の利益を保証しません。")
            else:
                st.warning("暫定診断基準を満たしていません。実売買条件として採用しないでください。")
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("取引数", int(summary.get("trade_count", 0)), border=True)
            win_rate = _number(summary.get("win_rate"))
            b2.metric("勝率", "—" if win_rate is None else f"{win_rate * 100:.1f}%",
                      border=True)
            pf = _number(summary.get("profit_factor"))
            pf_text = "—" if pf is None else "∞" if math.isinf(pf) else f"{pf:.2f}"
            b3.metric("Profit Factor", pf_text, border=True)
            avg_return = _number(summary.get("avg_net_return"))
            b4.metric("平均純損益率", "—" if avg_return is None
                      else f"{avg_return * 100:.2f}%", border=True)
            max_dd = _number(summary.get("max_drawdown"))
            b5.metric("最大DD（日次時価）", "—" if max_dd is None
                      else f"-{max_dd * 100:.2f}%", border=True)
            reasons = summary.get("reasons") or []
            warnings = summary.get("warnings") or []
            if reasons:
                st.markdown("**診断メモ**" if summary.get("passed") else "**未達理由**")
                for reason in reasons:
                    st.write(f"- {reason}")
            if warnings:
                st.caption(" / ".join(str(item) for item in warnings))
            folds = diagnostic.get("folds")
            trades = diagnostic.get("trades")
            if isinstance(folds, pd.DataFrame) and not folds.empty:
                st.markdown("#### 検証区間別")
                st.dataframe(folds, hide_index=True, use_container_width=True)
            if isinstance(trades, pd.DataFrame) and not trades.empty:
                st.markdown("#### 仮想取引明細")
                st.dataframe(trades, hide_index=True, use_container_width=True)
            st.caption(str(summary.get("note") or ""))


# ============================================================ アラートタブ
with tab_alerts:
    st.info("アラートは**アプリを開いている間だけ**判定します。閉じている間の監視や、"
            "メール・スマートフォンへの通知は行いません。", icon="ℹ️")
    saved_alerts = alerts_lib.load()

    with st.expander("➕ アラートを追加", expanded=not saved_alerts):
        a1, a2 = st.columns([1, 2])
        alert_ticker = a1.text_input("ティッカー", key="alert_ticker").strip().upper()
        selectable_kinds = [key for key in alerts_lib.KINDS if key != "rule_sell"]
        kind_labels = {key: alerts_lib.KINDS[key]["label"] for key in selectable_kinds}
        alert_kind = a2.selectbox(
            "条件", selectable_kinds, format_func=lambda key: kind_labels[key],
            key="alert_kind")
        needs_value = alerts_lib.KINDS[alert_kind]["needs_value"]
        b1, b2 = st.columns([1, 2])
        alert_value = b1.number_input(
            f"しきい値({alerts_lib.KINDS[alert_kind]['unit'] or '—'})",
            value=0.0, step=0.5, format="%.2f", key="alert_value",
            disabled=not needs_value)
        alert_note = b2.text_input("メモ（任意）", key="alert_note")
        if st.button("追加", key="add_alert"):
            if not alert_ticker:
                st.warning("ティッカーを入力してください。")
            else:
                saved_alerts.append(alerts_lib.new_alert(
                    alert_ticker, alert_kind,
                    alert_value if needs_value else None, alert_note))
                alerts_lib.save(saved_alerts)
                st.rerun()

    if not saved_alerts:
        st.caption("アラートはまだありません。")
    else:
        interval_label = st.selectbox("自動更新", list(REFRESH_CHOICES), index=0,
                                      key="alert_refresh")
        every = REFRESH_CHOICES[interval_label] or None
        st.caption("アラート巡回では未取得銘柄のmoomoo過去K線枠を新しく消費しません。")

        @st.fragment(run_every=every)
        def _alert_panel():
            now = pd.Timestamp.now(tz="America/New_York")
            st.caption(f"最終チェック: {now.strftime('%Y-%m-%d %H:%M:%S')} ET")
            contexts: dict[str, dict] = {}
            active_rule = store["rules"].get(store["active"], {})
            for symbol in alerts_lib.tickers(saved_alerts):
                item = _context(symbol, allow_new_quota=False, include_events=True)
                if item is None:
                    continue
                item = _with_latest_snapshot(item, symbol)
                gates = _external_gates(item, active_rule)
                entry = rules_lib.evaluate(
                    item["df"], active_rule, item["levels"], position_mode="entry",
                    external_gates=gates)
                holding = rules_lib.evaluate(
                    item["df"], active_rule, item["levels"], position_mode="holding",
                    external_gates=gates)
                item["entry_verdict"] = entry["verdict"]
                item["holding_verdict"] = holding["verdict"]
                contexts[symbol] = item

            fired, rows = [], []
            for alert in saved_alerts:
                context = contexts.get(alert["ticker"])
                if not alert.get("enabled"):
                    state, actual = "停止中", "—"
                elif context is None:
                    state, actual = "データなし", "—"
                else:
                    checked = alerts_lib.check(alert, context)
                    state = "成立" if checked["triggered"] else "監視中"
                    actual = checked["actual"]
                    if checked["triggered"]:
                        fired.append((alert, actual))
                rows.append({
                    "銘柄": alert["ticker"],
                    "条件": alerts_lib.describe(alert).split(": ", 1)[1],
                    "現在の値": actual,
                    "状態": state,
                    "メモ": alert.get("note", ""),
                })
            if fired:
                st.error(f"### 🔔 {len(fired)}件のアラートが成立しています")
                for alert, actual in fired:
                    note = f" — {alert['note']}" if alert.get("note") else ""
                    line = f"- **{alerts_lib.describe(alert)}**（現在 {actual}）{note}"
                    st.write(line.replace("$", "\\$"))
            else:
                st.success("成立しているアラートはありません。", icon="✅")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        _alert_panel()

        st.markdown("#### 管理")
        for alert in list(saved_alerts):
            c1, c2, c3 = st.columns([6, 1.2, 1.2])
            c1.write((alerts_lib.describe(alert)
                      + (f" — {alert['note']}" if alert.get("note") else ""))
                     .replace("$", "\\$"))
            enabled = c2.toggle("有効", value=alert.get("enabled", True),
                                key=f"enabled_{alert['id']}")
            if enabled != alert.get("enabled", True):
                alert["enabled"] = enabled
                alerts_lib.save(saved_alerts)
                st.rerun()
            if c3.button("削除", key=f"delete_{alert['id']}"):
                alerts_lib.save([item for item in saved_alerts
                                 if item["id"] != alert["id"]])
                st.rerun()

st.divider()
st.caption("判定、参考ストップ、検証、アラートはいずれも参考情報です。"
           "既定条件も利益を保証せず、売買の実行は行いません。")

"""売買判定(カスタムルール)とアラートのページ。判定表示のみ・発注機能なし。"""

import pandas as pd
import streamlit as st

from lib import (alerts as alerts_lib, data_fetcher, indicators, levels,
                 rules as rules_lib, ui, watchlist_store)

VERDICT_STYLE = {
    "BUY": ("🟢", "green", "買い条件が成立"),
    "SELL": ("🔴", "red", "売り条件が成立"),
    "CONFLICT": ("🟠", "orange", "買いと売りが同時成立"),
    "NEUTRAL": ("⚪", "gray", "様子見"),
}
REFRESH_CHOICES = {"自動更新しない": 0, "1分ごと": 60, "3分ごと": 180, "5分ごと": 300}


@st.cache_data(ttl=300, show_spinner=False)
def _context(ticker: str) -> dict | None:
    """判定に使うデータ(指標付き日足とサポレジ)をまとめて用意する。"""
    try:
        hist = data_fetcher.fetch_history(ticker, "2y")
    except data_fetcher.FetchError:
        return None
    if hist.empty:
        return None
    df = indicators.add_indicators(hist)
    view = indicators.slice_display(df, 182)
    return {"df": df, "levels": levels.find_levels(view)}


def _verdict_banner(verdict: str, ticker: str, rule_name: str):
    icon, _color, text = VERDICT_STYLE[verdict]
    body = f"### {icon} {ticker}: {text}"
    if verdict == "BUY":
        st.success(body)
    elif verdict == "SELL":
        st.error(body)
    elif verdict == "CONFLICT":
        st.warning(body)
    else:
        st.info(body)
    st.caption(f"ルール「{rule_name}」による判定です。"
               "このツールは注文を出しません。最終判断はご自身で行ってください。")


def _side_table(side: dict, title: str):
    st.markdown(f"**{title}** — {side['score']} / {side['total']}点"
                f"(合格 {side['threshold']}点)")
    st.progress(min(side["score"] / max(side["threshold"], 1), 1.0))
    if not side["checks"]:
        st.caption("条件が設定されていません。")
        return
    st.dataframe(pd.DataFrame([{
        "条件": c["label"], "必要": c["requirement"], "実測": c["actual_text"],
        "判定": "○" if c["ok"] else "×", "配点": c["points"],
    } for c in side["checks"]]), hide_index=True)


st.title("🎯 売買判定・アラート")
st.caption("自分で決めた条件で買い・売りを判定し、価格やRSIの節目を知らせます。"
           "**注文は一切出しません。**")

store = rules_lib.load()
rule_names = list(store["rules"])

tab_judge, tab_rules, tab_alerts = st.tabs(
    ["📊 判定", "⚙️ 判定基準の設定", "🔔 アラート"])

# ============================================================== 判定タブ
with tab_judge:
    c_tk, c_rule = st.columns([1, 1.4])
    ticker = c_tk.text_input("ティッカーシンボル", value="AAPL",
                             key="judge_ticker").strip().upper()
    active = c_rule.selectbox("使うルール", rule_names,
                              index=rule_names.index(store["active"])
                              if store["active"] in rule_names else 0)
    if active != store["active"]:
        rules_lib.save(store["rules"], active)
        store["active"] = active

    if not ticker:
        st.info("ティッカーシンボルを入力してください。")
    else:
        ctx = _context(ticker)
        if ctx is None:
            st.error(f"「{ticker}」のデータを取得できませんでした。"
                     "ティッカーが正しいか、時間をおいて再試行してください。")
        else:
            rule = store["rules"][active]
            res = rules_lib.evaluate(ctx["df"], rule, ctx["levels"])
            _verdict_banner(res["verdict"], ticker, active)

            price = float(ctx["df"]["Close"].iloc[-1])
            prev = float(ctx["df"]["Close"].iloc[-2]) if len(ctx["df"]) > 1 else price
            m1, m2, m3 = st.columns(3)
            m1.metric("現在値(終値)", f"${price:,.2f}",
                      f"{(price / prev - 1) * 100:+.2f}%" if prev else None,
                      border=True)
            m2.metric("買いスコア", f"{res['buy']['score']} / {res['buy']['total']}",
                      f"合格 {res['buy']['threshold']}点",
                      delta_color="off", border=True)
            m3.metric("売りスコア", f"{res['sell']['score']} / {res['sell']['total']}",
                      f"合格 {res['sell']['threshold']}点",
                      delta_color="off", border=True)

            col_b, col_s = st.columns(2)
            with col_b:
                _side_table(res["buy"], "🟢 買い条件")
            with col_s:
                _side_table(res["sell"], "🔴 売り条件")

            st.divider()
            st.markdown("#### ウォッチリストを一括判定")
            wl = watchlist_store.load()
            if not wl:
                st.caption("ウォッチリストが空です。「市場概況」ページで登録できます。")
            elif st.button("ウォッチリストを判定", key="judge_wl"):
                rows = []
                for t in wl:
                    c = _context(t)
                    if c is None:
                        rows.append({"銘柄": t, "判定": "取得失敗", "買い": None,
                                     "売り": None, "現在値": None})
                        continue
                    r = rules_lib.evaluate(c["df"], rule, c["levels"])
                    rows.append({
                        "銘柄": t,
                        "判定": {"BUY": "🟢 買い", "SELL": "🔴 売り",
                                 "CONFLICT": "🟠 競合", "NEUTRAL": "⚪ 様子見"}[r["verdict"]],
                        "買い": r["buy"]["score"], "売り": r["sell"]["score"],
                        "現在値": float(c["df"]["Close"].iloc[-1]),
                    })
                st.dataframe(pd.DataFrame(rows).style.format(
                    {"現在値": "${:,.2f}"}, na_rep="—"), hide_index=True)

# ====================================================== 判定基準の設定タブ
with tab_rules:
    st.caption("条件・しきい値・配点を自由に決められます。"
               "満たした条件の配点を合計し、合格点に達したら判定成立です。")

    c_sel, c_new = st.columns([2, 1])
    edit_name = c_sel.selectbox("編集するルール", rule_names, key="edit_rule")
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
                import copy
                store["rules"][name] = copy.deepcopy(store["rules"][base])
                rules_lib.save(store["rules"], name)
                st.rerun()

    rule = store["rules"][edit_name]
    edited = {"buy": {"conditions": [], "threshold": 0},
              "sell": {"conditions": [], "threshold": 0}}
    metric_ids = list(rules_lib.METRICS)
    metric_labels = {k: v["label"] for k, v in rules_lib.METRICS.items()}

    for side_key, side_title in (("buy", "🟢 買い条件"), ("sell", "🔴 売り条件")):
        st.markdown(f"#### {side_title}")
        side = rule.get(side_key, {})
        table = pd.DataFrame([{
            "指標": metric_labels.get(c["metric"], c["metric"]),
            "条件": c["op"], "しきい値": float(c["value"]),
            "配点": int(c.get("points", 0)),
        } for c in side.get("conditions", [])])
        if table.empty:
            table = pd.DataFrame({"指標": pd.Series(dtype=str),
                                  "条件": pd.Series(dtype=str),
                                  "しきい値": pd.Series(dtype=float),
                                  "配点": pd.Series(dtype=int)})
        out = st.data_editor(
            table, num_rows="dynamic", hide_index=True, key=f"editor_{side_key}",
            column_config={
                "指標": st.column_config.SelectboxColumn(
                    options=[metric_labels[m] for m in metric_ids], required=True),
                "条件": st.column_config.SelectboxColumn(
                    options=[">=", "<="], required=True,
                    help=">= は「以上で成立」、<= は「以下で成立」"),
                "しきい値": st.column_config.NumberColumn(format="%.2f", required=True),
                "配点": st.column_config.NumberColumn(min_value=0, max_value=100,
                                                      step=5, required=True),
            })
        conds = rules_lib.conditions_from_table(out)
        total = sum(c["points"] for c in conds)
        thr = st.slider(f"{side_title}の合格点", 0, max(total, 1),
                        min(int(side.get("threshold", 0)), max(total, 1)),
                        key=f"thr_{side_key}",
                        help=f"配点合計は {total}点です")
        edited[side_key] = {"conditions": conds, "threshold": int(thr)}

        with st.expander("指標の説明"):
            st.dataframe(pd.DataFrame([
                {"指標": v["label"], "単位": v["unit"] or "—", "説明": v["help"] or "—"}
                for v in rules_lib.METRICS.values()]), hide_index=True)

    problems = rules_lib.validate(edited)
    for p in problems:
        st.warning(p, icon="⚠️")

    c_save, c_del = st.columns([1, 1])
    if c_save.button("💾 このルールを保存", type="primary", key="save_rule",
                     use_container_width=True):
        store["rules"][edit_name] = edited
        rules_lib.save(store["rules"], store["active"])
        st.success(f"「{edit_name}」を保存しました。")
    if c_del.button("🗑️ このルールを削除", key="del_rule",
                    disabled=len(rule_names) <= 1, use_container_width=True):
        del store["rules"][edit_name]
        rules_lib.save(store["rules"], next(iter(store["rules"])))
        st.rerun()

# ============================================================ アラートタブ
with tab_alerts:
    st.info("アラートは**アプリを開いている間だけ**判定します。"
            "閉じている間の監視や、メール・スマホへの通知はできません。", icon="ℹ️")

    saved_alerts = alerts_lib.load()

    with st.expander("➕ アラートを追加", expanded=not saved_alerts):
        a1, a2 = st.columns([1, 2])
        a_ticker = a1.text_input("ティッカー", key="al_ticker").strip().upper()
        kind_labels = {k: v["label"] for k, v in alerts_lib.KINDS.items()}
        a_kind = a2.selectbox("条件", list(kind_labels),
                              format_func=lambda k: kind_labels[k], key="al_kind")
        needs = alerts_lib.KINDS[a_kind]["needs_value"]
        b1, b2 = st.columns([1, 2])
        a_value = b1.number_input(
            f"しきい値({alerts_lib.KINDS[a_kind]['unit'] or '—'})",
            value=0.0, step=0.5, format="%.2f", key="al_value",
            disabled=not needs)
        a_note = b2.text_input("メモ(任意)", key="al_note")
        if st.button("追加", key="al_add"):
            if not a_ticker:
                st.warning("ティッカーを入力してください。")
            else:
                saved_alerts.append(alerts_lib.new_alert(
                    a_ticker, a_kind, a_value if needs else None, a_note))
                alerts_lib.save(saved_alerts)
                st.rerun()

    if not saved_alerts:
        st.caption("アラートはまだありません。")
    else:
        interval = st.selectbox("自動更新", list(REFRESH_CHOICES), index=0,
                                key="al_refresh",
                                help="開いている間、この間隔で再判定します")
        every = REFRESH_CHOICES[interval] or None

        @st.fragment(run_every=every)
        def _alert_panel():
            now = pd.Timestamp.now(tz="America/New_York")
            st.caption(f"最終チェック: {now.strftime('%Y-%m-%d %H:%M:%S')} ET")

            ctxs: dict[str, dict] = {}
            for t in alerts_lib.tickers(saved_alerts):
                c = _context(t)
                if c is None:
                    continue
                rule = store["rules"].get(store["active"], {})
                c = dict(c)
                c["rule_verdict"] = rules_lib.evaluate(
                    c["df"], rule, c["levels"])["verdict"]
                ctxs[t] = c

            fired, quiet, rows = [], [], []
            for a in saved_alerts:
                ctx = ctxs.get(a["ticker"])
                if not a.get("enabled"):
                    state, actual = "停止中", "—"
                elif ctx is None:
                    state, actual = "データなし", "—"
                else:
                    r = alerts_lib.check(a, ctx)
                    state = "成立" if r["triggered"] else "監視中"
                    actual = r["actual"]
                    (fired if r["triggered"] else quiet).append(
                        (a, actual))
                rows.append({"銘柄": a["ticker"],
                             "条件": alerts_lib.describe(a).split(": ", 1)[1],
                             "現在の値": actual, "状態": state,
                             "メモ": a.get("note", "")})

            if fired:
                st.error(f"### 🔔 {len(fired)}件のアラートが成立しています")
                for a, actual in fired:
                    note = f" — {a['note']}" if a.get("note") else ""
                    # $が2つ以上あるとLaTeXとして解釈されるためエスケープする
                    line = f"- **{alerts_lib.describe(a)}**(現在 {actual}){note}"
                    st.write(line.replace("$", "\\$"))
            else:
                st.success("成立しているアラートはありません。", icon="✅")

            st.dataframe(pd.DataFrame(rows).style.map(
                lambda v: ("color: #d03b3b; font-weight: 700" if v == "成立"
                           else "color: #898781" if v in ("停止中", "データなし")
                           else ""), subset=["状態"]), hide_index=True)

        _alert_panel()

        st.markdown("#### 管理")
        for a in list(saved_alerts):
            c1, c2, c3 = st.columns([6, 1.2, 1.2])
            c1.write((alerts_lib.describe(a)
                      + (f" — {a['note']}" if a.get("note") else ""))
                     .replace("$", "\\$"))
            new_enabled = c2.toggle("有効", value=a.get("enabled", True),
                                    key=f"en_{a['id']}")
            if new_enabled != a.get("enabled", True):
                a["enabled"] = new_enabled
                alerts_lib.save(saved_alerts)
                st.rerun()
            if c3.button("削除", key=f"del_{a['id']}"):
                alerts_lib.save([x for x in saved_alerts if x["id"] != a["id"]])
                st.rerun()

st.divider()
st.caption("判定・アラートはいずれも参考情報です。売買の実行は行いません。"
           "しきい値は自由に変更できますが、有効性の検証はされていません。")

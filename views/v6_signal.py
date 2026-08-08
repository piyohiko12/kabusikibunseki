"""Route B V6 判定ページ(判定の表示のみ・発注機能なし)。"""

import pandas as pd
import streamlit as st

from lib import data_fetcher, ui, v6_signals as v6

ET = "America/New_York"

st.title("🤖 V6 判定")
st.caption("Route B V6 のルールを、そのまま計算して表示します。"
           "**このページは注文を出しません。** 判断は利用者が行ってください。")

with st.expander("⚠️ 使う前に必ずお読みください", expanded=False):
    st.markdown(
        "- **点数は上昇確率でも期待利益でもありません。** "
        "あらかじめ固定した短期トレンド・出来高・変動の条件が同時に確認された、"
        "という事実を示すだけです\n"
        "- このルールは**検証されていません**。利益を保証しません\n"
        "- 決算・ニュース・板の厚さ・ファンダメンタルズ・VIXなどは"
        "**一切見ていません**。固定44銘柄の外で急騰している銘柄も対象外です\n"
        "- データは Yahoo Finance の5分足で、**15〜20分遅延**しています。"
        "仕様が想定するリアルタイムのスナップショットではありません\n"
        "- 「日次2倍ETF」は1日単位で指数の約2倍を目指す商品です。"
        "複利効果により、複数日では2倍になりません")

col_run, col_note = st.columns([1, 3])
with col_run:
    run = st.button("🔄 判定を実行", type="primary", use_container_width=True)
with col_note:
    st.caption("67銘柄(SPY + セクターETF11 + 普通株44 + 日次2倍ETF11)の"
               "5分足をまとめて取得します。結果は5分間キャッシュされます。")

if not run and "v6_done" not in st.session_state:
    st.info("「判定を実行」を押すと、その時点のデータで判定します。")
    st.stop()
st.session_state["v6_done"] = True

try:
    frames = data_fetcher.fetch_intraday_batch(tuple(v6.universe()))
except data_fetcher.FetchError as e:
    st.error(f"データの取得に失敗しました: {e}")
    st.stop()

if not frames:
    st.error("データを取得できませんでした。米国市場の取引時間中に再試行してください。")
    st.stop()

now_et = pd.Timestamp.now(tz=ET)
res = v6.evaluate(frames, now_et)
data = res["data"]

# ------------------------------------------------------------ 判定の結論
decision = res["decision"]
if decision == "BUY":
    st.success(f"### 判定: BUY 候補  —  {res['choice']['symbol']}"
               f"({res['choice']['kind']})", icon="🟢")
    st.caption("ルール上の条件がすべて成立しています。"
               "実際に買うかどうかはご自身で判断してください。")
else:
    st.warning("### 判定: WAIT(条件不足)", icon="⏸️")
    for r in res["reasons"]:
        st.write(f"- {r}")
    st.caption("WAITは失敗ではなく、条件が足りないときの正常な判断です。")

# ------------------------------------------------------- データ健全性
with st.container(border=True):
    d1, d2, d3 = st.columns(3)
    d1.metric("取得できた銘柄", f"{len(frames)} / {len(v6.universe())}")
    d2.metric("最小サンプル数", f"{data['samples']}本",
              f"必要 {v6.MIN_SAMPLES}本",
              delta_color="off" if data["samples"] >= v6.MIN_SAMPLES else "inverse")
    d3.metric("データ時刻(ET)",
              data["latest"].tz_convert(ET).strftime("%H:%M")
              if data["latest"] is not None else "—")
    if data["issues"]:
        for msg in data["issues"]:
            st.warning(msg, icon="⚠️")
    else:
        st.caption("✅ データ健全性の検査に問題はありません。")

st.divider()

# ------------------------------------------------------------ セクター
st.subheader("① セクター選定")
if not res["sectors"]:
    st.info("セクターを採点できるデータがありません。")
else:
    sec_df = pd.DataFrame([{
        "セクター": f"{s['jp']}({s['etf']})",
        "得点": s["score"],
        **{c["label"]: ("○" if c["ok"] else "×") for c in s["checks"]},
    } for s in res["sectors"]])
    def _score_color(v):
        if v >= v6.STRONG_SECTOR_FOR_2X:
            return "color: #006300; font-weight: 700"
        if v >= v6.SECTOR_PASS:
            return "color: #0ca30c"
        return "color: #898781"

    st.dataframe(sec_df.style.map(_score_color, subset=["得点"]),
                 hide_index=True)
    top = res["selected_sector"]
    chip = ui.chip(f"最高得点: {top['jp']} {top['score']}点",
                   "green" if top["score"] >= v6.SECTOR_PASS else "red")
    st.markdown(chip + f"  合格ライン {v6.SECTOR_PASS}点", unsafe_allow_html=True)

    with st.expander(f"{top['jp']} の得点内訳"):
        st.dataframe(pd.DataFrame([{
            "条件": f"{c['key']}. {c['label']}", "配点": c["points"],
            "判定": "○" if c["ok"] else "×", "実測": c["detail"],
        } for c in top["checks"]]), hide_index=True)

st.divider()

# -------------------------------------------------------------- 銘柄
st.subheader("② 銘柄選定")
if res["stock"] is None:
    st.info("セクターが合格しなかったため、銘柄選定は行いません。")
else:
    c_stock, c_etf = st.columns(2)

    with c_stock:
        st.markdown("**普通株の候補**")
        rows = res["stock"]["ranked"]
        if not rows:
            st.caption("採点できる普通株がありません。")
        else:
            st.dataframe(pd.DataFrame([{
                "銘柄": r["symbol"], "得点": r["score"],
                "現値": r["liquidity"]["price"],
                "観測ドル出来高": r["liquidity"]["dollar_volume"],
                "流動性": "○" if r["liquidity"]["ok"] else "×",
            } for r in sorted(rows, key=lambda x: -x["score"])]).style.format({
                "現値": "${:,.2f}", "観測ドル出来高": "${:,.0f}"}),
                hide_index=True)
            st.caption(f"合格ライン {v6.STOCK_PASS}点 / "
                       f"流動性は${v6.MIN_PRICE:.2f}以上かつ"
                       f"${v6.MIN_DOLLAR_VOLUME:,.0f}以上")

    with c_etf:
        st.markdown("**日次2倍ETF**")
        e = res["etf2x"]
        if not e or not e["row"]:
            st.caption("2倍ETFのデータがありません。")
        else:
            r = e["row"]
            st.metric(r["symbol"], f"{r['score']}点",
                      f"流動性 {'○' if r['liquidity']['ok'] else '×'}",
                      delta_color="off", border=True)
            st.caption(
                f"優先条件: セクター{v6.STRONG_SECTOR_FOR_2X}点以上"
                f"({'○' if e['strong_sector'] else '×'}) かつ "
                f"2倍ETF{v6.ETF2X_PASS}点以上"
                f"({'○' if r['score'] >= v6.ETF2X_PASS else '×'})")
            if e["passed"]:
                st.success("強セクター条件が成立 → 2倍ETFを優先", icon="⚡")

    if res["choice"]:
        st.markdown(
            ui.chip(f"選択: {res['choice']['symbol']}"
                    f"({res['choice']['kind']}・{res['choice']['score']}点)",
                    "blue") + f"  {res['choice']['why']}",
            unsafe_allow_html=True)

st.divider()

# ------------------------------------------------------------ 買いスコア
st.subheader("③ 買い判断")
if res["buy"] is None:
    st.info("銘柄が選ばれなかったため、買いスコアは計算しません。")
else:
    b = res["buy"]
    b1, b2 = st.columns([1, 2])
    b1.metric("買いスコア", f"{b['score']}点",
              f"合格ライン {v6.BUY_PASS}点",
              delta_color="off" if b["passed"] else "inverse", border=True)
    with b2:
        st.dataframe(pd.DataFrame([{
            "条件": f"{c['key']}. {c['label']}", "配点": c["points"],
            "判定": "○" if c["ok"] else "×", "実測": c["detail"],
        } for c in b["checks"]]), hide_index=True)

    hhmm = now_et.strftime("%H:%M")
    in_window = v6.BUY_WINDOW[0] <= hhmm <= v6.BUY_WINDOW[1]
    st.caption(f"買付時間 {v6.BUY_WINDOW[0]}〜{v6.BUY_WINDOW[1]} ET: "
               f"{'○ 時間内' if in_window else '× 時間外'}(現在 {hhmm} ET)")

st.divider()

# ------------------------------------------------------------ 退出判定
st.subheader("④ 保有中の退出判定")
st.caption("すでに保有している場合の退出条件を確認できます。"
           "買値と保有時間を入力してください(このツールは売買を行いません)。")

hold_sym = st.selectbox("保有している銘柄", sorted(frames),
                        index=sorted(frames).index(res["choice"]["symbol"])
                        if res["choice"] and res["choice"]["symbol"] in frames else 0)
e1, e2 = st.columns(2)
entry_price = e1.number_input("買値(ドル)", min_value=0.0, step=0.01, value=0.0,
                              format="%.2f")
held = e2.number_input("保有サンプル数(5分単位)", min_value=0, step=1, value=0,
                       help="18以上で最大保有時間による退出条件が成立します")

if entry_price > 0:
    hdf = frames[hold_sym]
    last = float(hdf["Close"].iloc[-1])
    ret = last / entry_price - 1
    x1, x2, x3 = st.columns(3)
    x1.metric("現在値", f"${last:,.2f}", f"{ret * 100:+.2f}%", border=True)
    x2.metric("損切りライン", f"${entry_price * (1 + v6.STOP_LOSS):,.2f}",
              "−0.60%", delta_color="off", border=True)
    x3.metric("利確ライン", f"${entry_price * (1 + v6.TAKE_PROFIT):,.2f}",
              "+1.00%", delta_color="off", border=True)

    sigs = v6.exit_signals(hdf, entry_price, int(held), now_et=now_et)
    if sigs:
        st.error("### 退出条件が成立しています", icon="🔴")
        for s in sigs:
            st.write(f"- **{s['key']}. {s['label']}** — {s['detail']}")
        st.caption("複数成立しても、ルール上の売りは1回だけです。")
    else:
        st.success("退出条件はまだ成立していません(HOLD)。", icon="🟢")
        st.caption(f"監視中: 損切り −0.60% / 利確 +1.00% / "
                   f"トレーリング(最高値+0.50%到達後 −0.35%) / "
                   f"最大保有18サンプル / 15:50 ET / 買いスコア30点以下")
else:
    st.info("買値を入力すると、損切り・利確・トレーリング・保有時間・"
            "強制退出・シグナル反転の6条件を判定します。")

st.divider()
st.caption("このページはRoute B V6のルールを再現した**判定表示のみ**の機能です。"
           "注文・口座接続・自動売買は一切行いません。"
           "投資判断はご自身の責任で行ってください。")

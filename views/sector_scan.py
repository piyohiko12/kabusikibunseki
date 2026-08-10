"""セクター選定ページ: いま強いセクターと、その中で強い銘柄を並べる。"""

import html

import pandas as pd
import streamlit as st

from lib import sector_scan, ui

TONE_COLOR = {"up": "#0ca30c", "down": "#d03b3b", "flat": "#7d8ca3"}
TONE_CHIP = {"up": "green", "down": "red", "flat": "gray"}

st.title("🔥 セクター・銘柄選定")
st.caption("SPYに対する相対力で、いま強いセクターと銘柄を並べます。"
           "『市場全体より強く動いているのはどこか』を見るための順位表です。")

st.warning(sector_scan.EDGE_NOTE, icon="⚠️")

ranking = sector_scan.sector_ranking()
if not ranking:
    st.error("セクターデータを取得できませんでした。"
             "ネットワーク接続を確認して、しばらく後に再試行してください。")
    st.stop()


def bar(score: float, tone: str) -> str:
    width = max(0, min(100, score))
    color = TONE_COLOR.get(tone, "#7d8ca3")
    return (f'<div style="background:#eceae6;border-radius:6px;height:9px;'
            f'width:100%;position:relative;">'
            f'<div style="background:{color};width:{width}%;height:100%;'
            f'border-radius:6px;"></div>'
            f'<div style="position:absolute;left:50%;top:-2px;width:1px;'
            f'height:13px;background:#9a978f;"></div></div>')


def pct(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.2f}%"


# ------------------------------------------------------- セクターランキング
st.subheader("🗺️ セクターの強さ(強い順)")
st.caption("中央の縦線が中立(50点)です。線より右なら市場平均より強い動き。")

for i, s in enumerate(ranking, 1):
    c_rank, c_name, c_bar, c_num = st.columns([0.4, 1.7, 2.4, 2.6])
    c_rank.markdown(f"**{i}**")
    c_name.markdown(
        f"**{s['jp']}**（{s['etf']}）<br>"
        + ui.chip(s["label"], TONE_CHIP.get(s["tone"], "gray")),
        unsafe_allow_html=True)
    c_bar.markdown(f"<div style='padding-top:12px'>{bar(s['score'], s['tone'])}"
                   f"<div style='font-size:0.72rem;color:#7d8ca3'>"
                   f"{s['score']:.1f}点</div></div>", unsafe_allow_html=True)
    c_num.markdown(
        f"<div style='font-size:0.8rem;padding-top:6px'>"
        f"当日 <b>{pct(s['ret1'])}</b>(対SPY {pct(s['rel1'])})<br>"
        f"5日 {pct(s['ret5'])}(対SPY {pct(s['rel5'])})・"
        f"20日 {pct(s['ret20'])}(対SPY {pct(s['rel20'])})<br>"
        # trend_note には "EMA20<EMA50" のような不等号が入る。
        # エスケープしないとHTMLタグとみなされて以降の文字ごと消える。
        f"<span style='color:#7d8ca3'>{html.escape(s['trend_note'])}・"
        f"出来高比 {s['vol_ratio']:.2f}倍</span></div>",
        unsafe_allow_html=True)
    st.markdown("")

with st.expander("スコアの内訳を見る"):
    breakdown = pd.DataFrame([
        {"セクター": s["jp"], **s["parts"], "合計": s["score"]}
        for s in ranking])
    st.dataframe(breakdown, hide_index=True, width="stretch")
    st.caption("相対力(5日)25点 / 相対力(20日)20点 / 当日20点 / "
               "トレンド20点 / 出来高15点。各要素は中立で満点の半分が入ります。")

st.divider()

# --------------------------------------------------------- セクター内の銘柄
st.subheader("🎯 セクター内の銘柄")

names = [f"{s['jp']}({s['etf']})" for s in ranking]
default_name = names[0] if names else None
picked = st.selectbox("セクターを選ぶ", names, index=0,
                      help="上のランキング順に並んでいます")
etf = ranking[names.index(picked)]["etf"] if picked else None

col_n, col_only = st.columns([1, 2])
with col_n:
    limit = st.slider("表示銘柄数", 3, 12, 6)

members = sector_scan.sector_members(etf, limit) if etf else []
if not members:
    st.info("このセクターの銘柄データを取得できませんでした。")
else:
    sector_row = next(s for s in ranking if s["etf"] == etf)
    if sector_row["score"] < 47:
        st.info(f"※ {sector_row['jp']}セクター自体は「{sector_row['label']}」です。"
                "以下はその中での相対順位なので、市場全体では弱い可能性があります。",
                icon="ℹ️")
    table = pd.DataFrame([{
        "ティッカー": m["ticker"],
        "強さ": m["label"],
        "スコア": m["score"],
        "株価": f"${m['price']:,.2f}",
        "当日": pct(m["ret1"]),
        "5日": pct(m["ret5"]),
        "20日": pct(m["ret20"]),
        "対セクター(5日)": pct(m["rel5"]),
        "出来高比": f"{m['vol_ratio']:.2f}倍",
        "トレンド": m["trend_note"],
    } for m in members])
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("「対セクター」は所属セクターETFに対する相対リターン。"
               "プラスならセクターの中でも強い動きです。")

    links = " ・ ".join(f"[{m['ticker']}](/?ticker={m['ticker']})"
                       for m in members)
    st.markdown(f"**銘柄分析で開く:** {links}")

st.divider()

# --------------------------------------------------- 上位セクターの注目銘柄
st.subheader("⭐ 上位3セクターの注目銘柄")
st.caption("強い順に3セクターを取り、それぞれの中で強い3銘柄を並べます。")

picks = sector_scan.top_picks(ranking, sectors=3, per_sector=3)
if not picks:
    st.info("銘柄データを取得できませんでした。")
else:
    pick_table = pd.DataFrame([{
        "セクター": p["sector_jp"],
        "セクター強さ": f"{p['sector_label']}({p['sector_score']:.0f})",
        "ティッカー": p["ticker"],
        "銘柄強さ": f"{p['label']}({p['score']:.0f})",
        "株価": f"${p['price']:,.2f}",
        "当日": pct(p["ret1"]),
        "5日": pct(p["ret5"]),
        "対セクター(5日)": pct(p["rel5"]),
    } for p in picks])
    st.dataframe(pick_table, hide_index=True, width="stretch")

st.caption("※ ここに出るのは「現時点で相対的に強い」銘柄であって、"
           "買い推奨ではありません。エントリーの可否は、銘柄分析ページの"
           "サポレジ・当日トレンド・寄付予想とあわせてご自身で判断してください。")

import streamlit as st

from lib import data_fetcher, moomoo_client, settings_store

st.set_page_config(
    page_title="米国株式分析ツール",
    page_icon="📈",
    layout="wide",
)

pages = [
    st.Page("views/trade_desk.py", title="今日のトレードデスク", icon="🧭"),
    st.Page("views/stock_analysis.py", title="銘柄分析", icon="📈", default=True),
    st.Page("views/compare.py", title="銘柄比較", icon="⚖️"),
    st.Page("views/market.py", title="市場概況", icon="🌐"),
    st.Page("views/signals.py", title="売買判定・アラート", icon="🎯"),
    st.Page("views/v6_signal.py", title="V6判定", icon="🤖"),
    st.Page("views/portfolio.py", title="ポートフォリオ管理", icon="💼"),
]
nav = st.navigation(pages)

# ------------------------------------------------- サイドバー: moomoo連携の設定
STATE_ICON = {"ok": "🟢", "off": "⚪"}

with st.sidebar:
    st.divider()
    state = moomoo_client.status()
    icon = STATE_ICON.get(state["state"], "🟠")
    with st.expander(f"{icon} moomooリアルタイム連携", expanded=False):
        saved = settings_store.load()
        enabled = st.toggle("有効にする", value=bool(saved.get("moomoo_enabled", False)),
                            help="moomoo OpenDから、遅延のない株価・板情報・歩み値を取得します。"
                                 "オフのときは従来どおりYahoo Finance(15〜20分遅延)のみを使います。")
        if enabled != bool(saved.get("moomoo_enabled", False)):
            settings_store.save(moomoo_enabled=enabled)
            moomoo_client.status.clear()
            st.rerun()

        if enabled:
            c1, c2 = st.columns([2, 1])
            host = c1.text_input("OpenDのホスト",
                                 value=saved.get("moomoo_host") or moomoo_client.DEFAULT_HOST)
            port = c2.number_input("ポート", min_value=1, max_value=65535, step=1,
                                   value=int(saved.get("moomoo_port")
                                             or moomoo_client.DEFAULT_PORT))
            if (host != (saved.get("moomoo_host") or moomoo_client.DEFAULT_HOST)
                    or int(port) != int(saved.get("moomoo_port")
                                        or moomoo_client.DEFAULT_PORT)):
                settings_store.save(moomoo_host=host, moomoo_port=int(port))
                moomoo_client.status.clear()
                st.rerun()

            chart_hist = st.toggle(
                "過去K線もmoomooを使用（枠保護あり）",
                value=bool(saved.get("moomoo_chart_history", False)),
                help=("オフ（既定）では過去K線をYahoo Financeから取得します。オンでも、"
                      "取得済み銘柄・ローカルキャッシュを優先し、新しい銘柄は予約枠を"
                      "超える場合に限って明示的な単一銘柄表示から取得します。"),
            )
            if chart_hist != bool(saved.get("moomoo_chart_history", False)):
                settings_store.save(moomoo_chart_history=chart_hist)
                data_fetcher.fetch_chart_history.clear()
                st.rerun()

            reserve = st.number_input(
                "履歴K線の予約枠",
                min_value=0,
                max_value=2000,
                step=5,
                value=int(saved.get("moomoo_history_reserve", 10)),
                help=("過去30日間に未取得の銘柄は、残り枠がこの数以下なら"
                      "moomooへ取りに行かずYahooへ切り替えます。取得済み銘柄の"
                      "再利用とローカルキャッシュは枠を追加消費しません。"),
            )
            if int(reserve) != int(saved.get("moomoo_history_reserve", 10)):
                settings_store.save(moomoo_history_reserve=int(reserve))
                st.rerun()
            if chart_hist:
                st.caption("取得済み銘柄とキャッシュは追加枠なしで再利用します。"
                           "未取得銘柄は残り枠が予約枠を超える場合だけ取得します。")
            else:
                st.caption("過去K線はYahoo Financeを使い、moomooの履歴枠を消費しません。")

            if state["state"] == "ok":
                st.success(state["message"], icon="🟢")
                quota = moomoo_client.history_quota()
                if quota and quota.get("remain") is not None:
                    st.caption(f"履歴K線の残りクォータ: {quota['remain']}"
                               f"(使用済み {quota.get('used', '—')} / "
                               f"予約 {int(reserve)})")
                    if int(quota["remain"]) <= int(reserve):
                        st.warning("予約枠に達したため、新しい銘柄の過去K線は"
                                   "Yahoo Financeへ自動切替します。", icon="🛡️")
            else:
                st.warning(state["message"], icon="⚠️")
                st.caption("OpenDを起動してログインすると使えます。"
                           "このツールは相場データの取得のみを行い、発注は一切しません。")
        else:
            st.caption("オフの間はYahoo Financeのみを使います(15〜20分遅延)。")

nav.run()

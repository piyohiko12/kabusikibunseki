import streamlit as st

from lib import moomoo_client, settings_store

st.set_page_config(
    page_title="米国株式分析ツール",
    page_icon="📈",
    layout="wide",
)

pages = [
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

            if state["state"] == "ok":
                st.success(state["message"], icon="🟢")
                quota = moomoo_client.history_quota()
                if quota and quota.get("remain") is not None:
                    st.caption(f"履歴K線の残りクォータ: {quota['remain']}"
                               f"(使用済み {quota.get('used', '—')})")
            else:
                st.warning(state["message"], icon="⚠️")
                st.caption("OpenDを起動してログインすると使えます。"
                           "このツールは相場データの取得のみを行い、発注は一切しません。")
        else:
            st.caption("オフの間はYahoo Financeのみを使います(15〜20分遅延)。")

nav.run()

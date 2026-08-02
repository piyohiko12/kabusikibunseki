import streamlit as st

st.set_page_config(
    page_title="米国株式分析ツール",
    page_icon="📈",
    layout="wide",
)

pages = [
    st.Page("views/stock_analysis.py", title="銘柄分析", icon="📈", default=True),
    st.Page("views/compare.py", title="銘柄比較", icon="⚖️"),
    st.Page("views/market.py", title="市場概況", icon="🌐"),
    st.Page("views/portfolio.py", title="ポートフォリオ管理", icon="💼"),
]
st.navigation(pages).run()

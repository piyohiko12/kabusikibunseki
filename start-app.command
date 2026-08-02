#!/bin/bash
# 米国株式分析ツール ランチャー(macOS / Linux)
# Finderでダブルクリックすると起動します。終了はターミナルで Ctrl+C。
cd "$(dirname "$0")" || exit 1

PY=""
for cand in python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3 が見つかりません。https://www.python.org/downloads/ からインストールするか、"
  echo "Homebrew を使っている場合は  brew install python@3.12  を実行してください。"
  read -r -p "Enterキーで閉じます..."
  exit 1
fi

# 初回起動時は仮想環境を自動作成
if [ ! -x ".venv/bin/python" ]; then
  echo "初回セットアップ中です(数分かかります)..."
  "$PY" -m venv .venv || { echo "仮想環境の作成に失敗しました"; read -r; exit 1; }
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt || {
    echo "依存パッケージのインストールに失敗しました"; read -r; exit 1; }
  echo "セットアップ完了。"
fi

echo "============================================"
echo " 米国株式分析ツール  -  http://localhost:8501"
echo " ブラウザが自動で開きます。"
echo " 終了するには Ctrl+C を押すか、このウィンドウを閉じてください。"
echo "============================================"

( sleep 3; open http://localhost:8501 >/dev/null 2>&1 || \
  xdg-open http://localhost:8501 >/dev/null 2>&1 ) &

.venv/bin/python -m streamlit run app.py --server.port 8501

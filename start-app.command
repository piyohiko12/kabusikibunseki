#!/bin/bash
# 米国株式分析ツール ランチャー(macOS / Linux)
# Finderでダブルクリックすると起動します。終了はターミナルで Ctrl+C。
cd "$(dirname "$0")" || exit 1

# Python 3.10以上を探す。macOS標準の /usr/bin/python3 は 3.9系のことがあり、
# それで仮想環境を作ると起動時に分かりにくいエラーになるため必ず版を確認する。
PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
  command -v "$cand" >/dev/null 2>&1 || continue
  if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10以上が見つかりません。"
  echo "  Homebrew:  brew install python@3.12"
  echo "  または https://www.python.org/downloads/ からインストールしてください。"
  echo "(macOS標準の python3 は 3.9系のため、このツールでは使えません)"
  echo "インストール後は新しいターミナル/Finderから開き直してください。"
  read -r -p "Enterキーで閉じます..."
  exit 1
fi

# 仮想環境が無い場合と、requirements.txt が変わった場合(git pull後など)にセットアップ
STAMP=".venv/requirements.installed.txt"
if [ ! -x ".venv/bin/python" ]; then
  echo "初回セットアップ中です(数分かかります)..."
  rm -rf .venv
  "$PY" -m venv .venv || { echo "仮想環境の作成に失敗しました"; read -r; exit 1; }
  NEED_INSTALL=1
elif ! cmp -s requirements.txt "$STAMP"; then
  echo "依存パッケージが更新されています。追加インストール中..."
  NEED_INSTALL=1
else
  NEED_INSTALL=0
fi

if [ "$NEED_INSTALL" = 1 ]; then
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt || {
    echo "依存パッケージのインストールに失敗しました"; read -r; exit 1; }
  cp requirements.txt "$STAMP"
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

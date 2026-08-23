# Claude共同編集ルール

- Git管理対象のコード・テスト・文書を変更した場合、テスト後かつ完了報告前に、
  macOS/Linuxは `.venv/bin/python scripts/add_edit_log.py --editor claude ...`、
  Windowsは `.venv\Scripts\python.exe scripts\add_edit_log.py --editor claude ...` を
  1回実行してください。
- 読み取り専用の調査だけではログを追加しません。
- 既存ログは編集・削除せず、訂正も新しいログとして追加してください。
- 変更ファイルと `docs/edit-log/entries/` の新規ログを同じコミットへ含めてください。
- 秘密情報、個人データ、絶対ローカルパスはログへ記録しないでください。
- 詳細な形式と例は `docs/edit-log/README.md` を正本として参照してください。

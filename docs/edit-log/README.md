# 共同編集ログの運用

Codex・Claude・手動編集で「誰が、何を、どこまで確認したか」を引き継ぐための
Git管理ログです。分析ツールのサイドバーにある「編集ログ」ページから閲覧できます。

## 正本と競合回避

- 正本は `docs/edit-log/entries/*.json` です。
- **1つのまとまった作業につき、新しいJSONを1ファイルだけ追加**します。
- 既存エントリは編集・削除しません。訂正時は `--supersedes` を指定して新規追加します。
- 単一のCHANGELOGへ全員で追記しないため、CodexとClaudeの同時作業でも
  Git競合が起きにくい構成です。
- 変更コードとログファイルは同じコミットへ含めます。

導入前の履歴を編集者名から推測して作ることはしません。過去の変更はGit履歴を参照し、
この機能を導入した時点から記録します。

## 変更時の手順

1. 作業開始前に最新ログとGit状態を確認します。
2. コード・テスト・文書を変更します。
3. 必要なテストや画面確認を実行します。
4. 完了報告の前に、次のCLIを1回実行します。
5. 変更ファイルと生成されたログを同じコミットへ含めます。

```bash
.venv/bin/python scripts/add_edit_log.py \
  --editor codex \
  --summary "編集ログ機能を追加" \
  --change "Git共有の追記専用ログを追加" \
  --change "分析ツール内に閲覧ページを追加" \
  --file lib/edit_log.py \
  --file views/edit_log.py \
  --verification-status passed \
  --validation "全ユニットテスト成功" \
  --validation "ブラウザで編集ログページを確認"
```

Claudeは `--editor claude`、人が直接編集した場合は `--editor manual` を使います。
テストを実行していない場合は、既定の `--verification-status not_run` のままにし、
未実行の確認を成功と記録しません。

Windows PowerShellでは、同じ引数をバッククォートで改行できます。

```powershell
.venv\Scripts\python.exe scripts\add_edit_log.py `
  --editor claude `
  --summary "変更の短い要約" `
  --change "実装した内容" `
  --file views/example.py
```

主な引数:

| 引数 | 内容 |
|---|---|
| `--summary` | 今回の変更を表す短い要約 |
| `--change` | 実装・修正内容。複数指定可 |
| `--file` | 変更したGit相対パス。複数指定可 |
| `--validation` | 実際に行った確認と結果 |
| `--follow-up` | 残課題や次の担当者への注意 |
| `--status` | `completed / in_progress / blocked` |
| `--verification-status` | `passed / partial / failed / not_run` |
| `--supersedes` | 訂正対象となる過去ログのID |

`base_commit` と作業ブランチはGitから自動取得します。変更自身を含むコミットIDは
ログ作成時点では決まっていないため、既定では空欄です。コミット後に過去ログを
書き換える必要はありません。

## 公開リポジトリで記録してはいけないもの

このリポジトリと編集ログはGitHubで共有されます。次の情報は書かないでください。

- APIキー、トークン、パスワード、Cookie、環境変数の生値
- 口座ID、保有銘柄、注文、資産額などの個人情報
- `/Users/...` や `C:/Users/...` などの端末固有の絶対パス
- 外部APIの応答全文、秘密情報を含み得る例外payload
- `data/` 配下の個人データやキャッシュ内容

`--file` はリポジトリ相対パスだけを受け付けます。ログ画面は読み取り専用で、
Yahoo・moomoo・OKXなどの市場APIや注文APIを呼びません。

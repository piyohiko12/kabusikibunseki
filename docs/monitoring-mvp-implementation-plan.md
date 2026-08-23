# 統合監視フロー 実装指示書（ChatGPT/Codex向け）

| 項目 | 内容 |
|---|---|
| 元になる判断書 | `monitoring-improvement-decision-packet.md`（`IMPLEMENT_GO_MVP`で採択） |
| この指示書の役割 | 採択後の実装手順。判断書の「Required before implementation」を作業へ落としたもの |
| 実装担当 | ChatGPT/Codex |
| 作成 | Claude（調査・設計レビュー担当） |
| 現状の確認日 | 2026-08-23 |

## 0. 最初に読む：この指示書の前提

判断書は `agent/moomoo-chart-enhancements` @ `ff2e7e1` 単独を前提にしていたが、
実際には **`main` が別系統で先行しており、2つの開発線が分岐している**。
MVPのスキーマ（`session_instance_id`）に焼き付く問題なので、
**フェーズAを終えるまでフェーズCへ進んではいけない。**

```text
df6c40b（分岐点）
   ├── main                          2コミット先行
   │     lib/sessions.py, lib/gap.py, lib/intraday.py, lib/sector_scan.py
   │     views/sector_scan.py, tests/test_sessions.py, test_sector_scan.py
   │     テスト53件
   └── agent/moomoo-chart-enhancements  15コミット先行（ff2e7e1）
         lib/realtime_signal.py, lib/session_intelligence.py,
         lib/market_intelligence.py, views/trade_desk.py, views/board.py ほか
         テスト458件
```

作業ブランチは `agent/moomoo-chart-enhancements` を使う。
最終的に `main` をこのブランチへ取り込む方向で統合する。

### 全フェーズ共通の禁止事項（判断書の安全契約）

- 取引context・注文API・`unlock_trade` を追加しない
- `request_history_kline` を監視・実績の経路で呼ばない（`get_cur_kline` のみ）
- 口座ID・保有数量・注文・APIキー・SNS本文・生DataFrameを保存しない
- DB書込み失敗が売買判定の計算結果を変えない
- 自動注文・自動監視登録をしない（登録は必ず利用者の明示操作）
- 既存の日足ルール・V6判定・legacy `alerts.json` の挙動を変えない

---

# フェーズA：統合と土台の修正（判断書 R1〜R5）

**このフェーズはMVPの前提。スキーマを書く前に必ず完了させること。**
見積り 1.5〜2.5人日。判断書の5〜7人日には含まれていない。

## A-1. main を agent ブランチへ取り込む

### 作業

```bash
git checkout agent/moomoo-chart-enhancements
git merge main
```

衝突は5ファイル。以下の方針で解消する。**調査済みなのでこのとおりに実施してよい。**

#### `app.py`

両方のページを残す。順序も以下のとおり。

```python
    st.Page("views/market.py", title="市場概況", icon="🌐"),
    st.Page("views/sector_scan.py", title="セクター・銘柄選定", icon="🔥"),
    st.Page("views/board.py", title="情報掲示板", icon="📋"),
```

#### `lib/moomoo_fetcher.py` — `fetch_history`

**agent側のクォータ保護機構を土台にし、main側の `prepost` を引数として足す。**
引数順は `(ticker, period, interval, prepost=False, allow_new_quota=False)`。

セッション決定は main 側のロジックを採用（`Session.ALL` は古いSDKに無いので `getattr` 退避）:

```python
        ktype = getattr(KLType, INTERVAL_MAP[interval])
        if code.startswith("US.") and interval in {"1m", "5m", "15m", "1h"}:
            session = (getattr(Session, "ALL", Session.RTH) if prepost
                       else Session.RTH)
        else:
            session = Session.NONE
```

**重要（見落としやすい）**: `prepost` の有無でセッション範囲が変わるので、
永続キャッシュのキーを分けないと時間外込みのデータが通常足のキャッシュを汚染する。

```python
    cache_interval = f"{interval}+ext" if prepost else interval
    cached = _read_history_cache(code, period, cache_interval)
    ...
            _write_history_cache(result, code, period, cache_interval, meta)
```

読み出し・書き込みの**両方**を `cache_interval` に揃えること。

#### `lib/data_fetcher.py`

- 衝突箇所1（`fetch_realtime_snapshot` の直後）:
  main側の古い `fetch_chart_history` は**削除**する。
  agent側の `_decision_number` 以下のセッション価格ブロックを残す。
  （agent側に高機能な `fetch_chart_history` が後方に別途あるため）
- 衝突箇所2（`_fetch_chart_history_cached` 内）:
  agent側を採用し、`prepost` を通す。

```python
        return fetch_history(ticker, period, interval, prepost), {
            "source": "Yahoo Finance", "code": ticker,
            "fetched_at": None, "cache_status": "fallback",
            "quota": None, "remain": None, "fallback_reason": None,
        }
    try:
        moomoo = moomoo_fetcher.fetch_history(
            ticker, period, interval, prepost=prepost,
            allow_new_quota=allow_new_quota)
        moomoo_meta = dict(moomoo.attrs.get("moomoo_meta", {}))
```

- `_fetch_chart_history_cached` と公開 `fetch_chart_history` の両方に
  `prepost: bool = False` を追加し、引き渡す。
  公開側の引数順は `(ticker, period, interval="1d", prepost=False, allow_new_quota=False)`。
- `fetch_history` にも `prepost` を追加（`yf.Ticker().history(..., prepost=prepost)`）。

#### `views/stock_analysis.py`

- import: 両方の名前をマージ。main側から `gap`, `intraday` を追加。
- タブ: agent側の6タブ構成に「当日・寄付」を挿入して7タブにする。

```python
(tab_today, tab_chart, tab_day, tab_news, tab_board, tab_orderflow,
 tab_derivatives) = st.tabs([
    "今日", "チャート", "当日・寄付", "ニュース", "情報一覧", "板・需給", "関連市場",
])
tab_tape = tab_flow = tab_orderflow
```

- 3つ目の衝突（1892行目付近、約200行）はチャート描画〜タブ本体。
  agent側のUI簡素化を優先しつつ、main側の以下は必ず残す:
  - `extended_hours` トグルとチャートへの受け渡し
  - `with tab_day:` ブロック全体（当日トレンド + 寄付の見通し）
  - デイトレプリセットの `"extended_hours": True`

#### `README.md`

両方の機能説明を残す。目次・ファイル構成に以下を漏らさず含める:
`lib/sessions.py`, `lib/gap.py`, `lib/intraday.py`, `lib/sector_scan.py`,
`lib/realtime_signal.py`, `lib/session_intelligence.py`,
`views/sector_scan.py`, `views/trade_desk.py`, `views/board.py`

### 受け入れ条件（A-1）

- [ ] `pytest tests/ -q` が **511件前後（458+53）すべて成功**
- [ ] `streamlit run app.py` で全ページが例外なく開く
- [ ] `git grep -n "<<<<<<<\|>>>>>>>"` が0件

---

## A-2. `lib/sessions.py` のDSTバグ修正（判断書 R3）

### 不具合（Claudeが実測で再現済み）

`lib/sessions.py:71` の `to_et()` が `ambiguous="NaT"` を使っている。
11月の冬時間切替で1時台が2回ある日、**naiveなタイムスタンプに対して `NaT` を返す**。

再現結果（naive index 16本、うち2本が11/01 01:00台）:

| 症状 | 実測 |
|---|---|
| `trading_day` が NaT | 16本中2本 |
| `latest_day_slice` が取りこぼす | 該当2本が消える |
| `intraday.vwap` が NaN | 16本中2本 |

moomoo由来のK線は `_normalise_history` が **naive index** を作るため、
夜間・24時間取引のバーで実際に踏む。

### 修正

```python
# lib/sessions.py の to_et() 内
    if idx.tz is None:
        return idx.tz_localize(ET, nonexistent="shift_forward",
                               ambiguous=True)   # "NaT" から変更
```

`ambiguous=True` は重複時間帯で夏時間側（最初の出現）を選ぶ。
セッション判定は壁時計時刻で行うためどちらを選んでも同じ区分になり、
NaTが発生しないぶん安全。

### 受け入れ条件（A-2）

- [ ] 3月・11月のDST切替日について、naive/aware両方のindexで
      `trading_day` が NaT を返さない回帰テストを追加
- [ ] 同テストで `vwap` にNaNが出ないことを確認
- [ ] 既存の `tests/test_sessions.py` が引き続き全件成功

---

## A-3. セッション定義を1本化する（判断書 R2・R4）

### 現状：定義が3系統ある

| 場所 | 名称 | 境界 | 休日・早期引け |
|---|---|---|---|
| `lib/session_intelligence.py` | premarket / regular / afterhours / overnight | 定数 | **あり**（NYSE休日、13:00早期引け） |
| `lib/sessions.py`（main由来） | pre / regular / after / overnight | `BOUNDS` dict | なし |
| `lib/realtime_signal.py` | 同上 | `time(4,0)` 等を**ハードコード** | 注入は可能 |

`session_instance_id = session種別 + scheduled_open_utc` は
**dedupe_key に焼き込まれる永続値**。定義が割れたまま実装すると後から直せない。

### 方針

**`lib/session_intelligence.py` を唯一の正とする。**（休日・早期引けを持つため）

1. セッション名を `premarket / regular / afterhours / overnight` に統一する。
   `lib/sessions.py` の `PRE="pre"`, `AFTER="after"` を
   `"premarket"`, `"afterhours"` へ変更する。
   `SESSION_LABELS` / `SESSION_SHORT` の日本語表記（プレマーケット等）は変えない。
2. `lib/sessions.py` の境界判定を `session_intelligence` の定数から導出する。
   独自の `BOUNDS` 定数は削除し、`session_intelligence` を参照する。
   `sessions.py` は「DataFrameのindexを一括分類する層」として残す
   （`session_intelligence` は「現在時刻を判定する層」で役割が違うため統合はしない）。
3. `lib/realtime_signal.py` の `_session_name` / `_session_bounds` の
   ハードコード値を `session_intelligence` 由来の注入値に置き換える。
   既に `session` 引数で `regular_close` を受け取れる設計なので、
   **フォールバック定数を削除**して未指定時はエラーにする（黙って16:00にしない）。
4. `session_instance_id` を生成する関数を `session_intelligence` に新設する。

```python
def session_instance_id(session_name: str, day: date,
                        extra_holidays=()) -> str | None:
    """session種別 + scheduled_open_utc の安定ID。休場日はNone。

    早期引け・臨時休場を反映した予定時刻から作る。
    全モジュールはこの関数だけを使い、独自に組み立ててはいけない。
    """
```

### 受け入れ条件（A-3）

- [ ] `git grep -n "time(9, 30)\|time(4, 0)\|time(16, 0)\|time(20, 0)"` の結果が
      `lib/session_intelligence.py` のみ
- [ ] 同一時刻に対し `sessions.classify` と
      `session_intelligence.detect_current_session` が同じセッション種別を返す
      （4セッション×通常日・早期引け日・休場日でテスト）
- [ ] 早期引け日（13:00引け）で `session_instance_id` が
      通常日と異なる値を返し、かつ同一日内では安定
- [ ] 既存511件が全件成功

---

## A-4. 購読枠の式を実装可能な形に確定する（判断書 R5）

判断書 §9 の式:

```text
subscription_reserve = max(20枠, 総枠の20%)
admission_capacity   = max(0, remain - subscription_reserve)
```

**問題**: `_query_subscription_quota` は `total_used / own_used / remain` しか返さない。
`総枠` が直接得られない。

### 修正

`総枠 = total_used + remain` と定義し、コードとコメントに明記する。

```python
def admission_capacity(quota: dict) -> int:
    """新規K_1M購読に使ってよい枠数。確認できないときは0（fail-closed）。"""
    total_used, remain = quota.get("total_used"), quota.get("remain")
    if total_used is None or remain is None:
        return 0
    total = total_used + remain
    reserve = max(20, int(total * 0.2))
    return max(0, remain - reserve)
```

`_query_subscription_quota` は既に「応答が曖昧なら必ずNone+理由」を返す
fail-closed 実装になっている（`lib/moomoo_client.py:299`）。この性質を壊さないこと。

### 受け入れ条件（A-4）

- [ ] 総枠・残数のどちらかが取得できない場合に `capacity_wait` になるテスト
- [ ] 予約枠を超えて購読しないテスト（境界値 remain=reserve、remain=reserve+1）

---

## A-5. フェーズA完了時のコミット

```bash
git commit -m "Unify session layer and integrate main before monitoring MVP"
```

**この時点のコミットハッシュを記録し、判断書の `base_commit` を差し替える。**

---

# フェーズB：文書の修正（判断書 R6）

## B-1. primary endpoint の定義に限定条件を明記

判断書 §12.3 の primary endpoint を以下に差し替える。

> primary endpoint: 各sessionの15分 gross BBO markout 平均。
> ただし **判定時点でセッション残り時間が15分以上あった actionable episode に
> 限定した条件付き量**である。`CENSORED` は構造的欠測として分母から外すため、
> 引け際15分以内に発生したシグナルはこの指標に含まれない。
> 引け際の値動きは中盤と性質が異なるため、この限定を外した主張はしない。

理由: `coverage_15m = observed_15m / (executable - censored_15m)` は
分母からCENSOREDを引くが、CENSOREDは**ランダム欠測ではない**（必ず引け際に偏る）。
除外自体は正しいが、何を測った数字なのかを定義文に書く必要がある。

## B-2. 必要標本数の現実的な見積りを追記

判断書 §12.3 の `max(300, power計算結果)` について、300を計画の基準にしないよう注記する。

MDE=10bps、α=0.0125（Bonferroni 4session）、power=80%、
二方向cluster（design effect 3を仮定）でのClaudeの試算:

| 15分markoutのσ | 必要標本（DE=3） | 1日5件observedでの所要 |
|---:|---:|---:|
| 40bps | 約535件 | 約107取引日（約5ヶ月） |
| 60bps | 約1,204件 | 約241取引日（約11ヶ月） |
| 80bps | 約2,141件 | 約428取引日（約20ヶ月） |

判断書 §12.3 が「4〜8週間のshadow運用は運用品質と完全版へ進むかの判断期間であり、
収益性や勝率改善を確定する期間ではない」と書いている位置づけは正しい。
`EVIDENCE_SIGNAL_QUALITY_POSITIVE` は複数四半期規模の目標として扱う。

---

# フェーズC：MVP実装

判断書 §15 の推奨実装順に従う。**1タスク＝1セッション**で進め、
タスク間で長い会話履歴を持ち越さない（判断書 §14 のトークン方針）。

## 固定仕様（判断書で採択済み・変更不可）

| 項目 | 値 |
|---|---|
| 監視モード | in-app のみ |
| 目的 | 新規購入（entry）のみ。保有監視は完全版へ |
| 条件engine | `realtime_signal` の `BUY_READY` のみ |
| 候補 | セッション別top5、主表示3件 |
| K_1M製品上限 | 5銘柄 + SPY（実枠は `admission_capacity` で縮小） |
| 監視間隔 | 15秒 / quote鮮度上限 15秒 |
| outcome | 15分後 + 当該session close |
| 保存 | `data/monitoring.sqlite3`（WAL） |
| 自動注文 | なし |

---

## C-1. SQLiteスキーマと状態遷移reducer

新規 `lib/monitoring_store.py`（DB I/O）と `lib/monitoring_state.py`（純粋関数）に分ける。
**reducerは副作用なしの純粋関数にする**（テストしやすさとDB失敗時の分離のため）。

### テーブル（判断書 §8）

`candidates` / `monitor_cases` / `condition_config_versions` /
`monitor_state` / `monitor_events` / `outcomes` / `monitor_heartbeat`

### 必須契約

- WAL、`PRAGMA foreign_keys=ON`、busy timeout、短いtransaction
- active caseの部分unique index:
  `(symbol, session_instance_id, purpose, condition_engine) WHERE status='active'`
- `origin_candidate_id` / `candidate_policy_hash` は **nullableな由来情報**。
  一意性の判定に使わない
- `dedupe_key TEXT NOT NULL UNIQUE`。種別ごとの生成規則:
  - barイベント: `case_id + condition_config_version_id + session_instance_id + event_type + source_bar_time`
  - gap: `case_id + gap_start + gap_end`
  - 設定変更: `old_version_id + new_version_id`
  - 停止: `request_id`
- `outcomes`: `UNIQUE(trigger_monitor_event_id, horizon_key, cost_model_version)`
- outcome状態は `pending → observed / late / censored / data_missing` を
  transaction内でCAS/upsert
- **すべてUTC保存**。ET/JSTは表示時のみ変換

### 状態遷移reducer（判断書 §7.1）

`last_observed_state` と `episode_open` を**別のフィールドとして持つ**。

| 遷移 | 動作 |
|---|---|
| 初期 or 確認済みFALSE → TRUE | 新しいepisodeを開始 |
| TRUE → TRUE | 同じepisodeを継続。新着を増やさない |
| open中の TRUE → UNKNOWN | episodeを閉じない。判定不能として保留 |
| open中の UNKNOWN → TRUE | 同じepisodeを再開。**新規発火にしない** |
| UNKNOWN → FALSE / TRUE → FALSE | 確認済み解除。episodeを閉じて再arm |
| session終了 | `session_expired` でepisodeとcaseを終了 |
| 長時間gap | episodeを解除せず `monitoring_gap` |
| 設定変更 | 旧版のままepisode終了、新版で別versionとして開始 |

判定の投影:

| 監査状態 | realtime判定 |
|---|---|
| `TRUE` | `BUY_READY` かつ `actionable=True` |
| `FALSE` | データ正常時の `BUY_SETUP` / `NEUTRAL` |
| `UNKNOWN` | `DATA_WAIT`、安全条件待ち、stale、capacity不足、監視gap |

### DB書込み失敗時（判断書 §7.1）

- 計算済み判定は**書き換えない**
- ただし「新しく成立」として**確定通知しない**。
  「条件は成立・監査保存に失敗」と表示する
- outcomeを作らない。同じdedupe keyで保存を再試行する

### 受け入れ条件（C-1）

- [ ] 同一確定足を10回評価してもepisodeが1件、eventも1件
- [ ] `TRUE → UNKNOWN → TRUE` で新規発火が増えない
- [ ] 複数プロセスから同時に同じdedupe keyを書いても1行
- [ ] DB書込みを強制失敗させても判定結果が変わらず、通知が「保存に失敗」になる
- [ ] reducerが純粋関数（DBなしでテストできる）

---

## C-2. realtime BUY_READY から監査eventを記録

`lib/realtime_signal.py` の**判定ロジックは変更しない**。
出力を C-1 のreducerへ流す配線だけを追加する。

### 現状の問題（Claudeが確認済み）

`views/stock_analysis.py:502` で
`realtime_memories = st.session_state.setdefault("realtime_signal_memories", {})`
としており、**memory（2本確認・cooldown）がブラウザーsessionにしかない**。
再読込・タブを閉じると消える。これがSQLite永続化の主目的。

### 作業

1. 監視case登録時に `condition_config_versions` へ不変snapshotを保存する。
   保存内容: `schema_version`, engine version,
   正規化済み `DEFAULT_CONFIG` + 明示override, `actionable_sessions`
2. `realtime_signal.evaluate_realtime_signal` の戻り値 `memory` を
   `monitor_state` テーブルへ永続化し、次回はそこから読む
3. 判定結果を監査三状態へ投影し、reducer経由で `monitor_events` へ記録

### 受け入れ条件（C-2）

- [ ] ブラウザーを再読込しても2本確認とcooldownが継続する
- [ ] 複数タブから同じ銘柄を開いてもeventが重複しない
- [ ] legacy `alerts.json` の挙動が一切変わらない

---

## C-3. 成立履歴UIとcaseのライフサイクル

`views/signals.py` を拡張する（新規画面は作らない）。

- 表示状態: `監視中 / 新しく成立 / 成立中 / 解除 / 判断できない / 監視外`
- case操作: 登録・停止・期限切れ
- 有効期限: 登録した当該セッションの予定終了時刻。**自動延長しない**
- 満員時: 既存caseを追い出さず `capacity_wait`
- 停止後も、発火済みoutcomeは予定horizonが完了・欠損確定するまで観測を続ける

### UX Gate（判断書 §13.2）

- [ ] `購入前チェックへ / 監視に追加 / 通知・実績を見る` が一続き
- [ ] `新規成立 / 成立中 / 判断できない / capacity_wait` を日本語で区別
- [ ] 「実際の注文・約定・損益ではありません」を**常時表示**
- [ ] デスクトップと **390px幅** で主操作を確認
- [ ] 表が横スクロールしても主結論と停止操作へ到達できる

---

## C-4. セッション別top5候補 + 立会top-movers adapter

### 候補policy（判断書 §7.2・固定）

| 項目 | 値 |
|---|---|
| 対象 | データ源が返した米国株・ETF cohort（「米国市場全銘柄」と表現しない） |
| 方向 | 上昇率上位のみ |
| 価格 | 現セッション価格 $1〜$1,000、停止中でない |
| 鮮度 | rank取得batch ageが60秒以内 |
| 流動性 | session出来高2万株以上 **かつ** 価格×session出来高が100万ドル以上 |
| 除外 | オプション・先物・暗号資産 |
| 並び順 | source rank → 売買代金降順 → ticker昇順 |
| 公開数 | top5（主表示3件） |

- 候補発見は rank + snapshot のみ。**K_1M購読を使わない**
- 時間外snapshotは**候補表示専用**。発火・実績の根拠にしない
- candidate一意キー: `(symbol, session_instance_id, candidate_policy_hash)`
  60秒更新ごとに新IDを作らない
- upsertする列: `first_published_at / last_seen_at / latest_rank /
  best_rank / latest_price / latest_volume` のみ
- 毎分のrawランキング履歴は保存しない
- policy hash に universe・閾値・sort・source を含める

### 新規作業

プレ・アフター・夜間は現行session rankを再利用できるが、
**立会rankは現行adapterの対象外**。同じ出力契約を持つ
立会top-movers adapterを新規に作る。

---

## C-5. K_1M一括調停（最大5銘柄 + SPY）

- process全体のunique symbol unionを**一つの安定tuple**としてbatch取得
- 同一銘柄の複数caseは1取得を共有
- 既存の60/120秒lease制約（`_CURRENT_KLINE_MIN_LEASE_SECONDS`）を維持
- 既購読銘柄は追加枠に数えないが、process全体の利用中として管理
- 枠不足は `capacity_wait`。**他画面の購読を自動解除しない**
- SPYも1銘柄として数える
- A-4 の `admission_capacity` を使う

---

## C-6. 15分・session close の outcome

### 観測窓（判断書 §7.4・厳守）

```text
source_bar_time      = 確定1分足の開始時刻。dedupe/証拠用。entry起点にしない
decision_observed_at = 確定足の終了後、BUY_READYを初めて計算・記録できた時刻
entry Ask            = decision_observed_at から60秒以内に初めて観測した
                       fresh な同session Ask
entry_observed_at    = そのAskを実際に観測した時刻
15分後 Bid           = entry_observed_at+15分 から60秒以内に初めて観測した
                       fresh な同session Bid
session close        = 予定終了時刻の60秒前〜終了時刻に最後に観測した
                       fresh な同session Bid
```

**先読み禁止**: `decision_observed_at >= source_bar_time + 1分` を必ず検証する。
確定足を知る前のAskをentryに選んではいけない。

15分は**シグナル足開始からではなく、entry Askを観測してから**の仮想保有時間。

### 欠損分類（補完禁止）

| 状況 | 分類 |
|---|---|
| entry Askが窓内にない | `UNEXECUTABLE` |
| 15分後Bidが窓を過ぎて得られた | `LATE`（主評価から分離） |
| entry+15分 ≧ 予定session終了時刻 | `CENSORED`（構造的欠測） |
| session closeの同session Bidがない | `DATA_MISSING` |
| 利用者停止・通信gap・アプリ停止 | `DATA_MISSING`（`CENSORED`にしない） |

**next session価格・Close・Last・midpointへ補完してはいけない。**

### 指標

```text
gross_bbo_markout = exit_bid / entry_ask - 1
net_markout       = gross_bbo_markout - additional_cost_bps / 10_000
```

- primary は `gross_bbo_markout`
- `net_markout` は **0 / 5 / 10bps の3列すべてを常に表示**する
  （都合のよい列だけ出さない）
- cost gridと式を `cost_model_version` へ保存
- SPY超過markoutは、entry/exit endpointから60秒以内に同sessionのSPY Ask/Bidを
  取得できた場合のみ。揃わなければ `unavailable`。**0へ補完しない**

### 表示

「実際の注文・約定・損益ではありません」を常に併記する。

---

## C-7. テストと最終確認

### 技術Release Gate（判断書 §13.1 全項目）

- [ ] 既存511件（458+53）と新規テストがすべて成功
- [ ] 同一確定足を繰り返し評価してもepisodeが1件
- [ ] `TRUE / FALSE / UNKNOWN` と open episode を別管理
- [ ] 複数タブ・複数fragmentから同一eventを重複保存しない
- [ ] 同一trigger・horizonのoutcomeを二重保存・二重集計しない
- [ ] 手動登録と候補登録をまたいで同一銘柄・sessionのactive caseを重複作成しない
- [ ] entry/exit窓と同session制約を固定し、Close/Lastへ補完しない
- [ ] `source_bar_time` と `decision_observed_at` を分離し、確定前Askを採用しない
- [ ] プレ・立会・アフター・夜間とovernightの日付跨ぎを分離
- [ ] wrong-session、stale、停止銘柄でactionable eventを作らない
- [ ] eligibleからoutcomeまでの全分母・欠損理由をreplayで100%保存
- [ ] realtime engine / candidate policy / cost model / condition config の
      versionを保存
- [ ] DB書込み失敗が売買判定を変更しない
- [ ] DB保存失敗時に監査済み新着通知・outcomeを作らず再試行する
- [ ] 停止gap中の発火を再演しない
- [ ] process全体の購読上限と予約枠を超えない
- [ ] 15秒poll / 15秒quote freshness / 90秒遅延SLO を
      設定版・UI・テストで一致させる
- [ ] candidate policyの数値閾値・ID寿命・除外理由をreplayで再現できる
- [ ] 口座・注文・秘密情報を保存しない
- [ ] 過去K線API・取引context・注文APIを追加しない

### フェーズA由来の追加Gate

- [ ] 早期引け日（13:00引け）と臨時休場日で、`session_instance_id`・
      `CENSORED`判定・session close観測窓が正しい
- [ ] DST切替の両端で、naive/aware両方のindexから `trading_day` が NaTを返さない
- [ ] 全モジュールが同一時刻に対して同一の session種別と
      `scheduled_open_utc` を返す

### 実地確認

4セッション（プレ・立会・アフター・夜間）と390pxブラウザーで確認する。

---

# 作業順まとめ

| 順 | 作業 | 目安 | 前提 |
|---|---|---:|---|
| A-1 | main を agent へ取り込む | 0.5〜1.0日 | — |
| A-2 | sessions.py DSTバグ修正 | 0.25日 | A-1 |
| A-3 | セッション定義の1本化 | 0.75〜1.25日 | A-2 |
| A-4 | 購読枠の式を確定 | 0.25日 | A-1 |
| B-1/B-2 | 判断書の文言修正 | 0.25日 | — |
| C-1 | SQLite + reducer | 1.25〜1.75日 | **A完了必須** |
| C-2 | event記録の配線 | 1.0〜1.5日 | C-1 |
| C-3 | 成立履歴UIとcase | 0.75〜1.0日 | C-2 |
| C-4 | 候補top5 + 立会adapter | 1.0〜1.5日 | A-3 |
| C-5 | K_1M一括調停 | 0.75〜1.0日 | A-4, C-1 |
| C-6 | outcome台帳 | 1.5〜2.5日 | C-2, C-5 |
| C-7 | テスト・実地確認 | 1.5〜2.75日 | 全部 |

合計 **9.75〜14.75人日**（フェーズA 1.75〜2.75 + MVP 8.0〜12.0）。
判断書の「MVP 5〜7人日」はフェーズAを含まず、
C-4〜C-6を最小構成にした場合の下限値である。

---

# ChatGPTへの運用上のお願い

判断書 §14 のトークン方針に従う。

- **この指示書を正本とし、長い会話履歴を次のタスクへ再送しない**
- 1タスク＝1セッション。タスクID（A-1、C-3 など）で区切る
- `rg` で対象を特定してから、必要な範囲だけ読む
- 開発中は対象テストのみ、フェーズ終了時に全テスト
- 成功ログは要約し、失敗時だけ該当traceを読む
- 高推論はDB・セッション・金融判定・最終監査に限定する
- **ClaudeとChatGPTが同じファイルを同時に編集しない。**
  この指示書に基づく実装はChatGPTが担当し、
  Claudeはレビューと検証を担当する

## 各タスク完了時に報告してほしいこと

1. タスクID
2. 変更ファイル
3. テスト結果（件数と成否）
4. 判断書の固定仕様から外れた点があれば、その理由
5. 次タスクへの申し送り

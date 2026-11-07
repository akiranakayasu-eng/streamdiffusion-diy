# samples

試験用の入力動画と、設定を変えたときの比較。動画（*.mp4）はリポジトリに入れない。

## t-index などの比較（2026-11-07、MacBook Air M1・MPS・float16）

入力: 仮の動画 `synthetic.mp4`（512×512、24fps、色の円 3 つが回る）。
共通: `--model stabilityai/sd-turbo`、`--prompt "watercolor painting of colorful balloons"`、`--max-frames 24`、`--no-window`。
比較画像 `compare_tindex.png` は各設定の出力 12 フレーム目（左端が入力）。

| 設定 | 引数 | 平均 fps | 出力の遅れ（フレーム） | 見た目 |
|---|---|---|---|---|
| A | `--t-index 32,45`（既定） | 1.5 | 1〜4 でばらつく | ほぼ入力のまま |
| B | `--t-index 20,40` | 1.4 | 1〜3 でばらつく | 円に小さな紐が付く程度 |
| C | `--t-index 10,30` | 1.4 | 1〜4 でばらつく | 光沢と紐のある風船になる |
| D | `--t-index 20 --size 256` | 6.6 | 0〜1 | 風船らしいが粗い |
| E | `--t-index 10,30 --guidance 1.5 --cfg-type self` | 1.4 | 0〜3 でばらつく | 輪郭線の濃いイラスト調の風船 |
| F | `--t-index 32,45 --no-ssf` | 0.9 | 常に 1 | A と同じ |

- A〜E は類似度フィルタ（SSF）が入っていた。背景が大半の動画では前の出力を使い回すため、右が止まって遅れが 1〜4 フレームにばらつく。
- SSF を切った F は遅れが Stream Batch の n−1＝1 フレームに収まった。
- SSF による省略がある分、A〜E の fps は F より高く出ている。
- この結果から、動画ファイル入力では SSF を既定で切るようにした（`--ssf` で入れる）。
- 遅れは、出力の赤い円の位置に最も近い入力フレームを探して数えた。

## 機種ごとの速さ（2026-11-07）

| 実行先 | 設定 | fps | メモ |
|---|---|---|---|
| MacBook Air M1（MPS） | 512・strong（10,30） | 約 0.9 | SSF なし |
| MacBook Air M1（MPS） | 256・1 段（20） | 6.6 | |
| MacBook Pro M4・16GB（MPS） | 512・strong | 1.7 | |
| MacBook Pro M4・16GB（MPS） | 384・strong | 2.9 | |
| MacBook Pro M4・16GB（MPS） | 256・1 段 | 7.9 | |
| Modal（us-east）← 日本の Pro、1 枚ずつ往復 | 512・strong | 2.3 | 往復の中央値 428 ms |
| Modal（us-east）← 日本の Pro、1 枚ずつ往復 | 256・strong | 2.5 | 往復の中央値 385 ms |
| Modal（us-east）← 日本の Pro、1 枚ずつ往復 | 256・weak | 2.7 | 往復の中央値 362 ms |

- Modal 版は大きさを変えても fps がほぼ同じ。遅さは GPU ではなく日本と米国東部の往復（約 0.4 秒）で決まっていた。
- そこでページを「返事を待たずに最大 3 枚まで先に送る」作りに変えた（apps/web/index.html の MAX_INFLIGHT）。
- Modal の接続から使えるまで 13.6 秒（コンテナ起動とモデルの読み込み）。
- これ以降、Modal 版は使うたびに Modal の Dict `sdstream-stats` に記録が残る。

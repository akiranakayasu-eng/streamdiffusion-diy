# streamdiffusion-diy

StreamDiffusion（リアルタイム画像生成のパイプライン）を、論文の仕組みから自前で組み直した最小実装です。
**試作・未検証**: 実モデルでの動作はまだ確かめていません。

## 入っている仕組み

- **Stream Batch**: ノイズ除去の n 段をフレームごとにずらしてバッチにし、UNet 1 回で 1 フレーム出す
- **RCFG**（Residual Classifier-Free Guidance）: self / onetime / full
- **確率的類似度フィルタ**（SSF）: 入力が前とほぼ同じなら計算を飛ばす
- **事前計算**: プロンプト埋め込み・ノイズ・各段の係数
- **入出力のスレッド分け**、**Tiny AutoEncoder（TAESD）**
- 実行先は CUDA → Apple Silicon（MPS）→ CPU の順に自動で選ぶ

## 版（1 つのリポで共通の本体を使う）

| 版 | 計算する所 | TouchDesigner への出力 | 状態 |
|---|---|---|---|
| Mac | Apple Silicon の GPU（MPS） | Syphon（予定） | 試作 |
| Windows | NVIDIA の GPU（CUDA、TensorRT は予定） | Spout（予定） | 試作 |
| Modal | クラウドの GPU（L4、空きが無ければ A10G）。カメラと表示はスマホや PC のブラウザ、映像は WebSocket | ブラウザに表示 | 試作 |

`sdstream/` が共通の本体、`apps/` が版ごとの入口です。入力はカメラか動画ファイル（`--input`）、結果は画面か動画ファイル（`--output`、左が入力・右が生成）。

## 使い方

```bash
pip install -r requirements.txt
python test_dummy.py                       # モデル無しで仕組みを確かめる
python apps/local_stream.py --prompt "oil painting, portrait"     # カメラ。q で終了
python apps/local_stream.py --input samples/test.mp4 --output out.mp4   # 動画ファイル
```

既定のモデルは `stabilityai/sd-turbo` と `madebyollin/taesd`。SD1.5 系 + LCM-LoRA なら
`--model <SD1.5 系> --lcm-lora latent-consistency/lcm-lora-sdv1-5`。
モデルの重みはこのリポジトリに含みません。各モデルの利用規約に従ってください。

## Modal 版（スマホのブラウザで使う）

```bash
pip install modal && modal setup                                  # 1 回だけ
modal secret create sdstream-key SDSTREAM_KEY=<長い合言葉>          # 1 回だけ
modal deploy apps/modal_app.py                                     # 表示された URL を使う
```

表示された URL に `?key=<合言葉>` を付けてスマホで開き、カメラか動画ファイルを選んで「開始」。
課金の事故を防ぐため、1 回の接続は最長 10 分、30 秒映像が来なければ停止、画面を離れても停止、
ページは自動で再接続しません。使われなくなったコンテナは 60 秒で止まります（最大 1 台）。
完全に止めるときは `modal app stop streamdiffusion-diy`。

## 元にした論文

仕組みは次の論文によります。コードは公式実装を写さず、論文から書き起こしました。

- Akio Kodaira ほか, "StreamDiffusion: A Pipeline-level Solution for Real-Time Interactive Generation", *ICCV 2025*, pp. 12371–12380. DOI: [10.1109/ICCV51701.2025.01150](https://doi.org/10.1109/ICCV51701.2025.01150)
  - 論文（CVF Open Access）: https://openaccess.thecvf.com/content/ICCV2025/html/Kodaira_StreamDiffusion_A_Pipeline-level_Solution_for_Real-Time_Interactive_Generation_ICCV_2025_paper.html
  - プレプリント: [arXiv:2312.12491](https://arxiv.org/abs/2312.12491)（2023 年 12 月初出、2025 年 7 月改訂）
- 公式実装: https://github.com/cumulo-autumn/StreamDiffusion (Apache-2.0)

論文では RTX 4090 1 枚で最大 91.07 fps（diffusers の AutoPipeline の 59.56 倍以上）と報告されています。

```bibtex
@inproceedings{kodaira2025streamdiffusion,
  title     = {StreamDiffusion: A Pipeline-level Solution for Real-Time Interactive Generation},
  author    = {Kodaira, Akio and others},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {12371--12380},
  year      = {2025},
  doi       = {10.1109/ICCV51701.2025.01150}
}
```

## ライセンス

Apache-2.0

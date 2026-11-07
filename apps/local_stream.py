"""カメラまたは動画ファイル → 自作 StreamDiffusion → 画面・動画ファイル（img2img）。

入出力の待ち行列: カメラのときは読み込みを別スレッドにし、古いフレームは
捨てて遅れをためない。動画ファイルのときは全フレームを順に処理する。

  pip install -r requirements.txt
  python apps/local_stream.py --prompt "oil painting, portrait"            # カメラ
  python apps/local_stream.py --input samples/test.mp4 --output out.mp4    # 動画
  python apps/local_stream.py --input samples/test.mp4 --output out.mp4 --no-window
  q キーで終わる。出力の動画は左が入力、右が生成。
"""
from __future__ import annotations

import argparse
import os
import sys
import queue
import threading
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sdstream.models import SDParts
from sdstream.similarity_filter import SimilarityFilter
from sdstream.stream import StreamConfig, StreamDiffusionCore


def square(frame, size: int):
    """中央を正方形に切り出して size×size にする。"""
    h, w = frame.shape[:2]
    s = min(h, w)
    frame = frame[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    return cv2.resize(frame, (size, size))


def capture_loop(cap, q_in: queue.Queue, stop: threading.Event, size: int):
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            continue
        frame = square(frame, size)
        if q_in.full():               # 古いフレームは捨てる（遅れをためない）
            try:
                q_in.get_nowait()
            except queue.Empty:
                pass
        q_in.put(frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a portrait, oil painting, masterpiece")
    ap.add_argument("--model", default="stabilityai/sd-turbo")
    ap.add_argument("--lcm-lora", default=None)
    ap.add_argument("--t-index", default="32,45")
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--cfg-type", default="self")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--input", default=None, help="動画ファイル（省略時はカメラ）")
    ap.add_argument("--output", default=None, help="結果を書き出す動画ファイル（.mp4）")
    ap.add_argument("--max-frames", type=int, default=0, help="0 なら最後まで")
    ap.add_argument("--no-window", action="store_true", help="画面に出さない")
    ap.add_argument("--no-ssf", action="store_true")
    args = ap.parse_args()

    parts = SDParts(args.model, lcm_lora=args.lcm_lora)
    cfg = StreamConfig(t_index_list=[int(i) for i in args.t_index.split(",")],
                       guidance_scale=args.guidance, cfg_type=args.cfg_type,
                       latent_shape=(4, args.size // 8, args.size // 8))
    core = StreamDiffusionCore(parts.unet_fn, cfg, parts.device, parts.dtype)
    core.set_prompt(parts.embed(args.prompt), parts.embed(""))
    ssf = None if args.no_ssf else SimilarityFilter()

    stop = threading.Event()
    if args.input:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            raise SystemExit(f"動画を開けません: {args.input}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        def next_frame():
            ok, f = cap.read()
            return square(f, args.size) if ok else None
    else:
        cap = cv2.VideoCapture(args.camera)
        src_fps = 24.0
        q_in: queue.Queue = queue.Queue(maxsize=1)
        threading.Thread(target=capture_loop, args=(cap, q_in, stop, args.size),
                         daemon=True).start()

        def next_frame():
            return q_in.get()

    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (args.size * 2, args.size))

    last = None
    t0, frames = time.time(), 0
    while True:
        frame = next_frame()
        if frame is None:          # 動画の終わり
            break
        img = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1)[None] / 255.0
        if ssf is not None and last is not None and ssf.should_skip(img):
            out = last
        else:
            lat = parts.encode(img)
            out_lat = core.step(lat)
            rgb = parts.decode(out_lat)[0].permute(1, 2, 0).float().cpu().numpy()
            out = (rgb[:, :, ::-1] * 255).astype(np.uint8)
            last = out
        frames += 1
        if frames % 30 == 0:
            fps = frames / (time.time() - t0)
            print(f"{fps:.1f} fps")
        pair = np.hstack([frame, out])
        if writer is not None:
            writer.write(pair)
        if not args.no_window:
            cv2.imshow("sdstream", pair)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.max_frames and frames >= args.max_frames:
            break
    if frames:
        print(f"{frames} フレーム、平均 {frames / (time.time() - t0):.1f} fps")
    stop.set()
    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

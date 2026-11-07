"""カメラ → 自作 StreamDiffusion → 画面、の最小デモ（img2img）。

入出力の待ち行列: カメラ読み込みと表示を別スレッドにして、
GPU（UNet と VAE）を止めずに回す。

  pip install torch diffusers transformers accelerate opencv-python
  python apps/local_camera.py --prompt "oil painting, portrait"
  q キーで終わる。
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


def capture_loop(cap, q_in: queue.Queue, stop: threading.Event, size: int):
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        s = min(h, w)
        frame = frame[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
        frame = cv2.resize(frame, (size, size))
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
    ap.add_argument("--no-ssf", action="store_true")
    args = ap.parse_args()

    parts = SDParts(args.model, lcm_lora=args.lcm_lora)
    cfg = StreamConfig(t_index_list=[int(i) for i in args.t_index.split(",")],
                       guidance_scale=args.guidance, cfg_type=args.cfg_type,
                       latent_shape=(4, args.size // 8, args.size // 8))
    core = StreamDiffusionCore(parts.unet_fn, cfg, parts.device, parts.dtype)
    core.set_prompt(parts.embed(args.prompt), parts.embed(""))
    ssf = None if args.no_ssf else SimilarityFilter()

    cap = cv2.VideoCapture(args.camera)
    q_in: queue.Queue = queue.Queue(maxsize=1)
    stop = threading.Event()
    threading.Thread(target=capture_loop, args=(cap, q_in, stop, args.size),
                     daemon=True).start()

    last = None
    t0, frames = time.time(), 0
    while True:
        frame = q_in.get()
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
        cv2.imshow("sdstream", np.hstack([frame, out]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    stop.set()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

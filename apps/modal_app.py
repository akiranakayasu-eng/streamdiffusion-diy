"""Modal 版: GPU は Modal、カメラと表示はスマホや PC のブラウザ。

ブラウザでページを開く → スマホのカメラか動画ファイルの映像を 1 枚ずつ JPEG で
WebSocket に送る → Modal の GPU で自作 StreamDiffusion が変換 → JPEG で返す。

課金の事故を防ぐ仕掛け（スキル modal-cost-safety の教訓: 開いたままのページが
再接続し続けてアイドル停止が効かなかった）
  - 1 回の接続は最長 10 分（SESSION_MAX_SEC）。過ぎたらサーバー側から切る。
  - 30 秒フレームが来なければ切る（IDLE_SEC）。
  - ページは自動で再接続しない。画面を隠す・スリープでも止まる。
  - 使われなくなったら 60 秒でコンテナを止める（scaledown_window）。
  - コンテナは最大 1 台（max_containers=1）。同時に使えるのは 1 人。
  - URL に合言葉（?key=...）が無いと使えない。合言葉は Modal の Secret に置く。

準備（1 回だけ）
  modal secret create sdstream-key SDSTREAM_KEY=<好きな長い文字列>
起動（リポジトリの一番上で）
  modal deploy apps/modal_app.py      # 表示された URL に ?key=<合言葉> を付けて開く
止める
  modal app stop streamdiffusion-diy
"""

import asyncio
import io
import os
import time
import traceback
from pathlib import Path

import modal

APP_NAME = "streamdiffusion-diy"
MODEL_ID = "stabilityai/sd-turbo"
VAE_ID = "madebyollin/taesd"
SESSION_MAX_SEC = 600
IDLE_SEC = 30
STRENGTH = {"weak": (32, 45), "normal": (20, 40), "strong": (10, 30)}
SIZES = (256, 384, 512)

WEB_DIR = Path(__file__).parent / "web"


def download_models():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, allow_patterns=[
        "tokenizer/*", "text_encoder/*", "unet/*", "*.json"],
        ignore_patterns=["*.bin", "*.ckpt", "*fp16*.safetensors"])
    snapshot_download(VAE_ID)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "diffusers", "transformers", "accelerate",
                 "huggingface_hub", "fastapi[standard]", "pillow", "numpy")
    .env({"HF_HOME": "/models/hf"})
    .run_function(download_models)          # 重みはイメージに焼き込む（起動を速く）
    .add_local_python_source("sdstream")
    .add_local_dir(WEB_DIR, "/root/web")
)

app = modal.App(APP_NAME)
# 使うたびの速さの記録（modal dict items sdstream-stats で見る）
stats_dict = modal.Dict.from_name("sdstream-stats", create_if_missing=True)


@app.cls(
    image=image,
    gpu=["L4", "A10G"],          # L4 が空いていなければ A10G
    scaledown_window=60,
    timeout=SESSION_MAX_SEC + 120,
    max_containers=1,
    secrets=[modal.Secret.from_name("sdstream-key")],
)
@modal.concurrent(max_inputs=4)  # ページの取得と WebSocket を同時に受けるため
class Server:
    @modal.enter()
    def load(self):
        import torch

        from sdstream.models import SDParts

        self.parts = SDParts(MODEL_ID, vae_id=VAE_ID,
                             device=torch.device("cuda"), dtype=torch.float16)
        self.busy = asyncio.Lock()
        self.uncond = self.parts.embed("")
        self._configure("watercolor painting", "strong", 512)
        # 1 回空回しして、最初のフレームの待ちを短くする
        self._process_array(torch.zeros(1, 3, 512, 512))

    # ---- 設定 ------------------------------------------------------------
    def _configure(self, prompt: str, strength: str, size: int,
                   negative: str = "", guidance: float = 1.0):
        """negative が空なら CFG なし（速い）。入っていれば RCFG onetime で効かせる。"""
        from sdstream.stream import StreamConfig, StreamDiffusionCore

        size = size if size in SIZES else 512
        negative = (negative or "").strip()
        guidance = max(1.0, min(float(guidance or 1.0), 5.0))
        if negative and guidance <= 1.0:
            guidance = 1.5               # ネガティブを入れたのに効かない、を防ぐ
        cfg = StreamConfig(t_index_list=STRENGTH.get(strength, STRENGTH["strong"]),
                           guidance_scale=guidance,
                           cfg_type="onetime" if negative else "self",
                           latent_shape=(4, size // 8, size // 8))
        self.core = StreamDiffusionCore(self.parts.unet_fn, cfg,
                                        self.parts.device, self.parts.dtype)
        uncond = self.parts.embed(negative) if negative else self.uncond
        self.core.set_prompt(self.parts.embed(prompt), uncond)
        self.size = size
        self.strength = strength
        self.negative = bool(negative)
        self.guidance = guidance

    # ---- 1 フレーム ------------------------------------------------------
    def _process_array(self, img):
        lat = self.parts.encode(img)
        out = self.parts.decode(self.core.step(lat))
        return out

    def _process_jpeg(self, data: bytes) -> bytes:
        import numpy as np
        import torch
        from PIL import Image

        im = Image.open(io.BytesIO(data)).convert("RGB")
        if im.size != (self.size, self.size):
            im = im.resize((self.size, self.size))
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)[None] / 255.0
        y = self._process_array(x)[0].permute(1, 2, 0).float().cpu().numpy()
        buf = io.BytesIO()
        Image.fromarray((y * 255).astype("uint8")).save(buf, "JPEG", quality=85)
        return buf.getvalue()

    # ---- 記録 ------------------------------------------------------------
    async def _record(self, st: dict, ua: str):
        import json

        import torch

        n = st["frames"]
        if n < 2:
            return
        dur = st["last"] - st["first"]
        rec = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "gpu": torch.cuda.get_device_name(0),
            "size": self.size, "strength": self.strength,
            "t_index": list(STRENGTH.get(self.strength, STRENGTH["strong"])),
            "negative": self.negative, "guidance": self.guidance,
            "frames": n, "seconds": round(dur, 1),
            "fps": round((n - 1) / dur, 2) if dur > 0 else None,   # 往復を含む実際の速さ
            "gpu_ms": round(st["gpu_sec"] / n * 1000, 1),           # 1 枚の GPU 処理時間
            "client": "mobile" if "Mobile" in ua else "desktop",
        }
        print("STATS " + json.dumps(rec, ensure_ascii=False))
        try:
            await stats_dict.put.aio(f"{rec['time']} {rec['client']}", rec)
        except Exception:
            traceback.print_exc()

    # ---- Web ---------------------------------------------------------------
    @modal.asgi_app()
    def web(self):
        import json

        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse, PlainTextResponse

        api = FastAPI()
        key = os.environ["SDSTREAM_KEY"]
        page = Path("/root/web/index.html").read_text(encoding="utf-8")

        @api.get("/")
        async def index():
            return HTMLResponse(page)   # 合言葉は WebSocket で確かめる

        @api.get("/health")
        async def health():
            return PlainTextResponse("ok")

        @api.websocket("/ws")
        async def ws(socket: WebSocket):
            await socket.accept()
            if socket.query_params.get("key") != key:
                print("ws: 合言葉が違う")
                await socket.send_text(json.dumps({"error": "合言葉が違います"}))
                await socket.close(code=1008, reason="bad key")
                return
            if self.busy.locked():
                print("ws: 使用中")
                await socket.send_text(json.dumps({"error": "ほかの人が使用中です"}))
                await socket.close(code=1013, reason="busy")
                return
            async with self.busy:
                start = time.time()
                st = {"frames": 0, "gpu_sec": 0.0, "first": None, "last": None}
                await socket.send_text(json.dumps(
                    {"ready": True, "session_max_sec": SESSION_MAX_SEC}))
                try:
                    while True:
                        left = SESSION_MAX_SEC - (time.time() - start)
                        if left <= 0:
                            await socket.send_text(json.dumps(
                                {"stopped": "10 分たったので止めました"}))
                            break
                        try:
                            msg = await asyncio.wait_for(
                                socket.receive(), timeout=min(IDLE_SEC, left))
                        except asyncio.TimeoutError:
                            await socket.send_text(json.dumps(
                                {"stopped": "映像が届かないので止めました"}))
                            break
                        if msg.get("type") == "websocket.disconnect":
                            break
                        if msg.get("text"):
                            s = json.loads(msg["text"])
                            # 設定が変わる前の区間を記録して数え直す
                            await self._record(st, socket.headers.get("user-agent", ""))
                            st = {"frames": 0, "gpu_sec": 0.0, "first": None, "last": None}
                            await asyncio.to_thread(
                                self._configure, s.get("prompt", "watercolor painting"),
                                s.get("strength", "strong"), int(s.get("size", 512)),
                                s.get("negative", ""), float(s.get("guidance", 1.0)))
                            await socket.send_text(json.dumps({"configured": True}))
                        elif msg.get("bytes"):
                            t1 = time.time()
                            out = await asyncio.to_thread(self._process_jpeg, msg["bytes"])
                            t2 = time.time()
                            st["frames"] += 1
                            st["gpu_sec"] += t2 - t1
                            st["first"] = st["first"] or t1
                            st["last"] = t2
                            await socket.send_bytes(out)
                except WebSocketDisconnect:
                    pass
                except Exception as e:  # 原因を Modal の記録とページの両方に出す
                    traceback.print_exc()
                    try:
                        await socket.send_text(json.dumps({"error": f"サーバーのエラー: {e}"}))
                    except Exception:
                        pass
                await self._record(st, socket.headers.get("user-agent", ""))
                try:
                    await socket.close(code=1000, reason="done")
                except Exception:
                    pass

        return api

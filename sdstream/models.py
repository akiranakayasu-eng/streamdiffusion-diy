"""実モデルの読み込み（diffusers の部品だけを使い、パイプラインは使わない）。

既定は SD-Turbo（1〜2 段で絵になる）+ Tiny AutoEncoder（TAESD、VAE の
軽量版）。SD1.5 系の重み + LCM-LoRA にも切り替えられる。
"""
from __future__ import annotations

import torch


def pick_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


class SDParts:
    """UNet・テキストエンコーダ・VAE を持ち、StreamDiffusionCore に渡す関数を作る。"""

    def __init__(self, model_id: str = "stabilityai/sd-turbo",
                 lcm_lora: str | None = None,
                 vae_id: str = "madebyollin/taesd",
                 device=None, dtype=None):
        from diffusers import AutoencoderTiny, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        d, t = pick_device()
        self.device, self.dtype = device or d, dtype or t
        kw = dict(torch_dtype=self.dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", **kw).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", **kw).to(self.device)
        if lcm_lora:  # 例: "latent-consistency/lcm-lora-sdv1-5"（peft が要る）
            self.unet.load_lora_adapter(lcm_lora, prefix="unet")
            self.unet.fuse_lora()
        self.vae = AutoencoderTiny.from_pretrained(vae_id, **kw).to(self.device)
        self.scale = float(getattr(self.vae.config, "scaling_factor", 1.0))

    @torch.no_grad()
    def embed(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer(prompt, padding="max_length", truncation=True,
                             max_length=self.tokenizer.model_max_length,
                             return_tensors="pt").input_ids.to(self.device)
        return self.text_encoder(ids)[0]

    @torch.no_grad()
    def unet_fn(self, x, t, cond):
        return self.unet(x, t, encoder_hidden_states=cond, return_dict=False)[0]

    @torch.no_grad()
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """img: [1,3,H,W]、値は 0..1。"""
        x = img.to(self.device, self.dtype) * 2 - 1
        return self.vae.encode(x).latents * self.scale

    @torch.no_grad()
    def decode(self, lat: torch.Tensor) -> torch.Tensor:
        x = self.vae.decode(lat / self.scale).sample
        return (x / 2 + 0.5).clamp(0, 1)

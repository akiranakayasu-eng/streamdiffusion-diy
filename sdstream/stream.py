"""自作 StreamDiffusion のコア（モデルに依存しない部分）。

論文: Kodaira et al., "StreamDiffusion: A Pipeline-level Solution for
Real-time Interactive Generation" (arXiv:2312.12491) の仕組みを、
公式コードを写さずに最小限で組み直したもの。

入れてある仕組み
  1. Stream Batch   … ノイズ除去の段（t_index_list の長さ = n 段）を
                      フレームごとにずらしてバッチにまとめ、UNet を 1 回呼ぶ
                      ごとに 1 フレーム出す（遅れは n-1 フレーム）。
  2. RCFG           … 否定プロンプトの UNet を回さずに「残差ノイズ」で
                      CFG を近似する（self: UNet n 回 / onetime: n+1 回）。
  3. 事前計算       … プロンプトの埋め込み、ノイズ、係数 α・β・c_skip・c_out
                      を最初に 1 回だけ作る。
類似度フィルタ（SSF）は similarity_filter.py、モデルの読み込みは models.py。

UNet・VAE は呼び出し関数として受け取るので、ダミーでも動く（test_dummy.py）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import torch

# unet_fn(x_t[B,4,h,w], t[B], cond[B,77,D]) -> eps[B,4,h,w]
UNetFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def sd_alphas_cumprod(num_train_timesteps: int = 1000,
                      beta_start: float = 0.00085,
                      beta_end: float = 0.012) -> torch.Tensor:
    """Stable Diffusion 1.x / 2.x / Turbo の scaled_linear の ᾱ_t。"""
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5,
                           num_train_timesteps, dtype=torch.float64) ** 2
    return torch.cumprod(1.0 - betas, dim=0)


def lcm_timesteps(num_inference_steps: int = 50,
                  num_train_timesteps: int = 1000) -> list[int]:
    """LCM の元の時刻列（50 段なら 999, 979, …, 19）。"""
    c = num_train_timesteps // num_inference_steps
    return [num_train_timesteps - 1 - c * i for i in range(num_inference_steps)]


def lcm_boundary(t: torch.Tensor, sigma_data: float = 0.5,
                 timestep_scaling: float = 10.0):
    """LCM の境界条件 c_skip, c_out。"""
    s = t * timestep_scaling
    c_skip = sigma_data ** 2 / (s ** 2 + sigma_data ** 2)
    c_out = s / (s ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out


@dataclass
class StreamConfig:
    t_index_list: Sequence[int] = (32, 45)   # 50 段の LCM 時刻列のどこを使うか
    num_inference_steps: int = 50
    guidance_scale: float = 1.0              # 1.0 以下なら CFG なし
    cfg_type: Literal["none", "self", "onetime", "full"] = "self"
    delta: float = 1.0                       # RCFG の残差ノイズの強さ
    add_noise: bool = True
    seed: int = 2
    latent_shape: tuple[int, int, int] = (4, 64, 64)  # 512x512 のとき


class StreamDiffusionCore:
    """Stream Batch + RCFG の本体。1 回 step() を呼ぶと 1 フレーム出る。"""

    def __init__(self, unet_fn: UNetFn, cfg: StreamConfig,
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.unet_fn = unet_fn
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype
        self.n = len(cfg.t_index_list)
        self._precompute()

    # ---- 事前計算 -------------------------------------------------------
    def _precompute(self) -> None:
        cfg, dev, dt = self.cfg, self.device, self.dtype
        ac = sd_alphas_cumprod()
        all_t = lcm_timesteps(cfg.num_inference_steps)
        ts = [all_t[i] for i in cfg.t_index_list]
        self.timesteps = torch.tensor(ts, device=dev, dtype=torch.long)

        a = torch.tensor([ac[t].item() for t in ts], dtype=torch.float64)
        shp = (self.n, 1, 1, 1)
        self.alpha = a.sqrt().view(shp).to(dev, dt)          # √ᾱ_t
        self.beta = (1 - a).sqrt().view(shp).to(dev, dt)     # √(1-ᾱ_t)
        c_skip, c_out = lcm_boundary(torch.tensor(ts, dtype=torch.float64))
        self.c_skip = c_skip.view(shp).to(dev, dt)
        self.c_out = c_out.view(shp).to(dev, dt)

        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        self.init_noise = torch.randn((self.n, *cfg.latent_shape),
                                      generator=g).to(dev, dt)
        self.stock_noise = self.init_noise.clone()
        # 段 2..n で途中まで除去されたフレームの待ち行列
        self.buffer = torch.zeros((max(self.n - 1, 0), *cfg.latent_shape),
                                  device=dev, dtype=dt)
        self.cond: Optional[torch.Tensor] = None
        self.uncond: Optional[torch.Tensor] = None
        self.frames_in = 0

    def set_prompt(self, cond: torch.Tensor,
                   uncond: Optional[torch.Tensor] = None) -> None:
        """cond/uncond は [1,77,D]。バッチ分を先に並べておく。"""
        self.cond = cond.to(self.device, self.dtype).repeat(self.n, 1, 1)
        if uncond is not None:
            self.uncond = uncond.to(self.device, self.dtype)

    # ---- 1 段ぶんの計算 -------------------------------------------------
    def _denoise(self, x_t: torch.Tensor, eps: torch.Tensor,
                 idx: slice = slice(None)) -> torch.Tensor:
        x0 = (x_t - self.beta[idx] * eps) / self.alpha[idx]
        return self.c_out[idx] * x0 + self.c_skip[idx] * x_t

    def _unet(self, x_t: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        use_cfg = cfg.guidance_scale > 1.0 and cfg.cfg_type != "none"
        if not use_cfg:
            return self.unet_fn(x_t, self.timesteps, self.cond)

        if cfg.cfg_type == "full":           # 普通の CFG（UNet 2n 回）
            assert self.uncond is not None
            u = self.uncond.repeat(self.n, 1, 1)
            out = self.unet_fn(torch.cat([x_t, x_t]),
                               torch.cat([self.timesteps, self.timesteps]),
                               torch.cat([u, self.cond]))
            e_u, e_c = out.chunk(2)
        elif cfg.cfg_type == "onetime":      # 先頭の段だけ否定を回す（n+1 回）
            assert self.uncond is not None
            out = self.unet_fn(torch.cat([x_t[:1], x_t]),
                               torch.cat([self.timesteps[:1], self.timesteps]),
                               torch.cat([self.uncond, self.cond]))
            e_c = out[1:]
            e_u = self.stock_noise * cfg.delta
            e_u = torch.cat([out[:1], e_u[1:]])
        else:                                # self: 残差ノイズだけ（n 回）
            e_c = self.unet_fn(x_t, self.timesteps, self.cond)
            e_u = self.stock_noise * cfg.delta

        eps = e_u + cfg.guidance_scale * (e_c - e_u)
        self._update_stock_noise(eps)
        return eps

    def _update_stock_noise(self, eps: torch.Tensor) -> None:
        """RCFG: 次のフレームで使う「仮想の否定ノイズ」を更新する。

        各段で予測した x0 から、次の段の x_t に戻したときに残るノイズ成分を
        次の段の stock_noise にする（段がずれるので 1 つ送る）。
        """
        scaled = self.beta * self.stock_noise
        dx = self._denoise(scaled, eps)
        ones = torch.ones_like(self.alpha[:1])
        a_next = torch.cat([self.alpha[1:], ones])
        b_next = torch.cat([self.beta[1:], ones])
        dx = a_next * dx / b_next
        shifted = torch.cat([self.init_noise[1:], self.init_noise[:1]])
        self.stock_noise = shifted + dx

    # ---- 1 フレーム -----------------------------------------------------
    def step(self, latent_in: torch.Tensor) -> torch.Tensor:
        """latent_in: [1,4,h,w]（img2img の入力。txt2img なら 0 でよい）。

        戻り値: 予測した x0 の潜在 [1,4,h,w]。最初の n-1 回は
        まだ途中のフレームなので、呼ぶ側で捨てるか、そのまま出す。
        """
        assert self.cond is not None, "set_prompt() を先に呼ぶ"
        latent_in = latent_in.to(self.device, self.dtype)
        # 入力を最初の段の時刻までノイズを足す
        x_new = self.alpha[0] * latent_in + self.beta[0] * self.init_noise[:1]
        if self.n > 1:
            x_t = torch.cat([x_new, self.buffer])        # [n,4,h,w]
            self.stock_noise = torch.cat(
                [self.init_noise[:1], self.stock_noise[:-1]])
        else:
            x_t = x_new

        eps = self._unet(x_t)
        x0 = self._denoise(x_t, eps)                     # 全段まとめて

        if self.n > 1:
            out = x0[-1:]                                # 最後の段が完成品
            nxt = self.alpha[1:] * x0[:-1]               # 次の段へ送る
            if self.cfg.add_noise:
                nxt = nxt + self.beta[1:] * self.init_noise[1:]
            self.buffer = nxt
        else:
            out = x0
        self.frames_in += 1
        return out

    @property
    def warmed_up(self) -> bool:
        return self.frames_in >= self.n

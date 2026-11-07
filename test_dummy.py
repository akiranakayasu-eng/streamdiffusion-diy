"""GPU もモデルも無しで、自作 StreamDiffusion の仕組みが正しいか確かめる。

  python test_dummy.py
"""
import torch

from sdstream.stream import StreamConfig, StreamDiffusionCore
from sdstream.similarity_filter import SimilarityFilter

torch.manual_seed(0)
W = torch.randn(4, 4) * 0.1


class DummyUNet:
    """サンプルごとに独立な、時刻と条件に依存する適当な関数。呼ばれた枚数を数える。"""
    def __init__(self):
        self.samples = 0

    def __call__(self, x, t, cond):
        self.samples += x.shape[0]
        tt = (t.float() / 1000).view(-1, 1, 1, 1)
        c = cond.mean(dim=(1, 2)).view(-1, 1, 1, 1)
        return torch.tanh(torch.einsum("ij,bjhw->bihw", W, x) * (1 + tt) + c)


def sequential_reference(core, lat):
    """1 フレームを n 段の LCM で普通に（バッチにせず）除去する。"""
    a, b = core.alpha, core.beta
    x = a[0] * lat + b[0] * core.init_noise[:1]
    for i in range(core.n):
        eps = core.unet_fn(x, core.timesteps[i:i + 1], core.cond[:1])
        x0 = core._denoise(x, eps, slice(i, i + 1))
        if i + 1 < core.n:
            x = a[i + 1] * x0 + b[i + 1] * core.init_noise[i + 1:i + 2]
    return x0


def test_stream_batch_matches_sequential():
    for t_idx in [(45,), (32, 45), (0, 16, 32, 45)]:
        u = DummyUNet()
        cfg = StreamConfig(t_index_list=t_idx, latent_shape=(4, 8, 8))
        core = StreamDiffusionCore(u, cfg)
        core.set_prompt(torch.randn(1, 77, 16))
        frames = [torch.randn(1, 4, 8, 8) for _ in range(8)]
        outs = [core.step(f) for f in frames]
        n = core.n
        for k in range(len(frames) - n + 1):
            ref = sequential_reference(core, frames[k])
            got = outs[k + n - 1]            # n-1 フレーム遅れて出る
            assert torch.allclose(ref, got, atol=1e-5), (t_idx, k)
        print(f"ok  Stream Batch {len(t_idx)} 段: 逐次計算と一致（遅れ {n-1} フレーム）")


def test_unet_cost():
    n = 2
    for cfg_type, expect in [("none", n), ("self", n), ("onetime", n + 1), ("full", 2 * n)]:
        u = DummyUNet()
        cfg = StreamConfig(t_index_list=(32, 45), guidance_scale=1.5,
                           cfg_type=cfg_type, latent_shape=(4, 8, 8))
        core = StreamDiffusionCore(u, cfg)
        core.set_prompt(torch.randn(1, 77, 16), torch.randn(1, 77, 16))
        for _ in range(10):
            out = core.step(torch.randn(1, 4, 8, 8))
            assert torch.isfinite(out).all()
        per = u.samples / 10
        assert per == expect, (cfg_type, per)
        print(f"ok  CFG={cfg_type:8s}: UNet {per:.0f} 枚/フレーム")


def test_ssf():
    f = SimilarityFilter(threshold=0.98, max_skip=5, seed=0)
    x = torch.rand(3, 64, 64)
    skips = sum(f.should_skip(x) for _ in range(60))
    assert 0 < skips <= 50, skips                       # 同じ絵はほぼ飛ばす、上限 5 連続
    f2 = SimilarityFilter(seed=0)
    moves = sum(f2.should_skip(torch.rand(3, 64, 64)) for _ in range(60))
    assert moves == 0, moves                            # 違う絵は飛ばさない
    print(f"ok  類似度フィルタ: 同じ絵 60 枚中 {skips} 枚を省略、違う絵は 0 枚")


if __name__ == "__main__":
    test_stream_batch_matches_sequential()
    test_unet_cost()
    test_ssf()

"""確率的な類似度フィルタ（Stochastic Similarity Filter, SSF）。

入力フレームが前回とほぼ同じなら UNet を回さず前回の出力を使う。
ただし完全に止めると動き出しに反応しないので、似ているほど高い確率で
飛ばす（確率的）。静止したカメラ映像で GPU の負荷を下げる仕組み。
"""
from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn.functional as F


class SimilarityFilter:
    def __init__(self, threshold: float = 0.98, max_skip: int = 10,
                 seed: Optional[int] = None):
        self.threshold = threshold      # この類似度を超えたら飛ばし始める
        self.max_skip = max_skip        # 連続で飛ばす上限
        self.prev: Optional[torch.Tensor] = None
        self.skipped = 0
        self.rng = random.Random(seed)

    def should_skip(self, frame: torch.Tensor) -> bool:
        """frame: 入力画像のテンソル（形は何でもよい）。"""
        x = frame.detach().flatten().float()
        if self.prev is None or self.prev.shape != x.shape:
            self.prev = x
            return False
        sim = F.cosine_similarity(x, self.prev, dim=0).item()
        # sim が threshold → 1 に近づくほど飛ばす確率が 0 → 1 に上がる
        p = max(0.0, (sim - self.threshold) / (1.0 - self.threshold))
        if self.skipped < self.max_skip and self.rng.random() < p:
            self.skipped += 1
            return True
        self.prev = x
        self.skipped = 0
        return False

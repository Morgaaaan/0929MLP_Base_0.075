# -*- coding: utf-8 -*-
"""
model.py
- 簡潔版 MLP：數值特徵 + 多個類別欄位（經穩定雜湊 -> embedding）後 concat，接多層全連接。
- 可用於二分類或多分類（num_classes>=2）。
"""

from typing import Dict, List
import torch
import torch.nn as nn

class CatEmbedding(nn.Module):
    def __init__(self, buckets: Dict[str, int], emb_dim: int = 16):
        super().__init__()
        self.columns = list(buckets.keys())
        self.embs = nn.ModuleDict({
            c: nn.Embedding(num_embeddings=buckets[c], embedding_dim=emb_dim)
            for c in self.columns
        })
        self.out_dim = emb_dim * len(self.columns)

    def forward(self, x_cat: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.columns:
            # 無類別欄位
            batch = next(iter(x_cat.values())).shape[0] if x_cat else 0
            return torch.zeros((batch, 0), device=next(iter(x_cat.values())).device if x_cat else "cpu")
        embs = [self.embs[c](x_cat[c]) for c in self.columns]
        return torch.cat(embs, dim=-1)

class MLPClassifier(nn.Module):
    def __init__(self,
                 input_num_dim: int,
                 cat_buckets: Dict[str, int],
                 num_classes: int = 2,
                 hidden: List[int] = [256, 128],
                 dropout: float = 0.2,
                 cat_emb_dim: int = 16,
                 bn: bool = True):
        super().__init__()
        self.cat_block = CatEmbedding(cat_buckets, emb_dim=cat_emb_dim)
        in_dim = input_num_dim + self.cat_block.out_dim

        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev if layers else in_dim, num_classes)

    def forward(self, x_num: torch.Tensor, x_cat: Dict[str, torch.Tensor]) -> torch.Tensor:
        cat = self.cat_block(x_cat) if x_cat else torch.zeros((x_num.shape[0], 0), device=x_num.device)
        x = torch.cat([x_num, cat], dim=-1)
        if len(self.backbone) > 0:
            x = self.backbone(x)
        logits = self.head(x)
        return logits

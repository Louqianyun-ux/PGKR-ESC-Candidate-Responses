# coding=utf-8

import torch
import torch.nn as nn


def ranking_loss(scores: torch.Tensor, labels: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    device = scores.device
    loss_fn = nn.MarginRankingLoss(margin=margin, reduction="mean")

    total_loss = []
    batch_size, _ = scores.size()

    for b in range(batch_size):
        s = scores[b]
        l = labels[b]

        valid_mask = l.ge(0)
        s = s[valid_mask]
        l = l[valid_mask]

        if s.numel() <= 1:
            continue

        pos_idx = (l == 1).nonzero(as_tuple=False).view(-1)
        neg_idx = (l == 0).nonzero(as_tuple=False).view(-1)

        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue

        pos_scores = s[pos_idx]
        neg_scores = s[neg_idx]

        pos_expand = pos_scores.unsqueeze(1).expand(-1, neg_scores.size(0)).reshape(-1)
        neg_expand = neg_scores.unsqueeze(0).expand(pos_scores.size(0), -1).reshape(-1)

        target = torch.ones_like(pos_expand, device=device)
        loss = loss_fn(pos_expand, neg_expand, target)
        total_loss.append(loss)

    if len(total_loss) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(total_loss).mean()

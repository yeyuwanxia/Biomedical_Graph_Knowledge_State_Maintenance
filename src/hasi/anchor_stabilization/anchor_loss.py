from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistributedAnchorTarget:
    node: int
    target: torch.Tensor
    weight: float


class AnchorStabilizationLoss(nn.Module):
    """Embedding-level anchor loss used by HASI."""

    def __init__(self, lambda1: float = 2.0, lambda2: float = 0.5):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.register_buffer("primary_snapshot", torch.empty(0), persistent=False)
        self.register_buffer("secondary_snapshot", torch.empty(0), persistent=False)
        self.primary_nodes: list[int] = []
        self.secondary_nodes: list[int] = []
        self.distributed_targets: Dict[int, DistributedAnchorTarget] = {}

    def set_snapshots(
        self,
        embeddings: torch.Tensor,
        primary_nodes: Iterable[int],
        secondary_nodes: Iterable[int],
    ) -> None:
        self.primary_nodes = [int(node) for node in primary_nodes]
        self.secondary_nodes = [int(node) for node in secondary_nodes]
        device = embeddings.device
        self.primary_snapshot = self._snapshot(embeddings, self.primary_nodes).to(device)
        self.secondary_snapshot = self._snapshot(embeddings, self.secondary_nodes).to(device)

    def register_distributed_target(self, node: int, target: torch.Tensor, weight: float) -> None:
        self.distributed_targets[int(node)] = DistributedAnchorTarget(
            node=int(node),
            target=target.detach().clone(),
            weight=float(weight),
        )

    def clear_distributed_targets(self) -> None:
        self.distributed_targets.clear()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        loss = embeddings.new_tensor(0.0)

        if self.primary_nodes:
            current = embeddings[self.primary_nodes]
            loss = loss + self.lambda1 * F.mse_loss(current, self.primary_snapshot.to(embeddings.device))

        if self.secondary_nodes:
            current = embeddings[self.secondary_nodes]
            loss = loss + self.lambda2 * F.mse_loss(current, self.secondary_snapshot.to(embeddings.device))

        for target in self.distributed_targets.values():
            current = embeddings[target.node]
            loss = loss + target.weight * F.mse_loss(current, target.target.to(embeddings.device))

        return loss

    @staticmethod
    def _snapshot(embeddings: torch.Tensor, nodes: list[int]) -> torch.Tensor:
        if not nodes:
            return embeddings.new_empty((0, embeddings.shape[-1]))
        return embeddings[nodes].detach().clone()

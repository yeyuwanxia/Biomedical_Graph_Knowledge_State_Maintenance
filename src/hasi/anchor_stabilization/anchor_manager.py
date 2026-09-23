from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Set


@dataclass
class AnchorSets:
    primary: Set[int] = field(default_factory=set)
    secondary: Set[int] = field(default_factory=set)
    regular: Set[int] = field(default_factory=set)

    def layer_of(self, node: int) -> str:
        if node in self.primary:
            return "primary"
        if node in self.secondary:
            return "secondary"
        return "regular"


class AnchorManager:
    """Owns Primary/Secondary/Regular anchor membership."""

    def __init__(self, primary_ratio: float = 0.01, secondary_ratio: float = 0.05):
        self.primary_ratio = primary_ratio
        self.secondary_ratio = secondary_ratio
        self.anchors = AnchorSets()

    def classify_from_scores(self, scores: Mapping[int, float]) -> AnchorSets:
        ranked = sorted(scores, key=lambda node: scores[node], reverse=True)
        n_nodes = len(ranked)
        if n_nodes == 0:
            self.anchors = AnchorSets()
            return self.anchors

        primary_count = max(1, int(round(n_nodes * self.primary_ratio)))
        secondary_count = max(primary_count, int(round(n_nodes * self.secondary_ratio)))
        self.anchors = AnchorSets(
            primary=set(ranked[:primary_count]),
            secondary=set(ranked[primary_count:secondary_count]),
            regular=set(ranked[secondary_count:]),
        )
        return self.anchors

    def remove_forgetting_targets(self, nodes: Iterable[int]) -> None:
        for node in nodes:
            self.anchors.primary.discard(node)
            self.anchors.secondary.discard(node)
            self.anchors.regular.discard(node)

    def add_secondary(self, nodes: Iterable[int]) -> None:
        for node in nodes:
            self.anchors.regular.discard(node)
            self.anchors.primary.discard(node)
            self.anchors.secondary.add(node)

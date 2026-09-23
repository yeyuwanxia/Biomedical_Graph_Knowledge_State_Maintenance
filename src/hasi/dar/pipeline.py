from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import networkx as nx

from hasi.graph_compute import ShortestPathComputer


@dataclass(frozen=True)
class DARConfig:
    enabled: bool = True
    k: int = 5
    strategy: str = "distributed"
    min_distance: int = 2
    small_component_threshold: int = 10
    lambda2: float = 0.5
    gumbel_tau: float = 0.1
    alpha_score: float = 0.6
    beta_score: float = 0.4
    max_search_radius: Optional[int] = None
    seed: int = 42
    compute_backend: str = "auto"
    device: Optional[str] = None


@dataclass
class DeletionContext:
    deleted_node: int
    neighbors: List[int]
    candidate_distances: Dict[int, int]
    component_quotas: Dict[int, int]
    component_nodes: Dict[int, List[int]] = field(default_factory=dict)
    preselected_by_component: Dict[int, List[int]] = field(default_factory=dict)
    preselected_candidates: List[int] = field(default_factory=list)
    candidate_hub_scores: Dict[int, float] = field(default_factory=dict)
    strategy: str = "distributed"
    original_min_distance: int = 0
    final_min_distance: int = 0
    distance_backend: Dict[str, object] = field(default_factory=dict)


@dataclass
class DistributedAnchor:
    node: int
    weight: float
    target_kind: str
    cached_distance: int
    component_id: Optional[int] = None


class DARPipeline:
    """Two-phase Distributed Anchor Replacement for Primary hub deletion."""

    def __init__(self, config: Optional[DARConfig] = None):
        self.config = config or DARConfig()
        self.distance_computer = ShortestPathComputer(
            backend=self.config.compute_backend,
            device=self.config.device,
        )

    def run_phase1(
        self,
        graph: nx.Graph,
        deleted_node: int,
        hub_scores: Mapping[int, float],
        excluded_nodes: Optional[Iterable[int]] = None,
    ) -> DeletionContext:
        """Cache distances and preselect candidates while the deleted hub exists."""

        if deleted_node not in graph:
            raise ValueError(f"deleted_node {deleted_node} must exist before DAR phase1")

        self.distance_computer.clear()
        excluded = {int(node) for node in (excluded_nodes or set())}
        excluded.add(int(deleted_node))
        cutoff = self.config.max_search_radius
        distances = self.distance_computer.single_source(graph, deleted_node, cutoff=cutoff)
        distances = {int(node): int(distance) for node, distance in distances.items()}
        neighbors = sorted(int(node) for node in graph.neighbors(deleted_node))

        graph_after = graph.copy()
        graph_after.remove_node(deleted_node)
        component_nodes = self._component_nodes_after_deletion(graph_after)
        quotas = self._component_quotas(component_nodes)
        strategy = self._strategy()

        preselected_by_component: Dict[int, List[int]] = {}
        final_min = self.config.min_distance
        if strategy == "distributed":
            for comp_id, quota in quotas.items():
                count = max(1, 2 * int(quota))
                candidates = self._candidate_pool(
                    component_nodes.get(comp_id, []),
                    hub_scores,
                    distances,
                    excluded,
                    min_distance=self.config.min_distance,
                    require_hub_percentile=True,
                )
                if len(candidates) < count:
                    candidates = self._candidate_pool(
                        component_nodes.get(comp_id, []),
                        hub_scores,
                        distances,
                        excluded,
                        min_distance=self.config.min_distance,
                        require_hub_percentile=False,
                    )
                selected, used_min = self._distributed_select(
                    graph_after,
                    candidates,
                    hub_scores,
                    count,
                    rng=self._rng(deleted_node, comp_id),
                )
                final_min = min(final_min, used_min)
                preselected_by_component[int(comp_id)] = selected
        else:
            candidates = self._candidate_pool(
                graph_after.nodes,
                hub_scores,
                distances,
                excluded,
                min_distance=self.config.min_distance if strategy == "privacy_constrained" else 0,
                require_hub_percentile=False,
            )
            selected = self._select_by_strategy(
                graph_after,
                candidates,
                hub_scores,
                distances,
                deleted_node,
                max(2 * self.config.k, self.config.k),
                strategy,
                rng=self._rng(deleted_node, 0),
            )
            preselected_by_component[0] = selected

        preselected = self._unique(
            node for nodes in preselected_by_component.values() for node in nodes
        )
        return DeletionContext(
            deleted_node=int(deleted_node),
            neighbors=neighbors,
            candidate_distances=distances,
            component_quotas=quotas,
            component_nodes=component_nodes,
            preselected_by_component=preselected_by_component,
            preselected_candidates=preselected,
            candidate_hub_scores={int(node): float(hub_scores.get(node, 0.0)) for node in preselected},
            strategy=strategy,
            original_min_distance=int(self.config.min_distance),
            final_min_distance=int(final_min),
            distance_backend=dict(self.distance_computer.last_diagnostics),
        )

    def run_phase2(self, context: DeletionContext, repaired_graph: nx.Graph) -> List[DistributedAnchor]:
        """Select final replacement anchors on the repaired graph."""

        strategy = context.strategy or self._strategy()
        selected_by_component: Dict[int, List[int]] = {}
        final_min = context.final_min_distance or self.config.min_distance

        if strategy == "distributed":
            for comp_id, quota in context.component_quotas.items():
                candidates = [
                    int(node)
                    for node in context.preselected_by_component.get(comp_id, [])
                    if int(node) in repaired_graph
                ]
                selected, used_min = self._distributed_select(
                    repaired_graph,
                    candidates,
                    context.candidate_hub_scores,
                    int(quota),
                    rng=self._rng(context.deleted_node, comp_id + 10_000),
                )
                final_min = min(final_min, used_min)
                selected_by_component[int(comp_id)] = selected
            selected = self._unique(
                node for nodes in selected_by_component.values() for node in nodes
            )
            if len(selected) < self.config.k:
                fallback = [
                    node
                    for node in context.preselected_candidates
                    if node in repaired_graph and node not in selected
                ]
                more, used_min = self._distributed_select(
                    repaired_graph,
                    fallback,
                    context.candidate_hub_scores,
                    self.config.k - len(selected),
                    rng=self._rng(context.deleted_node, 20_000),
                )
                final_min = min(final_min, used_min)
                selected.extend(more)
        else:
            candidates = [node for node in context.preselected_candidates if node in repaired_graph]
            selected = self._select_by_strategy(
                repaired_graph,
                candidates,
                context.candidate_hub_scores,
                context.candidate_distances,
                context.deleted_node,
                self.config.k,
                strategy,
                rng=self._rng(context.deleted_node, 30_000),
            )
            selected_by_component[0] = selected

        selected = selected[: self.config.k]
        weight = self.config.lambda2 / max(1, len(selected))
        node_to_component = {
            int(node): int(comp_id)
            for comp_id, nodes in selected_by_component.items()
            for node in nodes
        }

        anchors: List[DistributedAnchor] = []
        for node in selected:
            distance = context.candidate_distances.get(int(node), 10**9)
            anchors.append(
                DistributedAnchor(
                    node=int(node),
                    weight=weight,
                    target_kind="h_new" if distance == 1 else "h_orig",
                    cached_distance=int(distance),
                    component_id=node_to_component.get(int(node)),
                )
            )
        context.final_min_distance = int(final_min)
        return anchors

    def clear_distance_cache(self) -> None:
        self.distance_computer.clear()

    def _strategy(self) -> str:
        strategy = str(self.config.strategy or "distributed").lower()
        allowed = {"hubscore", "proximity_weighted", "privacy_constrained", "distributed"}
        if strategy not in allowed:
            raise ValueError(f"Unsupported DAR strategy: {self.config.strategy!r}")
        return strategy

    def _component_nodes_after_deletion(self, graph_after: nx.Graph) -> Dict[int, List[int]]:
        components = [
            sorted(int(node) for node in component)
            for component in nx.connected_components(graph_after)
            if len(component) >= self.config.small_component_threshold
        ]
        if not components:
            nodes = sorted(int(node) for node in graph_after.nodes)
            return {0: nodes} if nodes else {}
        components.sort(key=len, reverse=True)
        return {idx: component for idx, component in enumerate(components)}

    def _component_quotas(self, component_nodes: Mapping[int, Sequence[int]]) -> Dict[int, int]:
        k = max(1, int(self.config.k))
        components = [(int(cid), len(nodes)) for cid, nodes in component_nodes.items() if nodes]
        if not components:
            return {0: k}
        components.sort(key=lambda item: item[1], reverse=True)
        if len(components) >= k:
            return {cid: 1 for cid, _ in components[:k]}

        quotas = {cid: 1 for cid, _ in components}
        remaining = k - len(components)
        total = float(sum(size for _, size in components)) or 1.0
        remainders: List[tuple[float, int]] = []
        for cid, size in components:
            raw_extra = remaining * (float(size) / total)
            extra = int(math.floor(raw_extra))
            quotas[cid] += extra
            remainders.append((raw_extra - extra, cid))
        used = sum(quotas.values())
        for _, cid in sorted(remainders, reverse=True):
            if used >= k:
                break
            quotas[cid] += 1
            used += 1
        while sum(quotas.values()) > k:
            largest = max(quotas, key=lambda cid: quotas[cid])
            quotas[largest] -= 1
        return quotas

    def _candidate_pool(
        self,
        candidate_nodes: Iterable[int],
        hub_scores: Mapping[int, float],
        distances: Mapping[int, int],
        excluded: set[int],
        *,
        min_distance: int,
        require_hub_percentile: bool,
    ) -> List[int]:
        candidates = [
            int(node)
            for node in candidate_nodes
            if int(node) not in excluded and distances.get(int(node), 10**9) >= min_distance
        ]
        if require_hub_percentile and candidates:
            threshold = self._percentile([hub_scores.get(node, 0.0) for node in candidates], 70.0)
            filtered = [node for node in candidates if hub_scores.get(node, 0.0) >= threshold]
            if filtered:
                candidates = filtered
        return self._unique(candidates)

    def _select_by_strategy(
        self,
        graph: nx.Graph,
        candidates: Sequence[int],
        hub_scores: Mapping[int, float],
        distances: Mapping[int, int],
        deleted_node: int,
        count: int,
        strategy: str,
        *,
        rng: random.Random,
    ) -> List[int]:
        candidates = [int(node) for node in candidates if int(node) in graph]
        if not candidates or count <= 0:
            return []
        if strategy == "hubscore":
            return sorted(candidates, key=lambda node: hub_scores.get(node, 0.0), reverse=True)[:count]
        if strategy == "proximity_weighted":
            return sorted(
                candidates,
                key=lambda node: hub_scores.get(node, 0.0) / max(1.0, float(distances.get(node, 10**6))),
                reverse=True,
            )[:count]
        if strategy == "privacy_constrained":
            filtered = [node for node in candidates if distances.get(node, 10**9) >= self.config.min_distance]
            if len(filtered) < count:
                filtered = candidates
            return sorted(filtered, key=lambda node: hub_scores.get(node, 0.0), reverse=True)[:count]
        selected, _ = self._distributed_select(graph, candidates, hub_scores, count, rng=rng)
        return selected

    def _distributed_select(
        self,
        graph: nx.Graph,
        candidates: Sequence[int],
        hub_scores: Mapping[int, float],
        count: int,
        *,
        rng: random.Random,
    ) -> tuple[List[int], int]:
        all_candidates = self._unique(node for node in candidates if int(node) in graph)
        selected: List[int] = []
        current_min_dist = int(self.config.min_distance)
        if not all_candidates or count <= 0:
            return selected, current_min_dist

        while len(selected) < count and current_min_dist >= 0:
            made_progress = True
            while len(selected) < count and made_progress:
                available = [
                    node
                    for node in all_candidates
                    if node not in selected
                    and all(
                        self._selection_distance(graph, node, chosen) >= current_min_dist
                        for chosen in selected
                    )
                ]
                if not available:
                    made_progress = False
                    break
                scored = self._score_distributed_candidates(graph, available, selected, hub_scores, rng)
                chosen = max(scored, key=lambda item: item[1])[0]
                selected.append(int(chosen))
            if len(selected) >= count:
                break
            current_min_dist -= 1

        if len(selected) < count:
            for node in all_candidates:
                if node not in selected:
                    selected.append(int(node))
                if len(selected) >= count:
                    break
        return selected[:count], max(0, current_min_dist)

    def _score_distributed_candidates(
        self,
        graph: nx.Graph,
        candidates: Sequence[int],
        selected: Sequence[int],
        hub_scores: Mapping[int, float],
        rng: random.Random,
    ) -> List[tuple[int, float]]:
        hub_norm = self._normalize({node: hub_scores.get(node, 0.0) for node in candidates})
        if selected:
            diversity_raw = {
                node: min(
                    self._selection_distance(graph, node, chosen)
                    for chosen in selected
                )
                for node in candidates
            }
            cap = max(1, graph.number_of_nodes())
            diversity_raw = {node: min(value, cap) for node, value in diversity_raw.items()}
            div_norm = self._normalize(diversity_raw)
        else:
            div_norm = {int(node): 1.0 for node in candidates}
        return [
            (
                int(node),
                self.config.alpha_score * hub_norm.get(node, 0.0)
                + self.config.beta_score * div_norm.get(node, 0.0)
                + self._gumbel(rng),
            )
            for node in candidates
        ]

    def _gumbel(self, rng: random.Random) -> float:
        tau = max(0.0, float(self.config.gumbel_tau))
        if tau == 0.0:
            return 0.0
        u = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
        return -math.log(-math.log(u)) * tau

    def _rng(self, deleted_node: int, salt: int) -> random.Random:
        return random.Random(int(self.config.seed) + int(deleted_node) * 1_000_003 + int(salt))

    @staticmethod
    def _normalize(values: Mapping[int, float]) -> Dict[int, float]:
        if not values:
            return {}
        min_value = min(float(value) for value in values.values())
        max_value = max(float(value) for value in values.values())
        if max_value == min_value:
            return {int(node): 0.0 for node in values}
        return {
            int(node): (float(value) - min_value) / (max_value - min_value)
            for node, value in values.items()
        }

    @staticmethod
    def _percentile(values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percentile / 100.0
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    @staticmethod
    def _unique(nodes: Iterable[int]) -> List[int]:
        seen: set[int] = set()
        result: List[int] = []
        for node in nodes:
            node = int(node)
            if node in seen:
                continue
            seen.add(node)
            result.append(node)
        return result

    def _safe_distance(self, graph: nx.Graph, source: int, target: int) -> int:
        return self.distance_computer.distance(graph, source, target)

    def _selection_distance(self, graph: nx.Graph, candidate: int, selected: int) -> int:
        if graph.is_directed():
            return self._safe_distance(graph, candidate, selected)
        # Undirected distance is symmetric. Querying from the few selected
        # anchors lets the exact BFS cache serve every candidate.
        return self._safe_distance(graph, selected, candidate)

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set

import networkx as nx

from hasi.graph_compute import TorchSparseGraph, resolve_backend


class PPRComputer:
    """Personalized PageRank based ERF approximation."""

    def __init__(
        self,
        alpha: float = 0.15,
        k_steps: int = 3,
        threshold: float = 0.01,
        *,
        backend: str = "auto",
        device: Optional[str] = None,
    ):
        self.alpha = alpha
        self.k_steps = k_steps
        self.threshold = threshold
        self.backend = backend
        self.device = device
        self.last_diagnostics: Dict[str, Any] = {}
        self.last_backend_diagnostics: Dict[str, Any] = {}
        self._torch_graphs: Dict[tuple[nx.Graph, int, int], TorchSparseGraph] = {}

    def compute_seed_ppr(self, graph: nx.Graph, seeds: Iterable[int]) -> Dict[int, float]:
        seeds = list(dict.fromkeys(int(seed) for seed in seeds if int(seed) in graph))
        if not seeds:
            self.last_backend_diagnostics = {
                "requested_backend": str(self.backend),
                "used_backend": "none",
                "device": str(self.device or "none"),
                "fallback_reason": "no valid seeds",
            }
            return {}

        decision = resolve_backend(self.backend, self.device)
        if decision.used == "torch":
            try:
                sparse_graph = self._torch_graph(graph, decision.device)
                rank_tensor = sparse_graph.ppr_steps(seeds, self.alpha, self.k_steps)
                values = rank_tensor.detach().cpu().tolist()
                self.last_backend_diagnostics = decision.as_dict()
                return {int(node): float(value) for node, value in zip(sparse_graph.nodes, values)}
            except (RuntimeError, NotImplementedError) as exc:
                if decision.requested != "auto":
                    raise
                self.last_backend_diagnostics = {
                    "requested_backend": "auto",
                    "used_backend": "cpu",
                    "device": "cpu",
                    "fallback_reason": f"torch PPR failed: {type(exc).__name__}: {exc}",
                }
                return self._compute_seed_ppr_cpu(graph, seeds)

        self.last_backend_diagnostics = decision.as_dict()
        return self._compute_seed_ppr_cpu(graph, seeds)

    def _compute_seed_ppr_cpu(self, graph: nx.Graph, seeds: list[int]) -> Dict[int, float]:

        rank = {node: 0.0 for node in graph.nodes}
        seed_mass = 1.0 / len(seeds)
        for seed in seeds:
            rank[seed] = seed_mass

        for _ in range(self.k_steps):
            next_rank = {node: 0.0 for node in graph.nodes}
            for node, value in rank.items():
                neighbors = list(graph.neighbors(node))
                if not neighbors:
                    next_rank[node] += (1.0 - self.alpha) * value
                    continue
                share = (1.0 - self.alpha) * value / len(neighbors)
                for neighbor in neighbors:
                    next_rank[neighbor] += share
            for seed in seeds:
                next_rank[seed] += self.alpha * seed_mass
            rank = next_rank

        return rank

    def _torch_graph(self, graph: nx.Graph, device: str) -> TorchSparseGraph:
        key = (graph, graph.number_of_nodes(), graph.number_of_edges())
        sparse_graph = self._torch_graphs.get(key)
        if sparse_graph is None or str(sparse_graph.device) != str(device):
            sparse_graph = TorchSparseGraph(graph, device)
            self._torch_graphs[key] = sparse_graph
        return sparse_graph

    def affected_region(
        self,
        graph: nx.Graph,
        seeds: Iterable[int],
        *,
        excluded_nodes: Optional[Iterable[int]] = None,
    ) -> Set[int]:
        """Return a non-degenerate, deterministic PPR-affected region.

        Multi-target requests dilute each seed's probability mass by ``1 / |S|``.
        A fixed absolute threshold can therefore return an empty region solely
        because the request contains many targets. We preserve the threshold as
        the primary selector and use the highest-ranked eligible nodes only when
        it selects fewer than one node per valid seed.
        """

        valid_seeds = list(dict.fromkeys(int(seed) for seed in seeds if int(seed) in graph))
        excluded = {int(node) for node in (excluded_nodes or [])}
        ppr = self.compute_seed_ppr(graph, valid_seeds)
        ranked = sorted(
            (node for node in ppr if node not in excluded),
            key=lambda node: (-ppr[node], int(node)),
        )
        threshold_region = {node for node in ranked if ppr[node] > self.threshold}
        minimum_size = min(len(ranked), len(valid_seeds))
        fallback_used = len(threshold_region) < minimum_size

        if fallback_used:
            selected = set(ranked[:minimum_size])
        else:
            selected = threshold_region

        self.last_diagnostics = {
            "selection_policy": "absolute_threshold_with_ranked_minimum",
            "valid_seed_count": len(valid_seeds),
            "excluded_node_count": len(excluded.intersection(graph.nodes)),
            "threshold": float(self.threshold),
            "threshold_selected_count": len(threshold_region),
            "minimum_region_size": minimum_size,
            "fallback_used": fallback_used,
            "returned_region_size": len(selected),
            "compute_backend": dict(self.last_backend_diagnostics),
        }
        return selected

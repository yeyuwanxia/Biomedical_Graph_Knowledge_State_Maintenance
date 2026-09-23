from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Optional

import networkx as nx
import torch


@dataclass(frozen=True)
class BackendDecision:
    requested: str
    used: str
    device: str
    fallback_reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested_backend": self.requested,
            "used_backend": self.used,
            "device": self.device,
            "fallback_reason": self.fallback_reason,
        }


def resolve_backend(backend: str, device: Optional[str]) -> BackendDecision:
    requested = str(backend or "auto").lower()
    if requested not in {"auto", "cpu", "torch"}:
        raise ValueError(f"Unsupported graph compute backend: {backend!r}")

    requested_device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if requested == "cpu":
        return BackendDecision(requested=requested, used="cpu", device="cpu")
    if requested == "torch":
        torch_device = torch.device(requested_device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA graph backend requested but CUDA is unavailable: {requested_device}")
        return BackendDecision(requested=requested, used="torch", device=str(torch_device))

    torch_device = torch.device(requested_device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        return BackendDecision(requested=requested, used="torch", device=str(torch_device))
    reason = None
    if torch_device.type == "cuda":
        reason = "CUDA requested but unavailable; using NetworkX CPU backend"
    return BackendDecision(requested=requested, used="cpu", device="cpu", fallback_reason=reason)


class TorchSparseGraph:
    """Sparse adjacency representation preserving NetworkX neighbor semantics."""

    def __init__(self, graph: nx.Graph, device: str):
        self.nodes = list(graph.nodes)
        self.node_to_index = {node: idx for idx, node in enumerate(self.nodes)}
        self.device = torch.device(device)
        self.num_nodes = len(self.nodes)
        self.last_pagerank_iterations = 0
        self.last_eigenvector_iterations = 0
        self.last_personalized_pagerank_iterations = 0

        sources: list[int] = []
        targets: list[int] = []
        directed = graph.is_directed()
        for source, target in graph.edges:
            source_idx = self.node_to_index[source]
            target_idx = self.node_to_index[target]
            sources.append(source_idx)
            targets.append(target_idx)
            if not directed and source_idx != target_idx:
                sources.append(target_idx)
                targets.append(source_idx)

        if sources:
            source_tensor = torch.tensor(sources, dtype=torch.long, device=self.device)
            target_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
            indices = torch.stack([target_tensor, source_tensor], dim=0)
            values = torch.ones(len(sources), dtype=torch.float64, device=self.device)
            self.adjacency = torch.sparse_coo_tensor(
                indices,
                values,
                (self.num_nodes, self.num_nodes),
                dtype=torch.float64,
                device=self.device,
            ).coalesce()
            self.bfs_adjacency = self.adjacency.to(dtype=torch.float32)
            self.out_degree = torch.bincount(source_tensor, minlength=self.num_nodes).to(torch.float64)
        else:
            empty_indices = torch.empty((2, 0), dtype=torch.long, device=self.device)
            empty_values = torch.empty((0,), dtype=torch.float64, device=self.device)
            self.adjacency = torch.sparse_coo_tensor(
                empty_indices,
                empty_values,
                (self.num_nodes, self.num_nodes),
                dtype=torch.float64,
                device=self.device,
            ).coalesce()
            self.bfs_adjacency = self.adjacency.to(dtype=torch.float32)
            self.out_degree = torch.zeros(self.num_nodes, dtype=torch.float64, device=self.device)

    def ppr_steps(self, seeds: list[Hashable], alpha: float, k_steps: int) -> torch.Tensor:
        rank = torch.zeros(self.num_nodes, dtype=torch.float64, device=self.device)
        seed_indices = torch.tensor(
            [self.node_to_index[seed] for seed in seeds],
            dtype=torch.long,
            device=self.device,
        )
        seed_mass = 1.0 / len(seeds)
        rank[seed_indices] = seed_mass
        restart = torch.zeros_like(rank)
        restart[seed_indices] = float(alpha) * seed_mass
        dangling = self.out_degree == 0

        for _ in range(int(k_steps)):
            normalized = torch.zeros_like(rank)
            non_dangling = ~dangling
            normalized[non_dangling] = rank[non_dangling] / self.out_degree[non_dangling]
            propagated = torch.sparse.mm(self.adjacency, normalized.unsqueeze(1)).squeeze(1)
            next_rank = (1.0 - float(alpha)) * propagated + restart
            next_rank[dangling] += (1.0 - float(alpha)) * rank[dangling]
            rank = next_rank
        return rank

    def pagerank(
        self,
        *,
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        if self.num_nodes == 0:
            return torch.empty(0, dtype=torch.float64, device=self.device)
        rank = torch.full(
            (self.num_nodes, 1),
            1.0 / self.num_nodes,
            dtype=torch.float64,
            device=self.device,
        )
        teleport = torch.full_like(rank, 1.0 / self.num_nodes)
        dangling = self.out_degree == 0
        for iteration in range(1, int(max_iter) + 1):
            propagated = self._propagate(rank)
            dangling_mass = rank[dangling].sum() if bool(dangling.any().item()) else rank.new_tensor(0.0)
            next_rank = float(alpha) * (
                propagated + dangling_mass * teleport
            ) + (1.0 - float(alpha)) * teleport
            error = torch.sum(torch.abs(next_rank - rank))
            rank = next_rank
            if float(error.item()) < self.num_nodes * float(tol):
                self.last_pagerank_iterations = iteration
                return rank.squeeze(1)
        raise nx.PowerIterationFailedConvergence(int(max_iter))

    def eigenvector_centrality(
        self,
        *,
        max_iter: int = 200,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        if self.num_nodes == 0:
            return torch.empty(0, dtype=torch.float64, device=self.device)
        vector = torch.full(
            (self.num_nodes, 1),
            1.0 / self.num_nodes,
            dtype=torch.float64,
            device=self.device,
        )
        for iteration in range(1, int(max_iter) + 1):
            next_vector = vector + torch.sparse.mm(self.adjacency, vector)
            norm = torch.linalg.vector_norm(next_vector)
            if float(norm.item()) == 0.0:
                norm = next_vector.new_tensor(1.0)
            next_vector = next_vector / norm
            error = torch.sum(torch.abs(next_vector - vector))
            vector = next_vector
            if float(error.item()) < self.num_nodes * float(tol):
                self.last_eigenvector_iterations = iteration
                return vector.squeeze(1)
        raise nx.PowerIterationFailedConvergence(int(max_iter))

    def personalized_pagerank_off_diagonal_mass(
        self,
        seeds: list[Hashable],
        *,
        alpha: float = 0.15,
        max_iter: int = 50,
        tol: float = 1e-6,
        batch_size: int = 64,
    ) -> Dict[Hashable, float]:
        valid_seeds = list(dict.fromkeys(seed for seed in seeds if seed in self.node_to_index))
        if not valid_seeds:
            return {}
        if self.num_nodes == 0:
            return {}

        scores: Dict[Hashable, float] = {}
        max_iterations_used = 0
        dangling = self.out_degree == 0
        for start in range(0, len(valid_seeds), max(1, int(batch_size))):
            batch = valid_seeds[start : start + max(1, int(batch_size))]
            seed_indices = torch.tensor(
                [self.node_to_index[seed] for seed in batch],
                dtype=torch.long,
                device=self.device,
            )
            columns = torch.arange(len(batch), dtype=torch.long, device=self.device)
            rank = torch.full(
                (self.num_nodes, len(batch)),
                1.0 / self.num_nodes,
                dtype=torch.float64,
                device=self.device,
            )
            teleport = torch.zeros_like(rank)
            teleport[seed_indices, columns] = 1.0
            active = torch.ones(len(batch), dtype=torch.bool, device=self.device)

            for iteration in range(1, int(max_iter) + 1):
                propagated = self._propagate(rank)
                next_rank = float(alpha) * propagated + (1.0 - float(alpha)) * teleport
                if bool(dangling.any().item()):
                    dangling_mass = rank[dangling].sum(dim=0)
                    next_rank[seed_indices, columns] += float(alpha) * dangling_mass
                error = torch.sum(torch.abs(next_rank - rank), dim=0)
                rank[:, active] = next_rank[:, active]
                active &= error >= self.num_nodes * float(tol)
                if not bool(active.any().item()):
                    max_iterations_used = max(max_iterations_used, iteration)
                    break
            else:
                raise nx.PowerIterationFailedConvergence(int(max_iter))

            total_mass = rank.sum(dim=0)
            self_mass = rank[seed_indices, columns]
            off_diagonal = (total_mass - self_mass).detach().cpu().tolist()
            scores.update({seed: float(value) for seed, value in zip(batch, off_diagonal)})

        self.last_personalized_pagerank_iterations = max_iterations_used
        return scores

    def _propagate(self, values: torch.Tensor) -> torch.Tensor:
        normalized = torch.zeros_like(values)
        non_dangling = self.out_degree > 0
        normalized[non_dangling] = values[non_dangling] / self.out_degree[non_dangling].unsqueeze(1)
        return torch.sparse.mm(self.adjacency, normalized)

    def bfs(self, source: Hashable, cutoff: Optional[int] = None) -> Dict[Hashable, int]:
        if source not in self.node_to_index:
            raise nx.NodeNotFound(f"Source {source!r} is not in graph")
        if self.num_nodes == 0:
            return {}

        source_idx = self.node_to_index[source]
        visited = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        frontier = torch.zeros_like(visited)
        distances = torch.full((self.num_nodes,), -1, dtype=torch.int64, device=self.device)
        visited[source_idx] = True
        frontier[source_idx] = True
        distances[source_idx] = 0
        depth = 0

        while bool(frontier.any().item()) and (cutoff is None or depth < int(cutoff)):
            reached = torch.sparse.mm(
                self.bfs_adjacency,
                frontier.to(torch.float32).unsqueeze(1),
            ).squeeze(1) > 0
            frontier = reached & ~visited
            if not bool(frontier.any().item()):
                break
            depth += 1
            distances[frontier] = depth
            visited |= frontier

        distances_cpu = distances.cpu().tolist()
        return {
            node: int(distance)
            for node, distance in zip(self.nodes, distances_cpu)
            if int(distance) >= 0
        }


class ShortestPathComputer:
    """Exact unweighted shortest paths with cached CPU or torch BFS results."""

    def __init__(self, backend: str = "auto", device: Optional[str] = None):
        self.decision = resolve_backend(backend, device)
        self._graph_keys: Dict[nx.Graph, tuple[nx.Graph, int, int]] = {}
        self._torch_graphs: Dict[tuple[nx.Graph, int, int], TorchSparseGraph] = {}
        self._distance_cache: Dict[
            tuple[nx.Graph, int, int, Hashable, Optional[int]],
            Dict[Hashable, int],
        ] = {}
        self.last_diagnostics: Dict[str, Any] = self.decision.as_dict()

    def clear(self) -> None:
        self._graph_keys.clear()
        self._torch_graphs.clear()
        self._distance_cache.clear()

    def single_source(
        self,
        graph: nx.Graph,
        source: Hashable,
        cutoff: Optional[int] = None,
    ) -> Dict[Hashable, int]:
        graph_key = self._graph_key(graph)
        key = (*graph_key, source, cutoff)
        cached = self._distance_cache.get(key)
        if cached is not None:
            self.last_diagnostics = self.decision.as_dict() | {"cache_hit": True}
            return dict(cached)

        try:
            if self.decision.used == "torch":
                distances = self._torch_graph(graph).bfs(source, cutoff=cutoff)
            else:
                distances = dict(nx.single_source_shortest_path_length(graph, source, cutoff=cutoff))
        except (RuntimeError, NotImplementedError) as exc:
            if self.decision.requested != "auto":
                raise
            distances = dict(nx.single_source_shortest_path_length(graph, source, cutoff=cutoff))
            self.decision = BackendDecision(
                requested="auto",
                used="cpu",
                device="cpu",
                fallback_reason=f"torch BFS failed: {type(exc).__name__}: {exc}",
            )

        normalized = {node: int(distance) for node, distance in distances.items()}
        self._distance_cache[key] = normalized
        self.last_diagnostics = self.decision.as_dict() | {"cache_hit": False}
        return dict(normalized)

    def distance(self, graph: nx.Graph, source: Hashable, target: Hashable) -> int:
        if source not in graph or target not in graph:
            return 10**9
        graph_key = self._graph_key(graph)
        key = (*graph_key, source, None)
        cached = self._distance_cache.get(key)
        if cached is None:
            self.single_source(graph, source)
            cached = self._distance_cache.get(key, {})
        else:
            self.last_diagnostics = self.decision.as_dict() | {"cache_hit": True}
        return int(cached.get(target, 10**9))

    def _torch_graph(self, graph: nx.Graph) -> TorchSparseGraph:
        key = self._graph_key(graph)
        sparse_graph = self._torch_graphs.get(key)
        if sparse_graph is None:
            sparse_graph = TorchSparseGraph(graph, self.decision.device)
            self._torch_graphs[key] = sparse_graph
        return sparse_graph

    def _graph_key(self, graph: nx.Graph) -> tuple[nx.Graph, int, int]:
        key = self._graph_keys.get(graph)
        if key is None:
            key = (graph, graph.number_of_nodes(), graph.number_of_edges())
            self._graph_keys[graph] = key
        return key

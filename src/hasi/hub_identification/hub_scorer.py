from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set

import networkx as nx

from hasi.graph_compute import BackendDecision, TorchSparseGraph, resolve_backend


ScoreMap = Dict[int, float]


@dataclass(frozen=True)
class HubScoreConfig:
    """Weights and ratios for HASI hub identification."""

    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3
    filter_ratio: float = 0.1
    primary_ratio: float = 0.01
    secondary_ratio: float = 0.05
    ppr_alpha: float = 0.15
    ppr_max_iter: int = 50
    ppr_tol: float = 1e-6
    compute_backend: str = "cpu"
    compute_device: Optional[str] = None
    ppr_batch_size: int = 64


class HubScorer:
    """Compute HASI HubScore for graph nodes.

    The implementation is intentionally dependency-light: topology scores use
    NetworkX, and model-aware gradient scores can be supplied by the caller.
    If no gradient scores are supplied, the gradient term is treated as zero
    rather than silently inventing task information.
    """

    def __init__(self, config: Optional[HubScoreConfig] = None):
        self.config = config or HubScoreConfig()
        self.backend_decision = resolve_backend(
            self.config.compute_backend, self.config.compute_device
        )
        self._torch_graphs: Dict[tuple[nx.Graph, int, int], TorchSparseGraph] = {}
        self.last_diagnostics: Dict[str, object] = self.backend_decision.as_dict()

    def compute_hub_scores(
        self,
        graph: nx.Graph,
        gradient_scores: Optional[Mapping[int, float]] = None,
        candidate_nodes: Optional[Iterable[int]] = None,
    ) -> ScoreMap:
        """Return weighted HubScore for every node in the graph.

        Filter-and-refine is applied by first ranking nodes with a centrality
        score, then computing ERF/PPR influence only for the retained set.
        Non-candidate nodes remain in the returned map with score 0.0 so
        downstream anchor classification can stay total over V.
        """

        nodes = list(candidate_nodes) if candidate_nodes is not None else list(graph.nodes)
        if not nodes:
            return {}

        centrality = self.compute_centrality_scores(graph, nodes)
        retained = self._retain_candidates(centrality)
        gradient = self._normalize_dict(dict(gradient_scores or {}), retained)
        erf = self.compute_erf_influence(graph, retained)

        scores = {int(node): 0.0 for node in graph.nodes}
        for node in retained:
            scores[int(node)] = (
                self.config.alpha * gradient.get(node, 0.0)
                + self.config.beta * centrality.get(node, 0.0)
                + self.config.gamma * erf.get(node, 0.0)
            )
        return scores

    def classify_anchors(self, scores: Mapping[int, float]) -> tuple[Set[int], Set[int], Set[int]]:
        """Split nodes into Primary, Secondary, and Regular anchors."""

        ranked = sorted(scores, key=lambda node: scores[node], reverse=True)
        n_nodes = len(ranked)
        if n_nodes == 0:
            return set(), set(), set()

        primary_count = max(1, int(round(n_nodes * self.config.primary_ratio)))
        secondary_count = max(primary_count, int(round(n_nodes * self.config.secondary_ratio)))

        primary = set(ranked[:primary_count])
        secondary = set(ranked[primary_count:secondary_count])
        regular = set(ranked[secondary_count:])
        return primary, secondary, regular

    def compute_centrality_scores(self, graph: nx.Graph, nodes: Optional[Sequence[int]] = None) -> ScoreMap:
        """Compute normalized topology centrality score.

        The default follows the HASI document: PageRank, degree, and
        eigenvector-style centrality. Eigenvector centrality can fail to
        converge on some graphs, so PageRank is used as the stable fallback.
        """

        nodes = list(nodes) if nodes is not None else list(graph.nodes)
        if not nodes:
            return {}

        degree = dict(graph.degree())
        if self.backend_decision.used == "torch" and graph.number_of_edges():
            try:
                sparse_graph = self._torch_graph(graph)
                pagerank_values = sparse_graph.pagerank(alpha=0.85, max_iter=100, tol=1e-6)
                eigen_values = sparse_graph.eigenvector_centrality(max_iter=200, tol=1e-6)
                pagerank = {
                    int(node): float(value)
                    for node, value in zip(sparse_graph.nodes, pagerank_values.detach().cpu().tolist())
                }
                eigen = {
                    int(node): float(value)
                    for node, value in zip(sparse_graph.nodes, eigen_values.detach().cpu().tolist())
                }
                self.last_diagnostics.update(
                    {
                        "centrality_backend": "torch_sparse",
                        "pagerank_iterations": sparse_graph.last_pagerank_iterations,
                        "eigenvector_iterations": sparse_graph.last_eigenvector_iterations,
                    }
                )
            except (RuntimeError, NotImplementedError, nx.NetworkXException) as exc:
                if self.backend_decision.requested != "auto":
                    raise
                self._fallback_to_cpu(exc)
                pagerank, eigen = self._centrality_cpu(graph)
        else:
            pagerank, eigen = self._centrality_cpu(graph)

        pr_norm = self._normalize_dict(pagerank, nodes)
        degree_norm = self._normalize_dict(degree, nodes)
        eigen_norm = self._normalize_dict(eigen, nodes)

        return {
            int(node): 0.6 * pr_norm.get(node, 0.0)
            + 0.3 * degree_norm.get(node, 0.0)
            + 0.1 * eigen_norm.get(node, 0.0)
            for node in nodes
        }

    def compute_erf_influence(self, graph: nx.Graph, nodes: Iterable[int]) -> ScoreMap:
        """Approximate ERF influence with personalized PageRank mass."""

        valid_nodes = [int(node) for node in nodes if node in graph]
        if self.backend_decision.used == "torch" and valid_nodes:
            try:
                sparse_graph = self._torch_graph(graph)
                raw_scores = sparse_graph.personalized_pagerank_off_diagonal_mass(
                    valid_nodes,
                    alpha=self.config.ppr_alpha,
                    max_iter=self.config.ppr_max_iter,
                    tol=self.config.ppr_tol,
                    batch_size=self.config.ppr_batch_size,
                )
                self.last_diagnostics.update(
                    {
                        "erf_backend": "torch_sparse_batched",
                        "erf_batch_size": int(self.config.ppr_batch_size),
                        "erf_max_iterations_used": sparse_graph.last_personalized_pagerank_iterations,
                        "erf_seed_count": len(valid_nodes),
                    }
                )
            except (RuntimeError, NotImplementedError, nx.NetworkXException) as exc:
                if self.backend_decision.requested != "auto":
                    raise
                self._fallback_to_cpu(exc)
                raw_scores = self._erf_cpu(graph, valid_nodes)
        else:
            raw_scores = self._erf_cpu(graph, valid_nodes)
        return self._normalize_dict(raw_scores, raw_scores.keys())

    def _torch_graph(self, graph: nx.Graph) -> TorchSparseGraph:
        key = (graph, graph.number_of_nodes(), graph.number_of_edges())
        sparse_graph = self._torch_graphs.get(key)
        if sparse_graph is None or str(sparse_graph.device) != str(self.backend_decision.device):
            sparse_graph = TorchSparseGraph(graph, self.backend_decision.device)
            self._torch_graphs[key] = sparse_graph
        return sparse_graph

    def _centrality_cpu(self, graph: nx.Graph) -> tuple[ScoreMap, ScoreMap]:
        pagerank = (
            nx.pagerank(graph, alpha=0.85)
            if graph.number_of_edges()
            else {int(node): 1.0 for node in graph.nodes}
        )
        try:
            eigen = nx.eigenvector_centrality(graph, max_iter=200)
        except (nx.NetworkXException, ZeroDivisionError):
            eigen = pagerank
        self.last_diagnostics["centrality_backend"] = "networkx_cpu"
        return pagerank, eigen

    def _erf_cpu(self, graph: nx.Graph, nodes: Iterable[int]) -> ScoreMap:
        raw_scores: ScoreMap = {}
        for node in nodes:
            ppr = nx.pagerank(
                graph,
                alpha=self.config.ppr_alpha,
                personalization={n: 1.0 if n == node else 0.0 for n in graph.nodes},
                max_iter=self.config.ppr_max_iter,
                tol=self.config.ppr_tol,
            )
            raw_scores[int(node)] = sum(ppr.values()) - ppr.get(node, 0.0)
        self.last_diagnostics.update(
            {
                "erf_backend": "networkx_cpu_per_seed",
                "erf_seed_count": len(raw_scores),
            }
        )
        return raw_scores

    def _fallback_to_cpu(self, exc: Exception) -> None:
        self.backend_decision = BackendDecision(
            requested="auto",
            used="cpu",
            device="cpu",
            fallback_reason=f"torch HubScore failed: {type(exc).__name__}: {exc}",
        )
        self.last_diagnostics = self.backend_decision.as_dict()

    def _retain_candidates(self, centrality: Mapping[int, float]) -> Set[int]:
        if self.config.filter_ratio >= 1.0:
            return set(centrality.keys())
        count = max(1, int(round(len(centrality) * self.config.filter_ratio)))
        ranked = sorted(centrality, key=lambda node: centrality[node], reverse=True)
        return set(ranked[:count])

    @staticmethod
    def _normalize_dict(values: Mapping[int, float], nodes: Iterable[int]) -> ScoreMap:
        nodes = list(nodes)
        if not nodes:
            return {}
        selected = {int(node): float(values.get(node, 0.0)) for node in nodes}
        min_value = min(selected.values())
        max_value = max(selected.values())
        if max_value == min_value:
            return {node: 0.0 for node in selected}
        return {node: (value - min_value) / (max_value - min_value) for node, value in selected.items()}

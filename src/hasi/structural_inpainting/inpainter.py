from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, Optional, Sequence

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class InpaintingDecision:
    should_inpaint: bool
    reason: str


@dataclass(frozen=True)
class InpaintingStats:
    method: str
    status: str
    candidates_scored: int = 0
    edges_added: int = 0
    repair_budget: int = 0
    train_edges: int = 0
    final_loss: Optional[float] = None


class StructuralInpainter:
    """Generative structural inpainting with a lightweight masked GAE backend.

    The backend follows the open graph-autoencoder pattern used by PyTorch
    Geometric's GAE model: a GCN encoder maps nodes to latent embeddings, and an
    inner-product decoder scores candidate edges. Training uses a MaskGAE-style
    masked edge reconstruction objective on the currently retained graph.
    """

    def __init__(
        self,
        mode: str = "full",
        method: str = "mgae",
        cc_drop_threshold: float = 0.30,
        min_damage_ratio: float = 0.10,
        hidden_channels: int = 64,
        embedding_channels: int = 32,
        train_epochs: int = 80,
        lr: float = 0.01,
        mask_ratio: float = 0.15,
        edge_threshold: float = 0.50,
        max_added_edges: int = 256,
        repair_ratio: float = 0.35,
        max_candidate_nodes: int = 512,
        max_candidate_edges: int = 20000,
        seed: int = 42,
        device: Optional[str] = None,
        timing_callback=None,
    ):
        self.mode = mode
        self.method = method
        self.cc_drop_threshold = cc_drop_threshold
        self.min_damage_ratio = min_damage_ratio
        self.hidden_channels = int(hidden_channels)
        self.embedding_channels = int(embedding_channels)
        self.train_epochs = int(train_epochs)
        self.lr = float(lr)
        self.mask_ratio = float(mask_ratio)
        self.edge_threshold = float(edge_threshold)
        self.max_added_edges = int(max_added_edges)
        self.repair_ratio = float(repair_ratio)
        self.max_candidate_nodes = int(max_candidate_nodes)
        self.max_candidate_edges = int(max_candidate_edges)
        self.seed = int(seed)
        self.device = device
        self.timing_callback = timing_callback
        self.last_stats = InpaintingStats(method=method, status="not_run")

    def should_trigger(
        self,
        graph_before: nx.Graph,
        graph_after: nx.Graph,
        affected_nodes: Iterable[int],
        has_hub_to_hub_deletion: bool = False,
    ) -> InpaintingDecision:
        if self.mode == "none":
            return InpaintingDecision(False, "inpainting disabled")
        if has_hub_to_hub_deletion:
            return InpaintingDecision(True, "hub-to-hub deletion")

        affected = [node for node in affected_nodes if node in graph_before]
        if not affected:
            return InpaintingDecision(False, "empty affected region")

        before_cc = nx.average_clustering(graph_before.subgraph(affected)) if len(affected) > 1 else 0.0
        after_cc = nx.average_clustering(graph_after.subgraph([node for node in affected if node in graph_after])) if len(affected) > 1 else 0.0
        if before_cc > 0 and (before_cc - after_cc) / before_cc > self.cc_drop_threshold:
            return InpaintingDecision(True, "clustering coefficient drop")

        if nx.number_connected_components(graph_after) > nx.number_connected_components(graph_before):
            return InpaintingDecision(True, "component fragmentation")

        return InpaintingDecision(False, "damage below threshold")

    def apply(
        self,
        graph: nx.Graph,
        affected_nodes: Iterable[int],
        *,
        graph_before: Optional[nx.Graph] = None,
        node_features=None,
        forbidden_edges: Optional[Iterable[tuple[int, int]]] = None,
        repair_edges: Optional[Iterable[tuple[int, int]]] = None,
    ) -> nx.Graph:
        """Return a repaired graph with predicted local bridge edges added."""

        if self.mode == "none":
            self.last_stats = InpaintingStats(method=self.method, status="disabled")
            return graph.copy()
        if self.method != "mgae":
            self.last_stats = InpaintingStats(method=self.method, status="unsupported_method")
            return graph.copy()

        backend = _MaskedGraphAutoencoderInpainter(
            hidden_channels=self.hidden_channels,
            embedding_channels=self.embedding_channels,
            epochs=self.train_epochs,
            lr=self.lr,
            mask_ratio=self.mask_ratio,
            edge_threshold=self.edge_threshold,
            max_added_edges=self.max_added_edges,
            repair_ratio=self.repair_ratio,
            max_candidate_nodes=self.max_candidate_nodes,
            max_candidate_edges=self.max_candidate_edges,
            seed=self.seed,
            device=self.device,
            timing_callback=self.timing_callback,
        )
        repaired, stats = backend.repair(
            graph,
            affected_nodes,
            graph_before=graph_before,
            node_features=node_features,
            forbidden_edges=forbidden_edges,
            repair_edges=repair_edges,
        )
        self.last_stats = stats
        return repaired


class _MaskedGraphAutoencoderInpainter:
    def __init__(
        self,
        *,
        hidden_channels: int,
        embedding_channels: int,
        epochs: int,
        lr: float,
        mask_ratio: float,
        edge_threshold: float,
        max_added_edges: int,
        repair_ratio: float,
        max_candidate_nodes: int,
        max_candidate_edges: int,
        seed: int,
        device: Optional[str],
        timing_callback=None,
    ):
        self.hidden_channels = hidden_channels
        self.embedding_channels = embedding_channels
        self.epochs = epochs
        self.lr = lr
        self.mask_ratio = mask_ratio
        self.edge_threshold = edge_threshold
        self.max_added_edges = max_added_edges
        self.repair_ratio = repair_ratio
        self.max_candidate_nodes = max_candidate_nodes
        self.max_candidate_edges = max_candidate_edges
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.timing_callback = timing_callback

    def _clock(self):
        if self.timing_callback is None:
            return 0.0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def repair(
        self,
        graph: nx.Graph,
        affected_nodes: Iterable[int],
        *,
        graph_before: Optional[nx.Graph],
        node_features,
        forbidden_edges: Optional[Iterable[tuple[int, int]]],
        repair_edges: Optional[Iterable[tuple[int, int]]],
    ) -> tuple[nx.Graph, InpaintingStats]:
        repaired = graph.copy()
        nodes = sorted(int(node) for node in repaired.nodes)
        if len(nodes) < 2 or repaired.number_of_edges() == 0:
            return repaired, InpaintingStats(self._method, "empty_graph")

        node_to_row = {node: row for row, node in enumerate(nodes)}
        row_to_node = {row: node for node, row in node_to_row.items()}
        features = _feature_matrix(repaired, nodes, node_features).to(self.device)
        positive_edges = _unique_edge_rows(repaired, node_to_row)
        if not positive_edges:
            return repaired, InpaintingStats(self._method, "no_positive_edges")

        train_start = self._clock()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        model = _MGAEEncoder(features.shape[1], self.hidden_channels, self.embedding_channels).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        edge_set = set(positive_edges)
        final_loss = 0.0

        for _ in range(max(1, self.epochs)):
            masked_edges, observed_edges = _mask_edges(positive_edges, self.mask_ratio, generator)
            if not masked_edges:
                masked_edges, observed_edges = positive_edges[:1], positive_edges[1:]
            negative_edges = _sample_negative_edges(len(nodes), edge_set, len(masked_edges), generator)
            if not negative_edges:
                continue

            edge_index = _edge_index(observed_edges, len(nodes), self.device)
            pos = torch.tensor(masked_edges, dtype=torch.long, device=self.device)
            neg = torch.tensor(negative_edges, dtype=torch.long, device=self.device)

            model.train()
            optimizer.zero_grad()
            z = model(features, edge_index)
            logits = torch.cat([_decode(z, pos), _decode(z, neg)], dim=0)
            labels = torch.cat(
                [
                    torch.ones(pos.shape[0], device=self.device),
                    torch.zeros(neg.shape[0], device=self.device),
                ],
                dim=0,
            )
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())

        if self.timing_callback is not None:
            self.timing_callback("support_scorer_train", self._clock() - train_start)
        score_start = self._clock()
        candidate_edges = self._candidate_edges(
            repaired,
            affected_nodes,
            node_to_row,
            forbidden_edges,
            repair_edges,
            generator,
        )
        repair_budget = self._repair_budget(
            graph_before,
            repaired,
            affected_nodes,
            forbidden_edges,
            repair_edges,
        )
        if not candidate_edges or repair_budget <= 0:
            if self.timing_callback is not None:
                self.timing_callback("support_candidates_score_add", self._clock() - score_start)
            return repaired, InpaintingStats(
                self._method,
                "no_candidates",
                candidates_scored=len(candidate_edges),
                repair_budget=repair_budget,
                train_edges=len(positive_edges),
                final_loss=final_loss,
            )

        model.eval()
        with torch.no_grad():
            full_edge_index = _edge_index(positive_edges, len(nodes), self.device)
            z = model(features, full_edge_index)
            candidate_rows = torch.tensor(
                [(node_to_row[u], node_to_row[v]) for u, v in candidate_edges],
                dtype=torch.long,
                device=self.device,
            )
            probabilities = torch.sigmoid(_decode(z, candidate_rows)).detach().cpu().tolist()

        scored = sorted(
            zip(candidate_edges, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        selected = [(u, v) for (u, v), score in scored if score >= self.edge_threshold][:repair_budget]
        if not selected and scored:
            selected = [scored[0][0]]

        for source, target in selected:
            repaired.add_edge(row_to_node[node_to_row[source]], row_to_node[node_to_row[target]])

        if self.timing_callback is not None:
            self.timing_callback("support_candidates_score_add", self._clock() - score_start)

        return repaired, InpaintingStats(
            self._method,
            "ok",
            candidates_scored=len(candidate_edges),
            edges_added=len(selected),
            repair_budget=repair_budget,
            train_edges=len(positive_edges),
            final_loss=final_loss,
        )

    @property
    def _method(self) -> str:
        return "mgae"

    def _candidate_edges(
        self,
        graph: nx.Graph,
        affected_nodes: Iterable[int],
        node_to_row: dict[int, int],
        forbidden_edges: Optional[Iterable[tuple[int, int]]],
        repair_edges: Optional[Iterable[tuple[int, int]]],
        generator: torch.Generator,
    ) -> list[tuple[int, int]]:
        affected = {int(node) for node in affected_nodes if int(node) in graph}
        for source, target in repair_edges or []:
            if int(source) in graph:
                affected.add(int(source))
            if int(target) in graph:
                affected.add(int(target))
        expanded = set(affected)
        for node in affected:
            expanded.update(int(neighbor) for neighbor in graph.neighbors(node))
        candidates_nodes = sorted(expanded)
        if len(candidates_nodes) > self.max_candidate_nodes:
            candidates_nodes = sorted(
                candidates_nodes,
                key=lambda node: graph.degree(node),
                reverse=True,
            )[: self.max_candidate_nodes]

        existing = _canonical_edge_set(graph.edges)
        forbidden = _canonical_edge_set(forbidden_edges or [])
        candidates: list[tuple[int, int]] = []
        for idx, source in enumerate(candidates_nodes):
            if source not in node_to_row:
                continue
            for target in candidates_nodes[idx + 1 :]:
                if target not in node_to_row:
                    continue
                edge = _canonical_edge(source, target)
                if edge in existing or edge in forbidden:
                    continue
                candidates.append(edge)

        if len(candidates) <= self.max_candidate_edges:
            return candidates
        order = torch.randperm(len(candidates), generator=generator)[: self.max_candidate_edges].tolist()
        return [candidates[idx] for idx in order]

    def _repair_budget(
        self,
        graph_before: Optional[nx.Graph],
        graph_after: nx.Graph,
        affected_nodes: Iterable[int],
        forbidden_edges: Optional[Iterable[tuple[int, int]]],
        repair_edges: Optional[Iterable[tuple[int, int]]],
    ) -> int:
        if graph_before is None:
            return max(1, min(self.max_added_edges, graph_after.number_of_edges() // 100 + 1))

        affected_before = {int(node) for node in affected_nodes if int(node) in graph_before}
        before_edges = _canonical_edge_set(graph_before.subgraph(affected_before).edges)
        affected_after = {node for node in affected_before if node in graph_after}
        after_edges = _canonical_edge_set(graph_after.subgraph(affected_after).edges)
        lost_edges = before_edges - after_edges

        explicit_edges = _canonical_edge_set(repair_edges or [])
        if explicit_edges:
            deleted_explicit_edges = {
                edge for edge in explicit_edges if graph_before.has_edge(*edge) and not graph_after.has_edge(*edge)
            }
            lost_edges.update(deleted_explicit_edges)

        if not lost_edges:
            return 0
        budget = max(1, int(round(len(lost_edges) * self.repair_ratio)))
        return min(self.max_added_edges, budget)


class _MGAEEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.lin2 = nn.Linear(hidden_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = _gcn_propagate(x, edge_index, self.lin1)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        return _gcn_propagate(h, edge_index, self.lin2)


def _gcn_propagate(x: torch.Tensor, edge_index: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
    num_nodes = x.shape[0]
    loops = torch.arange(num_nodes, device=x.device)
    source = torch.cat([edge_index[0], loops])
    target = torch.cat([edge_index[1], loops])
    support = layer(x)
    degree = torch.bincount(target, minlength=num_nodes).to(x.dtype).clamp(min=1.0)
    norm = degree[source].pow(-0.5) * degree[target].pow(-0.5)
    out = support.new_zeros((num_nodes, support.shape[1]))
    out.index_add_(0, target, support[source] * norm.unsqueeze(-1))
    return out


def _decode(z: torch.Tensor, edge_rows: torch.Tensor) -> torch.Tensor:
    return (z[edge_rows[:, 0]] * z[edge_rows[:, 1]]).sum(dim=-1)


def _feature_matrix(graph: nx.Graph, nodes: Sequence[int], node_features) -> torch.Tensor:
    if node_features is not None:
        if hasattr(node_features, "detach"):
            feature_tensor = node_features.detach().cpu().float()
        else:
            feature_tensor = torch.as_tensor(node_features, dtype=torch.float32)
        valid = [node for node in nodes if 0 <= node < feature_tensor.shape[0]]
        if len(valid) == len(nodes):
            return feature_tensor[valid]

    degree = torch.tensor([graph.degree(node) for node in nodes], dtype=torch.float32).unsqueeze(1)
    clustering = torch.tensor([nx.clustering(graph, node) for node in nodes], dtype=torch.float32).unsqueeze(1)
    bias = torch.ones((len(nodes), 1), dtype=torch.float32)
    if degree.numel() and float(degree.max()) > 0:
        degree = degree / degree.max()
    return torch.cat([bias, degree, clustering], dim=1)


def _unique_edge_rows(graph: nx.Graph, node_to_row: dict[int, int]) -> list[tuple[int, int]]:
    edges = []
    for source, target in graph.edges:
        if source not in node_to_row or target not in node_to_row:
            continue
        row_edge = _canonical_edge(node_to_row[int(source)], node_to_row[int(target)])
        edges.append(row_edge)
    return sorted(set(edges))


def _edge_index(edges: Sequence[tuple[int, int]], num_nodes: int, device: torch.device) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    directed = []
    for source, target in edges:
        if 0 <= source < num_nodes and 0 <= target < num_nodes:
            directed.append((source, target))
            directed.append((target, source))
    if not directed:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(directed, dtype=torch.long, device=device).t().contiguous()


def _mask_edges(
    positive_edges: Sequence[tuple[int, int]],
    mask_ratio: float,
    generator: torch.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    count = max(1, int(round(len(positive_edges) * mask_ratio)))
    count = min(count, len(positive_edges))
    order = torch.randperm(len(positive_edges), generator=generator).tolist()
    masked_ids = set(order[:count])
    masked = [edge for idx, edge in enumerate(positive_edges) if idx in masked_ids]
    observed = [edge for idx, edge in enumerate(positive_edges) if idx not in masked_ids]
    return masked, observed


def _sample_negative_edges(
    num_nodes: int,
    positive_edges: set[tuple[int, int]],
    count: int,
    generator: torch.Generator,
) -> list[tuple[int, int]]:
    if num_nodes < 2 or count <= 0:
        return []
    negatives: set[tuple[int, int]] = set()
    max_possible = num_nodes * (num_nodes - 1) // 2 - len(positive_edges)
    target_count = min(count, max(0, max_possible))
    attempts = 0
    while len(negatives) < target_count and attempts < target_count * 50 + 100:
        pair = torch.randint(0, num_nodes, (2,), generator=generator).tolist()
        attempts += 1
        if pair[0] == pair[1]:
            continue
        edge = _canonical_edge(pair[0], pair[1])
        if edge in positive_edges or edge in negatives:
            continue
        negatives.add(edge)
    return sorted(negatives)


def _canonical_edge(source: int, target: int) -> tuple[int, int]:
    source = int(source)
    target = int(target)
    return (source, target) if source <= target else (target, source)


def _canonical_edge_set(edges: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    return {_canonical_edge(source, target) for source, target in edges if int(source) != int(target)}

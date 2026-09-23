from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence

import networkx as nx
import torch
import torch.nn.functional as F

from evaluation.mia import PrivacyEvaluator
from data import clone_data
from hasi.anchor_stabilization import AnchorManager, AnchorSets, AnchorStabilizationLoss
from hasi.dar import DARConfig, DARPipeline
from hasi.erf_partitioning import PPRComputer
from hasi.hub_identification import HubScoreConfig, HubScorer
from hasi.structural_inpainting import StructuralInpainter


EDGE_FORGET_LOSS_MODES = frozenset({"original_kl", "uniform", "none"})
NODE_FORGET_LOSS_MODES = frozenset(
    {"original_kl", "post_removal_kl", "uniform", "none"}
)


@dataclass
class HASIConfig:
    hub_identification: HubScoreConfig = field(default_factory=HubScoreConfig)
    erf_alpha: float = 0.15
    erf_k_steps: int = 3
    erf_threshold: float = 0.01
    inpainting_mode: str = "full"
    inpainting_method: str = "mgae"
    inpainting_cc_drop_threshold: float = 0.30
    inpainting_min_damage_ratio: float = 0.10
    inpainting_hidden_channels: int = 64
    inpainting_embedding_channels: int = 32
    inpainting_train_epochs: int = 80
    inpainting_lr: float = 0.01
    inpainting_mask_ratio: float = 0.15
    inpainting_edge_threshold: float = 0.50
    inpainting_max_added_edges: int = 256
    inpainting_repair_ratio: float = 0.35
    inpainting_max_candidate_nodes: int = 512
    inpainting_max_candidate_edges: int = 20000
    dar: DARConfig = field(default_factory=DARConfig)
    anchor_mode: str = "hierarchical"
    anchor_lambda1: float = 2.0
    anchor_lambda2: float = 0.5
    finetune_epochs: int = 50
    finetune_lr: float = 0.01
    forget_weight: float = 0.1
    edge_forget_loss_mode: str = "original_kl"
    node_forget_loss_mode: str = "uniform"
    subgraph_finetune: bool = True
    subgraph_min_nodes: int = 5000
    feature_drift_threshold: float = 1e-6
    feature_anchor_to_h_new: bool = True
    graph_compute_backend: str = "auto"
    graph_compute_device: Optional[str] = None


class HASIUnlearner:
    """Coordinator for the HASI pipeline."""

    def __init__(self, model: Any = None, data: Any = None, graph: Optional[nx.Graph] = None, config: Optional[HASIConfig] = None):
        self.model = model
        self.data = data
        self.graph = graph or self._graph_from_data(data)
        self.config = config or HASIConfig()
        self.config.anchor_mode = str(self.config.anchor_mode).lower()
        if self.config.anchor_mode not in {"hierarchical", "none"}:
            raise ValueError(f"Unsupported anchor_mode: {self.config.anchor_mode!r}")
        self.config.edge_forget_loss_mode = str(self.config.edge_forget_loss_mode).lower()
        if self.config.edge_forget_loss_mode not in EDGE_FORGET_LOSS_MODES:
            raise ValueError(
                f"Unsupported edge_forget_loss_mode: {self.config.edge_forget_loss_mode!r}"
            )
        self.config.node_forget_loss_mode = str(self.config.node_forget_loss_mode).lower()
        if self.config.node_forget_loss_mode not in NODE_FORGET_LOSS_MODES:
            raise ValueError(
                f"Unsupported node_forget_loss_mode: {self.config.node_forget_loss_mode!r}"
            )
        self.hub_scorer = HubScorer(self.config.hub_identification)
        self.anchor_manager = AnchorManager(
            primary_ratio=self.config.hub_identification.primary_ratio,
            secondary_ratio=self.config.hub_identification.secondary_ratio,
        )
        self.ppr = PPRComputer(
            self.config.erf_alpha,
            self.config.erf_k_steps,
            self.config.erf_threshold,
            backend=self.config.graph_compute_backend,
            device=self.config.graph_compute_device,
        )
        self.inpainter = StructuralInpainter(
            mode=self.config.inpainting_mode,
            method=self.config.inpainting_method,
            cc_drop_threshold=self.config.inpainting_cc_drop_threshold,
            min_damage_ratio=self.config.inpainting_min_damage_ratio,
            hidden_channels=self.config.inpainting_hidden_channels,
            embedding_channels=self.config.inpainting_embedding_channels,
            train_epochs=self.config.inpainting_train_epochs,
            lr=self.config.inpainting_lr,
            mask_ratio=self.config.inpainting_mask_ratio,
            edge_threshold=self.config.inpainting_edge_threshold,
            max_added_edges=self.config.inpainting_max_added_edges,
            repair_ratio=self.config.inpainting_repair_ratio,
            max_candidate_nodes=self.config.inpainting_max_candidate_nodes,
            max_candidate_edges=self.config.inpainting_max_candidate_edges,
        )
        self.dar = DARPipeline(self.config.dar)
        self.privacy_evaluator = PrivacyEvaluator()
        self.hub_scores: Dict[int, float] = {}

    def preprocess(
        self,
        gradient_scores: Optional[Dict[int, float]] = None,
        *,
        precomputed_hub_scores: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        if self.graph is None:
            raise ValueError("HASIUnlearner needs a NetworkX graph or data with edge_index")
        if precomputed_hub_scores is None:
            self.hub_scores = self.hub_scorer.compute_hub_scores(self.graph, gradient_scores=gradient_scores)
            hub_score_source = "computed"
        else:
            expected_nodes = {int(node) for node in self.graph.nodes}
            provided_nodes = {int(node) for node in precomputed_hub_scores}
            if expected_nodes != provided_nodes:
                raise ValueError(
                    "Cached HubScore node set mismatch: "
                    f"expected {len(expected_nodes)}, received {len(provided_nodes)}"
                )
            self.hub_scores = {int(node): float(score) for node, score in precomputed_hub_scores.items()}
            hub_score_source = "cache"
        if self._anchor_enabled():
            anchors = self.anchor_manager.classify_from_scores(self.hub_scores)
        else:
            anchors = AnchorSets(regular=set(int(node) for node in self.graph.nodes))
            self.anchor_manager.anchors = anchors
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "anchor_mode": self.config.anchor_mode,
            "anchor_enabled": self._anchor_enabled(),
            "num_primary": len(anchors.primary),
            "num_secondary": len(anchors.secondary),
            "num_regular": len(anchors.regular),
            "hub_score_source": hub_score_source,
            "hub_score_compute": dict(self.hub_scorer.last_diagnostics),
            "graph_compute_backend": self.config.graph_compute_backend,
            "graph_compute_device": self.config.graph_compute_device,
        }

    def plan_node_unlearning(self, forget_nodes: Iterable[int]) -> Dict[str, Any]:
        self._ensure_preprocessed()
        forget_nodes = [int(node) for node in forget_nodes]
        if self._anchor_enabled():
            primary_forget = [node for node in forget_nodes if node in self.anchor_manager.anchors.primary]
        else:
            primary_forget = []
        affected_region = self.ppr.affected_region(
            self.graph,
            forget_nodes,
            excluded_nodes=forget_nodes,
        )
        affected_region_diagnostics = dict(self.ppr.last_diagnostics)

        dar_contexts = []
        if self._anchor_enabled() and self.config.dar.enabled:
            for node in primary_forget:
                dar_contexts.append(
                    self.dar.run_phase1(
                        self.graph,
                        node,
                        self.hub_scores,
                        excluded_nodes=self.anchor_manager.anchors.primary | self.anchor_manager.anchors.secondary,
                    )
                )

        return {
            "forget_nodes": forget_nodes,
            "primary_forget_nodes": primary_forget,
            "affected_region": sorted(affected_region),
            "affected_region_diagnostics": affected_region_diagnostics,
            "dar_enabled": bool(self._anchor_enabled() and self.config.dar.enabled),
            "dar_contexts": dar_contexts,
            "dar_anchors": [],
        }

    def unlearn_nodes(
        self,
        forget_nodes: Iterable[int],
        trainer: Any = None,
        finetune_epochs: Optional[int] = None,
        finetune_lr: Optional[float] = None,
        forget_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute node unlearning on the in-memory graph/data."""

        self._ensure_graph()
        plan = self.plan_node_unlearning(forget_nodes)
        forget_nodes = plan["forget_nodes"]

        graph_before = self.graph.copy()
        graph_after = graph_before.copy()
        node_repair_edges = self._node_repair_edges(graph_before, forget_nodes)
        graph_after.remove_nodes_from([node for node in forget_nodes if node in graph_after])
        data_after = self._data_without_nodes(self.data, forget_nodes) if self.data is not None else None

        decision = self.inpainter.should_trigger(
            graph_before,
            graph_after,
            plan["affected_region"],
            has_hub_to_hub_deletion=bool(plan["primary_forget_nodes"]),
        )
        if decision.should_inpaint:
            graph_after = self.inpainter.apply(
                graph_after,
                plan["affected_region"],
                graph_before=graph_before,
                node_features=getattr(data_after, "x", None),
                forbidden_edges=node_repair_edges,
                repair_edges=node_repair_edges,
            )

        data_after = self._replace_edge_index_from_graph(data_after, graph_after)
        dar_anchors = []
        self.dar.clear_distance_cache()
        for context in plan["dar_contexts"]:
            dar_anchors.extend(self.dar.run_phase2(context, graph_after))

        training = self._fine_tune(
            trainer,
            data_after,
            forget_nodes=forget_nodes,
            dar_anchors=dar_anchors,
            finetune_epochs=finetune_epochs,
            finetune_lr=finetune_lr,
            forget_weight=forget_weight,
            affected_region=plan["affected_region"],
            anchor_excluded_nodes=forget_nodes,
            forget_loss_mode=self.config.node_forget_loss_mode,
        )

        self.graph = graph_after
        self.data = data_after
        self.anchor_manager.remove_forgetting_targets(forget_nodes)
        return self._result(
            "node",
            forget_nodes,
            plan["affected_region"],
            training,
            inpainting={
                "triggered": decision.should_inpaint,
                "reason": decision.reason,
                "stats": self.inpainter.last_stats,
            },
            primary_forget_nodes=plan["primary_forget_nodes"],
            dar_contexts=plan["dar_contexts"],
            dar_anchors=dar_anchors,
            affected_region_diagnostics=plan["affected_region_diagnostics"],
        )

    def unlearn_edges(
        self,
        forget_edges: Iterable[tuple[int, int]],
        trainer: Any = None,
        finetune_epochs: Optional[int] = None,
        finetune_lr: Optional[float] = None,
        forget_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._ensure_preprocessed()
        forget_edges = [(int(u), int(v)) for u, v in forget_edges]
        affected_seeds = sorted({node for edge in forget_edges for node in edge})
        affected_region = sorted(self.ppr.affected_region(self.graph, affected_seeds))
        affected_region_diagnostics = dict(self.ppr.last_diagnostics)

        graph_before = self.graph.copy()
        graph_after = graph_before.copy()
        has_hub_to_hub = False
        hub_nodes = (
            self.anchor_manager.anchors.primary | self.anchor_manager.anchors.secondary
            if self._anchor_enabled()
            else set()
        )
        for source, target in forget_edges:
            has_hub_to_hub = has_hub_to_hub or (source in hub_nodes and target in hub_nodes)
            if graph_after.has_edge(source, target):
                graph_after.remove_edge(source, target)
        data_after = self._data_without_edges(self.data, forget_edges) if self.data is not None else None

        decision = self.inpainter.should_trigger(
            graph_before,
            graph_after,
            affected_region,
            has_hub_to_hub_deletion=has_hub_to_hub,
        )
        if decision.should_inpaint:
            graph_after = self.inpainter.apply(
                graph_after,
                affected_region,
                graph_before=graph_before,
                node_features=getattr(data_after, "x", None),
                forbidden_edges=forget_edges,
                repair_edges=forget_edges,
            )

        data_after = self._replace_edge_index_from_graph(data_after, graph_after)
        training = self._fine_tune(
            trainer,
            data_after,
            forget_nodes=affected_seeds,
            dar_anchors=[],
            finetune_epochs=finetune_epochs,
            finetune_lr=finetune_lr,
            forget_weight=forget_weight,
            affected_region=affected_region,
            anchor_excluded_nodes=[],
            forget_loss_mode=self.config.edge_forget_loss_mode,
        )

        self.graph = graph_after
        self.data = data_after
        return self._result(
            "edge",
            forget_edges,
            affected_region,
            training,
            inpainting={
                "triggered": decision.should_inpaint,
                "reason": decision.reason,
                "stats": self.inpainter.last_stats,
            },
            affected_region_diagnostics=affected_region_diagnostics,
        )

    def unlearn_features(
        self,
        forget_features: Iterable[int],
        trainer: Any = None,
        finetune_epochs: Optional[int] = None,
        finetune_lr: Optional[float] = None,
        forget_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._ensure_preprocessed()
        forget_features = [int(feature) for feature in forget_features]
        data_after = self._data_without_features(self.data, forget_features) if self.data is not None else None

        logits_orig = None
        anchor_snapshot = None
        affected_region = list(range(self.graph.number_of_nodes()))
        drift_stats: Dict[str, Any] = {"status": "not_computed"}
        if trainer is not None and self.data is not None and data_after is not None:
            logits_orig, embeddings_orig = trainer.predict_with_embeddings(self.data)
            _, embeddings_masked = trainer.predict_with_embeddings(data_after)
            affected_region, drift_stats = self._feature_affected_region(embeddings_orig, embeddings_masked)
            if self.config.feature_anchor_to_h_new:
                anchor_snapshot = self._mixed_anchor_snapshot(embeddings_orig, embeddings_masked, affected_region)

        training = self._fine_tune(
            trainer,
            data_after,
            forget_nodes=affected_region,
            dar_anchors=[],
            finetune_epochs=finetune_epochs,
            finetune_lr=finetune_lr,
            forget_weight=forget_weight,
            affected_region=affected_region,
            anchor_excluded_nodes=[],
            anchor_snapshot_embeddings=anchor_snapshot,
            logits_orig=logits_orig,
            forget_loss_mode="uniform",
        )

        self.data = data_after
        return self._result(
            "feature",
            forget_features,
            affected_region,
            training,
            inpainting={"triggered": False, "reason": "feature unlearning does not change topology"},
            feature_drift=drift_stats,
            anchor_target="h_new" if anchor_snapshot is not None else "h_orig",
        )

    def evaluate_privacy(self, *args, **kwargs):
        return self.privacy_evaluator.evaluate_from_logits(*args, **kwargs)

    def _fine_tune(
        self,
        trainer: Any,
        data_after: Any,
        forget_nodes: Sequence[int],
        dar_anchors: Sequence[Any],
        finetune_epochs: Optional[int],
        finetune_lr: Optional[float],
        forget_weight: Optional[float],
        *,
        affected_region: Optional[Sequence[int]] = None,
        anchor_excluded_nodes: Optional[Sequence[int]] = None,
        anchor_snapshot_embeddings: Optional[torch.Tensor] = None,
        logits_orig: Optional[torch.Tensor] = None,
        forget_loss_mode: str = "uniform",
    ) -> Optional[Dict[str, Any]]:
        if trainer is None or data_after is None:
            return None

        logits_before, embeddings_before = trainer.predict_with_embeddings(self.data)
        if logits_orig is None:
            logits_orig = logits_before
        logits_post_removal, embeddings_new = trainer.predict_with_embeddings(data_after)
        anchor_snapshot = anchor_snapshot_embeddings if anchor_snapshot_embeddings is not None else embeddings_before
        forget_loss_mode = str(forget_loss_mode).lower()
        if forget_loss_mode not in NODE_FORGET_LOSS_MODES:
            raise ValueError(f"Unsupported forget_loss_mode: {forget_loss_mode!r}")

        anchor_loss = None
        primary_nodes: list[int] = []
        secondary_nodes: list[int] = []
        if self._anchor_enabled():
            anchor_loss = AnchorStabilizationLoss(
                lambda1=self.config.anchor_lambda1,
                lambda2=self.config.anchor_lambda2,
            )
            excluded = {int(node) for node in (anchor_excluded_nodes or [])}
            primary_nodes = self._valid_nodes(self.anchor_manager.anchors.primary - excluded, anchor_snapshot)
            secondary_nodes = self._valid_nodes(self.anchor_manager.anchors.secondary - excluded, anchor_snapshot)
            anchor_loss.set_snapshots(anchor_snapshot, primary_nodes, secondary_nodes)

            for anchor in dar_anchors:
                node = int(anchor.node)
                if node >= embeddings_before.shape[0] or node >= embeddings_new.shape[0]:
                    continue
                target = embeddings_new[node] if anchor.target_kind == "h_new" else embeddings_before[node]
                anchor_loss.register_distributed_target(node, target, anchor.weight)

        active_forget_weight = self.config.forget_weight if forget_weight is None else float(forget_weight)
        valid_forget_nodes = self._valid_nodes(forget_nodes, logits_before)

        def extra_loss(logits: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
            loss = logits.sum() * 0.0
            if anchor_loss is not None:
                loss = loss + anchor_loss(embeddings)
            if active_forget_weight > 0 and valid_forget_nodes:
                if forget_loss_mode == "original_kl":
                    loss = loss + active_forget_weight * self._original_kl_forget_loss(logits, logits_orig, valid_forget_nodes)
                elif forget_loss_mode == "post_removal_kl":
                    loss = loss + active_forget_weight * self._original_kl_forget_loss(
                        logits,
                        logits_post_removal,
                        valid_forget_nodes,
                    )
                elif forget_loss_mode == "uniform":
                    loss = loss + active_forget_weight * self._uniform_forget_loss(logits, valid_forget_nodes)
                elif forget_loss_mode == "none":
                    pass
                else:
                    raise ValueError(f"Unsupported forget_loss_mode: {forget_loss_mode!r}")
            return loss

        adaptive_train_mask = self._adaptive_train_mask(data_after, affected_region)
        result = trainer.fine_tune(
            data_after,
            train_mask=adaptive_train_mask,
            epochs=finetune_epochs or self.config.finetune_epochs,
            lr=finetune_lr or self.config.finetune_lr,
            extra_loss_fn=extra_loss,
        )
        logits_after, embeddings_after = trainer.predict_with_embeddings(data_after)
        return {
            "fit": result.as_dict() if hasattr(result, "as_dict") else result,
            "logits_before_shape": list(logits_before.shape),
            "logits_after_shape": list(logits_after.shape),
            "embeddings_before_shape": list(embeddings_before.shape),
            "embeddings_after_shape": list(embeddings_after.shape),
            "forget_loss_mode": forget_loss_mode,
            "forget_loss_target": {
                "original_kl": "original_graph_logits",
                "post_removal_kl": "post_removal_logits",
                "uniform": "uniform_distribution",
                "none": "none",
            }[forget_loss_mode],
            "anchor_mode": self.config.anchor_mode,
            "anchor_enabled": self._anchor_enabled(),
            "primary_anchor_nodes": primary_nodes,
            "secondary_anchor_nodes": secondary_nodes,
            "adaptive_subgraph": adaptive_train_mask is not None,
            "adaptive_train_nodes": int(adaptive_train_mask.sum().item()) if adaptive_train_mask is not None else None,
        }

    def _anchor_enabled(self) -> bool:
        return self.config.anchor_mode != "none"

    def _ensure_preprocessed(self) -> None:
        self._ensure_graph()
        if not self.hub_scores:
            self.preprocess()

    def _ensure_graph(self) -> None:
        if self.graph is None:
            raise ValueError("HASIUnlearner needs a NetworkX graph or data with edge_index")

    def _adaptive_train_mask(self, data_after: Any, affected_region: Optional[Sequence[int]]) -> Optional[torch.Tensor]:
        if data_after is None or affected_region is None:
            return None
        num_nodes = int(getattr(data_after, "num_nodes", 0))
        if not self.config.subgraph_finetune or num_nodes <= self.config.subgraph_min_nodes:
            return None
        region_mask = torch.zeros(num_nodes, dtype=torch.bool)
        valid_nodes = [int(node) for node in affected_region if 0 <= int(node) < num_nodes]
        if not valid_nodes:
            return None
        region_mask[valid_nodes] = True
        if hasattr(data_after, "train_mask") and data_after.train_mask is not None:
            train_mask = data_after.train_mask
            if train_mask.dim() > 1:
                train_mask = train_mask[:, 0]
            train_mask = train_mask.detach().cpu().to(dtype=torch.bool)
            combined = train_mask & region_mask
            # Never train on validation/test labels merely because the affected
            # region has no retained training nodes. Returning None delegates to
            # the trainer's normal retained training mask.
            return combined if combined.sum().item() > 0 else None
        return region_mask

    def _feature_affected_region(self, embeddings_before: torch.Tensor, embeddings_after: torch.Tensor) -> tuple[list[int], Dict[str, Any]]:
        n_rows = min(int(embeddings_before.shape[0]), int(embeddings_after.shape[0]))
        if n_rows == 0:
            return [], {"status": "empty_embeddings"}
        drift = torch.linalg.norm(embeddings_before[:n_rows] - embeddings_after[:n_rows], dim=1)
        threshold = float(self.config.feature_drift_threshold)
        affected = torch.nonzero(drift > threshold, as_tuple=False).flatten().tolist()
        if not affected:
            affected = torch.nonzero(drift > 0, as_tuple=False).flatten().tolist()
        if not affected:
            affected = list(range(n_rows))
        return [int(node) for node in affected], {
            "status": "ok",
            "threshold": threshold,
            "affected_nodes": len(affected),
            "drift_mean": float(drift.mean().item()),
            "drift_max": float(drift.max().item()),
        }

    @staticmethod
    def _mixed_anchor_snapshot(
        embeddings_before: torch.Tensor,
        embeddings_after: torch.Tensor,
        h_new_nodes: Sequence[int],
    ) -> torch.Tensor:
        snapshot = embeddings_before.detach().clone()
        limit = min(snapshot.shape[0], embeddings_after.shape[0])
        valid_nodes = [int(node) for node in h_new_nodes if 0 <= int(node) < limit]
        if valid_nodes:
            snapshot[valid_nodes] = embeddings_after[valid_nodes].detach().clone()
        return snapshot

    @staticmethod
    def _valid_nodes(nodes: Iterable[int], embeddings: torch.Tensor) -> list[int]:
        limit = int(embeddings.shape[0])
        return sorted({int(node) for node in nodes if 0 <= int(node) < limit})

    @staticmethod
    def _uniform_forget_loss(logits: torch.Tensor, nodes: Sequence[int]) -> torch.Tensor:
        node_index = torch.tensor(list(nodes), dtype=torch.long, device=logits.device)
        selected = logits.index_select(0, node_index)
        log_prob = F.log_softmax(selected, dim=-1)
        target = torch.full_like(log_prob, 1.0 / log_prob.shape[-1])
        return F.kl_div(log_prob, target, reduction="batchmean")

    @staticmethod
    def _original_kl_forget_loss(logits: torch.Tensor, logits_orig: torch.Tensor, nodes: Sequence[int]) -> torch.Tensor:
        node_index = torch.tensor(list(nodes), dtype=torch.long, device=logits.device)
        selected = logits.index_select(0, node_index)
        orig = logits_orig.to(logits.device).index_select(0, node_index)
        log_prob = F.log_softmax(selected, dim=-1)
        target = F.softmax(orig, dim=-1)
        return F.kl_div(log_prob, target, reduction="batchmean")

    @staticmethod
    def _data_without_nodes(data: Any, nodes: Sequence[int]):
        data_after = clone_data(data)
        node_set = {int(node) for node in nodes}
        if hasattr(data_after, "x") and data_after.x is not None:
            data_after.x = data_after.x.clone()
            valid_nodes = [node for node in node_set if 0 <= node < data_after.x.shape[0]]
            if valid_nodes:
                data_after.x[valid_nodes] = 0
        if hasattr(data_after, "edge_index"):
            keep = _edge_keep_indices(data_after.edge_index, lambda u, v: u not in node_set and v not in node_set)
            _apply_edge_keep(data_after, keep)
        _disable_masks(data_after, node_set)
        return data_after

    @staticmethod
    def _node_repair_edges(graph: nx.Graph, forget_nodes: Sequence[int]) -> list[tuple[int, int]]:
        forget_set = {int(node) for node in forget_nodes}
        repair_edges: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for node in forget_set:
            if node not in graph:
                continue
            for source, target in graph.edges(node):
                source = int(source)
                target = int(target)
                if source in forget_set and target in forget_set:
                    continue
                edge = (source, target) if source <= target else (target, source)
                if edge in seen:
                    continue
                seen.add(edge)
                repair_edges.append((source, target))
        return repair_edges

    @staticmethod
    def _data_without_edges(data: Any, edges: Sequence[tuple[int, int]]):
        data_after = clone_data(data)
        edge_set = {(int(u), int(v)) for u, v in edges}
        edge_set |= {(v, u) for u, v in edge_set}
        if hasattr(data_after, "edge_index"):
            keep = _edge_keep_indices(data_after.edge_index, lambda u, v: (u, v) not in edge_set)
            _apply_edge_keep(data_after, keep)
        return data_after

    @staticmethod
    def _data_without_features(data: Any, features: Sequence[int]):
        data_after = clone_data(data)
        if data_after is None or not hasattr(data_after, "x"):
            return data_after
        data_after.x = data_after.x.clone()
        feature_ids = [idx for idx in {int(item) for item in features} if 0 <= idx < data_after.x.shape[1]]
        if feature_ids:
            data_after.x[:, feature_ids] = 0
        return data_after

    @staticmethod
    def _replace_edge_index_from_graph(data: Any, graph: Optional[nx.Graph]):
        if data is None or graph is None or not hasattr(data, "edge_index"):
            return data
        edges = []
        for source, target in graph.edges:
            edges.append((int(source), int(target)))
            edges.append((int(target), int(source)))
        if edges:
            data.edge_index = torch.tensor(edges, dtype=torch.long, device=data.edge_index.device).t().contiguous()
        else:
            data.edge_index = data.edge_index.new_empty((2, 0))
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = None
        return data

    def _result(
        self,
        kind: str,
        forget_targets: Any,
        affected_region: Sequence[int],
        training: Optional[Dict[str, Any]],
        **extra: Any,
    ) -> Dict[str, Any]:
        result = {
            "type": kind,
            "forget_targets": forget_targets,
            "affected_region_size": len(affected_region),
            "affected_region": list(affected_region),
            "training": training,
            "anchor_mode": self.config.anchor_mode,
            "anchor_enabled": self._anchor_enabled(),
            "num_nodes": self.graph.number_of_nodes() if self.graph is not None else None,
            "num_edges": self.graph.number_of_edges() if self.graph is not None else None,
        }
        result.update(extra)
        return result

    @staticmethod
    def _graph_from_data(data: Any) -> Optional[nx.Graph]:
        if data is None or not hasattr(data, "edge_index"):
            return None
        edge_index = data.edge_index
        if hasattr(edge_index, "detach"):
            edge_index = edge_index.detach().cpu().numpy()
        graph = nx.Graph()
        num_nodes = int(getattr(data, "num_nodes", 0))
        graph.add_nodes_from(range(num_nodes))
        for source, target in edge_index.T:
            graph.add_edge(int(source), int(target))
        return graph


def _edge_keep_indices(edge_index: torch.Tensor, keep_fn) -> torch.Tensor:
    pairs = edge_index.detach().cpu().t().tolist()
    keep = [idx for idx, (source, target) in enumerate(pairs) if keep_fn(int(source), int(target))]
    return torch.tensor(keep, dtype=torch.long, device=edge_index.device)


def _apply_edge_keep(data: Any, keep: torch.Tensor) -> None:
    num_edges = int(data.edge_index.shape[1])
    data.edge_index = data.edge_index[:, keep]
    if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.shape[0] == num_edges:
        data.edge_attr = data.edge_attr[keep]


def _disable_masks(data: Any, nodes: set[int]) -> None:
    for name in ("train_mask", "val_mask", "test_mask"):
        if not hasattr(data, name):
            continue
        mask = getattr(data, name)
        if mask is None:
            continue
        mask = mask.clone()
        valid_nodes = [node for node in nodes if 0 <= node < mask.shape[0]]
        if valid_nodes:
            mask[valid_nodes] = False
        setattr(data, name, mask)

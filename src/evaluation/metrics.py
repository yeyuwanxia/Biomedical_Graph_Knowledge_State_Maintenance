from __future__ import annotations

import heapq
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import networkx as nx
import numpy as np
import torch

from evaluation.mia import PrivacyEvaluator


EVALUATION_PROTOCOL_VERSION = "paper_eval_20260715_v1"
_CLUSTERING_GRAPH: Optional[nx.Graph] = None


def build_experiment_metrics(
    *,
    method: str,
    dataset: str,
    unlearning_type: str,
    data_before: Any,
    data_after: Any,
    logits_before: Optional[torch.Tensor],
    logits_after: Optional[torch.Tensor],
    embeddings_before: Optional[torch.Tensor],
    embeddings_after: Optional[torch.Tensor],
    forget_targets: Any,
    unlearn_time_seconds: Optional[float] = None,
    retrain_time_seconds: Optional[float] = None,
    online_wall_clock_seconds: Optional[float] = None,
    time_breakdown: Optional[dict[str, float]] = None,
    offline_preprocessing_seconds: Optional[float] = None,
    primary_anchor_nodes: Optional[Sequence[int]] = None,
    secondary_anchor_nodes: Optional[Sequence[int]] = None,
    mia_seed: int = 0,
    exact_retrain_logits=None,
    exact_retrain_embeddings=None,
    exact_retrain_reference: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    graph_before = graph_from_data(data_before)
    graph_after = graph_from_data(data_after)
    affected_nodes = infer_affected_nodes(
        unlearning_type, forget_targets, data_before, embeddings_before, embeddings_after, data_after=data_after
    )
    member_nodes = infer_member_nodes(unlearning_type, forget_targets, data_before)
    non_member_nodes = infer_non_member_nodes(
        data_before,
        member_nodes,
        len(member_nodes),
        seed=mia_seed,
        labels=getattr(data_before, "y", None) if data_before is not None else None,
    )

    metrics = {
        "evaluation_protocol": {
            "version": EVALUATION_PROTOCOL_VERSION,
            "medium_mia": "held_out_target_split",
            "edge_forgetting": "cosine_affinity_with_retained_control_v1",
            "exact_retrain_alignment": "test_mask_js_tv_v1",
            "feature_privacy": "not_applicable_global_feature_dimension_request",
        },
        "method": method,
        "dataset": dataset,
        "unlearning_type": unlearning_type,
        "forget_count": count_forget_targets(forget_targets),
        "utility": utility_metrics(data_before, data_after, logits_before, logits_after, "test_mask"),
        "validation_utility": utility_metrics(data_before, data_after, logits_before, logits_after, "val_mask"),
        "representation": representation_metrics(
            graph_before,
            embeddings_before,
            embeddings_after,
            affected_nodes,
            primary_anchor_nodes=primary_anchor_nodes,
            secondary_anchor_nodes=secondary_anchor_nodes,
        ),
        "structure": structural_metrics(
            graph_before,
            graph_after,
            excluded_nodes=member_nodes if unlearning_type == "node" else None,
        ),
        "privacy": privacy_metrics(
            logits_before,
            logits_after,
            embeddings_before,
            embeddings_after,
            graph_before,
            graph_after,
            member_nodes,
            non_member_nodes,
            unlearning_type,
            labels=getattr(data_before, "y", None) if data_before is not None else None,
        ),
        "efficiency": efficiency_metrics(
            unlearn_time_seconds,
            retrain_time_seconds,
            online_wall_clock_seconds=online_wall_clock_seconds,
            time_breakdown=time_breakdown,
            offline_preprocessing_seconds=offline_preprocessing_seconds,
        ),
    }
    if unlearning_type == "feature":
        metrics["feature_compliance"] = feature_compliance_metrics(
            data_before,
            data_after,
            forget_targets,
        )
    if unlearning_type == "edge":
        metrics["edge_forgetting"] = edge_forgetting_metrics(
            graph_before,
            graph_after,
            embeddings_before,
            embeddings_after,
            forget_targets,
            exact_retrain_embeddings=exact_retrain_embeddings,
            seed=mia_seed,
        )
        metrics["exact_retrain_alignment"] = exact_retrain_alignment_metrics(
            logits_before,
            logits_after,
            exact_retrain_logits,
            data_after,
            reference=exact_retrain_reference,
        )
    metrics["rq_summary"] = rq_summary(metrics)
    return metrics


def save_metrics(result: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(result), indent=2) + "\n", encoding="utf-8")
    return path


def default_metrics_path(
    root: str | Path,
    method: str,
    dataset: str,
    unlearning_type: str,
    ratio: float,
    *,
    selection: str | None = None,
    seed: int | str | None = None,
) -> Path:
    ratio_label = str(ratio).replace(".", "p")
    parts = [str(method), str(dataset), str(unlearning_type), f"r{ratio_label}"]
    if selection:
        parts.append(_path_label(selection))
    if seed is not None:
        parts.append(f"seed{seed}")
    name = "_".join(parts) + ".json"
    return default_results_dir(root, method) / name


def default_results_dir(root: str | Path, method: str) -> Path:
    normalized_method = _path_label(method)
    if normalized_method == "hasi":
        return Path(root) / "results" / "hasi"
    return Path(root) / "results" / "baselines" / normalized_method


def _path_label(value: str) -> str:
    label = str(value).strip().lower().replace(" ", "_")
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in label)


def utility_metrics(data_before, data_after, logits_before, logits_after, mask_name: str = "test_mask") -> dict[str, Any]:
    before = _classification_metrics(data_before, logits_before, mask_name)
    after = _classification_metrics(data_after, logits_after, mask_name)
    return {
        "accuracy_before": before["accuracy"],
        "accuracy_after": after["accuracy"],
        "accuracy_drop": _drop(before["accuracy"], after["accuracy"]),
        "f1_micro_before": before["f1_micro"],
        "f1_micro_after": after["f1_micro"],
        "f1_micro_drop": _drop(before["f1_micro"], after["f1_micro"]),
        "f1_macro_before": before["f1_macro"],
        "f1_macro_after": after["f1_macro"],
        "f1_macro_drop": _drop(before["f1_macro"], after["f1_macro"]),
        "num_eval_before": before["num_eval"],
        "num_eval_after": after["num_eval"],
    }


def representation_metrics(
    graph_before: nx.Graph,
    embeddings_before,
    embeddings_after,
    member_nodes: Sequence[int],
    *,
    primary_anchor_nodes: Optional[Sequence[int]] = None,
    secondary_anchor_nodes: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    if embeddings_before is None or embeddings_after is None:
        return {
            "embedding_l2_mean": None,
            "embedding_l2_median": None,
            "embedding_l2_max": None,
            "member_embedding_l2_mean": None,
            "neighbor_drift_mean": None,
            "neighbor_drift_max": None,
            "primary_anchor_drift_mean": None,
            "primary_anchor_drift_max": None,
            "secondary_anchor_drift_mean": None,
            "secondary_anchor_drift_max": None,
            "num_primary_anchor_nodes": 0,
            "num_secondary_anchor_nodes": 0,
            "status": "missing_embeddings",
        }

    before = _to_numpy(embeddings_before)
    after = _to_numpy(embeddings_after)
    n_rows = min(before.shape[0], after.shape[0])
    drift = np.linalg.norm(before[:n_rows] - after[:n_rows], axis=1)
    valid_members = [node for node in member_nodes if 0 <= node < n_rows]
    neighbor_nodes = sorted(
        {
            int(neighbor)
            for node in valid_members
            if node in graph_before
            for neighbor in graph_before.neighbors(node)
            if 0 <= int(neighbor) < n_rows
        }
    )
    primary_anchor_nodes = _valid_node_list(primary_anchor_nodes or [], n_rows)
    secondary_anchor_nodes = _valid_node_list(secondary_anchor_nodes or [], n_rows)
    return {
        "embedding_l2_mean": _safe_float(np.mean(drift)),
        "embedding_l2_median": _safe_float(np.median(drift)),
        "embedding_l2_max": _safe_float(np.max(drift)),
        "member_embedding_l2_mean": _safe_float(np.mean(drift[valid_members])) if valid_members else None,
        "neighbor_drift_mean": _safe_float(np.mean(drift[neighbor_nodes])) if neighbor_nodes else None,
        "neighbor_drift_max": _safe_float(np.max(drift[neighbor_nodes])) if neighbor_nodes else None,
        "primary_anchor_drift_mean": _safe_float(np.mean(drift[primary_anchor_nodes])) if primary_anchor_nodes else None,
        "primary_anchor_drift_max": _safe_float(np.max(drift[primary_anchor_nodes])) if primary_anchor_nodes else None,
        "secondary_anchor_drift_mean": _safe_float(np.mean(drift[secondary_anchor_nodes])) if secondary_anchor_nodes else None,
        "secondary_anchor_drift_max": _safe_float(np.max(drift[secondary_anchor_nodes])) if secondary_anchor_nodes else None,
        "num_member_nodes": len(valid_members),
        "num_neighbor_nodes": len(neighbor_nodes),
        "num_primary_anchor_nodes": len(primary_anchor_nodes),
        "num_secondary_anchor_nodes": len(secondary_anchor_nodes),
        "status": "ok",
    }


def edge_forgetting_metrics(
    graph_before: nx.Graph,
    graph_after: nx.Graph,
    embeddings_before,
    embeddings_after,
    forget_edges: Iterable[Sequence[int]],
    *,
    exact_retrain_embeddings=None,
    seed: int = 0,
) -> dict[str, Any]:
    valid_forget = _canonical_valid_edges(forget_edges, graph_before.number_of_nodes())
    present_after = sum(1 for source, target in valid_forget if graph_after.has_edge(source, target))
    base = {
        "status": "ok",
        "score_type": "cosine_affinity_0_1",
        "request_unit": "unique_undirected_edge",
        "forgotten_edge_count": len(valid_forget),
        "forgotten_edges_present_after": int(present_after),
        "request_applied": bool(valid_forget) and present_after == 0,
    }
    if not valid_forget:
        base["status"] = "missing_valid_forget_edges"
        return base
    if embeddings_before is None or embeddings_after is None:
        base["status"] = "missing_embeddings"
        return base

    before = _to_numpy(embeddings_before)
    after = _to_numpy(embeddings_after)
    row_limit = min(before.shape[0], after.shape[0])
    valid_forget = [
        edge for edge in valid_forget if edge[0] < row_limit and edge[1] < row_limit
    ]
    if not valid_forget:
        base["status"] = "forget_edges_out_of_embedding_range"
        return base

    forgotten_before = _edge_affinity_scores(before, valid_forget)
    forgotten_after = _edge_affinity_scores(after, valid_forget)
    forgotten_drop = forgotten_before - forgotten_after

    forgotten_set = set(valid_forget)
    retained_edges = sorted(
        {
            _canonical_edge(int(source), int(target))
            for source, target in graph_before.edges()
            if source != target
            and _canonical_edge(int(source), int(target)) not in forgotten_set
            and int(source) < row_limit
            and int(target) < row_limit
        }
    )
    control_count = min(len(valid_forget), len(retained_edges))
    if control_count:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(retained_edges), size=control_count, replace=False)
        control_edges = [retained_edges[int(index)] for index in np.sort(selected)]
        control_before = _edge_affinity_scores(before, control_edges)
        control_after = _edge_affinity_scores(after, control_edges)
        control_drop = control_before - control_after
    else:
        control_drop = np.asarray([], dtype=float)

    base.update(
        {
            "forgotten_edge_count_scored": len(valid_forget),
            "forgotten_score_before_mean": _safe_float(np.mean(forgotten_before)),
            "forgotten_score_after_mean": _safe_float(np.mean(forgotten_after)),
            "forgotten_score_drop_mean": _safe_float(np.mean(forgotten_drop)),
            "forgotten_score_decrease_fraction": _safe_float(np.mean(forgotten_drop > 0)),
            "retained_control_count": int(control_count),
            "retained_control_sampling": "seeded_uniform_retained_unique_undirected_edges",
            "retained_control_score_drop_mean": (
                _safe_float(np.mean(control_drop)) if control_count else None
            ),
            "targeted_drop_vs_control": (
                _safe_float(np.mean(forgotten_drop) - np.mean(control_drop))
                if control_count
                else None
            ),
        }
    )

    if exact_retrain_embeddings is None:
        base["exact_retrain_embedding_status"] = "not_provided"
        return base

    exact = _to_numpy(exact_retrain_embeddings)
    if exact.ndim != 2 or exact.shape[0] <= max(max(edge) for edge in valid_forget):
        base["exact_retrain_embedding_status"] = "shape_mismatch"
        return base
    exact_scores = _edge_affinity_scores(exact, valid_forget)
    base.update(
        {
            "exact_retrain_embedding_status": "ok",
            "forgotten_exact_retrain_score_mean": _safe_float(np.mean(exact_scores)),
            "forgotten_unlearned_to_retrain_abs_gap_mean": _safe_float(
                np.mean(np.abs(forgotten_after - exact_scores))
            ),
        }
    )
    return base


def exact_retrain_alignment_metrics(
    logits_before,
    logits_after,
    exact_retrain_logits,
    data,
    *,
    reference: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if exact_retrain_logits is None:
        return {"status": "not_provided"}
    if logits_before is None or logits_after is None:
        return {"status": "missing_logits"}

    before = _to_numpy(logits_before)
    after = _to_numpy(logits_after)
    exact = _to_numpy(exact_retrain_logits)
    if before.ndim != 2 or after.ndim != 2 or exact.ndim != 2:
        return {"status": "invalid_logit_rank"}
    if before.shape[1] != after.shape[1] or after.shape[1] != exact.shape[1]:
        return {
            "status": "class_dimension_mismatch",
            "before_shape": list(before.shape),
            "after_shape": list(after.shape),
            "exact_shape": list(exact.shape),
        }

    row_limit = min(before.shape[0], after.shape[0], exact.shape[0])
    eval_mask = np.ones(row_limit, dtype=bool)
    if data is not None and hasattr(data, "test_mask"):
        candidate = _to_numpy(data.test_mask).reshape(-1).astype(bool)
        eval_mask = candidate[:row_limit]
    if not np.any(eval_mask):
        return {"status": "empty_eval_mask"}

    before_prob = _softmax_rows(before[:row_limit][eval_mask])
    after_prob = _softmax_rows(after[:row_limit][eval_mask])
    exact_prob = _softmax_rows(exact[:row_limit][eval_mask])
    before_js = _js_divergence_rows(before_prob, exact_prob)
    after_js = _js_divergence_rows(after_prob, exact_prob)
    before_tv = 0.5 * np.abs(before_prob - exact_prob).sum(axis=1)
    after_tv = 0.5 * np.abs(after_prob - exact_prob).sum(axis=1)
    metadata = dict(reference or {})
    return {
        "status": "ok",
        "evaluation_scope": "test_mask",
        "num_eval": int(np.sum(eval_mask)),
        "original_to_retrain_js_mean": _safe_float(np.mean(before_js)),
        "unlearned_to_retrain_js_mean": _safe_float(np.mean(after_js)),
        "improvement_over_original_js": _safe_float(np.mean(before_js) - np.mean(after_js)),
        "original_to_retrain_tv_mean": _safe_float(np.mean(before_tv)),
        "unlearned_to_retrain_tv_mean": _safe_float(np.mean(after_tv)),
        "improvement_over_original_tv": _safe_float(np.mean(before_tv) - np.mean(after_tv)),
        "prediction_disagreement_rate": _safe_float(
            np.mean(after_prob.argmax(axis=1) != exact_prob.argmax(axis=1))
        ),
        "reference_schema_version": metadata.get("schema_version"),
        "reference_path": metadata.get("artifact_path") or metadata.get("path"),
        "reference_forget_set_sha256": metadata.get("forget_set_sha256"),
    }


def _canonical_valid_edges(
    edges: Iterable[Sequence[int]], num_nodes: int
) -> list[tuple[int, int]]:
    valid: set[tuple[int, int]] = set()
    for edge in edges:
        if len(edge) != 2:
            continue
        source, target = int(edge[0]), int(edge[1])
        if source == target or not (0 <= source < num_nodes and 0 <= target < num_nodes):
            continue
        valid.add(_canonical_edge(source, target))
    return sorted(valid)


def _canonical_edge(source: int, target: int) -> tuple[int, int]:
    return (source, target) if source < target else (target, source)


def _edge_affinity_scores(
    embeddings: np.ndarray, edges: Sequence[tuple[int, int]]
) -> np.ndarray:
    source = embeddings[[edge[0] for edge in edges]]
    target = embeddings[[edge[1] for edge in edges]]
    denom = np.linalg.norm(source, axis=1) * np.linalg.norm(target, axis=1)
    cosine = np.divide(
        np.sum(source * target, axis=1),
        denom,
        out=np.zeros(len(edges), dtype=float),
        where=denom > 0,
    )
    return np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def _js_divergence_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.clip(left, 1e-12, 1.0)
    right = np.clip(right, 1e-12, 1.0)
    middle = 0.5 * (left + right)
    return 0.5 * np.sum(left * np.log(left / middle), axis=1) + 0.5 * np.sum(
        right * np.log(right / middle), axis=1
    )


def structural_metrics(
    graph_before: nx.Graph,
    graph_after: nx.Graph,
    *,
    excluded_nodes: Optional[Iterable[int]] = None,
) -> dict[str, Any]:
    excluded = {int(node) for node in (excluded_nodes or [])}
    if excluded:
        retained_nodes = (set(graph_before.nodes) | set(graph_after.nodes)) - excluded
        graph_before = graph_before.subgraph(retained_nodes).copy()
        graph_after = graph_after.subgraph(retained_nodes).copy()

    before_cc, before_clustering_backend, before_clustering_workers = _average_clustering_exact(graph_before)
    after_cc, after_clustering_backend, after_clustering_workers = _average_clustering_exact(graph_after)
    before_components = nx.number_connected_components(graph_before) if graph_before.number_of_nodes() else 0
    after_components = nx.number_connected_components(graph_after) if graph_after.number_of_nodes() else 0
    return {
        "evaluation_scope": "retained_nodes" if excluded else "all_nodes",
        "excluded_node_count": len(excluded),
        "degree_kl_divergence": degree_kl_divergence(graph_before, graph_after),
        "clustering_coefficient_before": _safe_float(before_cc),
        "clustering_coefficient_after": _safe_float(after_cc),
        "clustering_coefficient_change": _safe_float(after_cc - before_cc),
        "clustering_backend": (
            before_clustering_backend
            if before_clustering_backend == after_clustering_backend
            else f"{before_clustering_backend}->{after_clustering_backend}"
        ),
        "clustering_workers": max(before_clustering_workers, after_clustering_workers),
        "component_count_before": int(before_components),
        "component_count_after": int(after_components),
        "component_count_change": int(after_components - before_components),
        "num_edges_before": graph_before.number_of_edges(),
        "num_edges_after": graph_after.number_of_edges(),
        "num_nodes_before": graph_before.number_of_nodes(),
        "num_nodes_after": graph_after.number_of_nodes(),
    }


def _average_clustering_exact(graph: nx.Graph) -> tuple[float, str, int]:
    if graph.number_of_nodes() <= 1:
        return 0.0, "networkx_serial", 1

    try:
        requested_workers = max(1, int(os.environ.get("STRUCTURAL_METRICS_WORKERS", "1")))
    except ValueError:
        requested_workers = 1
    try:
        min_edges = max(0, int(os.environ.get("STRUCTURAL_METRICS_PARALLEL_MIN_EDGES", "100000")))
    except ValueError:
        min_edges = 100000

    if (
        requested_workers <= 1
        or graph.number_of_edges() < min_edges
        or "fork" not in mp.get_all_start_methods()
    ):
        return float(nx.average_clustering(graph)), "networkx_serial", 1

    nodes = list(graph.nodes())
    workers = min(requested_workers, len(nodes))
    chunks = _degree_balanced_chunks(graph, workers)

    global _CLUSTERING_GRAPH
    _CLUSTERING_GRAPH = graph
    try:
        with mp.get_context("fork").Pool(processes=workers) as pool:
            partials = pool.map(_clustering_chunk, chunks)
    except (OSError, RuntimeError):
        return float(nx.average_clustering(graph)), "networkx_serial_fallback", 1
    finally:
        _CLUSTERING_GRAPH = None

    total = sum(partial_sum for partial_sum, _ in partials)
    count = sum(partial_count for _, partial_count in partials)
    return (float(total / count) if count else 0.0), "networkx_fork_pool_degree_balanced", workers


def _degree_balanced_chunks(graph: nx.Graph, workers: int) -> list[list[int]]:
    chunks: list[list[int]] = [[] for _ in range(workers)]
    queue = [(0, worker) for worker in range(workers)]
    heapq.heapify(queue)
    ranked_nodes = sorted(graph.degree(), key=lambda item: (-int(item[1]), int(item[0])))
    for node, degree in ranked_nodes:
        load, worker = heapq.heappop(queue)
        chunks[worker].append(int(node))
        weight = max(1, int(degree) * int(degree))
        heapq.heappush(queue, (load + weight, worker))
    return [chunk for chunk in chunks if chunk]


def _clustering_chunk(nodes: Sequence[int]) -> tuple[float, int]:
    if _CLUSTERING_GRAPH is None:
        raise RuntimeError("Clustering worker graph is unavailable.")
    values = nx.clustering(_CLUSTERING_GRAPH, nodes=nodes).values()
    values = list(values)
    return float(sum(values)), len(values)


def privacy_metrics(
    logits_before,
    logits_after,
    embeddings_before,
    embeddings_after,
    graph_before: nx.Graph,
    graph_after: nx.Graph,
    member_nodes: Sequence[int],
    non_member_nodes: Sequence[int],
    unlearning_type: str,
    labels=None,
) -> dict[str, Any]:
    base = {
        "weak_auc": None,
        "medium_auc": None,
        "strong_auc": None,
        "overall_mia_auc": None,
        "privacy_score": None,
        "strong_auc_null_mean": None,
        "strong_auc_null_std": None,
        "strong_auc_pvalue": None,
        "num_members": len(member_nodes),
        "num_non_members": len(non_member_nodes),
        "feature_proxy": False,
        "applicable": unlearning_type != "feature",
        "evaluation_type": "node_membership_inference",
        "status": "ok",
    }
    if unlearning_type == "feature":
        base.update(
            {
                "status": "not_applicable_global_feature_dimension_request",
                "evaluation_type": "feature_attribute_privacy_not_evaluated",
                "reason": (
                    "Global feature-dimension deletion has no semantically valid "
                    "node member/non-member partition."
                ),
            }
        )
        return base
    if logits_before is None or logits_after is None:
        base["status"] = "missing_logits"
        return base
    if not member_nodes or not non_member_nodes:
        base["status"] = "missing_member_or_non_member_nodes"
        return base
    row_limit = min(_to_numpy(logits_before).shape[0], _to_numpy(logits_after).shape[0])
    member_nodes = [node for node in member_nodes if 0 <= int(node) < row_limit]
    non_member_nodes = [node for node in non_member_nodes if 0 <= int(node) < row_limit]
    if not member_nodes or not non_member_nodes:
        base["status"] = "member_or_non_member_nodes_out_of_logit_range"
        base["num_members"] = len(member_nodes)
        base["num_non_members"] = len(non_member_nodes)
        return base

    evaluator = PrivacyEvaluator()
    degree_before = dict(graph_before.degree())
    degree_after = dict(graph_after.degree())
    result = evaluator.evaluate_from_logits(
        logits_before,
        logits_after,
        member_nodes,
        non_member_nodes,
        embeddings_before=embeddings_before,
        embeddings_after=embeddings_after,
        degree_before=degree_before,
        degree_after=degree_after,
        graph_before=graph_before,
        graph_after=graph_after,
        labels=labels,
    ).as_dict()
    base.update(result)
    return base


def feature_compliance_metrics(data_before, data_after, forget_features: Iterable[int]) -> dict[str, Any]:
    base = {
        "status": "ok",
        "request_applied": False,
        "forgotten_feature_count": 0,
        "forgotten_feature_ids": [],
        "forgotten_feature_nonzero_fraction_after": None,
        "forgotten_feature_residual_l1_ratio": None,
        "retained_feature_max_abs_diff": None,
        "ranking_metric": False,
        "note": "Input-deletion compliance check; not a model privacy metric.",
    }
    if data_before is None or data_after is None or not hasattr(data_before, "x") or not hasattr(data_after, "x"):
        base["status"] = "missing_features"
        return base

    before = _to_numpy(data_before.x)
    after = _to_numpy(data_after.x)
    if before.ndim != 2 or after.ndim != 2 or before.shape != after.shape:
        base["status"] = "feature_shape_mismatch"
        return base

    feature_ids = sorted({int(feature) for feature in forget_features if 0 <= int(feature) < before.shape[1]})
    base["forgotten_feature_count"] = len(feature_ids)
    base["forgotten_feature_ids"] = feature_ids
    if not feature_ids:
        base["status"] = "no_valid_forgotten_features"
        return base

    forgotten_before = np.asarray(before[:, feature_ids], dtype=float)
    forgotten_after = np.asarray(after[:, feature_ids], dtype=float)
    residual_l1 = float(np.abs(forgotten_after).sum())
    original_l1 = float(np.abs(forgotten_before).sum())
    residual_ratio = residual_l1 / original_l1 if original_l1 > 0 else (0.0 if residual_l1 == 0 else None)
    nonzero_fraction = float(np.mean(np.abs(forgotten_after) > 1e-9)) if forgotten_after.size else 0.0

    retained_ids = [idx for idx in range(before.shape[1]) if idx not in set(feature_ids)]
    retained_max_diff = 0.0
    if retained_ids:
        retained_max_diff = float(np.max(np.abs(before[:, retained_ids] - after[:, retained_ids])))

    base.update(
        {
            "request_applied": bool(residual_l1 <= 1e-9 and retained_max_diff <= 1e-9),
            "forgotten_feature_nonzero_fraction_after": _safe_float(nonzero_fraction),
            "forgotten_feature_residual_l1_ratio": _safe_float(residual_ratio),
            "retained_feature_max_abs_diff": _safe_float(retained_max_diff),
        }
    )
    return base


def efficiency_metrics(
    unlearn_time_seconds: Optional[float],
    retrain_time_seconds: Optional[float],
    *,
    online_wall_clock_seconds: Optional[float] = None,
    time_breakdown: Optional[dict[str, float]] = None,
    offline_preprocessing_seconds: Optional[float] = None,
) -> dict[str, Any]:
    speedup = None
    if unlearn_time_seconds and retrain_time_seconds:
        speedup = retrain_time_seconds / unlearn_time_seconds if unlearn_time_seconds > 0 else None
    return {
        "unlearn_time_seconds": _safe_float(unlearn_time_seconds),
        "retrain_time_seconds": _safe_float(retrain_time_seconds),
        "speedup_vs_retrain": _safe_float(speedup),
        "online_wall_clock_seconds": _safe_float(online_wall_clock_seconds),
        "time_breakdown": {str(k): _safe_float(v) for k, v in (time_breakdown or {}).items()},
        "offline_preprocessing_seconds": _safe_float(offline_preprocessing_seconds),
    }


def rq_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    utility = metrics["utility"]
    structure = metrics["structure"]
    privacy = metrics["privacy"]
    efficiency = metrics["efficiency"]
    representation = metrics["representation"]
    return {
        "RQ1_overall_utility": {
            "accuracy_after": utility["accuracy_after"],
            "accuracy_drop": utility["accuracy_drop"],
            "f1_macro_after": utility["f1_macro_after"],
            "f1_macro_drop": utility["f1_macro_drop"],
            "speedup_vs_retrain": efficiency["speedup_vs_retrain"],
        },
        "RQ2_anchor_stabilization": {
            "embedding_l2_mean": representation["embedding_l2_mean"],
            "member_embedding_l2_mean": representation["member_embedding_l2_mean"],
            "neighbor_drift_mean": representation["neighbor_drift_mean"],
        },
        "RQ3_partition_structure": {
            "degree_kl_divergence": structure["degree_kl_divergence"],
            "clustering_coefficient_change": structure["clustering_coefficient_change"],
            "component_count_change": structure["component_count_change"],
        },
        "RQ4_privacy_dar": {
            "weak_auc": privacy["weak_auc"],
            "medium_auc": privacy["medium_auc"],
            "strong_auc": privacy["strong_auc"],
            "overall_mia_auc": privacy["overall_mia_auc"],
            "privacy_score": privacy["privacy_score"],
        },
        "RQ5_inpainting_privacy_structure": {
            "overall_mia_auc": privacy["overall_mia_auc"],
            "degree_kl_divergence": structure["degree_kl_divergence"],
            "clustering_coefficient_change": structure["clustering_coefficient_change"],
        },
        "RQ6_unlearning_type_scope": {
            "unlearning_type": metrics["unlearning_type"],
            "accuracy_after": utility["accuracy_after"],
            "f1_macro_after": utility["f1_macro_after"],
            "overall_mia_auc": privacy["overall_mia_auc"],
        },
    }


def graph_from_data(data) -> nx.Graph:
    graph = nx.Graph()
    if data is None:
        return graph
    num_nodes = int(getattr(data, "num_nodes", data.x.shape[0] if hasattr(data, "x") else 0))
    graph.add_nodes_from(range(num_nodes))
    if not hasattr(data, "edge_index"):
        return graph
    edge_index = data.edge_index.detach().cpu().tolist()
    for source, target in zip(edge_index[0], edge_index[1]):
        graph.add_edge(int(source), int(target))
    return graph


def _valid_node_list(nodes: Iterable[int], limit: int) -> list[int]:
    return sorted({int(node) for node in nodes if 0 <= int(node) < limit})


def infer_member_nodes(
    unlearning_type: str,
    forget_targets: Any,
    data,
    embeddings_before=None,
    embeddings_after=None,
    data_after=None,
) -> list[int]:
    num_nodes = int(getattr(data, "num_nodes", data.x.shape[0] if hasattr(data, "x") else 0))
    if unlearning_type == "node":
        return _valid_nodes([int(node) for node in forget_targets], num_nodes)
    if unlearning_type == "edge":
        return _valid_nodes([node for edge in forget_targets for node in edge], num_nodes)
    # Global feature-dimension deletion has no node-level membership partition.
    return []


def infer_affected_nodes(
    unlearning_type: str,
    forget_targets: Any,
    data,
    embeddings_before=None,
    embeddings_after=None,
    data_after=None,
) -> list[int]:
    if unlearning_type != "feature":
        return infer_member_nodes(unlearning_type, forget_targets, data)

    num_nodes = int(getattr(data, "num_nodes", data.x.shape[0] if hasattr(data, "x") else 0))
    if data_after is not None and hasattr(data, "x") and hasattr(data_after, "x"):
        xb = _to_numpy(data.x)
        xa = _to_numpy(data_after.x)
        n = min(xb.shape[0], xa.shape[0], num_nodes)
        changed = np.where(np.abs(xb[:n] - xa[:n]).sum(axis=1) > 1e-9)[0]
        return [int(i) for i in changed]
    # Representation diagnostics may use drift as a fallback; privacy membership never does.
    if embeddings_before is not None and embeddings_after is not None:
        before = _to_numpy(embeddings_before)
        after = _to_numpy(embeddings_after)
        n_rows = min(before.shape[0], after.shape[0], num_nodes)
        drift = np.linalg.norm(before[:n_rows] - after[:n_rows], axis=1)
        count = max(1, min(n_rows // 4, 256))
        return [int(idx) for idx in np.argsort(drift)[-count:]]
    return list(range(max(0, min(num_nodes, 1))))


def infer_non_member_nodes(
    data,
    member_nodes: Sequence[int],
    count: int,
    *,
    seed: int = 0,
    labels=None,
    match_class: bool = True,
) -> list[int]:
    """Sample `count` non-member (retained) nodes as a FAIR control.

    Fixes the old bug where non-members were the lowest node ids (candidates[:count]),
    which confounds membership with node-id ordering (id often tracks class in KGs).
    Now: seeded random sampling, split-matched to train_mask when members all come
    from train_mask, and class-matched to the member class distribution when labels
    are available.
    """
    num_nodes = int(getattr(data, "num_nodes", data.x.shape[0] if hasattr(data, "x") else 0))
    members = set(int(node) for node in member_nodes)
    candidate_universe = _non_member_candidate_universe(data, members, num_nodes, count)
    candidates = [node for node in candidate_universe if node not in members]
    if count <= 0 or not candidates:
        return []

    rng = np.random.default_rng(seed)

    if match_class and labels is not None:
        y = _to_numpy(labels).reshape(-1).astype(int)
        member_class_counts: dict[int, int] = {}
        for node in members:
            if node < len(y):
                cls = int(y[node])
                member_class_counts[cls] = member_class_counts.get(cls, 0) + 1
        pool_by_class: dict[int, list[int]] = {}
        for node in candidates:
            if node < len(y):
                pool_by_class.setdefault(int(y[node]), []).append(node)
        chosen: list[int] = []
        for cls, k in member_class_counts.items():
            pool = pool_by_class.get(cls, [])
            rng.shuffle(pool)
            chosen.extend(pool[:k])
        if len(chosen) < count:  # top up if some classes were short
            chosen_set = set(chosen)
            remaining = [n for n in candidates if n not in chosen_set]
            rng.shuffle(remaining)
            chosen.extend(remaining[: count - len(chosen)])
        return chosen[:count]

    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return shuffled[:count]


def _non_member_candidate_universe(data, members: set[int], num_nodes: int, count: int) -> list[int]:
    all_nodes = list(range(num_nodes))
    train_mask = getattr(data, "train_mask", None)
    if train_mask is None or not members:
        return all_nodes

    mask = _to_numpy(train_mask).reshape(-1).astype(bool)
    limit = min(num_nodes, len(mask))
    if not all(0 <= node < limit and bool(mask[node]) for node in members):
        return all_nodes

    train_nodes = [int(node) for node in np.flatnonzero(mask[:limit])]
    retained_train_count = sum(1 for node in train_nodes if node not in members)
    if retained_train_count >= count:
        return train_nodes
    return all_nodes


def count_forget_targets(forget_targets: Any) -> int:
    if forget_targets is None:
        return 0
    if isinstance(forget_targets, (list, tuple, set)):
        return len(forget_targets)
    return 1


def degree_kl_divergence(graph_before: nx.Graph, graph_after: nx.Graph) -> Optional[float]:
    if graph_before.number_of_nodes() == 0 and graph_after.number_of_nodes() == 0:
        return 0.0
    before = _degree_distribution(graph_before)
    after = _degree_distribution(graph_after)
    max_len = max(len(before), len(after), 1)
    before = np.pad(before, (0, max_len - len(before)), constant_values=0.0)
    after = np.pad(after, (0, max_len - len(after)), constant_values=0.0)
    eps = 1e-12
    before = (before + eps) / np.sum(before + eps)
    after = (after + eps) / np.sum(after + eps)
    return _safe_float(np.sum(before * np.log(before / after)))


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(json_safe(item) for item in value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _safe_float(float(value))
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if hasattr(value, "as_dict"):
        return json_safe(value.as_dict())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return json_safe(value.__dict__)
    if isinstance(value, float):
        return _safe_float(value)
    return value


def _classification_metrics(data, logits, mask_name: str) -> dict[str, Any]:
    if data is None or logits is None or not hasattr(data, "y"):
        return {"accuracy": None, "f1_micro": None, "f1_macro": None, "num_eval": 0}
    y = data.y
    if y.dim() > 1:
        y = y.squeeze(-1)
    y_np = y.detach().cpu().numpy().astype(int)
    pred_np = logits.detach().cpu().argmax(dim=-1).numpy().astype(int)
    n_rows = min(len(y_np), len(pred_np))
    mask = getattr(data, mask_name, None)
    if mask is None:
        mask_np = np.ones(n_rows, dtype=bool)
    else:
        if mask.dim() > 1:
            mask = mask[:, 0]
        mask_np = mask.detach().cpu().numpy().astype(bool)[:n_rows]
    y_np = y_np[:n_rows][mask_np]
    pred_np = pred_np[:n_rows][mask_np]
    if len(y_np) == 0:
        return {"accuracy": None, "f1_micro": None, "f1_macro": None, "num_eval": 0}
    accuracy = float(np.mean(pred_np == y_np))
    return {
        "accuracy": _safe_float(accuracy),
        "f1_micro": _safe_float(_f1_micro(y_np, pred_np)),
        "f1_macro": _safe_float(_f1_macro(y_np, pred_np)),
        "num_eval": int(len(y_np)),
    }


def _f1_micro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    scores = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(scores)) if scores else 0.0


def _degree_distribution(graph: nx.Graph) -> np.ndarray:
    degrees = [degree for _, degree in graph.degree()]
    if not degrees:
        return np.array([1.0])
    counts = np.bincount(np.array(degrees, dtype=int))
    return counts.astype(float)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _valid_nodes(nodes: Iterable[int], num_nodes: int) -> list[int]:
    return sorted({int(node) for node in nodes if 0 <= int(node) < num_nodes})


def _drop(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None:
        return None
    return _safe_float(before - after)


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value

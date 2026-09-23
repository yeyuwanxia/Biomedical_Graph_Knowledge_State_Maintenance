from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np


@dataclass
class MIAResult:
    weak_auc: float
    medium_auc: float
    strong_auc: float
    overall_mia_auc: float
    privacy_score: float
    num_members: int
    num_non_members: int
    weak_feature_dim: int
    medium_feature_dim: int
    strong_feature_dim: int
    medium_uses_shadow: bool = False
    medium_evaluation: str = "held_out_target_split"
    medium_train_size: int = 0
    medium_eval_size: int = 0
    strong_auc_null_mean: float = 0.5
    strong_auc_null_std: float = 0.0
    strong_auc_pvalue: float = 1.0

    def as_dict(self) -> Dict[str, float | int | bool | str]:
        return {
            "weak_auc": self.weak_auc,
            "medium_auc": self.medium_auc,
            "strong_auc": self.strong_auc,
            "overall_mia_auc": self.overall_mia_auc,
            "privacy_score": self.privacy_score,
            "num_members": self.num_members,
            "num_non_members": self.num_non_members,
            "weak_feature_dim": self.weak_feature_dim,
            "medium_feature_dim": self.medium_feature_dim,
            "strong_feature_dim": self.strong_feature_dim,
            "medium_uses_shadow": self.medium_uses_shadow,
            "medium_evaluation": self.medium_evaluation,
            "medium_train_size": self.medium_train_size,
            "medium_eval_size": self.medium_eval_size,
            "strong_auc_null_mean": self.strong_auc_null_mean,
            "strong_auc_null_std": self.strong_auc_null_std,
            "strong_auc_pvalue": self.strong_auc_pvalue,
        }


class WeakAttacker:
    """Black-box attacker using confidence and entropy changes."""

    def features(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        prob_before = _softmax(before)
        prob_after = _softmax(after)
        confidence_change = prob_before.max(axis=1) - prob_after.max(axis=1)
        entropy_change = _entropy(prob_after) - _entropy(prob_before)
        return np.stack([confidence_change, np.abs(confidence_change), entropy_change], axis=1)

    def scores(self, features: np.ndarray, members: np.ndarray, non_members: np.ndarray) -> np.ndarray:
        return _score_rows(features, members, non_members)


class MediumAttacker:
    """Shadow-transfer attacker.

    If shadow features are supplied, a logistic attack classifier is trained on
    the shadow split and transferred to target rows. Without a shadow graph, the
    same classifier API falls back to a stratified target split so experiments
    remain runnable while still using a learned attacker instead of a raw norm.
    """

    def features(self, weak_features: np.ndarray, embeddings_before, embeddings_after) -> np.ndarray:
        n_rows = weak_features.shape[0]
        embedding_l2 = np.zeros(n_rows)
        if embeddings_before is not None and embeddings_after is not None:
            emb_before = _to_numpy(embeddings_before)
            emb_after = _to_numpy(embeddings_after)
            n_rows = min(n_rows, emb_before.shape[0], emb_after.shape[0])
            embedding_l2[:n_rows] = np.linalg.norm(emb_before[:n_rows] - emb_after[:n_rows], axis=1)
        return np.column_stack([weak_features, embedding_l2])

    def scores(
        self,
        features: np.ndarray,
        members: np.ndarray,
        non_members: np.ndarray,
        *,
        shadow_features=None,
        shadow_labels=None,
    ) -> tuple[np.ndarray, np.ndarray, bool, int]:
        target_rows = np.concatenate([features[members], features[non_members]], axis=0)
        y_true = np.concatenate([np.ones(len(members)), np.zeros(len(non_members))])
        target_indices = np.arange(len(y_true), dtype=int)
        if shadow_features is not None and shadow_labels is not None:
            train_x = _to_numpy(shadow_features)
            train_y = np.asarray(shadow_labels, dtype=float)
            return (
                _logistic_scores(train_x, train_y, target_rows),
                target_indices,
                True,
                int(train_x.shape[0]),
            )
        if len(y_true) >= 4 and len(np.unique(y_true)) == 2:
            train_idx = _stratified_half_indices(y_true)
            eval_idx = np.setdiff1d(target_indices, train_idx, assume_unique=True)
            if len(eval_idx) >= 2 and len(np.unique(y_true[eval_idx])) == 2:
                return (
                    _logistic_scores(target_rows[train_idx], y_true[train_idx], target_rows[eval_idx]),
                    eval_idx,
                    False,
                    int(len(train_idx)),
                )
        return _score_rows(features, members, non_members), target_indices, False, 0


class StrongAttacker:
    """White-box attacker using the 10 HASI framework features."""

    def features(
        self,
        before: np.ndarray,
        after: np.ndarray,
        embeddings_before,
        embeddings_after,
        degree_before: Dict[int, int],
        degree_after: Dict[int, int],
        *,
        graph_before=None,
        graph_after=None,
        labels=None,
    ) -> np.ndarray:
        weak = WeakAttacker().features(before, after)
        n_rows = before.shape[0]
        embedding_l2 = np.zeros(n_rows)
        embedding_cosine = np.zeros(n_rows)
        node_drift = np.zeros(n_rows)

        if embeddings_before is not None and embeddings_after is not None:
            emb_before = _to_numpy(embeddings_before)
            emb_after = _to_numpy(embeddings_after)
            n_embed = min(n_rows, emb_before.shape[0], emb_after.shape[0])
            delta = emb_before[:n_embed] - emb_after[:n_embed]
            node_drift[:n_embed] = np.linalg.norm(delta, axis=1)
            embedding_l2[:n_embed] = node_drift[:n_embed]
            denom = np.linalg.norm(emb_before[:n_embed], axis=1) * np.linalg.norm(emb_after[:n_embed], axis=1)
            denom = np.where(denom == 0, 1.0, denom)
            embedding_cosine[:n_embed] = 1.0 - np.sum(emb_before[:n_embed] * emb_after[:n_embed], axis=1) / denom

        degree_abs = np.array([degree_before.get(i, 0) - degree_after.get(i, 0) for i in range(n_rows)], dtype=float)
        degree_rel = degree_abs / np.maximum(1.0, np.array([degree_before.get(i, 0) for i in range(n_rows)], dtype=float))
        neighbor_avg, neighbor_max = _neighbor_drift_features(graph_before, node_drift, n_rows)
        homophily_change = _homophily_change(graph_before, graph_after, labels, n_rows)
        return np.column_stack(
            [
                weak[:, 0],
                weak[:, 1],
                weak[:, 2],
                embedding_l2,
                embedding_cosine,
                degree_abs,
                degree_rel,
                neighbor_avg,
                neighbor_max,
                homophily_change,
            ]
        )

    def scores(self, features: np.ndarray, members: np.ndarray, non_members: np.ndarray) -> np.ndarray:
        return _score_rows(features, members, non_members)


class PrivacyEvaluator:
    """Model-agnostic multi-tier MIA evaluator."""

    def __init__(self):
        self.weak_attacker = WeakAttacker()
        self.medium_attacker = MediumAttacker()
        self.strong_attacker = StrongAttacker()

    def evaluate_from_logits(
        self,
        logits_before,
        logits_after,
        member_indices: Iterable[int],
        non_member_indices: Iterable[int],
        embeddings_before=None,
        embeddings_after=None,
        degree_before: Optional[Dict[int, int]] = None,
        degree_after: Optional[Dict[int, int]] = None,
        graph_before=None,
        graph_after=None,
        labels=None,
        shadow_features=None,
        shadow_labels=None,
    ) -> MIAResult:
        before = _to_numpy(logits_before)
        after = _to_numpy(logits_after)
        row_limit = min(before.shape[0], after.shape[0])
        before = before[:row_limit]
        after = after[:row_limit]
        members = _valid_indices(member_indices, row_limit)
        non_members = _valid_indices(non_member_indices, row_limit)

        weak_features = self.weak_attacker.features(before, after)
        medium_features = self.medium_attacker.features(weak_features, embeddings_before, embeddings_after)
        strong_features = self.strong_attacker.features(
            before,
            after,
            embeddings_before,
            embeddings_after,
            degree_before or {},
            degree_after or {},
            graph_before=graph_before,
            graph_after=graph_after,
            labels=labels,
        )

        y_true = np.concatenate([np.ones(len(members)), np.zeros(len(non_members))])
        weak_scores = self.weak_attacker.scores(weak_features, members, non_members)
        medium_scores, medium_eval_indices, medium_uses_shadow, medium_train_size = self.medium_attacker.scores(
            medium_features,
            members,
            non_members,
            shadow_features=shadow_features,
            shadow_labels=shadow_labels,
        )
        strong_scores = self.strong_attacker.scores(strong_features, members, non_members)

        weak_auc = _attack_auc(y_true, weak_scores)
        medium_auc = _attack_auc(y_true[medium_eval_indices], medium_scores)
        strong_ranks = _average_ranks(strong_scores)
        strong_auc = _attack_auc_from_ranks(y_true, strong_ranks)
        privacy_score = max(0.0, 1.0 - 2.0 * abs(strong_auc - 0.5))

        # Reuse exact score ranks across label permutations. This is mathematically
        # identical to pairwise Mann-Whitney comparisons and scales to large graphs.
        null_rng = np.random.default_rng(12345)
        n_perm = 200
        null_aucs = np.empty(n_perm)
        for i in range(n_perm):
            null_aucs[i] = _attack_auc_from_ranks(
                null_rng.permutation(y_true), strong_ranks
            )
        strong_auc_null_mean = float(null_aucs.mean())
        strong_auc_null_std = float(null_aucs.std())
        strong_auc_pvalue = float((np.sum(null_aucs >= strong_auc) + 1) / (n_perm + 1))

        return MIAResult(
            weak_auc=float(weak_auc),
            medium_auc=float(medium_auc),
            strong_auc=float(strong_auc),
            overall_mia_auc=float(strong_auc),
            privacy_score=float(privacy_score),
            num_members=int(len(members)),
            num_non_members=int(len(non_members)),
            weak_feature_dim=int(weak_features.shape[1]),
            medium_feature_dim=int(medium_features.shape[1]),
            strong_feature_dim=int(strong_features.shape[1]),
            medium_uses_shadow=bool(medium_uses_shadow),
            medium_evaluation=(
                "shadow_transfer_target_eval"
                if medium_uses_shadow
                else "held_out_target_split" if medium_train_size else "unsupervised_fallback"
            ),
            medium_train_size=int(medium_train_size),
            medium_eval_size=int(len(medium_eval_indices)),
            strong_auc_null_mean=strong_auc_null_mean,
            strong_auc_null_std=strong_auc_null_std,
            strong_auc_pvalue=strong_auc_pvalue,
        )


def _valid_indices(indices: Iterable[int], row_limit: int) -> np.ndarray:
    return np.array([int(idx) for idx in indices if 0 <= int(idx) < row_limit], dtype=int)


def _score_rows(features: np.ndarray, members: np.ndarray, non_members: np.ndarray) -> np.ndarray:
    rows = np.concatenate([features[members], features[non_members]], axis=0)
    rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
    scale = rows.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    rows = (rows - rows.mean(axis=0)) / scale
    return np.linalg.norm(rows, axis=1)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(probabilities * np.log(probabilities), axis=1)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _neighbor_drift_features(graph, node_drift: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    avg = np.zeros(n_rows)
    max_value = np.zeros(n_rows)
    if graph is None:
        return avg, max_value
    for node in range(n_rows):
        if node not in graph:
            continue
        neighbors = [int(neighbor) for neighbor in graph.neighbors(node) if 0 <= int(neighbor) < n_rows]
        if not neighbors:
            continue
        values = node_drift[neighbors]
        avg[node] = float(np.mean(values))
        max_value[node] = float(np.max(values))
    return avg, max_value


def _homophily_change(graph_before, graph_after, labels, n_rows: int) -> np.ndarray:
    change = np.zeros(n_rows)
    if graph_before is None or graph_after is None or labels is None:
        return change
    label_values = _to_numpy(labels).reshape(-1)
    limit = min(n_rows, label_values.shape[0])
    for node in range(limit):
        before = _local_homophily(graph_before, node, label_values, limit)
        after = _local_homophily(graph_after, node, label_values, limit)
        change[node] = abs(after - before)
    return change


def _local_homophily(graph, node: int, labels: np.ndarray, limit: int) -> float:
    if node not in graph:
        return 0.0
    neighbors = [int(neighbor) for neighbor in graph.neighbors(node) if 0 <= int(neighbor) < limit]
    if not neighbors:
        return 0.0
    return float(np.mean(labels[neighbors] == labels[node]))


def _stratified_half_indices(y: np.ndarray, seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices: list[int] = []
    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label).copy()
        rng.shuffle(label_indices)
        take = max(1, len(label_indices) // 2)
        indices.extend(label_indices[:take].tolist())
    return np.array(sorted(indices), dtype=int)


def _logistic_scores(train_x: np.ndarray, train_y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    train_x = np.nan_to_num(np.asarray(train_x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    target_x = np.nan_to_num(np.asarray(target_x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    train_y = np.asarray(train_y, dtype=float).reshape(-1)
    if train_x.shape[0] == 0 or len(np.unique(train_y)) < 2:
        scale = target_x.std(axis=0)
        scale = np.where(scale == 0, 1.0, scale)
        centered = (target_x - target_x.mean(axis=0)) / scale
        return np.linalg.norm(centered, axis=1)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    x = (train_x - mean) / scale
    target = (target_x - mean) / scale
    x = np.column_stack([x, np.ones(x.shape[0])])
    target = np.column_stack([target, np.ones(target.shape[0])])
    weights = np.zeros(x.shape[1])
    lr = 0.1
    for _ in range(200):
        logits = np.clip(x @ weights, -30.0, 30.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        grad = x.T @ (pred - train_y) / max(1, x.shape[0])
        weights -= lr * grad
    return 1.0 / (1.0 + np.exp(-np.clip(target @ weights, -30.0, 30.0)))


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    if len(values) == 0:
        return ranks

    starts = np.concatenate(
        ([0], np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1)
    )
    ends = np.concatenate((starts[1:], [len(values)]))
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (float(start + 1) + float(end)) / 2.0
    return ranks


def _auc_from_ranks(y_true: np.ndarray, ranks: np.ndarray) -> float:
    labels = np.asarray(y_true).reshape(-1)
    ranks = np.asarray(ranks, dtype=float).reshape(-1)
    if len(labels) != len(ranks):
        raise ValueError("y_true and ranks must have the same length")
    positive = labels == 1
    num_positive = int(np.sum(positive))
    num_negative = int(len(labels) - num_positive)
    if num_positive == 0 or num_negative == 0:
        return 0.5

    rank_sum = float(np.sum(ranks[positive]))
    mann_whitney = rank_sum - num_positive * (num_positive + 1) / 2.0
    return float(np.clip(mann_whitney / (num_positive * num_negative), 0.0, 1.0))


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return _auc_from_ranks(y_true, _average_ranks(scores))


def _attack_auc_from_ranks(y_true: np.ndarray, ranks: np.ndarray) -> float:
    raw_auc = _auc_from_ranks(y_true, ranks)
    return max(raw_auc, 1.0 - raw_auc)


def _attack_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return _attack_auc_from_ranks(y_true, _average_ranks(scores))

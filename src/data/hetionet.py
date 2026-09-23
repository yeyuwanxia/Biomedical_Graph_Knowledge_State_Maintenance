from __future__ import annotations

import bz2
import gzip
import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


HETIONET_SMALL_NOSOURCE_VARIANT = "hetionet_small_nosource"
HETIONET_FULL_NOSOURCE_VARIANT = "hetionet_full_nosource"
RAW_FILE = "hetionet-v1.0.json.bz2"
PROCESSED_SMALL_NOSOURCE_DIR = "processed_small_nosource"
PROCESSED_FULL_NOSOURCE_DIR = "processed_full_nosource"
PROCESSED_DIR = PROCESSED_SMALL_NOSOURCE_DIR
DATA_FILE = "data.pt"
METADATA_FILE = "metadata.json"
DEFAULT_DOWNLOAD_URL = "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2"
DEFAULT_NODE_TYPES = (
    "Disease",
    "Gene",
    "Compound",
    "Pathway",
    "Side Effect",
    "Biological Process",
)
FEATURE_SCHEMA = [
    "log_degree",
    "sqrt_degree",
    "normalized_degree",
    "log_raw_in_degree",
    "log_raw_out_degree",
    "raw_in_share",
    "raw_out_share",
    "log_neighbor_degree_mean",
    "normalized_neighbor_degree_mean",
    "normalized_neighbor_degree_std",
    "normalized_neighbor_degree_min",
    "normalized_neighbor_degree_max",
    "neighbor_degree_ratio",
    "local_clustering_coefficient",
    "log_triangle_count",
    "normalized_triangle_count",
    "normalized_core_number",
    "log_core_number",
    "pagerank",
    "log_scaled_pagerank",
]


def hetionet_small_nosource_cache_paths(root: str | Path) -> list[Path]:
    base = _hetionet_root(root)
    return [base / PROCESSED_SMALL_NOSOURCE_DIR, base]


def hetionet_full_nosource_cache_paths(root: str | Path) -> list[Path]:
    base = _hetionet_root(root)
    return [base / PROCESSED_FULL_NOSOURCE_DIR, base]


def load_hetionet_small_nosource(
    root: str | Path,
    *,
    download: bool = True,
    node_types: tuple[str, ...] = DEFAULT_NODE_TYPES,
    max_nodes_per_class: int = 5_000,
    min_nodes_per_class: int = 500,
    selection_seed: int = 42,
    make_undirected: bool = True,
    deduplicate_edges: bool = True,
    remove_self_loops: bool = True,
    drop_isolated_nodes: bool = True,
    force_rebuild: bool = False,
):
    """Load or build a structure-only Hetionet node-type classification graph.

    The task label is the biomedical entity type. Features deliberately exclude
    entity-type one-hot, source/database indicators, and relation-type counts.
    """
    return _load_hetionet_nosource(
        root,
        download=download,
        dataset_name="hetionet-small-nosource",
        dataset_variant=HETIONET_SMALL_NOSOURCE_VARIANT,
        processed_dir_name=PROCESSED_SMALL_NOSOURCE_DIR,
        node_types=node_types,
        max_nodes_per_class=max_nodes_per_class,
        min_nodes_per_class=min_nodes_per_class,
        selection_seed=selection_seed,
        make_undirected=make_undirected,
        deduplicate_edges=deduplicate_edges,
        remove_self_loops=remove_self_loops,
        drop_isolated_nodes=drop_isolated_nodes,
        force_rebuild=force_rebuild,
        node_rule="sample major biomedical entity types with per-class caps",
    )


def load_hetionet_full_nosource(
    root: str | Path,
    *,
    download: bool = True,
    node_types: tuple[str, ...] | None = None,
    max_nodes_per_class: int | None = None,
    min_nodes_per_class: int = 1,
    selection_seed: int = 42,
    make_undirected: bool = True,
    deduplicate_edges: bool = True,
    remove_self_loops: bool = True,
    drop_isolated_nodes: bool = False,
    force_rebuild: bool = False,
):
    """Load or build the full structure-only Hetionet node-type graph.

    Unlike ``hetionet-small-nosource``, this variant keeps all raw Hetionet
    metanode types by default and applies no per-class cap. Features still
    exclude entity-type one-hot, source/database indicators, and relation-type
    counts.
    """
    return _load_hetionet_nosource(
        root,
        download=download,
        dataset_name="hetionet-full-nosource",
        dataset_variant=HETIONET_FULL_NOSOURCE_VARIANT,
        processed_dir_name=PROCESSED_FULL_NOSOURCE_DIR,
        node_types=node_types,
        max_nodes_per_class=max_nodes_per_class,
        min_nodes_per_class=min_nodes_per_class,
        selection_seed=selection_seed,
        make_undirected=make_undirected,
        deduplicate_edges=deduplicate_edges,
        remove_self_loops=remove_self_loops,
        drop_isolated_nodes=drop_isolated_nodes,
        force_rebuild=force_rebuild,
        node_rule="retain all raw Hetionet metanode types without per-class caps",
    )


def _load_hetionet_nosource(
    root: str | Path,
    *,
    download: bool,
    dataset_name: str,
    dataset_variant: str,
    processed_dir_name: str,
    node_types: tuple[str, ...] | None,
    max_nodes_per_class: int | None,
    min_nodes_per_class: int,
    selection_seed: int,
    make_undirected: bool,
    deduplicate_edges: bool,
    remove_self_loops: bool,
    drop_isolated_nodes: bool,
    force_rebuild: bool,
    node_rule: str,
):
    """Build a homogeneous, structure-only Hetionet node-type task."""

    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise SystemExit(
            f"torch-geometric is required to build/load {dataset_name}. "
            "Use the graphunlearning environment or install torch-geometric."
        ) from exc

    base = _hetionet_root(root)
    processed = base / processed_dir_name
    data_path = processed / DATA_FILE
    metadata_path = processed / METADATA_FILE
    if data_path.exists() and metadata_path.exists() and not force_rebuild:
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return data, metadata

    raw_path = _find_raw_file(base)
    if raw_path is None:
        if not download:
            raise FileNotFoundError(
                f"Missing Hetionet raw file under {base}. Expected {RAW_FILE} or hetionet-v1.0.json."
            )
        raw_path = _download_raw_file(base)

    raw = _read_hetionet_json(raw_path)
    nodes = raw.get("nodes")
    edges = raw.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError(f"Unexpected Hetionet JSON structure in {raw_path}")

    wanted_types = _resolve_node_types(raw, nodes, node_types)
    wanted_set = set(wanted_types)
    nodes_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_type_counts: Counter[str] = Counter()
    for node in nodes:
        kind = _canonical_type(str(node.get("kind", "")))
        if not kind:
            continue
        all_type_counts[kind] += 1
        if kind in wanted_set:
            nodes_by_type[kind].append(node)

    rng = random.Random(int(selection_seed))
    selected_nodes: list[dict[str, Any]] = []
    selected_counts_before_edges: dict[str, int] = {}
    skipped_types: dict[str, int] = {}
    for node_type in wanted_types:
        candidates = nodes_by_type.get(node_type, [])
        if len(candidates) < int(min_nodes_per_class):
            skipped_types[node_type] = len(candidates)
            continue
        if max_nodes_per_class is not None and len(candidates) > int(max_nodes_per_class):
            sampled_positions = set(rng.sample(range(len(candidates)), int(max_nodes_per_class)))
            candidates = [node for idx, node in enumerate(candidates) if idx in sampled_positions]
        selected_counts_before_edges[node_type] = len(candidates)
        selected_nodes.extend(candidates)

    if len(selected_counts_before_edges) < 2:
        raise ValueError(
            f"{dataset_name} needs at least two retained classes. "
            f"Retained={selected_counts_before_edges}, skipped={skipped_types}"
        )

    class_labels = [node_type for node_type in wanted_types if node_type in selected_counts_before_edges]
    class_mapping = {label: idx for idx, label in enumerate(class_labels)}
    selected_key_to_idx: dict[tuple[str, str], int] = {}
    node_kinds: list[str] = []
    node_identifiers: list[str] = []
    y_values: list[int] = []
    for node in selected_nodes:
        key = _node_key(node)
        if key is None or key in selected_key_to_idx:
            continue
        kind = _canonical_type(key[0])
        if kind not in class_mapping:
            continue
        selected_key_to_idx[(kind, key[1])] = len(node_kinds)
        node_kinds.append(kind)
        node_identifiers.append(key[1])
        y_values.append(class_mapping[kind])

    raw_in_degree = torch.zeros(len(node_kinds), dtype=torch.float32)
    raw_out_degree = torch.zeros(len(node_kinds), dtype=torch.float32)
    source_nodes: list[int] = []
    target_nodes: list[int] = []
    relation_counts: Counter[str] = Counter()
    kept_raw_edge_count = 0
    skipped_edge_count = 0

    for edge in edges:
        src_key = _edge_node_key(edge.get("source_id"))
        dst_key = _edge_node_key(edge.get("target_id"))
        if src_key is None or dst_key is None:
            skipped_edge_count += 1
            continue
        src_key = (_canonical_type(src_key[0]), src_key[1])
        dst_key = (_canonical_type(dst_key[0]), dst_key[1])
        src = selected_key_to_idx.get(src_key)
        dst = selected_key_to_idx.get(dst_key)
        if src is None or dst is None:
            continue
        kept_raw_edge_count += 1
        relation_counts[str(edge.get("kind", "unknown"))] += 1
        if remove_self_loops and src == dst:
            continue
        raw_out_degree[src] += 1.0
        raw_in_degree[dst] += 1.0
        source_nodes.append(src)
        target_nodes.append(dst)

    if not source_nodes:
        raise ValueError("No Hetionet edges remain after node selection.")

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_count_directed = int(edge_index.shape[1])
    if make_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_count_after_undirected = int(edge_index.shape[1])
    if deduplicate_edges:
        edge_index = torch.unique(edge_index, dim=1)
    edge_count_after_dedup = int(edge_index.shape[1])

    processed_degree = torch.bincount(edge_index[0], minlength=len(node_kinds)).to(torch.float32)
    keep_nodes = processed_degree > 0
    isolated_nodes_before_drop = int((~keep_nodes).sum().item())
    removed_isolated_nodes = isolated_nodes_before_drop if drop_isolated_nodes else 0
    if drop_isolated_nodes and removed_isolated_nodes:
        old_to_new = torch.full((len(node_kinds),), -1, dtype=torch.long)
        old_to_new[keep_nodes] = torch.arange(int(keep_nodes.sum().item()), dtype=torch.long)
        keep_edges = keep_nodes[edge_index[0]] & keep_nodes[edge_index[1]]
        edge_index = old_to_new[edge_index[:, keep_edges]]
        kept_indices = torch.nonzero(keep_nodes, as_tuple=False).view(-1).tolist()
        node_kinds = [node_kinds[idx] for idx in kept_indices]
        node_identifiers = [node_identifiers[idx] for idx in kept_indices]
        y_values = [y_values[idx] for idx in kept_indices]
        raw_in_degree = raw_in_degree[keep_nodes]
        raw_out_degree = raw_out_degree[keep_nodes]
        processed_degree = torch.bincount(edge_index[0], minlength=len(node_kinds)).to(torch.float32)

    y = torch.tensor(y_values, dtype=torch.long)
    x = _structure_features(edge_index, raw_in_degree, raw_out_degree, feature_dim=20)

    data = Data(x=x, edge_index=edge_index.contiguous(), y=y)
    data.num_nodes = int(x.shape[0])
    data.hetionet_node_type = node_kinds
    data.hetionet_node_identifier = node_identifiers

    class_counts = torch.bincount(y, minlength=len(class_mapping)).tolist()
    metadata: dict[str, Any] = {
        "dataset": dataset_name,
        "dataset_variant": dataset_variant,
        "source_dataset": "hetionet",
        "original_graph_type": "heterogeneous_biomedical_knowledge_graph",
        "graph_type": "homogeneous_projection",
        "task": "node_type_classification",
        "raw_path": str(raw_path),
        "download_url": DEFAULT_DOWNLOAD_URL,
        "processed_data": str(data_path),
        "construction_protocol": {
            "node_rule": node_rule,
            "edge_rule": "keep original Hetionet edges whose endpoints are selected nodes",
            "selection_seed": int(selection_seed),
            "max_nodes_per_class": int(max_nodes_per_class) if max_nodes_per_class is not None else None,
            "min_nodes_per_class": int(min_nodes_per_class),
            "drop_isolated_nodes": bool(drop_isolated_nodes),
            "uses_train_val_test_split": False,
        },
        "target_node_types": list(wanted_types),
        "skipped_node_types": skipped_types,
        "num_source_nodes": int(len(nodes)),
        "num_source_edges": int(len(edges)),
        "num_nodes_before_isolated_removal": int(len(keep_nodes)),
        "num_isolated_nodes_before_drop": isolated_nodes_before_drop,
        "removed_isolated_nodes": removed_isolated_nodes,
        "num_nodes": int(data.num_nodes),
        "num_edges_raw_kept": int(kept_raw_edge_count),
        "num_edges_directed": int(edge_count_directed),
        "num_edges_after_undirected": int(edge_count_after_undirected),
        "num_edges_processed": int(edge_index.shape[1]),
        "num_edges_after_dedup_before_isolated_removal": int(edge_count_after_dedup),
        "skipped_malformed_edges": int(skipped_edge_count),
        "make_undirected": bool(make_undirected),
        "deduplicate_edges": bool(deduplicate_edges),
        "remove_self_loops": bool(remove_self_loops),
        "edge_direction": "bidirectional_edge_index_entries" if make_undirected else "directed_edge_index_entries",
        "class_mapping": class_mapping,
        "class_counts_before_edges": selected_counts_before_edges,
        "class_counts": {
            label: int(class_counts[idx])
            for label, idx in sorted(class_mapping.items(), key=lambda item: item[1])
        },
        "all_node_type_counts": dict(sorted(all_type_counts.items())),
        "kept_relation_counts_raw": dict(sorted(relation_counts.items())),
        "feature_graph_scope": "constructed_homogeneous_graph",
        "feature_schema": FEATURE_SCHEMA,
        "feature_note": (
            "Entity-type one-hot, source/database indicators, and relation-type counts are intentionally "
            "excluded. Features are structure-only descriptors computed on the constructed homogeneous graph."
        ),
        "num_features": int(x.shape[1]),
        "feature_dim_requested": 20,
        "num_classes": int(len(class_mapping)),
    }

    processed.mkdir(parents=True, exist_ok=True)
    torch.save(data, data_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data, metadata


def _structure_features(
    edge_index: torch.Tensor,
    raw_in_degree: torch.Tensor,
    raw_out_degree: torch.Tensor,
    *,
    feature_dim: int,
) -> torch.Tensor:
    import networkx as nx

    num_nodes = int(raw_in_degree.numel())
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    edge_index_cpu = edge_index.detach().cpu()
    graph.add_edges_from(
        (int(src), int(dst))
        for src, dst in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
        if int(src) != int(dst)
    )

    degree = torch.tensor([float(graph.degree(node)) for node in range(num_nodes)], dtype=torch.float32)
    max_degree = degree.max().clamp(min=1.0)
    neighbor_mean = torch.zeros(num_nodes, dtype=torch.float32)
    neighbor_std = torch.zeros(num_nodes, dtype=torch.float32)
    neighbor_min = torch.zeros(num_nodes, dtype=torch.float32)
    neighbor_max = torch.zeros(num_nodes, dtype=torch.float32)
    for node in range(num_nodes):
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            continue
        values = degree[torch.tensor(neighbors, dtype=torch.long)]
        neighbor_mean[node] = values.mean()
        neighbor_std[node] = values.std(unbiased=False) if values.numel() > 1 else 0.0
        neighbor_min[node] = values.min()
        neighbor_max[node] = values.max()

    if graph.number_of_edges():
        clustering_map = nx.clustering(graph)
        triangle_map = nx.triangles(graph)
        core_map = nx.core_number(graph)
        pagerank_map = nx.pagerank(graph, alpha=0.85, max_iter=100, tol=1.0e-6)
    else:
        clustering_map = {node: 0.0 for node in range(num_nodes)}
        triangle_map = {node: 0 for node in range(num_nodes)}
        core_map = {node: 0 for node in range(num_nodes)}
        pagerank_map = {node: 1.0 / max(num_nodes, 1) for node in range(num_nodes)}

    clustering = torch.tensor([float(clustering_map[node]) for node in range(num_nodes)], dtype=torch.float32)
    triangles = torch.tensor([float(triangle_map[node]) for node in range(num_nodes)], dtype=torch.float32)
    core = torch.tensor([float(core_map[node]) for node in range(num_nodes)], dtype=torch.float32)
    pagerank = torch.tensor([float(pagerank_map[node]) for node in range(num_nodes)], dtype=torch.float32)

    raw_total = (raw_in_degree + raw_out_degree).clamp(min=1.0)
    columns = [
        torch.log1p(degree),
        torch.sqrt(degree),
        degree / max_degree,
        torch.log1p(raw_in_degree),
        torch.log1p(raw_out_degree),
        raw_in_degree / raw_total,
        raw_out_degree / raw_total,
        torch.log1p(neighbor_mean),
        _normalize_max(neighbor_mean),
        _normalize_max(neighbor_std),
        _normalize_max(neighbor_min),
        _normalize_max(neighbor_max),
        neighbor_mean / (degree + 1.0),
        clustering,
        torch.log1p(triangles),
        _normalize_max(triangles),
        _normalize_max(core),
        torch.log1p(core),
        pagerank,
        torch.log1p(pagerank * max(num_nodes, 1)),
    ]
    x = torch.stack(columns[:feature_dim], dim=1).to(torch.float32)
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_max(values: torch.Tensor) -> torch.Tensor:
    denom = values.max().clamp(min=1.0)
    return values / denom


def _hetionet_root(root: str | Path) -> Path:
    root_path = Path(root)
    legacy = root_path / "Planetoid" / "Hetionet"
    if legacy.exists():
        return legacy
    return root_path / "Hetionet"


def _find_raw_file(base: Path) -> Path | None:
    candidates = [
        base / RAW_FILE,
        base / "hetionet-v1.0.json",
        base / "hetionet-v1.0.json.gz",
        base / "hetionet.json",
        base / "hetionet.json.bz2",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _download_raw_file(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    raw_path = base / RAW_FILE
    urllib.request.urlretrieve(DEFAULT_DOWNLOAD_URL, raw_path)
    return raw_path


def _read_hetionet_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_node_types(raw: dict[str, Any], nodes: list[dict[str, Any]], node_types: tuple[str, ...] | None) -> tuple[str, ...]:
    if node_types is not None:
        return tuple(_canonical_type(name) for name in node_types)

    raw_types = raw.get("metanode_kinds")
    if isinstance(raw_types, list) and raw_types:
        return tuple(_canonical_type(str(name)) for name in raw_types)

    seen: list[str] = []
    seen_set: set[str] = set()
    for node in nodes:
        kind = _canonical_type(str(node.get("kind", "")))
        if kind and kind not in seen_set:
            seen.append(kind)
            seen_set.add(kind)
    return tuple(seen)


def _node_key(node: dict[str, Any]) -> tuple[str, str] | None:
    kind = node.get("kind")
    identifier = node.get("identifier", node.get("id"))
    if kind is None or identifier is None:
        return None
    return _canonical_type(str(kind)), str(identifier)


def _edge_node_key(value: Any) -> tuple[str, str] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return str(value[0]), str(value[1])
    if isinstance(value, dict):
        kind = value.get("kind") or value.get("type")
        identifier = value.get("identifier") or value.get("id")
        if kind is not None and identifier is not None:
            return str(kind), str(identifier)
    return None


def _canonical_type(value: str) -> str:
    cleaned = str(value).replace("_", " ").replace("-", " ").strip()
    aliases = {
        "SideEffect": "Side Effect",
        "Side effect": "Side Effect",
        "side effect": "Side Effect",
        "BiologicalProcess": "Biological Process",
        "Biological process": "Biological Process",
        "biological process": "Biological Process",
    }
    return aliases.get(cleaned, cleaned)

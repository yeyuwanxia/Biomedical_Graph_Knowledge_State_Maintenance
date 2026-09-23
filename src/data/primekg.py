from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


PRIMEKG_VARIANT = "primekg_homo"
PRIMEKG_FULL_NOSOURCE_VARIANT = "primekg_full_nosource"
PRIMEKG_DISEASE_GENE_SMALL_VARIANT = "primekg_disease_gene_small"
PRIMEKG_DISEASE_GENE_SMALL_NOSOURCE_VARIANT = "primekg_disease_gene_small_nosource"
RAW_FILE = "primekg.tab"
PROCESSED_DIR = "processed"
FULL_NOSOURCE_PROCESSED_DIR = "processed_full_nosource"
DISEASE_GENE_SMALL_PROCESSED_DIR = "processed_disease_gene_small"
DISEASE_GENE_SMALL_NOSOURCE_PROCESSED_DIR = "processed_disease_gene_small_nosource"
DATA_FILE = "data.pt"
METADATA_FILE = "metadata.json"


def primekg_cache_paths(root: str | Path) -> list[Path]:
    base = _primekg_root(root)
    return [base / PROCESSED_DIR, base]


def primekg_full_nosource_cache_paths(root: str | Path) -> list[Path]:
    base = _primekg_root(root)
    return [base / FULL_NOSOURCE_PROCESSED_DIR, base]


def primekg_disease_gene_small_cache_paths(root: str | Path) -> list[Path]:
    base = _primekg_root(root)
    return [base / DISEASE_GENE_SMALL_PROCESSED_DIR, base]


def primekg_disease_gene_small_nosource_cache_paths(root: str | Path) -> list[Path]:
    base = _primekg_root(root)
    return [base / DISEASE_GENE_SMALL_NOSOURCE_PROCESSED_DIR, base]


def load_primekg_homo(
    root: str | Path,
    *,
    min_class_count: int = 20,
    make_undirected: bool = True,
    deduplicate_edges: bool = True,
    remove_self_loops: bool = True,
    force_rebuild: bool = False,
    chunksize: int = 500_000,
):
    """Load or build a homogeneous node-type classification graph from PrimeKG.

    The task label is the biomedical entity type. Input features deliberately do
    not include entity-type one-hot features to avoid direct label leakage.
    """

    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise SystemExit(
            "torch-geometric is required to build/load PrimeKG-Homo. "
            "Use the graphunlearning environment or install torch-geometric."
        ) from exc

    base = _primekg_root(root)
    processed = base / PROCESSED_DIR
    data_path = processed / DATA_FILE
    metadata_path = processed / METADATA_FILE
    if data_path.exists() and metadata_path.exists() and not force_rebuild:
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return data, metadata

    raw_path = base / RAW_FILE
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing PrimeKG raw file: {raw_path}")

    import pandas as pd

    node_to_idx: dict[tuple[str, str, str], int] = {}
    node_types: list[str] = []
    node_sources: list[str] = []
    relation_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()

    for chunk in pd.read_csv(raw_path, usecols=_RAW_COLUMNS, chunksize=chunksize, dtype=str):
        _normalize_chunk(chunk)
        relation_counter.update(chunk["relation"].tolist())
        for side in ("x", "y"):
            ids = chunk[f"{side}_id"].tolist()
            types = chunk[f"{side}_type"].tolist()
            sources = chunk[f"{side}_source"].tolist()
            for node_id, node_type, source in zip(ids, types, sources):
                key = (str(node_type), str(source), str(node_id))
                if key in node_to_idx:
                    continue
                node_to_idx[key] = len(node_types)
                node_types.append(str(node_type))
                node_sources.append(str(source))
                type_counter[str(node_type)] += 1
                source_counter[str(source)] += 1

    relation_mapping = {name: idx for idx, name in enumerate(sorted(relation_counter))}
    source_mapping = {name: idx for idx, name in enumerate(sorted(source_counter))}
    class_mapping, merged_small_classes = _build_class_mapping(type_counter, int(min_class_count))

    num_nodes = len(node_types)
    num_relations = len(relation_mapping)
    num_sources = len(source_mapping)
    raw_in_degree = torch.zeros(num_nodes, dtype=torch.float32)
    raw_out_degree = torch.zeros(num_nodes, dtype=torch.float32)
    relation_in = torch.zeros((num_nodes, num_relations), dtype=torch.float32)
    relation_out = torch.zeros((num_nodes, num_relations), dtype=torch.float32)
    source_x = torch.empty(num_nodes, dtype=torch.long)
    y = torch.empty(num_nodes, dtype=torch.long)

    for idx, (node_type, source) in enumerate(zip(node_types, node_sources)):
        y[idx] = class_mapping[node_type] if node_type in class_mapping else class_mapping["other"]
        source_x[idx] = source_mapping[source]

    source_nodes: list[torch.Tensor] = []
    target_nodes: list[torch.Tensor] = []
    raw_edge_count = 0
    for chunk in pd.read_csv(raw_path, usecols=_RAW_COLUMNS, chunksize=chunksize, dtype=str):
        _normalize_chunk(chunk)
        raw_edge_count += int(len(chunk))
        src = torch.tensor(
            [
                node_to_idx[(str(node_type), str(source), str(node_id))]
                for node_type, source, node_id in zip(chunk["x_type"], chunk["x_source"], chunk["x_id"])
            ],
            dtype=torch.long,
        )
        dst = torch.tensor(
            [
                node_to_idx[(str(node_type), str(source), str(node_id))]
                for node_type, source, node_id in zip(chunk["y_type"], chunk["y_source"], chunk["y_id"])
            ],
            dtype=torch.long,
        )
        rel = torch.tensor([relation_mapping[str(value)] for value in chunk["relation"]], dtype=torch.long)

        raw_out_degree.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        raw_in_degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        relation_out.index_put_((src, rel), torch.ones_like(src, dtype=torch.float32), accumulate=True)
        relation_in.index_put_((dst, rel), torch.ones_like(dst, dtype=torch.float32), accumulate=True)
        source_nodes.append(src)
        target_nodes.append(dst)

    edge_index = torch.stack([torch.cat(source_nodes), torch.cat(target_nodes)], dim=0)
    edge_count_directed = int(edge_index.shape[1])
    if remove_self_loops:
        keep = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, keep]
    edge_count_after_self_loop_removal = int(edge_index.shape[1])
    if make_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_count_after_undirected = int(edge_index.shape[1])
    if deduplicate_edges:
        edge_index = torch.unique(edge_index, dim=1)
    edge_count_after_dedup = int(edge_index.shape[1])

    processed_degree = torch.bincount(edge_index[0], minlength=num_nodes).to(torch.float32)
    relation_total = relation_in + relation_out
    relation_total = _row_normalize(relation_total)
    source_features = torch.nn.functional.one_hot(source_x, num_classes=num_sources).to(torch.float32)
    x = torch.cat(
        [
            torch.log1p(processed_degree).unsqueeze(1),
            torch.log1p(raw_in_degree).unsqueeze(1),
            torch.log1p(raw_out_degree).unsqueeze(1),
            relation_total,
            source_features,
        ],
        dim=1,
    )

    data = Data(x=x, edge_index=edge_index.contiguous(), y=y)
    data.num_nodes = num_nodes
    data.primekg_node_type = node_types
    data.primekg_node_source = node_sources

    class_counts = torch.bincount(y, minlength=len(class_mapping)).tolist()
    metadata: dict[str, Any] = {
        "dataset": "primekg",
        "dataset_variant": PRIMEKG_VARIANT,
        "original_graph_type": "heterogeneous_biomedical_knowledge_graph",
        "task": "node_type_classification",
        "raw_path": str(raw_path),
        "processed_data": str(data_path),
        "num_nodes": int(num_nodes),
        "num_edges_raw": int(raw_edge_count),
        "num_edges_directed": int(edge_count_directed),
        "num_edges_after_self_loop_removal": int(edge_count_after_self_loop_removal),
        "num_edges_after_undirected": int(edge_count_after_undirected),
        "num_edges_processed": int(edge_count_after_dedup),
        "make_undirected": bool(make_undirected),
        "deduplicate_edges": bool(deduplicate_edges),
        "remove_self_loops": bool(remove_self_loops),
        "min_class_count": int(min_class_count),
        "merged_small_classes": merged_small_classes,
        "class_mapping": class_mapping,
        "class_counts_before_merge": dict(sorted(type_counter.items())),
        "class_counts_after_merge": {
            label: int(class_counts[idx])
            for label, idx in sorted(class_mapping.items(), key=lambda item: item[1])
        },
        "relation_mapping": relation_mapping,
        "source_mapping": source_mapping,
        "feature_graph_scope": "full_graph",
        "feature_schema": [
            "log_processed_degree",
            "log_raw_in_degree",
            "log_raw_out_degree",
            "normalized_relation_type_counts",
            "node_source_one_hot",
        ],
        "feature_note": "Entity-type one-hot is intentionally excluded because y is entity type.",
        "num_features": int(x.shape[1]),
        "num_classes": int(len(class_mapping)),
    }

    processed.mkdir(parents=True, exist_ok=True)
    torch.save(data, data_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data, metadata


def load_primekg_full_nosource(
    root: str | Path,
    *,
    min_class_count: int = 20,
    make_undirected: bool = True,
    deduplicate_edges: bool = True,
    remove_self_loops: bool = True,
    force_rebuild: bool = False,
    chunksize: int = 500_000,
):
    """Load the full PrimeKG graph without source-derived input features.

    This variant preserves the full graph, node ordering, and labels from
    ``load_primekg_homo``. It removes source one-hot and relation-type count
    features because both are strongly correlated with the node-type label.
    """

    base = _primekg_root(root)
    processed = base / FULL_NOSOURCE_PROCESSED_DIR
    data_path = processed / DATA_FILE
    metadata_path = processed / METADATA_FILE
    if data_path.exists() and metadata_path.exists() and not force_rebuild:
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return data, metadata

    source_data, source_metadata = load_primekg_homo(
        root,
        min_class_count=min_class_count,
        make_undirected=make_undirected,
        deduplicate_edges=deduplicate_edges,
        remove_self_loops=remove_self_loops,
        force_rebuild=False,
        chunksize=chunksize,
    )
    relation_width = len(source_metadata.get("relation_mapping", {}))
    source_width = len(source_metadata.get("source_mapping", {}))
    retained_width = 3
    expected_width = retained_width + relation_width + source_width
    actual_width = int(source_data.x.shape[1])
    if actual_width != expected_width:
        raise ValueError(
            "PrimeKG feature schema does not match the expected "
            f"3 + relation({relation_width}) + source({source_width}) layout: "
            f"actual={actual_width}, expected={expected_width}"
        )

    data = source_data.clone()
    data.x = source_data.x[:, :retained_width].clone()
    if hasattr(data, "primekg_node_source"):
        del data.primekg_node_source

    metadata = deepcopy(source_metadata)
    metadata.update(
        {
            "dataset": "primekg-full-nosource",
            "dataset_variant": PRIMEKG_FULL_NOSOURCE_VARIANT,
            "processed_data": str(data_path),
            "derived_from_processed_data": source_metadata.get("processed_data"),
            "feature_schema": [
                "log_processed_degree",
                "log_raw_in_degree",
                "log_raw_out_degree",
            ],
            "feature_note": (
                "Entity-type one-hot, node-source one-hot, and relation-type counts "
                "are intentionally excluded because y is entity type and both "
                "source and relation semantics are label-correlated."
            ),
            "feature_protocol": "structure_only_v1",
            "source_feature_removed": True,
            "removed_source_feature_count": int(source_width),
            "relation_type_features_removed": True,
            "removed_relation_feature_count": int(relation_width),
            "num_features": int(data.x.shape[1]),
        }
    )

    processed.mkdir(parents=True, exist_ok=True)
    torch.save(data, data_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data, metadata


def load_primekg_disease_gene_small(
    root: str | Path,
    *,
    make_undirected: bool = True,
    deduplicate_edges: bool = True,
    remove_self_loops: bool = True,
    force_rebuild: bool = False,
    chunksize: int = 500_000,
):
    """Load or build a compact biomedical PrimeKG disease-gene/protein graph.

    Nodes are endpoints of disease_protein edges. Edges are disease_protein
    plus protein_protein edges whose endpoints are disease-linked proteins.
    The rule is deterministic and independent of train/val/test splits.
    """

    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise SystemExit(
            "torch-geometric is required to build/load PrimeKG-DiseaseGene-Small. "
            "Use the graphunlearning environment or install torch-geometric."
        ) from exc

    base = _primekg_root(root)
    processed = base / DISEASE_GENE_SMALL_PROCESSED_DIR
    data_path = processed / DATA_FILE
    metadata_path = processed / METADATA_FILE
    if data_path.exists() and metadata_path.exists() and not force_rebuild:
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return data, metadata

    raw_path = base / RAW_FILE
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing PrimeKG raw file: {raw_path}")

    import pandas as pd

    node_to_idx: dict[tuple[str, str, str], int] = {}
    node_types: list[str] = []
    node_sources: list[str] = []
    disease_linked_proteins: set[tuple[str, str, str]] = set()
    relation_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    disease_protein_raw_edges = 0

    for chunk in pd.read_csv(raw_path, usecols=_RAW_COLUMNS, chunksize=chunksize, dtype=str):
        _normalize_chunk(chunk)
        chunk = chunk[chunk["relation"] == "disease_protein"]
        disease_protein_raw_edges += int(len(chunk))
        relation_counter.update(chunk["relation"].tolist())
        for side in ("x", "y"):
            ids = chunk[f"{side}_id"].tolist()
            types = chunk[f"{side}_type"].tolist()
            sources = chunk[f"{side}_source"].tolist()
            for node_id, node_type, source in zip(ids, types, sources):
                key = (str(node_type), str(source), str(node_id))
                if str(node_type) == "gene/protein":
                    disease_linked_proteins.add(key)
                if key in node_to_idx:
                    continue
                node_to_idx[key] = len(node_types)
                node_types.append(str(node_type))
                node_sources.append(str(source))
                type_counter[str(node_type)] += 1
                source_counter[str(source)] += 1

    if not node_to_idx:
        raise ValueError("No disease_protein edges found in PrimeKG raw file.")

    class_labels = ["disease", "gene/protein"]
    class_mapping = {label: idx for idx, label in enumerate(class_labels)}
    relation_mapping = {name: idx for idx, name in enumerate(["disease_protein", "protein_protein"])}
    source_mapping = {name: idx for idx, name in enumerate(sorted(source_counter))}

    num_nodes = len(node_types)
    num_relations = len(relation_mapping)
    num_sources = len(source_mapping)
    raw_in_degree = torch.zeros(num_nodes, dtype=torch.float32)
    raw_out_degree = torch.zeros(num_nodes, dtype=torch.float32)
    relation_in = torch.zeros((num_nodes, num_relations), dtype=torch.float32)
    relation_out = torch.zeros((num_nodes, num_relations), dtype=torch.float32)
    source_x = torch.empty(num_nodes, dtype=torch.long)
    y = torch.empty(num_nodes, dtype=torch.long)

    for idx, (node_type, source) in enumerate(zip(node_types, node_sources)):
        if node_type not in class_mapping:
            raise ValueError(f"Unexpected node type in disease-gene subgraph: {node_type!r}")
        y[idx] = class_mapping[node_type]
        source_x[idx] = source_mapping[source]

    source_nodes: list[torch.Tensor] = []
    target_nodes: list[torch.Tensor] = []
    kept_raw_edge_count = 0
    kept_relation_counts: Counter[str] = Counter()
    for chunk in pd.read_csv(raw_path, usecols=_RAW_COLUMNS, chunksize=chunksize, dtype=str):
        _normalize_chunk(chunk)
        relation = chunk["relation"]
        chunk = chunk[(relation == "disease_protein") | (relation == "protein_protein")]
        if chunk.empty:
            continue

        kept_rows: list[tuple[str, str, str, str, str, str, str]] = []
        for row in chunk[_RAW_COLUMNS].itertuples(index=False, name=None):
            rel, x_id, x_type, x_source, y_id, y_type, y_source = row
            src_key = (str(x_type), str(x_source), str(x_id))
            dst_key = (str(y_type), str(y_source), str(y_id))
            if rel == "disease_protein":
                if src_key in node_to_idx and dst_key in node_to_idx:
                    kept_rows.append(row)
            elif rel == "protein_protein":
                if src_key in disease_linked_proteins and dst_key in disease_linked_proteins:
                    kept_rows.append(row)

        if not kept_rows:
            continue

        kept_raw_edge_count += len(kept_rows)
        kept_relation_counts.update(str(row[0]) for row in kept_rows)
        src = torch.tensor([node_to_idx[(str(row[2]), str(row[3]), str(row[1]))] for row in kept_rows], dtype=torch.long)
        dst = torch.tensor([node_to_idx[(str(row[5]), str(row[6]), str(row[4]))] for row in kept_rows], dtype=torch.long)
        rel = torch.tensor([relation_mapping[str(row[0])] for row in kept_rows], dtype=torch.long)

        raw_out_degree.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        raw_in_degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        relation_out.index_put_((src, rel), torch.ones_like(src, dtype=torch.float32), accumulate=True)
        relation_in.index_put_((dst, rel), torch.ones_like(dst, dtype=torch.float32), accumulate=True)
        source_nodes.append(src)
        target_nodes.append(dst)

    edge_index = torch.stack([torch.cat(source_nodes), torch.cat(target_nodes)], dim=0)
    edge_count_directed = int(edge_index.shape[1])
    if remove_self_loops:
        keep = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, keep]
    edge_count_after_self_loop_removal = int(edge_index.shape[1])
    if make_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_count_after_undirected = int(edge_index.shape[1])
    if deduplicate_edges:
        edge_index = torch.unique(edge_index, dim=1)
    edge_count_after_dedup = int(edge_index.shape[1])

    processed_degree = torch.bincount(edge_index[0], minlength=num_nodes).to(torch.float32)
    relation_total = _row_normalize(relation_in + relation_out)
    source_features = torch.nn.functional.one_hot(source_x, num_classes=num_sources).to(torch.float32)
    x = torch.cat(
        [
            torch.log1p(processed_degree).unsqueeze(1),
            torch.log1p(raw_in_degree).unsqueeze(1),
            torch.log1p(raw_out_degree).unsqueeze(1),
            relation_total,
            source_features,
        ],
        dim=1,
    )

    data = Data(x=x, edge_index=edge_index.contiguous(), y=y)
    data.num_nodes = num_nodes
    data.primekg_node_type = node_types
    data.primekg_node_source = node_sources

    class_counts = torch.bincount(y, minlength=len(class_mapping)).tolist()
    metadata: dict[str, Any] = {
        "dataset": "primekg-disease-gene-small",
        "dataset_variant": PRIMEKG_DISEASE_GENE_SMALL_VARIANT,
        "source_dataset": "primekg",
        "original_graph_type": "heterogeneous_biomedical_knowledge_graph",
        "task": "node_type_classification",
        "raw_path": str(raw_path),
        "processed_data": str(data_path),
        "construction_protocol": {
            "node_rule": "keep endpoints of disease_protein edges",
            "edge_rule": "keep disease_protein edges and protein_protein edges whose endpoints are disease-linked proteins",
            "selection_seed": None,
            "deterministic": True,
            "uses_train_val_test_split": False,
        },
        "num_nodes": int(num_nodes),
        "num_edges_raw_disease_protein": int(disease_protein_raw_edges),
        "num_edges_raw_kept": int(kept_raw_edge_count),
        "num_edges_directed": int(edge_count_directed),
        "num_edges_after_self_loop_removal": int(edge_count_after_self_loop_removal),
        "num_edges_after_undirected": int(edge_count_after_undirected),
        "num_edges_processed": int(edge_count_after_dedup),
        "make_undirected": bool(make_undirected),
        "deduplicate_edges": bool(deduplicate_edges),
        "remove_self_loops": bool(remove_self_loops),
        "class_mapping": class_mapping,
        "class_counts": {
            label: int(class_counts[idx])
            for label, idx in sorted(class_mapping.items(), key=lambda item: item[1])
        },
        "kept_relation_counts_raw": dict(sorted(kept_relation_counts.items())),
        "relation_mapping": relation_mapping,
        "source_mapping": source_mapping,
        "feature_graph_scope": "constructed_subgraph",
        "feature_schema": [
            "log_processed_degree",
            "log_raw_in_degree",
            "log_raw_out_degree",
            "normalized_relation_type_counts",
            "node_source_one_hot",
        ],
        "feature_note": "Entity-type one-hot is intentionally excluded because y is entity type.",
        "num_features": int(x.shape[1]),
        "num_classes": int(len(class_mapping)),
    }

    processed.mkdir(parents=True, exist_ok=True)
    torch.save(data, data_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data, metadata


def load_primekg_disease_gene_small_nosource(
    root: str | Path,
    *,
    make_undirected: bool = True,
    deduplicate_edges: bool = True,
    remove_self_loops: bool = True,
    force_rebuild: bool = False,
    chunksize: int = 500_000,
):
    """Load or build the no-source PrimeKG disease-gene/protein graph.

    This variant keeps the same nodes, labels, and edges as
    load_primekg_disease_gene_small, but excludes node_source_one_hot
    from x because source is a strong proxy for the entity-type label.
    """

    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise SystemExit(
            "torch-geometric is required to build/load PrimeKG-DiseaseGene-Small-NoSource. "
            "Use the graphunlearning environment or install torch-geometric."
        ) from exc

    base = _primekg_root(root)
    processed = base / DISEASE_GENE_SMALL_NOSOURCE_PROCESSED_DIR
    data_path = processed / DATA_FILE
    metadata_path = processed / METADATA_FILE
    if data_path.exists() and metadata_path.exists() and not force_rebuild:
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return data, metadata

    raw_path = base / RAW_FILE
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing PrimeKG raw file: {raw_path}")

    import pandas as pd

    node_to_idx: dict[tuple[str, str, str], int] = {}
    node_types: list[str] = []
    node_sources: list[str] = []
    disease_linked_proteins: set[tuple[str, str, str]] = set()
    relation_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    disease_protein_raw_edges = 0

    for chunk in pd.read_csv(raw_path, usecols=_RAW_COLUMNS, chunksize=chunksize, dtype=str):
        _normalize_chunk(chunk)
        chunk = chunk[chunk["relation"] == "disease_protein"]
        disease_protein_raw_edges += int(len(chunk))
        relation_counter.update(chunk["relation"].tolist())
        for side in ("x", "y"):
            ids = chunk[f"{side}_id"].tolist()
            types = chunk[f"{side}_type"].tolist()
            sources = chunk[f"{side}_source"].tolist()
            for node_id, node_type, source in zip(ids, types, sources):
                key = (str(node_type), str(source), str(node_id))
                if str(node_type) == "gene/protein":
                    disease_linked_proteins.add(key)
                if key in node_to_idx:
                    continue
                node_to_idx[key] = len(node_types)
                node_types.append(str(node_type))
                node_sources.append(str(source))
                type_counter[str(node_type)] += 1
                source_counter[str(source)] += 1

    if not node_to_idx:
        raise ValueError("No disease_protein edges found in PrimeKG raw file.")

    class_labels = ["disease", "gene/protein"]
    class_mapping = {label: idx for idx, label in enumerate(class_labels)}
    relation_mapping = {name: idx for idx, name in enumerate(["disease_protein", "protein_protein"])}
    source_mapping = {name: idx for idx, name in enumerate(sorted(source_counter))}

    num_nodes = len(node_types)
    num_relations = len(relation_mapping)
    num_sources = len(source_mapping)
    raw_in_degree = torch.zeros(num_nodes, dtype=torch.float32)
    raw_out_degree = torch.zeros(num_nodes, dtype=torch.float32)
    relation_in = torch.zeros((num_nodes, num_relations), dtype=torch.float32)
    relation_out = torch.zeros((num_nodes, num_relations), dtype=torch.float32)
    source_x = torch.empty(num_nodes, dtype=torch.long)
    y = torch.empty(num_nodes, dtype=torch.long)

    for idx, (node_type, source) in enumerate(zip(node_types, node_sources)):
        if node_type not in class_mapping:
            raise ValueError(f"Unexpected node type in disease-gene subgraph: {node_type!r}")
        y[idx] = class_mapping[node_type]
        source_x[idx] = source_mapping[source]

    source_nodes: list[torch.Tensor] = []
    target_nodes: list[torch.Tensor] = []
    kept_raw_edge_count = 0
    kept_relation_counts: Counter[str] = Counter()
    for chunk in pd.read_csv(raw_path, usecols=_RAW_COLUMNS, chunksize=chunksize, dtype=str):
        _normalize_chunk(chunk)
        relation = chunk["relation"]
        chunk = chunk[(relation == "disease_protein") | (relation == "protein_protein")]
        if chunk.empty:
            continue

        kept_rows: list[tuple[str, str, str, str, str, str, str]] = []
        for row in chunk[_RAW_COLUMNS].itertuples(index=False, name=None):
            rel, x_id, x_type, x_source, y_id, y_type, y_source = row
            src_key = (str(x_type), str(x_source), str(x_id))
            dst_key = (str(y_type), str(y_source), str(y_id))
            if rel == "disease_protein":
                if src_key in node_to_idx and dst_key in node_to_idx:
                    kept_rows.append(row)
            elif rel == "protein_protein":
                if src_key in disease_linked_proteins and dst_key in disease_linked_proteins:
                    kept_rows.append(row)

        if not kept_rows:
            continue

        kept_raw_edge_count += len(kept_rows)
        kept_relation_counts.update(str(row[0]) for row in kept_rows)
        src = torch.tensor([node_to_idx[(str(row[2]), str(row[3]), str(row[1]))] for row in kept_rows], dtype=torch.long)
        dst = torch.tensor([node_to_idx[(str(row[5]), str(row[6]), str(row[4]))] for row in kept_rows], dtype=torch.long)
        rel = torch.tensor([relation_mapping[str(row[0])] for row in kept_rows], dtype=torch.long)

        raw_out_degree.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        raw_in_degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        relation_out.index_put_((src, rel), torch.ones_like(src, dtype=torch.float32), accumulate=True)
        relation_in.index_put_((dst, rel), torch.ones_like(dst, dtype=torch.float32), accumulate=True)
        source_nodes.append(src)
        target_nodes.append(dst)

    edge_index = torch.stack([torch.cat(source_nodes), torch.cat(target_nodes)], dim=0)
    edge_count_directed = int(edge_index.shape[1])
    if remove_self_loops:
        keep = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, keep]
    edge_count_after_self_loop_removal = int(edge_index.shape[1])
    if make_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_count_after_undirected = int(edge_index.shape[1])
    if deduplicate_edges:
        edge_index = torch.unique(edge_index, dim=1)
    edge_count_after_dedup = int(edge_index.shape[1])

    processed_degree = torch.bincount(edge_index[0], minlength=num_nodes).to(torch.float32)
    relation_total = _row_normalize(relation_in + relation_out)
    x = torch.cat(
        [
            torch.log1p(processed_degree).unsqueeze(1),
            torch.log1p(raw_in_degree).unsqueeze(1),
            torch.log1p(raw_out_degree).unsqueeze(1),
            relation_total,
        ],
        dim=1,
    )

    data = Data(x=x, edge_index=edge_index.contiguous(), y=y)
    data.num_nodes = num_nodes
    data.primekg_node_type = node_types
    data.primekg_node_source = node_sources

    class_counts = torch.bincount(y, minlength=len(class_mapping)).tolist()
    metadata: dict[str, Any] = {
        "dataset": "primekg-disease-gene-small-nosource",
        "dataset_variant": PRIMEKG_DISEASE_GENE_SMALL_NOSOURCE_VARIANT,
        "source_dataset": "primekg",
        "original_graph_type": "heterogeneous_biomedical_knowledge_graph",
        "task": "node_type_classification",
        "raw_path": str(raw_path),
        "processed_data": str(data_path),
        "construction_protocol": {
            "node_rule": "keep endpoints of disease_protein edges",
            "edge_rule": "keep disease_protein edges and protein_protein edges whose endpoints are disease-linked proteins",
            "selection_seed": None,
            "deterministic": True,
            "uses_train_val_test_split": False,
        },
        "num_nodes": int(num_nodes),
        "num_edges_raw_disease_protein": int(disease_protein_raw_edges),
        "num_edges_raw_kept": int(kept_raw_edge_count),
        "num_edges_directed": int(edge_count_directed),
        "num_edges_after_self_loop_removal": int(edge_count_after_self_loop_removal),
        "num_edges_after_undirected": int(edge_count_after_undirected),
        "num_edges_processed": int(edge_count_after_dedup),
        "make_undirected": bool(make_undirected),
        "deduplicate_edges": bool(deduplicate_edges),
        "remove_self_loops": bool(remove_self_loops),
        "class_mapping": class_mapping,
        "class_counts": {
            label: int(class_counts[idx])
            for label, idx in sorted(class_mapping.items(), key=lambda item: item[1])
        },
        "kept_relation_counts_raw": dict(sorted(kept_relation_counts.items())),
        "relation_mapping": relation_mapping,
        "source_mapping": source_mapping,
        "feature_graph_scope": "constructed_subgraph",
        "feature_schema": [
            "log_processed_degree",
            "log_raw_in_degree",
            "log_raw_out_degree",
            "normalized_relation_type_counts",
        ],
        "feature_note": "Entity-type one-hot and node-source one-hot are intentionally excluded because y is entity type and source is label-correlated.",
        "num_features": int(x.shape[1]),
        "num_classes": int(len(class_mapping)),
    }

    processed.mkdir(parents=True, exist_ok=True)
    torch.save(data, data_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data, metadata


_RAW_COLUMNS = [
    "relation",
    "x_id",
    "x_type",
    "x_source",
    "y_id",
    "y_type",
    "y_source",
]


def _primekg_root(root: str | Path) -> Path:
    root_path = Path(root)
    legacy = root_path / "Planetoid" / "PrimeKG"
    if legacy.exists():
        return legacy
    return root_path / "PrimeKG"


def _normalize_chunk(chunk) -> None:
    for col in _RAW_COLUMNS:
        chunk[col] = chunk[col].fillna("unknown").astype(str)


def _build_class_mapping(type_counter: Counter[str], min_class_count: int) -> tuple[dict[str, int], list[str]]:
    kept = sorted(node_type for node_type, count in type_counter.items() if count >= min_class_count)
    merged = sorted(node_type for node_type, count in type_counter.items() if count < min_class_count)
    labels = kept + (["other"] if merged else [])
    return {label: idx for idx, label in enumerate(labels)}, merged


def _row_normalize(values: torch.Tensor) -> torch.Tensor:
    denom = values.sum(dim=1, keepdim=True).clamp(min=1.0)
    return values / denom

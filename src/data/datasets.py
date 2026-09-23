from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


PLANETOID_NAMES = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}
PRIMEKG_NAMES = {"primekg", "primekg-homo"}
PRIMEKG_FULL_NOSOURCE_NAMES = {
    "primekg-full-nosource",
    "primekg-homo-nosource",
}
PRIMEKG_DISEASE_GENE_SMALL_NAMES = {
    "primekg-disease-gene-small",
    "primekg-diseasegene-small",
    "primekg-dg-small",
}
PRIMEKG_DISEASE_GENE_SMALL_NOSOURCE_NAMES = {
    "primekg-disease-gene-small-nosource",
    "primekg-diseasegene-small-nosource",
    "primekg-dg-small-nosource",
}
HETIONET_SMALL_NOSOURCE_NAMES = {
    "hetionet-small-nosource",
    "hetionet-nosource",
    "hetionet-small",
}
HETIONET_FULL_NOSOURCE_NAMES = {
    "hetionet-full-nosource",
    "hetionet-full",
}
PPI_HOMO_SL_FILTERED_NAMES = {
    "ppi-homo-sl-filtered",
    "ppi-homo-sl",
    "ppi-homo",
}
PPI_INDUCTIVE_SL_FILTERED_NAMES = {
    "ppi-inductive-sl-filtered",
    "ppi-inductive-sl",
    "ppi-inductive",
}
PPI_INDUCTIVE_SL_MOSTFREQ_FILTERED_NAMES = {
    "ppi-inductive-sl-mostfreq-filtered",
    "ppi-inductive-mostfreq",
    "ppi-inductive-sl-mostfreq",
}
PPI_INDUCTIVE_SL_BALANCED_FILTERED_EXAMPLES = {
    "ppi-inductive-sl-balanced20-filtered",
    "ppi-inductive-sl-balanced10-filtered",
}


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    data: object
    num_features: int
    num_classes: int
    root: Path


def load_dataset(name: str, root: str | Path = "data/raw", normalize: bool = True, download: bool = True) -> DatasetBundle:
    """Load a graph dataset into a PyG Data object.

    Supported names include cora, citeseer, pubmed, primekg, primekg-full-nosource,
    primekg-disease-gene-small, primekg-disease-gene-small-nosource,
    hetionet-small-nosource, hetionet-full-nosource,
    ppi-homo-sl-filtered, ppi-inductive-sl-filtered, ppi-inductive-sl-mostfreq-filtered,
    ppi-inductive-sl-balanced{K}-filtered, and reddit.
    The function keeps downloads in root.
    """

    normalized_name = normalize_dataset_name(name)
    root_path = Path(root)
    if not download and not dataset_cache_exists(normalized_name, root_path):
        raise SystemExit(
            f"Dataset {normalized_name!r} is not present under {root_path}. "
            "Run experiments/download_datasets.py first, or pass --allow_download."
        )

    if normalized_name in PLANETOID_NAMES:
        dataset = _load_planetoid(normalized_name, root_path, normalize)
        return DatasetBundle(
            name=normalized_name,
            data=dataset[0],
            num_features=int(dataset.num_features),
            num_classes=int(dataset.num_classes),
            root=root_path,
        )

    if normalized_name in PRIMEKG_NAMES:
        data, metadata = _load_primekg(root_path)
        return DatasetBundle(
            name="primekg",
            data=data,
            num_features=int(data.x.shape[1]),
            num_classes=int(metadata["num_classes"]),
            root=root_path,
        )

    if normalized_name in PRIMEKG_FULL_NOSOURCE_NAMES:
        data, metadata = _load_primekg_full_nosource(root_path)
        return DatasetBundle(
            name="primekg-full-nosource",
            data=data,
            num_features=int(data.x.shape[1]),
            num_classes=int(metadata["num_classes"]),
            root=root_path,
        )

    if normalized_name in PRIMEKG_DISEASE_GENE_SMALL_NAMES:
        data, metadata = _load_primekg_disease_gene_small(root_path)
        return DatasetBundle(
            name="primekg-disease-gene-small",
            data=data,
            num_features=int(data.x.shape[1]),
            num_classes=int(metadata["num_classes"]),
            root=root_path,
        )

    if normalized_name in PRIMEKG_DISEASE_GENE_SMALL_NOSOURCE_NAMES:
        data, metadata = _load_primekg_disease_gene_small_nosource(root_path)
        return DatasetBundle(
            name="primekg-disease-gene-small-nosource",
            data=data,
            num_features=int(data.x.shape[1]),
            num_classes=int(metadata["num_classes"]),
            root=root_path,
        )

    if normalized_name in HETIONET_SMALL_NOSOURCE_NAMES:
        data, metadata = _load_hetionet_small_nosource(root_path, download=download)
        return DatasetBundle(
            name="hetionet-small-nosource",
            data=data,
            num_features=int(data.x.shape[1]),
            num_classes=int(metadata["num_classes"]),
            root=root_path,
        )

    if normalized_name in HETIONET_FULL_NOSOURCE_NAMES:
        data, metadata = _load_hetionet_full_nosource(root_path, download=download)
        return DatasetBundle(
            name="hetionet-full-nosource",
            data=data,
            num_features=int(data.x.shape[1]),
            num_classes=int(metadata["num_classes"]),
            root=root_path,
        )

    if normalized_name in PPI_HOMO_SL_FILTERED_NAMES:
        data, metadata = _load_ppi_homo_sl_filtered(root_path)
        return DatasetBundle(
            name="ppi-homo-sl-filtered",
            data=data,
            num_features=int(data.x.shape[1]),
            num_classes=int(metadata["num_classes"]),
            root=root_path,
        )

    if normalized_name in PPI_INDUCTIVE_SL_FILTERED_NAMES:
        data, metadata = _load_ppi_inductive_sl_filtered(root_path)
        return DatasetBundle(
            name="ppi-inductive-sl-filtered",
            data=data,
            num_features=int(data.x.shape[1]),
            num_classes=int(metadata["num_classes"]),
            root=root_path,
        )

    if normalized_name in PPI_INDUCTIVE_SL_MOSTFREQ_FILTERED_NAMES:
        data, metadata = _load_ppi_inductive_sl_mostfreq_filtered(root_path)
        return DatasetBundle(
            name="ppi-inductive-sl-mostfreq-filtered",
            data=data,
            num_features=int(data.x.shape[1]),
            num_classes=int(metadata["num_classes"]),
            root=root_path,
        )

    balanced_target = _ppi_inductive_sl_balanced_target(normalized_name)
    if balanced_target is not None:
        data, metadata = _load_ppi_inductive_sl_balanced_filtered(root_path, balanced_target)
        return DatasetBundle(
            name=f"ppi-inductive-sl-balanced{balanced_target}-filtered",
            data=data,
            num_features=int(data.x.shape[1]),
            num_classes=int(metadata["num_classes"]),
            root=root_path,
        )

    if normalized_name == "reddit":
        dataset = _load_reddit(root_path, normalize)
        return DatasetBundle(
            name=normalized_name,
            data=dataset[0],
            num_features=int(dataset.num_features),
            num_classes=int(dataset.num_classes),
            root=root_path,
        )

    supported = ", ".join([*PLANETOID_NAMES, "primekg", "primekg-full-nosource", "primekg-disease-gene-small", "primekg-disease-gene-small-nosource", "hetionet-small-nosource", "hetionet-full-nosource", "ppi-homo-sl-filtered", "ppi-inductive-sl-filtered", "ppi-inductive-sl-mostfreq-filtered", "ppi-inductive-sl-balanced{K}-filtered", "reddit"])
    raise ValueError(f"Unsupported dataset {name!r}. Supported datasets: {supported}")


def normalize_dataset_name(name: str) -> str:
    return str(name).lower().replace("_", "-")


def dataset_cache_exists(name: str, root: str | Path = "data/raw") -> bool:
    """Return whether a dataset appears to be downloaded/processed locally."""

    normalized_name = normalize_dataset_name(name)
    root_path = Path(root)
    return any(_has_files(path) for path in dataset_cache_paths(normalized_name, root_path))


def dataset_cache_paths(name: str, root: str | Path = "data/raw") -> list[Path]:
    normalized_name = normalize_dataset_name(name)
    root_path = Path(root)
    if normalized_name in PLANETOID_NAMES:
        display = PLANETOID_NAMES[normalized_name]
        return [
            root_path / "Planetoid" / display / "processed",
            root_path / "Planetoid" / display / "raw",
        ]
    if normalized_name in PRIMEKG_NAMES:
        from .primekg import primekg_cache_paths

        return primekg_cache_paths(root_path)
    if normalized_name in PRIMEKG_FULL_NOSOURCE_NAMES:
        from .primekg import primekg_full_nosource_cache_paths

        return primekg_full_nosource_cache_paths(root_path)
    if normalized_name in PRIMEKG_DISEASE_GENE_SMALL_NAMES:
        from .primekg import primekg_disease_gene_small_cache_paths

        return primekg_disease_gene_small_cache_paths(root_path)
    if normalized_name in PRIMEKG_DISEASE_GENE_SMALL_NOSOURCE_NAMES:
        from .primekg import primekg_disease_gene_small_nosource_cache_paths

        return primekg_disease_gene_small_nosource_cache_paths(root_path)
    if normalized_name in HETIONET_SMALL_NOSOURCE_NAMES:
        from .hetionet import hetionet_small_nosource_cache_paths

        return hetionet_small_nosource_cache_paths(root_path)
    if normalized_name in HETIONET_FULL_NOSOURCE_NAMES:
        from .hetionet import hetionet_full_nosource_cache_paths

        return hetionet_full_nosource_cache_paths(root_path)
    if normalized_name in PPI_HOMO_SL_FILTERED_NAMES:
        return [root_path / "Planetoid" / "PPI" / "processed_ppi_homo_sl_filtered"]
    if normalized_name in PPI_INDUCTIVE_SL_FILTERED_NAMES:
        return [root_path / "Planetoid" / "PPI" / "processed_ppi_inductive_sl_filtered"]
    if normalized_name in PPI_INDUCTIVE_SL_MOSTFREQ_FILTERED_NAMES:
        return [root_path / "Planetoid" / "PPI" / "processed_ppi_inductive_sl_mostfreq_filtered"]
    balanced_target = _ppi_inductive_sl_balanced_target(normalized_name)
    if balanced_target is not None:
        return [root_path / "Planetoid" / "PPI" / f"processed_ppi_inductive_sl_balanced{balanced_target}_filtered"]
    if normalized_name == "reddit":
        return [
            root_path / "Reddit" / "processed",
            root_path / "Reddit" / "raw",
        ]
    supported = ", ".join([*PLANETOID_NAMES, "primekg", "primekg-full-nosource", "primekg-disease-gene-small", "primekg-disease-gene-small-nosource", "hetionet-small-nosource", "hetionet-full-nosource", "ppi-homo-sl-filtered", "ppi-inductive-sl-filtered", "ppi-inductive-sl-mostfreq-filtered", "ppi-inductive-sl-balanced{K}-filtered", "reddit"])
    raise ValueError(f"Unsupported dataset {name!r}. Supported datasets: {supported}")


def _ppi_inductive_sl_balanced_target(name: str) -> int | None:
    prefix = "ppi-inductive-sl-balanced"
    suffix = "-filtered"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    target = name[len(prefix) : -len(suffix)]
    if not target.isdigit():
        return None
    value = int(target)
    if value <= 0:
        return None
    return value


def _has_files(path: Path) -> bool:
    return path.exists() and any(item.is_file() for item in path.rglob("*"))


def clone_data(data):
    if hasattr(data, "clone"):
        return data.clone()
    return copy.deepcopy(data)


def apply_stratified_split(
    data,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """Return a clone with class-stratified train/val/test masks."""

    import torch

    total = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    train_ratio = float(train_ratio) / total
    val_ratio = float(val_ratio) / total
    test_ratio = float(test_ratio) / total

    split_data = clone_data(data)
    y = split_data.y
    if y.dim() > 1:
        y = y.squeeze(-1)
    y_cpu = y.detach().cpu()
    num_nodes = int(split_data.num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    for label in torch.unique(y_cpu).tolist():
        class_indices = torch.nonzero(y_cpu == int(label), as_tuple=False).view(-1)
        order = torch.randperm(class_indices.numel(), generator=generator)
        class_indices = class_indices[order]
        count = int(class_indices.numel())
        train_count = _ratio_count(count, train_ratio)
        val_count = _ratio_count(count - train_count, val_ratio / max(val_ratio + test_ratio, 1e-12))
        test_count = count - train_count - val_count
        if count >= 3:
            if train_count == 0:
                train_count, test_count = 1, max(0, test_count - 1)
            if val_count == 0:
                val_count, test_count = 1, max(0, test_count - 1)
            if test_count == 0:
                test_count = 1
                if train_count >= val_count and train_count > 1:
                    train_count -= 1
                elif val_count > 1:
                    val_count -= 1

        train_idx = class_indices[:train_count]
        val_idx = class_indices[train_count : train_count + val_count]
        test_idx = class_indices[train_count + val_count :]
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    split_data.train_mask = train_mask.to(device=split_data.x.device)
    split_data.val_mask = val_mask.to(device=split_data.x.device)
    split_data.test_mask = test_mask.to(device=split_data.x.device)
    return split_data


def induced_subgraph_from_mask(data, mask_name: str = "train_mask"):
    """Return the node-induced subgraph for a boolean node mask."""

    import torch

    mask = getattr(data, mask_name, None)
    if mask is None:
        raise ValueError(f"Data object has no {mask_name!r}.")
    if mask.dim() > 1:
        mask = mask[:, 0]
    mask = mask.detach().cpu().to(dtype=torch.bool)
    node_ids = torch.nonzero(mask, as_tuple=False).view(-1)
    if node_ids.numel() == 0:
        raise ValueError(f"Mask {mask_name!r} selects no nodes.")

    sub_data = clone_data(data)
    num_nodes = int(data.num_nodes)
    mapping = torch.full((num_nodes,), -1, dtype=torch.long)
    mapping[node_ids] = torch.arange(node_ids.numel(), dtype=torch.long)

    if hasattr(data, "x") and data.x is not None:
        sub_data.x = data.x[node_ids.to(data.x.device)].clone()
    if hasattr(data, "y") and data.y is not None:
        sub_data.y = data.y[node_ids.to(data.y.device)].clone()

    edge_index = data.edge_index.detach().cpu()
    keep = mask[edge_index[0]] & mask[edge_index[1]]
    kept_edges = edge_index[:, keep]
    sub_data.edge_index = mapping[kept_edges].to(device=data.edge_index.device)
    if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.shape[0] == data.edge_index.shape[1]:
        sub_data.edge_attr = data.edge_attr[keep.to(data.edge_attr.device)].clone()

    import torch as _torch

    sub_data.num_nodes = int(node_ids.numel())
    sub_data.train_mask = _torch.ones(sub_data.num_nodes, dtype=_torch.bool, device=sub_data.x.device)
    sub_data.val_mask = None
    sub_data.test_mask = None
    sub_data.original_node_ids = node_ids.clone()
    return sub_data


def select_forget_nodes(
    data,
    ratio: float = 0.1,
    node_ids: Optional[Iterable[int]] = None,
    seed: int = 42,
    mask_name: str = "train_mask",
) -> list[int]:
    if node_ids is not None:
        return _dedupe_ints(node_ids)

    import torch

    candidates = _mask_indices(getattr(data, mask_name, None))
    if not candidates:
        candidates = list(range(int(data.num_nodes)))

    count = _sample_count(len(candidates), ratio)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    order = torch.randperm(len(candidates), generator=generator)[:count].tolist()
    return [int(candidates[idx]) for idx in order]


def select_forget_edges(
    data,
    ratio: float = 0.1,
    edge_ids: Optional[Iterable[int]] = None,
    seed: int = 42,
) -> list[tuple[int, int]]:
    if edge_ids is not None:
        edge_index = data.edge_index.detach().cpu()
        return [
            (int(edge_index[0, int(idx)]), int(edge_index[1, int(idx)]))
            for idx in edge_ids
        ]

    import torch

    num_edges = int(data.edge_index.shape[1])
    count = _sample_count(num_edges, ratio)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    selected = torch.randperm(num_edges, generator=generator)[:count].tolist()
    edge_index = data.edge_index.detach().cpu()
    return [(int(edge_index[0, idx]), int(edge_index[1, idx])) for idx in selected]


def select_forget_features(
    data,
    ratio: float = 0.1,
    feature_ids: Optional[Iterable[int]] = None,
    seed: int = 42,
) -> list[int]:
    if feature_ids is not None:
        return _dedupe_ints(feature_ids)

    import torch

    num_features = int(data.x.shape[1])
    count = _sample_count(num_features, ratio)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return [int(idx) for idx in torch.randperm(num_features, generator=generator)[:count].tolist()]


def _load_primekg(root: Path):
    from .primekg import load_primekg_homo

    return load_primekg_homo(root)


def _load_primekg_full_nosource(root: Path):
    from .primekg import load_primekg_full_nosource

    return load_primekg_full_nosource(root)


def _load_primekg_disease_gene_small(root: Path):
    from .primekg import load_primekg_disease_gene_small

    return load_primekg_disease_gene_small(root)


def _load_primekg_disease_gene_small_nosource(root: Path):
    from .primekg import load_primekg_disease_gene_small_nosource

    return load_primekg_disease_gene_small_nosource(root)


def _load_planetoid(name: str, root: Path, normalize: bool):
    try:
        import torch_geometric.transforms as T
        from torch_geometric.datasets import Planetoid
    except ImportError as exc:
        raise SystemExit(
            "torch-geometric is required for Planetoid datasets. "
            "Install the project environment before loading graph datasets."
        ) from exc

    transform = T.NormalizeFeatures() if normalize else None
    return Planetoid(str(root / "Planetoid"), PLANETOID_NAMES[name], transform=transform)


def _load_hetionet_small_nosource(root: Path, *, download: bool = True):
    from .hetionet import load_hetionet_small_nosource

    return load_hetionet_small_nosource(root, download=download)


def _load_hetionet_full_nosource(root: Path, *, download: bool = True):
    from .hetionet import load_hetionet_full_nosource

    return load_hetionet_full_nosource(root, download=download)


def _load_ppi_homo_sl_filtered(root: Path):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required for PPI-Homo-SL-Filtered.") from exc

    processed_dir = root / "Planetoid" / "PPI" / "processed_ppi_homo_sl_filtered"
    data_path = processed_dir / "data.pt"
    metadata_path = processed_dir / "metadata.json"
    if not data_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing PPI-Homo-SL-Filtered artifacts under {processed_dir}. "
            "Run experiments/build_ppi_homo_sl_filtered.py first."
        )
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return data, metadata


def _load_ppi_inductive_sl_filtered(root: Path):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required for PPI-Inductive-SL-Filtered.") from exc

    processed_dir = root / "Planetoid" / "PPI" / "processed_ppi_inductive_sl_filtered"
    data_path = processed_dir / "data.pt"
    metadata_path = processed_dir / "metadata.json"
    if not data_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing PPI-Inductive-SL-Filtered artifacts under {processed_dir}. "
            "Run experiments/build_ppi_inductive_sl_filtered.py first."
        )
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return data, metadata


def _load_ppi_inductive_sl_mostfreq_filtered(root: Path):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required for PPI-Inductive-SL-MostFreq-Filtered.") from exc

    processed_dir = root / "Planetoid" / "PPI" / "processed_ppi_inductive_sl_mostfreq_filtered"
    data_path = processed_dir / "data.pt"
    metadata_path = processed_dir / "metadata.json"
    if not data_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing PPI-Inductive-SL-MostFreq-Filtered artifacts under {processed_dir}. "
            "Run experiments/build_ppi_inductive_sl_mostfreq_filtered.py first."
        )
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return data, metadata


def _load_ppi_inductive_sl_balanced_filtered(root: Path, target_classes: int):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required for PPI-Inductive-SL-Balanced-Filtered.") from exc

    processed_dir = root / "Planetoid" / "PPI" / f"processed_ppi_inductive_sl_balanced{int(target_classes)}_filtered"
    data_path = processed_dir / "data.pt"
    metadata_path = processed_dir / "metadata.json"
    if not data_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing PPI-Inductive-SL-Balanced{int(target_classes)}-Filtered artifacts under {processed_dir}. "
            "Run experiments/build_ppi_inductive_sl_balanced_filtered.py first."
        )
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return data, metadata


def _load_reddit(root: Path, normalize: bool):
    try:
        import torch_geometric.transforms as T
        from torch_geometric.datasets import Reddit
    except ImportError as exc:
        raise SystemExit(
            "torch-geometric is required for Reddit. "
            "Install the project environment before loading graph datasets."
        ) from exc

    transform = T.NormalizeFeatures() if normalize else None
    return Reddit(str(root / "Reddit"), transform=transform)


def _mask_indices(mask) -> list[int]:
    if mask is None:
        return []
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu()
    if getattr(mask, "dim", lambda: 1)() > 1:
        mask = mask[:, 0]
    return [int(idx) for idx, enabled in enumerate(mask.tolist()) if bool(enabled)]


def _sample_count(total: int, ratio: float) -> int:
    if total <= 0:
        return 0
    if ratio <= 0:
        return 0
    if ratio >= 1 and float(ratio).is_integer():
        return min(total, int(ratio))
    return max(1, min(total, int(round(total * ratio))))


def _ratio_count(total: int, ratio: float) -> int:
    if total <= 0 or ratio <= 0:
        return 0
    return max(0, min(total, int(round(total * ratio))))


def _dedupe_ints(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        item = int(value)
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered

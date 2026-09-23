#!/usr/bin/env python3
"""Prepare the paper's Hetionet inputs from a user-supplied local bz2 file.

This wrapper never downloads data or starts training. The frozen feature builder
is imported unchanged from src/data/hetionet.py. Run --help for usage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW_SHA256 = "a342ab57e9073e6c02bb5e109d1f16917e6f933be4e5c77ebbaebfa26b984c19"
ORIGINAL_CACHE_SHA256 = "7953370be1c45eed015d9683eaa34ce411de70366e229a653dd10e648db0a0a1"
CLASS_COUNTS = {
    "Anatomy": 402, "Biological Process": 11381, "Cellular Component": 1391,
    "Compound": 1552, "Disease": 137, "Gene": 20945,
    "Molecular Function": 2884, "Pathway": 1822, "Pharmacologic Class": 345,
    "Side Effect": 5734, "Symptom": 438,
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stage_raw_file(source: Path, destination: Path, expected_sha256: str) -> str:
    """Validate before copying; never replace a different existing file."""
    if not source.is_file():
        raise FileNotFoundError(f"Missing local Hetionet input: {source}. See docs/DATA_PREPARATION.md.")
    actual = sha256(source)
    if actual != expected_sha256:
        raise ValueError(f"Raw input SHA256 mismatch: {actual}; expected {expected_sha256}.")
    if destination.exists():
        if not destination.is_file() or sha256(destination) != expected_sha256:
            raise ValueError(f"Existing raw input differs; it was not overwritten: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids silently replacing an input from another run.
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if sha256(destination) != expected_sha256:
            raise ValueError(f"Copied input failed SHA256 verification: {destination}")
    return actual


def validate_cache(data, metadata, feature_schema) -> dict:
    """Check the published graph construction facts, not training results."""
    import torch

    expected = {
        "dataset": "hetionet-full-nosource", "dataset_variant": "hetionet_full_nosource",
        "num_source_nodes": 47031, "num_source_edges": 2250197,
        "num_nodes": 47031, "num_edges_processed": 4215418,
        "num_isolated_nodes_before_drop": 1873, "removed_isolated_nodes": 0,
        "num_features": 20, "num_classes": 11,
        "make_undirected": True, "deduplicate_edges": True, "remove_self_loops": True,
        "class_counts": CLASS_COUNTS,
        "class_mapping": {name: i for i, name in enumerate(CLASS_COUNTS)},
        "feature_schema": list(feature_schema),
        "feature_graph_scope": "constructed_homogeneous_graph",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Cache metadata does not match the paper protocol: {key}")
    protocol = metadata.get("construction_protocol", {})
    for key, value in {"selection_seed": 42, "max_nodes_per_class": None,
                       "min_nodes_per_class": 1, "drop_isolated_nodes": False,
                       "uses_train_val_test_split": False}.items():
        if key not in protocol or protocol[key] != value:
            raise ValueError(f"Unexpected construction protocol: {key}")
    if tuple(data.x.shape) != (47031, 20) or tuple(data.y.shape) != (47031,):
        raise ValueError("Unexpected node feature or label shape.")
    if tuple(data.edge_index.shape) != (2, 4215418) or int(data.num_nodes) != 47031:
        raise ValueError("Unexpected graph shape.")
    if data.x.dtype != torch.float32 or data.y.dtype != torch.int64 or data.edge_index.dtype != torch.int64:
        raise ValueError("Unexpected tensor dtype.")
    if not bool(torch.isfinite(data.x).all()):
        raise ValueError("Features contain NaN or infinite values.")
    if int(data.y.min()) != 0 or int(data.y.max()) != 10:
        raise ValueError("Unexpected class labels.")
    if torch.bincount(data.y, minlength=11).tolist() != list(CLASS_COUNTS.values()):
        raise ValueError("Unexpected per-class counts.")
    edges = data.edge_index
    if int(edges.min()) < 0 or int(edges.max()) >= 47031 or bool((edges[0] == edges[1]).any()):
        raise ValueError("Invalid endpoints or self-loops in the processed graph.")
    # Frozen torch.unique(dim=1) produces lexicographically sorted edge columns.
    codes = edges[0] * 47031 + edges[1]
    if not bool((codes[1:] > codes[:-1]).all()):
        raise ValueError("Edges are not sorted and deduplicated as in the frozen builder.")
    reverse_codes = edges[1] * 47031 + edges[0]
    positions = torch.searchsorted(codes, reverse_codes)
    if bool((positions >= codes.numel()).any()) or not torch.equal(codes[positions], reverse_codes):
        raise ValueError("Processed graph is not bidirectional.")
    isolated = int((torch.bincount(edges[0], minlength=47031) == 0).sum())
    if isolated != 1873:
        raise ValueError(f"Unexpected isolated-node count: {isolated}")
    digest = hashlib.sha256()
    for name in ("x", "edge_index", "y"):
        tensor = getattr(data, name).detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return {"num_nodes": 47031, "num_edges": 4215418, "num_features": 20,
            "num_classes": 11, "isolated_nodes": isolated,
            "tensor_content_sha256": digest.hexdigest()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, help="Local original hetionet-v1.0.json.bz2; no download is performed.")
    parser.add_argument("--check-only", action="store_true", help="Validate existing package-local inputs/cache without writing or rebuilding.")
    args = parser.parse_args(argv)
    if args.check_only and args.raw_file is not None:
        parser.error("--check-only checks package-local files; do not combine it with --raw-file")
    raw_root = ROOT / "data" / "raw"
    base = raw_root / "Hetionet"
    raw_path = base / "hetionet-v1.0.json.bz2"
    cache = base / "processed_full_nosource"
    data_path, metadata_path = cache / "data.pt", cache / "metadata.json"
    if (raw_root / "Planetoid" / "Hetionet").exists():
        raise ValueError("Legacy Planetoid/Hetionet path detected; use a clean extraction of this code package.")
    if args.check_only:
        if not raw_path.is_file() or sha256(raw_path) != RAW_SHA256:
            raise ValueError(f"Missing or mismatched package-local original input: {raw_path}")
        if not data_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Node cache is missing; run prepare_data.py with your local original bz2 input first.")
    else:
        stage_raw_file((args.raw_file or raw_path).expanduser().resolve(), raw_path, RAW_SHA256)
        if data_path.exists() != metadata_path.exists():
            raise ValueError("Incomplete existing cache; use a clean extraction rather than mixing runs.")
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    import networkx
    import scipy
    import torch_geometric
    from data.hetionet import FEATURE_SCHEMA, load_hetionet_full_nosource

    if data_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        data = torch.load(data_path, map_location="cpu", weights_only=False)
    else:
        print("Building the full graph and its 20 topology features; this can take substantial CPU time and RAM.", flush=True)
        data, metadata = load_hetionet_full_nosource(
            raw_root, download=False, node_types=None, max_nodes_per_class=None,
            min_nodes_per_class=1, selection_seed=42, make_undirected=True,
            deduplicate_edges=True, remove_self_loops=True, drop_isolated_nodes=False,
            force_rebuild=False,
        )
    facts = validate_cache(data, metadata, FEATURE_SCHEMA)
    cache_sha = sha256(data_path)
    receipt = {
        "raw_file": str(raw_path.relative_to(ROOT)), "raw_sha256": RAW_SHA256,
        "cache_file": str(data_path.relative_to(ROOT)), "cache_sha256": cache_sha,
        "original_cache_sha256_for_reference_only": ORIGINAL_CACHE_SHA256,
        "matches_original_cache_bytes": cache_sha == ORIGINAL_CACHE_SHA256,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "torch_geometric": torch_geometric.__version__,
                     "networkx": networkx.__version__, "scipy": scipy.__version__},
        **facts,
        "scope": "New reproduction inputs; no archived results or model weights were loaded.",
    }
    text = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    if not args.check_only:
        (ROOT / "data" / "preparation_receipt.json").write_text(text, encoding="utf-8")
    print(text, end="")
    print("Inputs validated. Run scripts/reproduce.py for task splits and experiments.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, ImportError) as exc:
        raise SystemExit(f"Data preparation failed: {exc}") from exc

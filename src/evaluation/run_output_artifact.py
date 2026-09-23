from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "run_output_artifact_v1"


def save_run_output_artifact(
    path: str | Path,
    *,
    logits,
    embeddings,
    trainer,
    data,
    method: str,
    dataset: str,
    unlearning_type: str,
    forget_set_path: str | Path | None,
    base_artifact_path: str | Path | None,
    seed: int,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Persist the post-request outputs needed for deterministic offline audits.

    The artifact is deliberately separate from the metrics JSON.  It binds the
    tensors to the base artifact and forget-set hashes so that paired JS/TV,
    prediction disagreement, and retained-only representation diagnostics cannot
    accidentally mix incompatible requests.
    """

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    forget_path = Path(forget_set_path) if forget_set_path is not None else None
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": "graph-unlearning formal runner",
        "method": str(method),
        "dataset": str(dataset),
        "unlearning_type": str(unlearning_type),
        "seed": int(seed),
        "forget_set_path": str(forget_path) if forget_path is not None else None,
        "forget_set_sha256": _file_sha256(forget_path) if forget_path is not None else None,
        "base_artifact_path": str(base_artifact_path) if base_artifact_path is not None else None,
        "base_metadata_sha256": _base_metadata_sha256(base_artifact_path),
        "model_config": dict(model_config),
        "num_nodes": int(getattr(data, "num_nodes", 0)) if data is not None else None,
        "logits_shape": list(logits.shape) if logits is not None else None,
        "embeddings_shape": list(embeddings.shape) if embeddings is not None else None,
        "mask_counts": _mask_counts(data),
        "artifact_path": str(artifact_path),
    }
    model_state = None
    if trainer is not None and getattr(trainer, "model", None) is not None:
        model_state = {
            str(key): value.detach().cpu()
            for key, value in trainer.model.state_dict().items()
        }
    torch.save(
        {
            "metadata": metadata,
            "logits": _cpu_tensor(logits),
            "embeddings": _cpu_tensor(embeddings),
            "model_state": model_state,
            "masks": _masks(data),
        },
        artifact_path,
    )
    sidecar = artifact_path.with_suffix(artifact_path.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_run_output_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"Invalid run-output artifact: {artifact_path}")
    if payload["metadata"].get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported run-output artifact schema: {artifact_path}")
    payload["path"] = str(artifact_path)
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_metadata_sha256(base_artifact_path: str | Path | None) -> str | None:
    if base_artifact_path is None:
        return None
    metadata_path = Path(base_artifact_path) / "metadata.json"
    return _file_sha256(metadata_path) if metadata_path.is_file() else None


def _cpu_tensor(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu()
    return torch.as_tensor(value).cpu()


def _masks(data) -> dict[str, torch.Tensor | None]:
    result: dict[str, torch.Tensor | None] = {}
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, name, None) if data is not None else None
        result[name] = mask.detach().cpu().to(dtype=torch.bool) if mask is not None else None
    return result


def _mask_counts(data) -> dict[str, int | None]:
    return {
        name: (int(mask.sum().item()) if mask is not None else None)
        for name, mask in _masks(data).items()
    }

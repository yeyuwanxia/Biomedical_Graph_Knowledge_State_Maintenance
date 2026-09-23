from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "exact_retrain_reference_v1"


def save_exact_retrain_reference(
    path: str | Path,
    *,
    logits,
    embeddings,
    dataset: str,
    unlearning_type: str,
    forget_set_path: str | Path,
    base_artifact_path: str | Path | None,
    seed: int,
    model_config: dict[str, Any],
    training: dict[str, Any] | None,
) -> dict[str, Any]:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": "experiments/run_baselines.py",
        "method": "retrain",
        "dataset": str(dataset),
        "unlearning_type": str(unlearning_type),
        "seed": int(seed),
        "forget_set_path": str(forget_set_path),
        "forget_set_sha256": file_sha256(forget_set_path),
        "base_artifact_path": str(base_artifact_path) if base_artifact_path is not None else None,
        "model_config": dict(model_config),
        "training": dict(training or {}),
        "artifact_path": str(artifact_path),
    }
    torch.save(
        {
            "metadata": metadata,
            "logits": _cpu_tensor(logits),
            "embeddings": _cpu_tensor(embeddings),
        },
        artifact_path,
    )
    sidecar_path = artifact_path.with_suffix(artifact_path.suffix + ".json")
    sidecar_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_exact_retrain_reference(
    path: str | Path,
    *,
    dataset: str,
    unlearning_type: str,
    forget_set_path: str | Path,
    base_artifact_path: str | Path | None,
) -> dict[str, Any]:
    artifact_path = Path(path)
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"Invalid exact-retrain artifact: {artifact_path}")

    metadata = payload["metadata"]
    mismatches: list[str] = []
    expected = {
        "schema_version": SCHEMA_VERSION,
        "method": "retrain",
        "dataset": str(dataset),
        "unlearning_type": str(unlearning_type),
        "forget_set_sha256": file_sha256(forget_set_path),
        "base_artifact_path": str(base_artifact_path) if base_artifact_path is not None else None,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            mismatches.append(f"{key}: expected {value!r}, got {metadata.get(key)!r}")
    if mismatches:
        raise ValueError(
            f"Exact-retrain artifact does not match this run ({artifact_path}): "
            + "; ".join(mismatches)
        )
    if payload.get("logits") is None:
        raise ValueError(f"Exact-retrain artifact has no logits: {artifact_path}")

    payload["path"] = str(artifact_path)
    return payload


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_tensor(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu()
    return torch.as_tensor(value).cpu()

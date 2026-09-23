from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import torch


CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class HubScoreCacheResult:
    hit: bool
    key: str
    path: Path
    scores: Optional[dict[int, float]] = None
    reason: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def as_dict(self) -> dict[str, Any]:
        metadata = dict(self.metadata or {})
        return {
            "enabled": True,
            "hit": self.hit,
            "key": self.key,
            "path": str(self.path),
            "reason": self.reason,
            "offline_preprocessing_seconds": metadata.get("offline_preprocessing_seconds"),
            "artifact_metadata": metadata,
        }


class HubScoreCache:
    """Content-addressed cache for final HubScore maps."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def lookup(self, identity: Mapping[str, Any]) -> HubScoreCacheResult:
        key = cache_key(identity)
        path = self._path(identity, key)
        if not path.is_file():
            return HubScoreCacheResult(hit=False, key=key, path=path, reason="not_found")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
                raise ValueError("schema_version_mismatch")
            if payload.get("key") != key or payload.get("identity") != _json_normalize(identity):
                raise ValueError("identity_mismatch")
            scores = {int(node): float(value) for node, value in payload.get("scores", {}).items()}
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("metadata_mismatch")
            return HubScoreCacheResult(
                hit=True,
                key=key,
                path=path,
                scores=scores,
                metadata=metadata,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return HubScoreCacheResult(
                hit=False,
                key=key,
                path=path,
                reason=f"invalid_cache:{type(exc).__name__}:{exc}",
            )

    def store(
        self,
        identity: Mapping[str, Any],
        scores: Mapping[int, float],
        *,
        build_seconds: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> HubScoreCacheResult:
        key = cache_key(identity)
        path = self._path(identity, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact_metadata = dict(metadata or {})
        artifact_metadata.setdefault("created_at", datetime.now().astimezone().isoformat(timespec="seconds"))
        if build_seconds is not None:
            artifact_metadata["offline_preprocessing_seconds"] = float(build_seconds)
        artifact_metadata = _json_normalize(artifact_metadata)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "key": key,
            "identity": _json_normalize(identity),
            "metadata": artifact_metadata,
            "scores": {str(int(node)): float(value) for node, value in scores.items()},
        }
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return HubScoreCacheResult(
            hit=False,
            key=key,
            path=path,
            reason="stored",
            metadata=artifact_metadata,
        )

    def _path(self, identity: Mapping[str, Any], key: str) -> Path:
        dataset = _safe_component(str(identity.get("dataset", "unknown")))
        seed = _safe_component(str(identity.get("training_seed", "unknown")))
        return self.root / dataset / f"seed{seed}" / f"{key}.json"


def cache_key(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_normalize(identity), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_tensors(tensors: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name]
        if value is None:
            continue
        tensor = value.detach().cpu() if hasattr(value, "detach") else torch.as_tensor(value)
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        if tensor.layout != torch.strided:
            tensor = tensor.coalesce()
            digest.update(tensor.indices().contiguous().view(torch.uint8).numpy().tobytes())
            digest.update(tensor.values().contiguous().view(torch.uint8).numpy().tobytes())
        else:
            digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def fingerprint_model(model: torch.nn.Module) -> str:
    return fingerprint_tensors({name: tensor for name, tensor in model.state_dict().items()})


def hub_cache_identity(
    *,
    dataset: str,
    training_seed: int,
    data: Any,
    model: torch.nn.Module,
    hub_config: Any,
    gradient_enabled: bool,
    gradient_passes: int,
    gradient_dropout: bool,
) -> dict[str, Any]:
    data_tensors = {
        "edge_index": getattr(data, "edge_index", None),
        "features": getattr(data, "x", None),
        "train_mask": getattr(data, "train_mask", None),
        "labels": getattr(data, "y", None),
    }
    config_payload = asdict(hub_config) if hasattr(hub_config, "__dataclass_fields__") else dict(hub_config)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": str(dataset),
        "training_seed": int(training_seed),
        "num_nodes": int(getattr(data, "num_nodes", 0)),
        "data_fingerprint": fingerprint_tensors(data_tensors),
        "model_fingerprint": fingerprint_model(model) if gradient_enabled else "gradient_disabled",
        "hub_config": config_payload,
        "gradient": {
            "enabled": bool(gradient_enabled),
            "passes": int(gradient_passes),
            "dropout": bool(gradient_dropout),
        },
    }


def _json_normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)

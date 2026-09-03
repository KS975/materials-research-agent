from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ARTIFACT_ROOT = Path("engine/artifacts")


@dataclass(frozen=True)
class EnginePathConfig:
    """Resolve engine-owned defaults without binding the core to a host CWD."""

    artifact_root: Path = DEFAULT_ARTIFACT_ROOT

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_root", Path(self.artifact_root))

    @classmethod
    def from_env(cls) -> EnginePathConfig:
        configured = os.environ.get("ENGINE_ARTIFACT_ROOT", "").strip()
        return cls(artifact_root=Path(configured) if configured else DEFAULT_ARTIFACT_ROOT)

    @property
    def dataset_dir(self) -> Path:
        return self.artifact_root / "datasets"

    @property
    def model_dir(self) -> Path:
        return self.artifact_root / "models"

    @property
    def optimization_dir(self) -> Path:
        return self.artifact_root / "optimizations"

    @property
    def model_registry_path(self) -> Path:
        return self.model_dir / "model-registry.json"


def default_artifact_dir(kind: str) -> str:
    config = EnginePathConfig.from_env()
    if kind == "dataset":
        return str(config.dataset_dir)
    if kind == "model":
        return str(config.model_dir)
    if kind == "optimization":
        return str(config.optimization_dir)
    raise ValueError(f"unknown artifact kind: {kind}")


def default_model_registry_path() -> str:
    return str(EnginePathConfig.from_env().model_registry_path)

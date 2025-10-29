"""Model persistence helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models.simple_nn import SimpleNeuralNetwork
from .utils.characters import CharacterSet


@dataclass
class ModelArtifact:
    model: SimpleNeuralNetwork
    character_set: CharacterSet
    image_size: int

    @property
    def input_dim(self) -> int:
        return self.model.input_dim

    def save(self, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        model_path = target_dir / "model.npz"
        metadata_path = target_dir / "metadata.json"

        self.model.save(model_path)
        metadata = {
            "image_size": self.image_size,
            "characters": list(self.character_set.chars),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, source_dir: Path) -> "ModelArtifact":
        model_path = source_dir / "model.npz"
        metadata_path = source_dir / "metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Missing model files under {source_dir}")

        model = SimpleNeuralNetwork.load(model_path)
        metadata = json.loads(metadata_path.read_text())
        character_set = CharacterSet.from_iterable(metadata["characters"])
        image_size = metadata["image_size"]
        return cls(model=model, character_set=character_set, image_size=image_size)

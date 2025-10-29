"""Inference helpers for applying a trained model."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .persistence import ModelArtifact


def load_bitmap_from_path(image_path: Path, image_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    resized = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def prepare_input(bitmap: np.ndarray) -> np.ndarray:
    if bitmap.ndim == 2:
        flattened = bitmap.reshape(1, -1)
    elif bitmap.ndim == 3 and bitmap.shape[0] == 1:
        flattened = bitmap.reshape(1, -1)
    else:
        raise ValueError("Bitmap must be a 2D array with shape (H, W)")
    return flattened.astype(np.float32)


def predict_bitmap(artifact: ModelArtifact, bitmap: np.ndarray) -> tuple[str, float, np.ndarray]:
    prepared = prepare_input(bitmap)
    probs = artifact.model.predict_proba(prepared)[0]
    predicted_index = int(np.argmax(probs))
    predicted_char = artifact.character_set.char_for(predicted_index)
    return predicted_char, float(probs[predicted_index]), probs

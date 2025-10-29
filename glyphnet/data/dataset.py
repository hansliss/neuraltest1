"""Dataset generation utilities."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..config import DEFAULT_CONFIG, PRINTABLE_LATIN1
from ..rendering import GlyphRenderingError, render_font_glyphs
from ..utils.characters import CharacterSet

try:  # Optional dependency, falls back to plain range if unavailable
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm always available via requirements
    tqdm = None  # type: ignore


@dataclass(slots=True)
class GlyphDataset:
    images: np.ndarray
    labels: np.ndarray
    font_indices: np.ndarray
    characters: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.images.ndim != 2:
            raise ValueError("images must be 2D (num_samples, flattened_pixels)")
        if self.labels.shape[0] != self.images.shape[0]:
            raise ValueError("labels must align with images")
        if self.font_indices.shape[0] != self.images.shape[0]:
            raise ValueError("font_indices must align with images")

    @property
    def num_samples(self) -> int:
        return self.images.shape[0]

    @property
    def input_dim(self) -> int:
        return self.images.shape[1]

    def batches(self, batch_size: int, shuffle: bool = True, seed: int | None = None):
        indices = np.arange(self.num_samples)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        for start in range(0, self.num_samples, batch_size):
            end = start + batch_size
            selected = indices[start:end]
            yield self.images[selected], self.labels[selected]

    def save(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            images=self.images,
            labels=self.labels,
            font_indices=self.font_indices,
            characters=np.array(self.characters, dtype=object),
        )

    @classmethod
    def load(cls, source: Path) -> "GlyphDataset":
        payload = np.load(source, allow_pickle=True)
        characters = tuple(payload["characters"].tolist())
        return cls(
            images=payload["images"],
            labels=payload["labels"],
            font_indices=payload["font_indices"],
            characters=characters,
        )


def build_character_set(chars: Iterable[str] | None = None) -> CharacterSet:
    return CharacterSet.from_iterable(chars or PRINTABLE_LATIN1)


def build_dataset_from_fonts(
    font_paths: Sequence[Path],
    character_set: CharacterSet | None = None,
    canvas_size: int | None = None,
    target_size: int | None = None,
    show_progress: bool = True,
) -> GlyphDataset:
    if not font_paths:
        raise ValueError("font_paths must not be empty")
    char_set = character_set or build_character_set()
    canvas = canvas_size or DEFAULT_CONFIG.canvas_size
    target = target_size or DEFAULT_CONFIG.image_size

    images: list[np.ndarray] = []
    labels: list[int] = []
    font_indices: list[int] = []

    iterator: Iterable[Path]
    if show_progress and tqdm is not None:
        iterator = tqdm(font_paths, desc="Rendering fonts", unit="font")
    else:
        iterator = font_paths

    skipped_fonts: list[Path] = []

    for font_idx, font_path in enumerate(iterator):
        try:
            glyphs = render_font_glyphs(
                font_path,
                char_set.chars,
                canvas_size=canvas,
                target_size=target,
            )
        except (GlyphRenderingError, OSError) as exc:
            skipped_fonts.append(font_path)
            if show_progress and tqdm is not None:
                tqdm.write(f"Skipping font {font_path} ({exc})")
            continue
        for ch, bitmap in glyphs.items():
            images.append(bitmap.reshape(1, -1))
            labels.append(char_set.index_of(ch))
            font_indices.append(font_idx)

    if skipped_fonts and (not show_progress or tqdm is None):
        print(f"Skipped {len(skipped_fonts)} fonts due to rendering errors.")

    if not images:
        raise RuntimeError("No glyphs were rendered. Check font availability and character coverage.")

    images_array = np.concatenate(images, axis=0)
    labels_array = np.asarray(labels, dtype=np.int64)
    font_indices_array = np.asarray(font_indices, dtype=np.int64)

    return GlyphDataset(images=images_array, labels=labels_array, font_indices=font_indices_array, characters=char_set.chars)


def _derive_cache_name(
    split_name: str,
    font_paths: Sequence[Path],
    character_set: CharacterSet,
    canvas_size: int,
    target_size: int,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(split_name.encode("utf-8"))
    for path in font_paths:
        hasher.update(str(path).encode("utf-8"))
    for ch in character_set.chars:
        hasher.update(ch.encode('utf-8'))
    hasher.update(str(canvas_size).encode())
    hasher.update(str(target_size).encode())
    digest = hasher.hexdigest()[:12]
    return f"{split_name}-{digest}.npz"


def load_or_build_dataset(
    cache_dir: Path,
    split_name: str,
    font_paths: Sequence[Path],
    character_set: CharacterSet | None = None,
    canvas_size: int | None = None,
    target_size: int | None = None,
    force_rebuild: bool = False,
    show_progress: bool = True,
) -> GlyphDataset:
    if not font_paths:
        raise ValueError(f"No fonts provided for split '{split_name}'")

    char_set = character_set or build_character_set()
    canvas = canvas_size or DEFAULT_CONFIG.canvas_size
    target = target_size or DEFAULT_CONFIG.image_size

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file_name = _derive_cache_name(split_name, font_paths, char_set, canvas, target)
    cache_path = cache_dir / cache_file_name

    if cache_path.exists() and not force_rebuild:
        return GlyphDataset.load(cache_path)

    dataset = build_dataset_from_fonts(
        font_paths,
        character_set=char_set,
        canvas_size=canvas,
        target_size=target,
        show_progress=show_progress,
    )
    dataset.save(cache_path)
    return dataset

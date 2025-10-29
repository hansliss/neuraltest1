"""Default configuration values for the glyph recognition project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


def _build_printable_latin1_characters() -> tuple[str, ...]:
    # Skip control codes for practicality; fonts rarely define glyphs for them.
    basic = [chr(c) for c in range(32, 127)]
    extended = [chr(c) for c in range(160, 256)]
    return tuple(basic + extended)


@dataclass(frozen=True)
class ProjectConfig:
    fonts_dir: Path
    artifacts_dir: Path
    cache_dir: Path
    image_size: int = 16
    canvas_size: int = 48

    @property
    def bitmap_shape(self) -> tuple[int, int]:
        return (self.image_size, self.image_size)


PRINTABLE_LATIN1: tuple[str, ...] = _build_printable_latin1_characters()
NUM_CLASSES = len(PRINTABLE_LATIN1)

DEFAULT_CONFIG = ProjectConfig(
    fonts_dir=Path("fonts"),
    artifacts_dir=Path("artifacts"),
    cache_dir=Path("artifacts/cache"),
)


def character_index_lookup(chars: Sequence[str] | None = None) -> dict[str, int]:
    lookup_chars: Iterable[str] = chars if chars is not None else PRINTABLE_LATIN1
    return {ch: idx for idx, ch in enumerate(lookup_chars)}

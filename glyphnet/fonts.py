"""Utilities for discovering and partitioning font files."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}


@dataclass(frozen=True)
class FontSplit:
    train: tuple[Path, ...]
    validation: tuple[Path, ...]
    test: tuple[Path, ...]

    def to_json(self) -> str:
        payload = {
            "train": [str(p) for p in self.train],
            "validation": [str(p) for p in self.validation],
            "test": [str(p) for p in self.test],
        }
        return json.dumps(payload, indent=2)


def collect_fonts(font_root: Path) -> list[Path]:
    root = font_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Font root {root} does not exist")

    fonts = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in FONT_EXTENSIONS]
    if not fonts:
        raise RuntimeError(f"No font files found under {root}")
    fonts.sort()
    return fonts


def split_fonts(
    fonts: Sequence[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 13,
) -> FontSplit:
    if train_ratio <= 0 or val_ratio < 0:
        raise ValueError("Ratios must be positive")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1 to leave room for the test split")

    fonts_list = list(fonts)
    rng = random.Random(seed)
    rng.shuffle(fonts_list)

    total = len(fonts_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train = tuple(fonts_list[:train_end])
    validation = tuple(fonts_list[train_end:val_end])
    test = tuple(fonts_list[val_end:])
    return FontSplit(train=train, validation=validation, test=test)


def save_split(font_split: FontSplit, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(font_split.to_json())


def load_split(path: Path) -> FontSplit:
    payload = json.loads(path.read_text())
    return FontSplit(
        train=tuple(Path(p) for p in payload["train"]),
        validation=tuple(Path(p) for p in payload["validation"]),
        test=tuple(Path(p) for p in payload["test"]),
    )

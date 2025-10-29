"""Font glyph rendering helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import DEFAULT_CONFIG


class GlyphRenderingError(Exception):
    """Raised when a glyph cannot be rendered for a given font."""


def _load_font(font_path: Path, canvas_size: int) -> ImageFont.FreeTypeFont:
    size = int(canvas_size * 0.75)
    size = max(size, 1)
    return ImageFont.truetype(str(font_path), size=size)


def render_glyph(
    font_path: Path,
    character: str,
    canvas_size: int | None = None,
    target_size: int | None = None,
) -> np.ndarray | None:
    canvas = canvas_size or DEFAULT_CONFIG.canvas_size
    target = target_size or DEFAULT_CONFIG.image_size

    try:
        font = _load_font(font_path, canvas)
    except OSError as exc:  # Font file unusable
        raise GlyphRenderingError(f"Failed to load font {font_path}") from exc

    image = Image.new("L", (canvas, canvas), color=0)
    drawer = ImageDraw.Draw(image)
    try:
        bbox = drawer.textbbox((0, 0), character, font=font)
    except OSError as exc:
        raise GlyphRenderingError(f"Could not render character {repr(character)} with {font_path}") from exc
    if bbox is None:
        return None

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width == 0 or height == 0:
        return None

    x = (canvas - width) / 2 - bbox[0]
    y = (canvas - height) / 2 - bbox[1]
    drawer.text((x, y), character, fill=255, font=font)

    image = image.resize((target, target), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32)
    arr /= 255.0
    return arr


def render_font_glyphs(
    font_path: Path,
    characters: Iterable[str],
    canvas_size: int | None = None,
    target_size: int | None = None,
) -> dict[str, np.ndarray]:
    bitmap_by_char: dict[str, np.ndarray] = {}
    for ch in characters:
        try:
            bitmap = render_glyph(font_path, ch, canvas_size=canvas_size, target_size=target_size)
        except GlyphRenderingError:
            continue
        if bitmap is None:
            continue
        bitmap_by_char[ch] = bitmap
    return bitmap_by_char

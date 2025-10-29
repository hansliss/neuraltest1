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
    kwargs = {}
    layout = getattr(ImageFont, "Layout", None)
    if layout is not None:
        kwargs["layout_engine"] = getattr(layout, "BASIC", None)
    try:
        return ImageFont.truetype(str(font_path), size=size, **{k: v for k, v in kwargs.items() if v is not None})
    except TypeError:
        # Older Pillow releases might not accept layout_engine; retry without it.
        return ImageFont.truetype(str(font_path), size=size)


def _measure_glyph_bbox(drawer: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, character: str) -> tuple[int, int, int, int] | None:
    try:
        bbox = drawer.textbbox((0, 0), character, font=font)
    except (ValueError, OSError):
        bbox = None
    if bbox is not None:
        return bbox

    # Fallback to font-provided bounding boxes to avoid layout engines that rely on libraqm.
    try:
        bbox = font.getbbox(character)
    except (AttributeError, ValueError, OSError):
        bbox = None
    if bbox is not None:
        return bbox

    try:
        mask = font.getmask(character, mode="L")
        bbox = mask.getbbox()
    except (AttributeError, TypeError, ValueError, OSError):
        bbox = None
    if bbox is not None:
        return bbox

    try:
        width, height = font.getsize(character)
    except Exception:
        return None
    if width == 0 or height == 0:
        return None
    return (0, 0, width, height)


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
    bbox = _measure_glyph_bbox(drawer, font, character)
    if bbox is None:
        return None

    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    if width == 0 or height == 0:
        return None

    x = (canvas - width) / 2 - x0
    y = (canvas - height) / 2 - y0
    try:
        drawer.text((x, y), character, fill=255, font=font)
    except OSError as exc:
        raise GlyphRenderingError(
            f"Failed to draw character {repr(character)} using font {font_path}"
        ) from exc

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

"""Public API for the glyph recognition project."""
from .config import DEFAULT_CONFIG, PRINTABLE_LATIN1, ProjectConfig
from .fonts import FontSplit, collect_fonts, split_fonts
from .inference import predict_bitmap
from .persistence import ModelArtifact
from .training import Trainer, TrainingConfig
from .utils.characters import CharacterSet

__all__ = [
    "DEFAULT_CONFIG",
    "PRINTABLE_LATIN1",
    "ProjectConfig",
    "FontSplit",
    "collect_fonts",
    "split_fonts",
    "Trainer",
    "TrainingConfig",
    "ModelArtifact",
    "CharacterSet",
    "predict_bitmap",
]

#!/usr/bin/env python3
"""Evaluate a trained model on a cached dataset split."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from glyphnet.data.dataset import load_or_build_dataset
from glyphnet.fonts import load_split
from glyphnet.persistence import ModelArtifact
from glyphnet.training import Trainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained glyph recognizer model")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing model.npz and metadata.json")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Cache directory used during training")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test", help="Which split to evaluate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifact = ModelArtifact.load(args.model_dir)
    split_path = args.model_dir / "font_split.json"
    font_split = load_split(split_path)
    fonts = getattr(font_split, args.split)

    dataset = load_or_build_dataset(
        cache_dir=args.cache_dir,
        split_name=args.split,
        font_paths=fonts,
        character_set=artifact.character_set,
        target_size=artifact.image_size,
        force_rebuild=False,
        show_progress=False,
    )

    trainer = Trainer(artifact.model, TrainingConfig(batch_size=args.batch_size))
    loss, accuracy = trainer.evaluate(dataset, batch_size=args.batch_size)
    print(f"{args.split.title()} loss: {loss:.4f}")
    print(f"{args.split.title()} accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

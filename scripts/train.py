#!/usr/bin/env python3
"""Train the glyph recognition neural network."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Sequence


from glyphnet.config import DEFAULT_CONFIG
from glyphnet.data.dataset import build_character_set, load_or_build_dataset
from glyphnet.fonts import FontSplit, collect_fonts, save_split, split_fonts
from glyphnet.models.simple_nn import SimpleNeuralNetwork
from glyphnet.persistence import ModelArtifact
from glyphnet.training import Trainer, TrainingConfig


def _limit_fonts(fonts: Sequence[Path], limit: int | None) -> Sequence[Path]:
    if limit is None or limit <= 0:
        return fonts
    return fonts[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple neural network for 16x16 Latin-1 glyph recognition.")
    parser.add_argument("--fonts-dir", type=Path, default=DEFAULT_CONFIG.fonts_dir, help="Path to the directory containing font files")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_CONFIG.artifacts_dir, help="Where to store trained models and logs")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CONFIG.cache_dir, help="Cache directory for rendered datasets")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio of fonts used for training")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Ratio of fonts used for validation")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate for gradient descent")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 penalty applied to weight matrices")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden layer width of the neural network")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--force-rebuild", action="store_true", help="Force regeneration of cached datasets")
    parser.add_argument("--max-fonts-train", type=int, default=None, help="Optional cap on the number of training fonts (useful for experiments)")
    parser.add_argument("--max-fonts-val", type=int, default=None, help="Optional cap on validation fonts")
    parser.add_argument("--max-fonts-test", type=int, default=None, help="Optional cap on test fonts")
    parser.add_argument("--image-size", type=int, default=DEFAULT_CONFIG.image_size, help="Target image resolution (defaults to 16)")
    parser.add_argument("--canvas-size", type=int, default=DEFAULT_CONFIG.canvas_size, help="Intermediate render canvas size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fonts = collect_fonts(args.fonts_dir)
    font_split = split_fonts(fonts, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)

    train_fonts = tuple(_limit_fonts(font_split.train, args.max_fonts_train))
    val_fonts = tuple(_limit_fonts(font_split.validation, args.max_fonts_val))
    test_fonts = tuple(_limit_fonts(font_split.test, args.max_fonts_test))

    char_set = build_character_set()

    train_ds = load_or_build_dataset(
        cache_dir=args.cache_dir,
        split_name="train",
        font_paths=train_fonts,
        character_set=char_set,
        canvas_size=args.canvas_size,
        target_size=args.image_size,
        force_rebuild=args.force_rebuild,
    )
    val_ds = load_or_build_dataset(
        cache_dir=args.cache_dir,
        split_name="validation",
        font_paths=val_fonts,
        character_set=char_set,
        canvas_size=args.canvas_size,
        target_size=args.image_size,
        force_rebuild=args.force_rebuild,
    )
    test_ds = load_or_build_dataset(
        cache_dir=args.cache_dir,
        split_name="test",
        font_paths=test_fonts,
        character_set=char_set,
        canvas_size=args.canvas_size,
        target_size=args.image_size,
        force_rebuild=args.force_rebuild,
    )

    model = SimpleNeuralNetwork(
        input_dim=train_ds.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=char_set.size,
        seed=args.seed,
        weight_scale=0.05,
    )

    trainer = Trainer(
        model,
        TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            shuffle_seed=args.seed,
        ),
    )
    trainer.fit(train_ds, val_ds)
    test_loss, test_accuracy = trainer.evaluate(test_ds)

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = args.artifacts_dir / f"run-{timestamp}"
    artifact = ModelArtifact(model=model, character_set=char_set, image_size=args.image_size)
    artifact.save(run_dir)

    split_json_path = run_dir / "font_split.json"
    save_split(FontSplit(train=train_fonts, validation=val_fonts, test=test_fonts), split_json_path)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "history": [
                    {
                        "epoch": result.epoch,
                        "train_loss": float(result.train_loss),
                        "train_accuracy": float(result.train_accuracy),
                        "val_loss": None if result.val_loss is None else float(result.val_loss),
                        "val_accuracy": None if result.val_accuracy is None else float(result.val_accuracy),
                    }
                    for result in trainer.history
                ],
            },
            indent=2,
        )
    )

    print(f"Training complete. Test accuracy: {test_accuracy:.3f}")
    print(f"Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()

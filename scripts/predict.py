#!/usr/bin/env python3
"""Run inference on a 16x16 grayscale bitmap using a trained model."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np

from glyphnet.inference import load_bitmap_from_path, predict_bitmap
from glyphnet.persistence import ModelArtifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the Latin-1 character rendered in a 16x16 grayscale bitmap")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing the trained model artifact")
    parser.add_argument("--image", type=Path, required=True, help="Path to a 16x16 grayscale image (PNG, JPEG, etc.)")
    parser.add_argument("--top-k", type=int, default=5, help="Show the top-k most likely characters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifact = ModelArtifact.load(args.model_dir)
    bitmap = load_bitmap_from_path(args.image, artifact.image_size)
    predicted_char, confidence, probs = predict_bitmap(artifact, bitmap)

    print(f"Prediction: {repr(predicted_char)} (code {ord(predicted_char)})")
    print(f"Confidence: {confidence:.4f}")

    top_k = min(args.top_k, len(probs))
    indices = np.argsort(probs)[::-1][:top_k]
    print("Top candidates:")
    for idx in indices:
        ch = artifact.character_set.char_for(int(idx))
        print(f"  {repr(ch):>6} (code {ord(ch):>3}) -> {probs[idx]:.4f}")


if __name__ == "__main__":
    main()

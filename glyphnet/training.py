"""Training loop orchestration for the glyph recognizer."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .data.dataset import GlyphDataset
from .models.simple_nn import SimpleNeuralNetwork


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 0.1
    weight_decay: float = 1e-4
    shuffle_seed: int = 42
    early_stopping_patience: int = 5
    gradient_clip_norm: float | None = 5.0


@dataclass
class TrainingEpochResult:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float | None = None
    val_accuracy: float | None = None


def _clip_gradients(grads: dict[str, np.ndarray], max_norm: float) -> dict[str, np.ndarray]:
    total_norm_sq = sum(float(np.sum(grad ** 2)) for grad in grads.values())
    total_norm = math.sqrt(total_norm_sq)
    if total_norm <= max_norm:
        return grads
    scale = max_norm / (total_norm + 1e-12)
    return {name: grad * scale for name, grad in grads.items()}


class Trainer:
    def __init__(self, model: SimpleNeuralNetwork, config: TrainingConfig | None = None) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.history: list[TrainingEpochResult] = []

    def fit(self, train_ds: GlyphDataset, val_ds: GlyphDataset | None = None) -> list[TrainingEpochResult]:
        best_val_loss = float("inf")
        patience_counter = 0
        best_params: Dict[str, np.ndarray] | None = None

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_accuracy = self._run_epoch(train_ds, epoch)

            val_loss = None
            val_accuracy = None
            if val_ds is not None:
                val_loss, val_accuracy = self.evaluate(val_ds)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_params = {name: param.copy() for name, param in self.model.params.items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        if best_params is not None:
                            self.model.params = best_params
                        break

            result = TrainingEpochResult(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )
            self.history.append(result)

        return self.history

    def _run_epoch(self, dataset: GlyphDataset, epoch: int) -> tuple[float, float]:
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        seed = self.config.shuffle_seed + epoch
        for batch_x, batch_y in dataset.batches(self.config.batch_size, shuffle=True, seed=seed):
            loss, grads = self.model.loss_and_gradients(batch_x, batch_y)
            if self.config.gradient_clip_norm is not None:
                grads = _clip_gradients(grads, self.config.gradient_clip_norm)
            self.model.apply_gradients(
                grads,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

            logits, _ = self.model.forward(batch_x)
            probs = self.model._softmax(logits)
            predictions = probs.argmax(axis=1)
            total_loss += loss * len(batch_x)
            total_correct += int(np.sum(predictions == batch_y))
            total_samples += len(batch_x)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def evaluate(self, dataset: GlyphDataset, batch_size: int | None = None) -> tuple[float, float]:
        bsz = batch_size or self.config.batch_size
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_x, batch_y in dataset.batches(bsz, shuffle=False):
            logits, _ = self.model.forward(batch_x)
            probs = self.model._softmax(logits)
            batch_loss = -np.log(probs[np.arange(len(batch_y)), batch_y] + 1e-12).sum()
            predictions = probs.argmax(axis=1)
            total_loss += batch_loss
            total_correct += int(np.sum(predictions == batch_y))
            total_samples += len(batch_x)

        return total_loss / total_samples, total_correct / total_samples

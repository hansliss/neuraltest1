"""A small feed-forward neural network implemented with NumPy."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass
class SimpleNeuralNetwork:
    input_dim: int
    hidden_dim: int
    output_dim: int
    seed: int = 1234
    weight_scale: float = 0.01
    params: Dict[str, np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.params = {
            "W1": rng.normal(0, self.weight_scale, size=(self.input_dim, self.hidden_dim)).astype(np.float32),
            "b1": np.zeros(self.hidden_dim, dtype=np.float32),
            "W2": rng.normal(0, self.weight_scale, size=(self.hidden_dim, self.output_dim)).astype(np.float32),
            "b2": np.zeros(self.output_dim, dtype=np.float32),
        }

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(logits_shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        W1, b1, W2, b2 = self.params["W1"], self.params["b1"], self.params["W2"], self.params["b2"]
        z1 = x @ W1 + b1
        hidden = self._relu(z1)
        logits = hidden @ W2 + b2
        cache = {
            "input": x,
            "z1": z1,
            "hidden": hidden,
        }
        return logits, cache

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(x)
        return self._softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    def loss_and_gradients(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        logits, cache = self.forward(x)
        probs = self._softmax(logits)
        n = x.shape[0]
        y_indices = y.astype(int)
        correct_logprobs = -np.log(probs[np.arange(n), y_indices] + 1e-12)
        loss = float(correct_logprobs.mean())

        grad_logits = probs
        grad_logits[np.arange(n), y_indices] -= 1.0
        grad_logits /= n

        hidden = cache["hidden"]
        z1 = cache["z1"]
        grad_W2 = hidden.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)

        grad_hidden = grad_logits @ self.params["W2"].T
        grad_hidden[z1 <= 0] = 0.0

        grad_W1 = cache["input"].T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)

        grads = {
            "W1": grad_W1.astype(np.float32),
            "b1": grad_b1.astype(np.float32),
            "W2": grad_W2.astype(np.float32),
            "b2": grad_b2.astype(np.float32),
        }
        return loss, grads

    def apply_gradients(self, grads: dict[str, np.ndarray], learning_rate: float, weight_decay: float = 0.0) -> None:
        for name, grad in grads.items():
            if weight_decay and name.startswith("W"):
                grad = grad + weight_decay * self.params[name]
            self.params[name] -= learning_rate * grad

    def save(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **self.params, input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim)

    @classmethod
    def load(cls, source: Path) -> "SimpleNeuralNetwork":
        payload = np.load(source, allow_pickle=True)
        model = cls(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            output_dim=int(payload["output_dim"]),
        )
        model.params = {
            "W1": payload["W1"],
            "b1": payload["b1"],
            "W2": payload["W2"],
            "b2": payload["b2"],
        }
        return model

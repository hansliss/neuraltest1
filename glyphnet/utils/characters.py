"""Helpers for working with character sets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CharacterSet:
    chars: tuple[str, ...]
    char_to_index: dict[str, int]

    @classmethod
    def from_iterable(cls, chars: Iterable[str]) -> "CharacterSet":
        unique: list[str] = []
        mapping: dict[str, int] = {}
        for ch in chars:
            if ch in mapping:
                continue
            mapping[ch] = len(unique)
            unique.append(ch)
        return cls(tuple(unique), mapping)

    @property
    def size(self) -> int:
        return len(self.chars)

    def index_of(self, ch: str) -> int:
        return self.char_to_index[ch]

    def indices_of(self, text: Sequence[str] | str) -> list[int]:
        if isinstance(text, str):
            return [self.char_to_index[ch] for ch in text]
        return [self.char_to_index[ch] for ch in text]

    def char_for(self, index: int) -> str:
        return self.chars[index]

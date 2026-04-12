"""Tokenizer boundary used by transcript state and model-specific adapters."""

from __future__ import annotations

from typing import Protocol, Sequence


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class UTF8Tokenizer:
    """A deterministic, dependency-free tokenizer for tests and local demos.

    Production must inject the exact model tokenizer. Byte tokens make local
    traces reproducible without downloading model assets.
    """

    def encode(self, text: str) -> list[int]:
        return [byte + 1 for byte in text.encode("utf-8")]

    def decode(self, token_ids: Sequence[int]) -> str:
        return bytes(token_id - 1 for token_id in token_ids).decode("utf-8")


def token_lcp(old: Sequence[int], new: Sequence[int], guard_tokens: int = 0) -> int:
    """Return a conservative token-level longest common prefix length."""

    prefix = 0
    for old_token, new_token in zip(old, new, strict=False):
        if old_token != new_token:
            break
        prefix += 1

    if old != new and guard_tokens:
        prefix = max(0, prefix - guard_tokens)
    return prefix


"""Logical Key-Value cache metadata and reference-counted block simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence


class Validity(str, Enum):
    COMMITTED = "committed"
    TENTATIVE = "tentative"
    INVALIDATED = "invalidated"


class Ownership(str, Enum):
    SPINE = "spine"
    SHARED = "shared"
    BRANCH_PRIVATE = "branch_private"


class Residency(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    EVICTED = "evicted"


@dataclass(slots=True)
class PhysicalBlock:
    block_id: int
    token_ids: list[int]
    ref_count: int = 1


@dataclass(slots=True)
class KVRef:
    logical_id: int
    token_start: int
    token_end: int
    validity: Validity
    ownership: Ownership
    residency: Residency
    physical_block_id: int
    version: int

    @property
    def length(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True, slots=True)
class RollbackResult:
    removed_tokens: int
    released_blocks: tuple[int, ...]
    copied_blocks: int


class BlockPool:
    def __init__(self, block_size: int = 16) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self._blocks: dict[int, PhysicalBlock] = {}
        self._next_block_id = 0
        self.blocks_allocated = 0
        self.blocks_freed = 0

    def allocate(self, token_ids: Sequence[int]) -> int:
        if not token_ids or len(token_ids) > self.block_size:
            raise ValueError("a block must contain between 1 and block_size tokens")
        block_id = self._next_block_id
        self._next_block_id += 1
        self._blocks[block_id] = PhysicalBlock(block_id, list(token_ids))
        self.blocks_allocated += 1
        return block_id

    def get(self, block_id: int) -> PhysicalBlock:
        try:
            return self._blocks[block_id]
        except KeyError as exc:
            raise KeyError(f"unknown physical block: {block_id}") from exc

    def incref(self, block_id: int) -> None:
        self.get(block_id).ref_count += 1

    def decref(self, block_id: int) -> bool:
        block = self.get(block_id)
        if block.ref_count <= 0:
            raise RuntimeError(f"block {block_id} has an invalid ref_count")
        block.ref_count -= 1
        if block.ref_count:
            return False
        del self._blocks[block_id]
        self.blocks_freed += 1
        return True

    def extend(self, block_id: int, token_ids: Sequence[int]) -> None:
        block = self.get(block_id)
        if block.ref_count != 1:
            raise RuntimeError("shared blocks must use copy-on-write")
        if len(block.token_ids) + len(token_ids) > self.block_size:
            raise ValueError("block capacity exceeded")
        block.token_ids.extend(token_ids)

    def truncate(self, block_id: int, length: int) -> None:
        block = self.get(block_id)
        if block.ref_count != 1:
            raise RuntimeError("shared blocks must use copy-on-write")
        if not 0 < length <= len(block.token_ids):
            raise ValueError("truncated block must retain at least one token")
        del block.token_ids[length:]

    def snapshot(self) -> dict[int, dict[str, Any]]:
        return {
            block_id: {
                "tokens": len(block.token_ids),
                "ref_count": block.ref_count,
            }
            for block_id, block in sorted(self._blocks.items())
        }


@dataclass(slots=True)
class _SessionKV:
    refs: list[KVRef] = field(default_factory=list)


class VersionedKVManager:
    """Owns logical cache references while an engine owns physical tensors.

    ``BlockPool`` is a deterministic stand-in used for local correctness tests.
    A vLLM integration maps the same allocate, share, rollback, and release
    transitions to the engine's cache manager.
    """

    def __init__(self, block_size: int = 16) -> None:
        self.pool = BlockPool(block_size)
        self._sessions: dict[str, _SessionKV] = {}
        self._branches: dict[str, list[KVRef]] = {}
        self._next_logical_id = 0
        self.copy_on_write_blocks = 0
        self.rollback_tokens = 0

    def _new_ref(
        self,
        start: int,
        end: int,
        validity: Validity,
        ownership: Ownership,
        block_id: int,
        version: int,
    ) -> KVRef:
        logical_id = self._next_logical_id
        self._next_logical_id += 1
        return KVRef(
            logical_id,
            start,
            end,
            validity,
            ownership,
            Residency.GPU,
            block_id,
            version,
        )

    def _session(self, session_id: str) -> _SessionKV:
        return self._sessions.setdefault(session_id, _SessionKV())

    def append_session(
        self,
        session_id: str,
        token_ids: Sequence[int],
        validity: Validity,
        version: int,
    ) -> tuple[KVRef, ...]:
        if validity is Validity.INVALIDATED:
            raise ValueError("cannot append invalidated tokens")
        refs = self._session(session_id).refs
        appended_refs: list[KVRef] = []
        remaining = list(token_ids)
        cursor = refs[-1].token_end if refs else 0

        if remaining and refs:
            last = refs[-1]
            block = self.pool.get(last.physical_block_id)
            space = self.pool.block_size - len(block.token_ids)
            if space and last.validity is validity and last.ownership is Ownership.SPINE:
                take = remaining[:space]
                if block.ref_count > 1:
                    replacement = self.pool.allocate(block.token_ids + take)
                    self.pool.decref(last.physical_block_id)
                    last.physical_block_id = replacement
                    self.copy_on_write_blocks += 1
                else:
                    self.pool.extend(last.physical_block_id, take)
                last.token_end += len(take)
                last.version = version
                cursor += len(take)
                del remaining[: len(take)]
                appended_refs.append(last)

        while remaining:
            chunk = remaining[: self.pool.block_size]
            del remaining[: len(chunk)]
            block_id = self.pool.allocate(chunk)
            ref = self._new_ref(
                cursor,
                cursor + len(chunk),
                validity,
                Ownership.SPINE,
                block_id,
                version,
            )
            refs.append(ref)
            appended_refs.append(ref)
            cursor += len(chunk)
        return tuple(appended_refs)

    def rollback_tentative(self, session_id: str, from_token: int) -> RollbackResult:
        refs = self._session(session_id).refs
        current_end = refs[-1].token_end if refs else 0
        frontier = max(
            (ref.token_end for ref in refs if ref.validity is Validity.COMMITTED),
            default=0,
        )
        if from_token < frontier:
            raise ValueError("ordinary revision cannot roll back committed tokens")
        if from_token >= current_end:
            return RollbackResult(0, (), 0)

        released: list[int] = []
        copied = 0
        kept: list[KVRef] = []
        for ref in refs:
            if ref.token_end <= from_token:
                kept.append(ref)
                continue
            if ref.token_start < from_token < ref.token_end:
                keep_length = from_token - ref.token_start
                block = self.pool.get(ref.physical_block_id)
                if block.ref_count > 1:
                    replacement = self.pool.allocate(block.token_ids[:keep_length])
                    self.pool.decref(ref.physical_block_id)
                    ref.physical_block_id = replacement
                    copied += 1
                    self.copy_on_write_blocks += 1
                else:
                    self.pool.truncate(ref.physical_block_id, keep_length)
                ref.token_end = from_token
                kept.append(ref)
                continue
            block_id = ref.physical_block_id
            ref.validity = Validity.INVALIDATED
            if self.pool.decref(block_id):
                released.append(block_id)

        self._session(session_id).refs = kept
        removed = current_end - from_token
        self.rollback_tokens += removed
        return RollbackResult(removed, tuple(released), copied)

    def commit_tentative(self, session_id: str, frontier: int, version: int) -> None:
        refs = self._session(session_id).refs
        current_end = refs[-1].token_end if refs else 0
        if frontier != current_end:
            raise ValueError("the first implementation commits the complete tentative tail")
        for ref in refs:
            if ref.token_end <= frontier:
                ref.validity = Validity.COMMITTED
                ref.version = version

    def fork_branch(
        self,
        branch_id: str,
        session_id: str,
        include_tentative: bool,
    ) -> tuple[KVRef, ...]:
        if branch_id in self._branches:
            raise ValueError(f"branch already exists: {branch_id}")
        branch_refs: list[KVRef] = []
        for source in self._session(session_id).refs:
            if not include_tentative and source.validity is not Validity.COMMITTED:
                continue
            self.pool.incref(source.physical_block_id)
            branch_refs.append(
                self._new_ref(
                    source.token_start,
                    source.token_end,
                    source.validity,
                    Ownership.SHARED,
                    source.physical_block_id,
                    source.version,
                )
            )
        self._branches[branch_id] = branch_refs
        return tuple(branch_refs)

    def append_branch(
        self,
        branch_id: str,
        token_ids: Sequence[int],
        version: int,
    ) -> tuple[KVRef, ...]:
        try:
            refs = self._branches[branch_id]
        except KeyError as exc:
            raise KeyError(f"unknown branch: {branch_id}") from exc
        remaining = list(token_ids)
        appended: list[KVRef] = []
        cursor = refs[-1].token_end if refs else 0

        if remaining and refs:
            last = refs[-1]
            block = self.pool.get(last.physical_block_id)
            space = self.pool.block_size - len(block.token_ids)
            if space:
                take = remaining[:space]
                if last.ownership is Ownership.SHARED:
                    replacement = self.pool.allocate(block.token_ids + take)
                    self.pool.decref(last.physical_block_id)
                    last.physical_block_id = replacement
                    last.ownership = Ownership.BRANCH_PRIVATE
                    self.copy_on_write_blocks += 1
                else:
                    self.pool.extend(last.physical_block_id, take)
                last.token_end += len(take)
                last.version = version
                cursor += len(take)
                del remaining[: len(take)]
                appended.append(last)

        while remaining:
            chunk = remaining[: self.pool.block_size]
            del remaining[: len(chunk)]
            block_id = self.pool.allocate(chunk)
            ref = self._new_ref(
                cursor,
                cursor + len(chunk),
                Validity.COMMITTED,
                Ownership.BRANCH_PRIVATE,
                block_id,
                version,
            )
            refs.append(ref)
            appended.append(ref)
            cursor += len(chunk)
        return tuple(appended)

    def release_branch(self, branch_id: str) -> tuple[int, ...]:
        refs = self._branches.pop(branch_id, None)
        if refs is None:
            return ()
        freed: list[int] = []
        for ref in refs:
            if self.pool.decref(ref.physical_block_id):
                freed.append(ref.physical_block_id)
        return tuple(freed)

    def session_refs(self, session_id: str) -> tuple[KVRef, ...]:
        return tuple(self._session(session_id).refs)

    def branch_refs(self, branch_id: str) -> tuple[KVRef, ...]:
        return tuple(self._branches[branch_id])

    def session_tokens(self, session_id: str) -> tuple[int, ...]:
        return self._tokens(self._session(session_id).refs)

    def branch_tokens(self, branch_id: str) -> tuple[int, ...]:
        return self._tokens(self._branches[branch_id])

    def _tokens(self, refs: Iterable[KVRef]) -> tuple[int, ...]:
        output: list[int] = []
        for ref in refs:
            output.extend(self.pool.get(ref.physical_block_id).token_ids)
        return tuple(output)

    def validate_ref_counts(self) -> None:
        expected: dict[int, int] = {}
        all_refs = [session.refs for session in self._sessions.values()]
        all_refs.extend(self._branches.values())
        for refs in all_refs:
            for ref in refs:
                expected[ref.physical_block_id] = expected.get(ref.physical_block_id, 0) + 1
        actual = {block_id: data["ref_count"] for block_id, data in self.pool.snapshot().items()}
        if expected != actual:
            raise AssertionError(f"reference count mismatch: expected={expected}, actual={actual}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "block_size": self.pool.block_size,
            "blocks": self.pool.snapshot(),
            "sessions": {key: len(value.refs) for key, value in sorted(self._sessions.items())},
            "branches": {key: len(value) for key, value in sorted(self._branches.items())},
            "copy_on_write_blocks": self.copy_on_write_blocks,
            "rollback_tokens": self.rollback_tokens,
        }


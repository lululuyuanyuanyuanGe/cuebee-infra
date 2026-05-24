from __future__ import annotations

import unittest

from cuebee.branch_graph import BranchGraph, BranchStatus, TaskKind
from cuebee.event_schema import TokenRange
from cuebee.kv_metadata import Validity, VersionedKVManager


class KVManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kv = VersionedKVManager(block_size=4)

    def test_append_rollback_and_commit(self) -> None:
        self.kv.append_session("s", [1, 2, 3, 4], Validity.COMMITTED, 1)
        self.kv.append_session("s", [5, 6, 7], Validity.TENTATIVE, 2)
        rollback = self.kv.rollback_tentative("s", 5)
        self.assertEqual(rollback.removed_tokens, 2)
        self.assertEqual(self.kv.session_tokens("s"), (1, 2, 3, 4, 5))
        self.kv.commit_tentative("s", 5, 3)
        self.assertTrue(all(ref.validity is Validity.COMMITTED for ref in self.kv.session_refs("s")))
        self.kv.validate_ref_counts()

    def test_branch_append_copy_on_write_preserves_spine(self) -> None:
        self.kv.append_session("s", [1, 2, 3, 4, 5, 6], Validity.COMMITTED, 1)
        self.kv.fork_branch("b", "s", include_tentative=False)
        self.kv.append_branch("b", [7], version=1)

        self.assertEqual(self.kv.session_tokens("s"), (1, 2, 3, 4, 5, 6))
        self.assertEqual(self.kv.branch_tokens("b"), (1, 2, 3, 4, 5, 6, 7))
        self.assertEqual(self.kv.copy_on_write_blocks, 1)
        self.kv.validate_ref_counts()

    def test_rollback_shared_tail_uses_copy_on_write(self) -> None:
        self.kv.append_session("s", [1, 2, 3, 4], Validity.COMMITTED, 1)
        self.kv.append_session("s", [5, 6, 7], Validity.TENTATIVE, 2)
        self.kv.fork_branch("b", "s", include_tentative=True)
        rollback = self.kv.rollback_tentative("s", 6)

        self.assertEqual(rollback.copied_blocks, 1)
        self.assertEqual(self.kv.session_tokens("s"), (1, 2, 3, 4, 5, 6))
        self.assertEqual(self.kv.branch_tokens("b"), (1, 2, 3, 4, 5, 6, 7))
        self.kv.release_branch("b")
        self.kv.validate_ref_counts()

    def test_cannot_rollback_committed_spine(self) -> None:
        self.kv.append_session("s", [1, 2, 3], Validity.COMMITTED, 1)
        with self.assertRaises(ValueError):
            self.kv.rollback_tentative("s", 2)


class BranchGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kv = VersionedKVManager(block_size=4)
        self.kv.append_session("s", [1, 2, 3, 4], Validity.COMMITTED, 1)
        self.kv.append_session("s", [5, 6], Validity.TENTATIVE, 2)
        self.graph = BranchGraph(self.kv)

    def test_revision_invalidates_tail_branch_but_not_memory(self) -> None:
        hint = self.graph.create("hint", "s", TaskKind.PROACTIVE_HINT, 2, 4, 6)
        memory = self.graph.create("memory", "s", TaskKind.MEMORY_EXTRACTION, 2, 4, 6)

        invalidated = self.graph.invalidate("s", TokenRange(5, 7), new_version=3)
        self.assertEqual(invalidated, ("hint",))
        self.assertEqual(hint.status, BranchStatus.INVALIDATED)
        self.assertEqual(memory.status, BranchStatus.QUEUED)
        self.kv.validate_ref_counts()

    def test_append_invalidates_exact_version_branch(self) -> None:
        hint = self.graph.create("hint", "s", TaskKind.PROACTIVE_HINT, 2, 4, 6)
        invalidated = self.graph.invalidate("s", TokenRange(6, 8), new_version=3)
        self.assertEqual(invalidated, ("hint",))
        self.assertEqual(hint.invalidated_by_version, 3)


if __name__ == "__main__":
    unittest.main()

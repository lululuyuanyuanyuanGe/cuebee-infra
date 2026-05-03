from __future__ import annotations

import unittest

from cuebee.event_schema import EventType, STTEvent
from cuebee.session_manager import SessionManager
from cuebee.tokenizer import UTF8Tokenizer


class SessionManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = UTF8Tokenizer()
        self.manager = SessionManager(self.tokenizer, retokenization_guard_tokens=1)

    def event(self, revision: int, event_type: EventType, text: str) -> STTEvent:
        return STTEvent("s1", "seg1", revision, event_type, text)

    def test_append_revise_commit_lifecycle(self) -> None:
        first = self.manager.apply(self.event(1, EventType.APPEND_PARTIAL, "budget is fifty"))
        revised = self.manager.apply(
            self.event(2, EventType.REVISE_PARTIAL, "budget is one hundred fifty")
        )
        committed = self.manager.apply(
            self.event(3, EventType.COMMIT_FINAL, "budget is one hundred fifty")
        )

        state = self.manager.get("s1")
        self.assertGreater(revised.rollback_tokens, 0)
        self.assertTrue(committed.committed)
        self.assertEqual(state.commit_frontier, len(self.tokenizer.encode(state.transcript_text())))
        self.assertEqual(state.tentative_tokens, [])

    def test_commit_frontier_is_append_only(self) -> None:
        self.manager.apply(self.event(1, EventType.COMMIT_FINAL, "first fact"))
        first_frontier = self.manager.get("s1").commit_frontier
        self.manager.apply(
            STTEvent("s1", "seg2", 1, EventType.COMMIT_FINAL, "second fact")
        )
        state = self.manager.get("s1")
        self.assertGreater(state.commit_frontier, first_frontier)
        self.assertEqual(state.commit_frontier, len(self.tokenizer.encode(state.transcript_text())))

    def test_two_tentative_segments_are_rejected(self) -> None:
        self.manager.apply(self.event(1, EventType.APPEND_PARTIAL, "unfinished"))
        with self.assertRaises(ValueError):
            self.manager.apply(
                STTEvent("s1", "seg2", 1, EventType.APPEND_PARTIAL, "overlap")
            )

    def test_speaker_relabel_changes_no_tokens(self) -> None:
        self.manager.apply(self.event(1, EventType.APPEND_PARTIAL, "hello"))
        before = self.manager.get("s1").all_tokens
        diff = self.manager.apply(
            STTEvent("s1", "metadata", 1, EventType.SPEAKER_RELABEL, "spk_001=Alice")
        )
        self.assertEqual(self.manager.get("s1").all_tokens, before)
        self.assertEqual(diff.rollback_tokens, 0)
        self.assertEqual(self.manager.get("s1").speaker_names["spk_001"], "Alice")

    def test_closed_session_rejects_updates(self) -> None:
        self.manager.apply(STTEvent("s1", "close", 1, EventType.SESSION_CLOSE))
        with self.assertRaises(RuntimeError):
            self.manager.apply(self.event(2, EventType.COMMIT_FINAL, "late"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from cuebee.event_schema import EventType, STTEvent, TokenRange
from cuebee.revision_adapter import RevisionAdapter
from cuebee.tokenizer import UTF8Tokenizer, token_lcp


class EventContractTests(unittest.TestCase):
    def test_round_trip_payload(self) -> None:
        event = STTEvent(
            session_id="session-1",
            segment_id="segment-1",
            revision=2,
            type=EventType.REVISE_PARTIAL,
            text="budget is 150",
            start_ms=10,
            end_ms=200,
            confidence=0.9,
            client_epoch=3,
            seq_no=4,
        )
        self.assertEqual(STTEvent.from_dict(event.to_dict()), event)

    def test_invalid_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TokenRange(2, 1)

    def test_adapter_normalizes_and_deduplicates(self) -> None:
        adapter = RevisionAdapter()
        first = STTEvent("s", "g", 1, EventType.REVISE_PARTIAL, "fifty")
        revised = STTEvent("s", "g", 2, EventType.APPEND_PARTIAL, "one fifty")

        self.assertEqual(adapter.normalize(first).type, EventType.APPEND_PARTIAL)  # type: ignore[union-attr]
        self.assertIsNone(adapter.normalize(first))
        self.assertEqual(adapter.normalize(revised).type, EventType.REVISE_PARTIAL)  # type: ignore[union-attr]
        self.assertEqual(adapter.duplicates_dropped, 1)

    def test_final_segment_rejects_late_partial(self) -> None:
        adapter = RevisionAdapter()
        final = STTEvent("s", "g", 1, EventType.COMMIT_FINAL, "final")
        late = STTEvent("s", "g", 2, EventType.REVISE_PARTIAL, "late")
        self.assertIsNotNone(adapter.normalize(final))
        self.assertIsNone(adapter.normalize(late))

    def test_utf8_tokenizer_round_trip(self) -> None:
        tokenizer = UTF8Tokenizer()
        text = "预算是一百五十万"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_guarded_lcp_backs_up(self) -> None:
        self.assertEqual(token_lcp([1, 2, 3], [1, 2, 4], guard_tokens=1), 1)


if __name__ == "__main__":
    unittest.main()

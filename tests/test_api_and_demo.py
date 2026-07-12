from __future__ import annotations

import unittest

from cuebee.api.server import Application
from cuebee.demo import run_demo


class ApplicationTests(unittest.TestCase):
    def test_stt_task_and_tick(self) -> None:
        app = Application()
        event = app.handle_stt(
            "s",
            {
                "segment_id": "seg",
                "revision": 1,
                "type": "commit_final",
                "text": "the answer is forty two",
            },
        )
        self.assertTrue(event["accepted"])
        task = app.submit_task(
            "s",
            {
                "kind": "user_query",
                "prompt": "what is the answer",
                "task_id": "q1",
            },
        )
        self.assertEqual(task["task_id"], "q1")
        output = app.tick()
        self.assertEqual(output["decision"], "allow")

    def test_speaker_endpoint_flow(self) -> None:
        app = Application()
        accepted = app.submit_audio(
            {
                "session_id": "s",
                "chunk_id": "c",
                "start_ms": 0,
                "end_ms": 500,
                "samples": [0.2] * 80,
                "now_ms": 0,
            }
        )
        self.assertEqual(accepted["status"], "accepted")
        output = app.flush_audio({"now_ms": 30, "force": True})
        self.assertEqual(output["segments"][0]["speaker_id"], "spk_001")


class DemoTests(unittest.TestCase):
    def test_demo_prevents_old_branch_and_allows_final_output(self) -> None:
        timeline = run_demo()
        revision = next(item for item in timeline if item["step"] == "revision_v42")
        output = next(item for item in timeline if item["step"] == "fresh_output")
        self.assertIn("search-v41", revision["invalidated_tasks"])
        self.assertEqual(output["decision"], "allow")


if __name__ == "__main__":
    unittest.main()


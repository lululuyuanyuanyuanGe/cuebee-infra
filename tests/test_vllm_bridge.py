from __future__ import annotations

import asyncio
import unittest

from cuebee.engine import SessionRevisionUpdate
from cuebee.vllm_bridge import VLLMRevisionBridge


def update(version: int, replace_from: int, tokens: tuple[int, ...]) -> SessionRevisionUpdate:
    return SessionRevisionUpdate(
        session_id="session",
        version=version,
        base_version=version - 1,
        replace_from=replace_from,
        replacement_token_ids=tokens,
        commit_frontier=4,
        tentative=True,
    )


class VLLMRevisionBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_orders_versioned_inputs_and_closes(self) -> None:
        bridge = VLLMRevisionBridge()
        stream = bridge.open_session(
            "session",
            sampling_params="params",
            input_factory=lambda item, params: (item.version, item.replace_from, params),
        )
        bridge.update_session(update(2, 8, (20, 21)))
        bridge.update_session(update(3, 9, (30,)))
        bridge.close_session_nowait("session")

        values = [value async for value in stream.inputs()]
        self.assertEqual(values, [(2, 8, "params"), (3, 9, "params")])

    async def test_applies_backpressure(self) -> None:
        bridge = VLLMRevisionBridge()
        bridge.open_session(
            "session",
            sampling_params=None,
            max_pending_updates=1,
            input_factory=lambda item, params: (item, params),
        )
        bridge.update_session(update(2, 8, (20,)))
        with self.assertRaises(asyncio.QueueFull):
            bridge.update_session(update(3, 9, (30,)))

    def test_update_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "base_version"):
            SessionRevisionUpdate(
                session_id="session",
                version=2,
                base_version=2,
                replace_from=0,
                replacement_token_ids=(1,),
                commit_frontier=0,
                tentative=True,
            )


if __name__ == "__main__":
    unittest.main()

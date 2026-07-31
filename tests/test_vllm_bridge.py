from __future__ import annotations

import asyncio
import unittest

from cuebee.engine import BranchForkUpdate, SessionRevisionUpdate
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

    async def test_opens_partial_block_branch(self) -> None:
        bridge = VLLMRevisionBridge()
        bridge.open_session(
            "session",
            sampling_params="params",
            input_factory=lambda item, params: (item, params),
        )
        update = BranchForkUpdate(
            source_session_id="session",
            branch_id="branch:search",
            version=8,
            fork_at=20,
            prompt_token_ids=(*range(20), 100, 101),
            commit_frontier=16,
        )
        stream = bridge.open_branch(
            update,
            sampling_params="params",
            input_factory=lambda item, params: (item, params),
        )
        stream.close_nowait()

        values = [value async for value in stream.inputs()]
        self.assertEqual(values, [(update, "params")])

        bridge.close_session_nowait("session")

    def test_rejects_branch_without_open_source(self) -> None:
        bridge = VLLMRevisionBridge()
        update = BranchForkUpdate(
            source_session_id="missing",
            branch_id="branch:search",
            version=8,
            fork_at=20,
            prompt_token_ids=(*range(20), 100),
            commit_frontier=16,
        )
        with self.assertRaisesRegex(KeyError, "source session"):
            bridge.open_branch(update, sampling_params="params")

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

"""Translate CueBee session revisions to the pinned in-process vLLM fork."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

from cuebee.engine import SessionRevisionUpdate

InputFactory = Callable[[SessionRevisionUpdate, Any], Any]
_CLOSED = object()


class VLLMInputStream:
    """Bounded asynchronous input stream consumed by ``AsyncLLM.generate``."""

    def __init__(
        self,
        sampling_params: Any,
        max_pending_updates: int = 64,
        input_factory: InputFactory | None = None,
    ) -> None:
        if max_pending_updates <= 0:
            raise ValueError("max_pending_updates must be positive")
        self.sampling_params = sampling_params
        self._queue: asyncio.Queue[SessionRevisionUpdate | object] = asyncio.Queue(
            max_pending_updates
        )
        self._input_factory = input_factory or self._build_vllm_input
        self._closed = False

    def publish_nowait(self, update: SessionRevisionUpdate) -> None:
        if self._closed:
            raise RuntimeError("vLLM input stream is closed")
        self._queue.put_nowait(update)

    async def publish(self, update: SessionRevisionUpdate) -> None:
        if self._closed:
            raise RuntimeError("vLLM input stream is closed")
        await self._queue.put(update)

    def close_nowait(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(_CLOSED)

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._queue.put(_CLOSED)

    async def inputs(self) -> AsyncGenerator[Any, None]:
        """Yield fork-native ``StreamingInput`` objects until closed."""
        while True:
            item = await self._queue.get()
            if item is _CLOSED:
                return
            assert isinstance(item, SessionRevisionUpdate)
            yield self._input_factory(item, self.sampling_params)

    @staticmethod
    def _build_vllm_input(update: SessionRevisionUpdate, sampling_params: Any) -> Any:
        try:
            from vllm import TokensPrompt  # type: ignore[import-not-found]
            from vllm.engine.protocol import StreamingInput  # type: ignore[import-not-found]
            from vllm.v1.engine import StreamingRevision  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "the pinned CueBee vLLM submodule must be installed to build inputs"
            ) from exc

        return StreamingInput(
            prompt=TokensPrompt(prompt_token_ids=list(update.replacement_token_ids)),
            sampling_params=sampling_params,
            revision=StreamingRevision(
                version=update.version,
                base_version=update.base_version,
                replace_from=update.replace_from,
                commit_frontier=update.commit_frontier,
                branch_id=update.branch_id,
            ),
        )


class VLLMRevisionBridge:
    """Own the long-lived input stream for each active CueBee session."""

    def __init__(self) -> None:
        self._streams: dict[str, VLLMInputStream] = {}

    def open_session(
        self,
        session_id: str,
        sampling_params: Any,
        *,
        max_pending_updates: int = 64,
        input_factory: InputFactory | None = None,
    ) -> VLLMInputStream:
        if not session_id:
            raise ValueError("session_id must not be empty")
        if session_id in self._streams:
            raise ValueError(f"session {session_id!r} is already open")
        stream = VLLMInputStream(
            sampling_params,
            max_pending_updates=max_pending_updates,
            input_factory=input_factory,
        )
        self._streams[session_id] = stream
        return stream

    def update_session(self, update: SessionRevisionUpdate) -> None:
        try:
            stream = self._streams[update.session_id]
        except KeyError as exc:
            raise KeyError(f"session {update.session_id!r} is not open") from exc
        stream.publish_nowait(update)

    def close_session_nowait(self, session_id: str) -> None:
        stream = self._streams.pop(session_id)
        stream.close_nowait()

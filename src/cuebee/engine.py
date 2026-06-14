"""Model engine boundary with deterministic and OpenAI-compatible backends."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True, slots=True)
class EngineRequest:
    task_id: str
    session_id: str
    base_version: int
    prompt: str
    prompt_token_ids: tuple[int, ...]
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class EngineResult:
    request_id: str
    text: str
    input_tokens: int
    output_tokens: int


class InferenceEngine(Protocol):
    def prefill(
        self,
        session_id: str,
        token_ids: Sequence[int],
        version: int,
        tentative: bool,
    ) -> None: ...

    def generate(self, request: EngineRequest) -> EngineResult: ...

    def abort(self, request_id: str) -> bool: ...


class DeterministicEngine:
    """Local backend that exercises state transitions without a model download."""

    def __init__(self) -> None:
        self.prefill_calls: list[tuple[str, tuple[int, ...], int, bool]] = []
        self.generate_calls: list[EngineRequest] = []
        self.aborted: set[str] = set()

    def prefill(
        self,
        session_id: str,
        token_ids: Sequence[int],
        version: int,
        tentative: bool,
    ) -> None:
        self.prefill_calls.append((session_id, tuple(token_ids), version, tentative))

    def generate(self, request: EngineRequest) -> EngineResult:
        self.generate_calls.append(request)
        digest = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()[:10]
        text = f"cuebee:{request.base_version}:{digest}"
        return EngineResult(
            request_id=request.task_id,
            text=text,
            input_tokens=len(request.prompt_token_ids),
            output_tokens=len(text.split(":")),
        )

    def abort(self, request_id: str) -> bool:
        self.aborted.add(request_id)
        return True


class OpenAICompatibleEngine:
    """Baseline adapter for a vLLM OpenAI-compatible completion endpoint.

    This adapter benefits from upstream automatic prefix caching but does not
    claim direct ownership of vLLM physical cache blocks. The version-specific
    in-process bridge is a separate integration boundary.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def prefill(
        self,
        session_id: str,
        token_ids: Sequence[int],
        version: int,
        tentative: bool,
    ) -> None:
        # OpenAI-compatible HTTP does not expose a prefill-only operation.
        # The full stable prompt is sent on generation and vLLM may reuse it.
        del session_id, token_ids, version, tentative

    def generate(self, request: EngineRequest) -> EngineResult:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": request.prompt,
                "max_tokens": request.max_output_tokens,
                "temperature": 0,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"inference request failed: {exc}") from exc
        choice = payload["choices"][0]
        usage = payload.get("usage", {})
        return EngineResult(
            request_id=payload.get("id", request.task_id),
            text=choice["text"],
            input_tokens=int(usage.get("prompt_tokens", len(request.prompt_token_ids))),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )

    def abort(self, request_id: str) -> bool:
        del request_id
        return False


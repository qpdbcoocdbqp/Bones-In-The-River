"""測試用的假 LLM：依序回傳預先寫好的回應，並記錄收到的 prompt。"""

from __future__ import annotations

from collections.abc import Callable


class FakeClient:
    model = "fake-model"

    def __init__(self, responses: list[str] | Callable[[str], str]) -> None:
        self._responses = responses
        self.prompts: list[str] = []

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        if callable(self._responses):
            return self._responses(prompt)
        return self._responses[len(self.prompts) - 1]

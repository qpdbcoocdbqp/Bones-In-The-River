from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(Exception):
    """Base class of LLM errors: missing config, connection failure, malformed response."""


class LLMTransportError(LLMError):
    """Connection, auth, timeout, HTTP errors. Retrying will not help, so no retry."""


class LLMFormatError(LLMError):
    """The server responded, but not with valid JSON or not matching the schema. Worth retrying."""


class LLMClient(Protocol):
    model: str

    def complete(self, prompt: str, system: str | None = None) -> str: ...


class OpenAICompatClient:
    """OpenAI-compatible endpoint (e.g. llama.cpp, vLLM, LM Studio).

    Configuration comes from .env or environment variables: BASE_URL, API_KEY, MODEL (the `model`
    argument overrides MODEL; the `base_url` / `api_key` arguments override BASE_URL / API_KEY).
    Environment variables take precedence over the .env file.
    """

    def __init__(
        self,
        model: str | None = None,
        env_file: Path = Path(".env"),
        timeout: float = 900.0,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        from dotenv import dotenv_values  # lazy: not needed unless this backend is used
        from openai import OpenAI

        file_values = dotenv_values(env_file) if env_file.exists() else {}

        def setting(name: str) -> str | None:
            return os.environ.get(name) or file_values.get(name)

        base_url = base_url or setting("BASE_URL")
        api_key = api_key or setting("API_KEY")
        resolved_model = model or setting("MODEL")
        missing = [
            n
            for n, v in (("BASE_URL", base_url), ("API_KEY", api_key), ("MODEL", resolved_model))
            if not v
        ]
        if missing or not resolved_model:
            raise LLMError(
                f"Missing settings: {', '.join(missing)} "
                f"(put them in {env_file} or the environment)"
            )
        self.model = resolved_model
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=0
            )
        # the SDK has many connection/timeout/HTTP error types; convert all to LLMTransportError
        except Exception as exc:
            raise LLMTransportError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc
        if not response.choices:
            raise LLMTransportError("LLM response has no choices")
        return response.choices[0].message.content or ""


def extract_json(text: str) -> object:
    """Extract the JSON object from the first `{` to the last `}` of the response (after removing
    <think> blocks)."""
    text = _THINK.sub("", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise LLMFormatError("No JSON object in response")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMFormatError(f"JSON parse failed: {exc}") from exc


def complete_json[T: BaseModel](
    client: LLMClient,
    prompt: str,
    schema: type[T],
    *,
    system: str | None = None,
    retries: int = 2,
) -> T:
    """Call the client and validate against the schema.

    Only format mismatches are retried (up to `retries` times), and the retry includes why the last
    one was rejected: with temperature 0,
    resending verbatim just returns the same response. Connection errors (LLMTransportError)
    propagate without retry.
    """
    rejected: str | None = None
    for _ in range(retries + 1):
        text = prompt
        if rejected:
            text += (
                f"\n\nYour previous answer was rejected ({rejected}). "
                "Answer again with ONLY the JSON object, in exactly the requested format."
            )
        raw = client.complete(text, system=system)
        try:
            return schema.model_validate(extract_json(raw))
        except LLMFormatError as exc:
            rejected = str(exc)
        except ValidationError as exc:
            rejected = f"does not match the schema: {str(exc)[:200]}"
    raise LLMFormatError(f"Still unparseable after {retries} retries: {rejected}")

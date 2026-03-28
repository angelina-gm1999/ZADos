"""
ZA-DOS LLM Interpretation Layer — Ollama HTTP integration.

Provides:
    call_llama()            — single HTTP call, no retry
    call_llama_with_retry() — exponential-backoff wrapper
    LLMCallError            — raised on unrecoverable failure

All communication is JSON over Ollama's /api/chat endpoint.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from zados.LLM_interpretation.constants import (
    OLLAMA_BASE_URL,
    OLLAMA_MAX_RETRIES,
    OLLAMA_MODEL,
)


class LLMCallError(Exception):
    """Raised when an Ollama call fails (connection error, HTTP error, bad JSON)."""


# ---------------------------------------------------------------------------
# Single call
# ---------------------------------------------------------------------------

def call_llama(
    messages: List[Dict[str, Any]],
    max_tokens: int = 800,
    temperature: float = 0.75,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Single Ollama HTTP POST to /api/chat.

    Parameters
    ----------
    messages    : OpenAI-format message list.
    max_tokens  : Maximum tokens to generate (mapped to ``num_predict``).
    temperature : Sampling temperature.
    tools       : Optional tool definitions for function-calling.

    Returns
    -------
    dict with at least ``"content": str``.
    If the model issued a tool call, the dict also contains
    ``"tool_calls": list``.

    Raises
    ------
    LLMCallError on any HTTP, connection, or JSON-decode failure.
    """
    payload: Dict[str, Any] = {
        "model":   OLLAMA_MODEL,
        "messages": messages,
        "stream":  False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if tools:
        payload["tools"] = tools

    body = json.dumps(payload).encode("utf-8")
    url  = f"{OLLAMA_BASE_URL}/api/chat"

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise LLMCallError(f"Ollama HTTP {exc.code}: {exc.reason}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise LLMCallError(f"Ollama connection error: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMCallError(f"Ollama returned non-JSON: {raw[:200]}") from exc

    message = data.get("message", {})
    content = message.get("content", "")

    result: Dict[str, Any] = {"content": content}
    if "tool_calls" in message:
        result["tool_calls"] = message["tool_calls"]

    return result


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def call_llama_with_retry(
    messages: List[Dict[str, Any]],
    max_tokens: int = 800,
    temperature: float = 0.75,
    tools: Optional[List[Dict[str, Any]]] = None,
    max_retries: int = OLLAMA_MAX_RETRIES,
) -> Dict[str, Any]:
    """
    call_llama() with exponential-backoff retry.

    Attempts up to ``max_retries`` times with delays of 1 s, 2 s, 4 s …
    Raises ``LLMCallError`` if all attempts fail.
    """
    last_exc: Optional[LLMCallError] = None

    for attempt in range(max_retries):
        try:
            return call_llama(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
            )
        except LLMCallError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)   # 1 s, 2 s, 4 s …

    raise LLMCallError(
        f"Ollama failed after {max_retries} attempts: {last_exc}"
    ) from last_exc

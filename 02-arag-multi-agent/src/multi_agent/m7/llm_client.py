"""M7 LLM Client: simple wrapper around vLLM OpenAI-compatible API.

Adapted from OPERA's VllmChatClient. Strips reasoning tags from responses.
"""

from __future__ import annotations

import logging
import re
import time
import threading
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── Reasoning tag stripping ──────────────────────────────────────────

REASONING_TAG_BLOCK_RE = re.compile(
    r"<(?:think|thnk)(?:\s[^>]*)?>.*?</(?:think|thnk)>",
    flags=re.IGNORECASE | re.DOTALL,
)
REASONING_TAG_OPEN_RE = re.compile(r"<(?:think|thnk)(?:\s[^>]*)?>", flags=re.IGNORECASE)
REASONING_TAG_CLOSE_RE = re.compile(r"</(?:think|thnk)>", flags=re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove reasoning tags AND untagged thinking preamble."""
    if not text or not isinstance(text, str):
        return ""
    text = REASONING_TAG_BLOCK_RE.sub("", text)
    text = REASONING_TAG_OPEN_RE.sub("", text)
    text = REASONING_TAG_CLOSE_RE.sub("", text)
    # If response contains JSON, extract from JSON onwards
    # Handles untagged thinking like "Okay, let's tackle..."
    json_start = text.find("{")
    if json_start > 0 and json_start < len(text) - 10:
        text = text[json_start:]
    elif text.find("[") > 0:
        arr_start = text.find("[")
        if arr_start < len(text) - 10:
            text = text[arr_start:]
    return text.strip()


# ── Token Tracker ────────────────────────────────────────────────────

class TokenTracker:
    """Thread-local token and API call counter."""

    def __init__(self):
        self._local = threading.local()

    def _ensure(self):
        if not hasattr(self._local, "prompt_tokens"):
            self._local.prompt_tokens = 0
            self._local.completion_tokens = 0
            self._local.api_calls = 0

    def reset(self):
        self._local.prompt_tokens = 0
        self._local.completion_tokens = 0
        self._local.api_calls = 0

    def record_call(self, prompt_tokens: int, completion_tokens: int):
        self._ensure()
        self._local.prompt_tokens += prompt_tokens
        self._local.completion_tokens += completion_tokens
        self._local.api_calls += 1

    def snapshot(self) -> dict[str, int]:
        self._ensure()
        return {
            "prompt_tokens": self._local.prompt_tokens,
            "completion_tokens": self._local.completion_tokens,
            "total_tokens": self._local.prompt_tokens + self._local.completion_tokens,
            "api_calls": self._local.api_calls,
        }


token_tracker = TokenTracker()


# ── VllmChatClient ───────────────────────────────────────────────────

class VllmChatClient:
    """Simple OpenAI-compatible chat client for vLLM."""

    def __init__(
        self,
        model: str = "Qwen3-8B",
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "dummy",
        timeout: int = 600,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        system_prompt: str = "You are a helpful AI assistant.",
        enable_thinking: bool = False,
    ) -> str:
        """Single chat completion call. Returns content string."""
        payload = {
            "model": self.model,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.95,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                # Track tokens
                usage = data.get("usage", {})
                token_tracker.record_call(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )

                return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                last_error = exc
                sleep_s = 2 ** attempt
                logger.warning(
                    "vLLM request failed (attempt %d/3): %s", attempt + 1, exc,
                )
                time.sleep(sleep_s)

        raise RuntimeError(f"vLLM request failed after retries: {last_error}")

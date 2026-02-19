from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str

    @staticmethod
    def from_env() -> Optional["LLMConfig"]:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        return LLMConfig(api_key=api_key, base_url=base_url, model=model)


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig, timeout_s: float = 20.0) -> None:
        self.config = config
        self.timeout_s = timeout_s

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {e.code}: {raw}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"LLM connection error: {e}") from e


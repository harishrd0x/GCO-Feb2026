from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    base_url: str
    model: str

    @staticmethod
    def from_env() -> Optional["OpenAIConfig"]:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        return OpenAIConfig(api_key=api_key, base_url=base_url, model=model)


@dataclass(frozen=True)
class AzureOpenAIConfig:
    endpoint: str
    api_key: str
    api_version: str
    deployment: str

    @staticmethod
    def from_env() -> Optional["AzureOpenAIConfig"]:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
        api_key = os.getenv("AZURE_OPENAI_KEY", "").strip()
        api_version = os.getenv("AZURE_API_VERSION", "").strip()
        deployment = (
            os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
            or os.getenv("OPENAI_MODEL", "").strip()
            or "gpt-4o-mini"
        )

        if not (endpoint and api_key and api_version):
            return None
        return AzureOpenAIConfig(
            endpoint=endpoint, api_key=api_key, api_version=api_version, deployment=deployment
        )


class ChatCompletionsClient(Protocol):
    model: str

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class OpenAICompatibleClient:
    def __init__(self, config: OpenAIConfig, timeout_s: float = 20.0) -> None:
        self.config = config
        self.model = config.model
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


class AzureOpenAIClient:
    def __init__(self, config: AzureOpenAIConfig, timeout_s: float = 20.0) -> None:
        self.config = config
        # In Azure, “model” is effectively the deployment name.
        self.model = config.deployment
        self.timeout_s = timeout_s

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = (
            f"{self.config.endpoint}/openai/deployments/{self.config.deployment}/chat/completions"
            f"?api-version={self.config.api_version}"
        )
        # Azure uses the deployment in the URL; omit "model" to avoid schema issues.
        payload = dict(payload)
        payload.pop("model", None)

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"api-key": self.config.api_key, "Content-Type": "application/json"},
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


def build_llm_client_from_env() -> Optional[ChatCompletionsClient]:
    """
    Prefers Azure OpenAI config if present, otherwise falls back to OpenAI-compatible config.
    Returns None when not configured.
    """
    az = AzureOpenAIConfig.from_env()
    if az:
        return AzureOpenAIClient(az)
    oa = OpenAIConfig.from_env()
    if oa:
        return OpenAICompatibleClient(oa)
    return None


"""
LLM client abstraction layer.
Swap providers by changing config/llm.yaml — no code changes needed.
Currently supports: Ollama, vLLM (OpenAI-compatible endpoint)
"""

import logging
import requests
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Base Client ─────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    """Abstract base for all LLM clients."""

    @abstractmethod
    def complete(self, messages: list[dict]) -> str:
        """Send messages and return text response."""
        pass

    @abstractmethod
    def ping(self) -> bool:
        """Check if LLM service is available."""
        pass


# ─── Ollama Client ───────────────────────────────────────────────────────────

class OllamaClient(BaseLLMClient):
    """
    Client for Ollama local LLM serving.
    Uses OpenAI-compatible /v1/chat/completions endpoint.
    """

    def __init__(self, config: dict):
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "qwen2.5:7b")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 800)
        self.timeout = config.get("timeout", 60)

    def complete(self, messages: list[dict]) -> str:
        """Send chat completion request to Ollama."""
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s. "
                "Consider increasing timeout in config/llm.yaml"
            )
        except requests.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Ollama response format: {e}")

    def ping(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code != 200:
                return False
            # check if our model is pulled
            models = [m["name"] for m in response.json().get("models", [])]
            model_base = self.model.split(":")[0]
            return any(model_base in m for m in models)
        except Exception:
            return False


# ─── vLLM Client ─────────────────────────────────────────────────────────────

class VLLMClient(BaseLLMClient):
    """
    Client for vLLM serving (production-grade local deployment).
    Also uses OpenAI-compatible endpoint — same interface as Ollama.
    """

    def __init__(self, config: dict):
        self.base_url = config.get("base_url", "http://localhost:8000")
        self.model = config.get("model", "Qwen/Qwen2.5-7B-Instruct")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 800)
        self.timeout = config.get("timeout", 60)

    def complete(self, messages: list[dict]) -> str:
        """Send chat completion request to vLLM."""
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.Timeout:
            raise TimeoutError(
                f"vLLM request timed out after {self.timeout}s"
            )
        except requests.HTTPError as e:
            raise RuntimeError(f"vLLM HTTP error: {e}")

    def ping(self) -> bool:
        """Check if vLLM service is running."""
        try:
            response = requests.get(
                f"{self.base_url}/v1/models",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False


# ─── Factory ─────────────────────────────────────────────────────────────────

class LLMClientFactory:
    """
    Creates LLM client from config.
    Register new providers here — no other code changes needed.
    """
    _registry: dict[str, type] = {
        "ollama": OllamaClient,
        "vllm": VLLMClient,
    }

    @classmethod
    def create(cls, config: dict) -> BaseLLMClient:
        provider = config.get("provider", "ollama")
        if provider not in cls._registry:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        logger.info(
            f"Creating LLM client: provider={provider}, "
            f"model={config.get('model')}"
        )
        return cls._registry[provider](config)

    @classmethod
    def register(cls, name: str, client_class: type):
        """Register a new provider — extensible without core changes."""
        cls._registry[name] = client_class

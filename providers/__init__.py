import os

from .base import Provider
from .local import LocalProvider
from .mock import MockProvider
from .openai import OpenAIProvider


def get_provider() -> Provider:
    backend = os.getenv("LLM_BACKEND", "mock").lower()
    providers = {"mock": MockProvider, "local": LocalProvider, "openai": OpenAIProvider}
    if backend not in providers:
        raise ValueError(f"Unknown LLM_BACKEND={backend!r}")
    return providers[backend]()


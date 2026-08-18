from __future__ import annotations

from .base import GenerationRequest, Provider


class LocalProvider(Provider):
    def generate(self, request: GenerationRequest) -> dict:
        raise NotImplementedError("Configure a local model adapter before selecting LLM_BACKEND=local")


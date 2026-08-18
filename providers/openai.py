from __future__ import annotations

from .base import GenerationRequest, Provider


class OpenAIProvider(Provider):
    """Non-operational adapter boundary: this MVP never makes a paid API request."""

    def generate(self, request: GenerationRequest) -> dict:
        raise RuntimeError(
            "OpenAI calls are intentionally disabled in the mock-first MVP; "
            "implement and explicitly authorize this adapter later."
        )


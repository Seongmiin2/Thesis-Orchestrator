from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationRequest:
    agent: str
    mission: dict[str, Any]
    context: dict[str, Any]


class Provider(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        """Return a structured agent response without mutating research state."""


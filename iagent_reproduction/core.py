"""Published iAgent/i²Agent control flow, with all memory scoped to one user."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Item:
    item_id: str
    title: str
    text: str

    @property
    def document(self) -> str:
        return f"{self.title}. {self.text}"


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item: Item
    timestamp: int
    review: str = ""
    instruction: str = ""


@dataclass(frozen=True)
class ParsedInstruction:
    internal_knowledge: str
    keywords: tuple[str, ...]
    use_tools: bool


@dataclass(frozen=True)
class DynamicMemory:
    profile: str
    interests: tuple[str, ...]


class Backend(Protocol):
    def parse(self, instruction: str) -> ParsedInstruction: ...
    def retrieve(self, keywords: tuple[str, ...]) -> str: ...
    def rerank(self, *, instruction: str, internal: str, external: str, static_memory: str,
               candidates: list[Item], dynamic: DynamicMemory | None = None) -> list[str]: ...
    def update_profile(self, previous: str, positive: Interaction, negative: Item) -> str: ...
    def extract_dynamic(self, profile: str, static_memory: str, instruction: str,
                        internal: str, external: str) -> DynamicMemory: ...


def static_memory(history: list[Interaction], max_items: int = 20) -> str:
    """X_SU: text representation of user history. Truncation is explicit and deterministic."""
    recent = sorted(history, key=lambda row: row.timestamp)[-max_items:]
    return "\n".join(f"- {row.item.document}; feedback: {row.review}" for row in recent) or "No prior history."


def reflect(original: list[Item], proposed_ids: list[str]) -> list[str]:
    """Paper's self-reflection invariant: output must be a permutation of input slate.

    Missing, duplicated, or hallucinated IDs are rejected and replaced by the
    original platform order. This deterministic guard makes failures measurable.
    """
    expected = [item.item_id for item in original]
    return proposed_ids if len(proposed_ids) == len(expected) and set(proposed_ids) == set(expected) else expected


class IAgent:
    def __init__(self, backend: Backend):
        self.backend = backend

    def rank(self, history: list[Interaction], instruction: str, candidates: list[Item]) -> list[str]:
        parsed = self.backend.parse(instruction)
        external = self.backend.retrieve(parsed.keywords) if parsed.use_tools else ""
        proposal = self.backend.rerank(instruction=instruction, internal=parsed.internal_knowledge,
                                       external=external, static_memory=static_memory(history), candidates=candidates)
        return reflect(candidates, proposal)


class I2Agent(IAgent):
    """i²Agent: feedback changes only this instance's individual profile F^T."""

    def __init__(self, backend: Backend):
        super().__init__(backend)
        self.profile = "No individual profile has been learned yet."

    def learn_feedback(self, history: list[Interaction], negative: Item) -> None:
        if not history:
            return
        positive = max(history, key=lambda row: row.timestamp)
        self.profile = self.backend.update_profile(self.profile, positive, negative)

    def rank(self, history: list[Interaction], instruction: str, candidates: list[Item]) -> list[str]:
        parsed = self.backend.parse(instruction)
        external = self.backend.retrieve(parsed.keywords) if parsed.use_tools else ""
        dynamic = self.backend.extract_dynamic(self.profile, static_memory(history), instruction,
                                               parsed.internal_knowledge, external)
        proposal = self.backend.rerank(instruction=instruction, internal=parsed.internal_knowledge,
                                       external=external, static_memory=static_memory(history), candidates=candidates,
                                       dynamic=dynamic)
        return reflect(candidates, proposal)

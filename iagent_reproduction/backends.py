"""Deterministic teaching backend and a Qwen OpenAI-compatible implementation seam."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .core import DynamicMemory, Interaction, Item, ParsedInstruction

STOP = {"the", "and", "for", "with", "that", "this", "from", "into", "want", "need", "would", "like"}


def terms(text: str) -> set[str]:
    return {word.casefold() for word in re.findall(r"[A-Za-z]{4,}", text) if word.casefold() not in STOP}


class TeachingBackend:
    """Zero-cost stand-in: exposes data flow but makes no claim of LLM quality."""

    def parse(self, instruction: str) -> ParsedInstruction:
        keys = tuple(sorted(terms(instruction)))[:8]
        return ParsedInstruction("The instruction expresses constraints represented by its key terms.", keys, bool(keys))

    def retrieve(self, keywords: tuple[str, ...]) -> str:
        return "Retrieved domain cues: " + ", ".join(keywords)

    def rerank(self, *, instruction: str, internal: str, external: str, static_memory: str,
               candidates: list[Item], dynamic: DynamicMemory | None = None) -> list[str]:
        context = terms(" ".join([instruction, internal, external, static_memory, dynamic.profile if dynamic else "",
                                  " ".join(dynamic.interests) if dynamic else ""]))
        return [item.item_id for item in sorted(candidates, key=lambda item: len(context & terms(item.document)), reverse=True)]

    def update_profile(self, previous: str, positive: Interaction, negative: Item) -> str:
        cues = sorted(terms(positive.item.document + " " + positive.review))[:8]
        return f"This individual has positive feedback around {', '.join(cues)}. Previous profile: {previous[:180]}"

    def extract_dynamic(self, profile: str, static_memory: str, instruction: str,
                        internal: str, external: str) -> DynamicMemory:
        interest = tuple(sorted(terms(instruction + " " + internal)))[:8]
        return DynamicMemory(profile=f"Current profile: {profile}", interests=interest)


PARSER_PROMPT = """Analyze a recommender user instruction. Return JSON only with keys internal_knowledge (string), keywords (array of strings), and use_tools (boolean). Instruction: {instruction}"""
RERANK_PROMPT = """Rerank ONLY the supplied candidate IDs for the user's instruction. Return JSON array of every ID exactly once, no prose.
Instruction: {instruction}\nInternal knowledge: {internal}\nExternal knowledge: {external}\nStatic memory: {static_memory}\nDynamic memory: {dynamic}\nCandidates:\n{candidates}"""
PROFILE_PROMPT = """Update only this user's preference profile from feedback. Output one concise paragraph only. Previous profile: {previous}\nPositive item: {positive}\nReview: {review}\nRandom negative contrast: {negative}"""
EXTRACT_PROMPT = """Extract instruction-relative dynamic memory. Return JSON only with profile (string) and interests (array of strings).\nProfile: {profile}\nStatic memory: {static_memory}\nInstruction: {instruction}\nInternal: {internal}\nExternal: {external}"""


@dataclass
class QwenBackend:
    model: str = "qwen-plus"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str | None = None
    max_static_chars: int = 6000
    max_candidate_chars: int = 600

    def __post_init__(self) -> None:
        self.api_key = (self.api_key or os.environ.get("IAGENT_API_KEY")
                        or os.environ.get("DASHSCOPE_API_KEY"))
        if not self.api_key:
            raise RuntimeError("Set IAGENT_API_KEY (local vLLM) or DASHSCOPE_API_KEY; never store it in Git.")

    def _chat(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install API support: pip install -e .[qwen]") from exc
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        request = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": "Follow the output contract exactly."},
                         {"role": "user", "content": prompt}],
        }
        # DashScope accepts this switch; a stock vLLM OpenAI server should not
        # receive provider-specific request fields.
        if "dashscope" in self.base_url.casefold():
            request["extra_body"] = {"enable_thinking": False}
        answer = client.chat.completions.create(**request)
        if not answer.choices[0].message.content:
            raise RuntimeError("Empty Qwen response")
        return answer.choices[0].message.content

    @staticmethod
    def _json(text: str):
        return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I))

    @staticmethod
    def _clip(text: object, limit: int) -> str:
        """Bound prompt growth deterministically while preserving a readable prefix."""
        normalized = " ".join(str(text).split())
        return normalized if len(normalized) <= limit else normalized[:limit - 1] + "…"

    def parse(self, instruction: str) -> ParsedInstruction:
        row = self._json(self._chat(PARSER_PROMPT.format(instruction=instruction)))
        return ParsedInstruction(str(row["internal_knowledge"]), tuple(map(str, row.get("keywords", []))), bool(row.get("use_tools")))

    def retrieve(self, keywords: tuple[str, ...]) -> str:
        # The paper leaves concrete tools open. Keep this deterministic until a
        # source-approved retrieval tool is selected for the experiment.
        return "No external retriever configured. Keywords: " + ", ".join(keywords)

    def rerank(self, *, instruction: str, internal: str, external: str, static_memory: str,
               candidates: list[Item], dynamic: DynamicMemory | None = None) -> list[str]:
        listing = "\n".join(
            f"{item.item_id} | {self._clip(item.document, self.max_candidate_chars)}"
            for item in candidates
        )
        memory = "" if dynamic is None else f"{dynamic.profile}; interests={list(dynamic.interests)}"
        return list(map(str, self._json(self._chat(RERANK_PROMPT.format(instruction=instruction, internal=internal,
            external=external, static_memory=self._clip(static_memory, self.max_static_chars),
            dynamic=memory, candidates=listing)))))

    def update_profile(self, previous: str, positive: Interaction, negative: Item) -> str:
        return self._chat(PROFILE_PROMPT.format(
            previous=self._clip(previous, self.max_static_chars),
            positive=self._clip(positive.item.document, self.max_candidate_chars),
            review=self._clip(positive.review, self.max_candidate_chars),
            negative=self._clip(negative.document, self.max_candidate_chars),
        )).strip()

    def extract_dynamic(self, profile: str, static_memory: str, instruction: str,
                        internal: str, external: str) -> DynamicMemory:
        row = self._json(self._chat(EXTRACT_PROMPT.format(profile=profile,
            static_memory=self._clip(static_memory, self.max_static_chars),
            instruction=instruction, internal=internal, external=external)))
        return DynamicMemory(str(row["profile"]), tuple(map(str, row.get("interests", []))))

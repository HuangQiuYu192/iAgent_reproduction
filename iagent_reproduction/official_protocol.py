"""Runnable, checkpointed adaptation of the authors' released iAgent protocol.

This module deliberately keeps the public implementation's prompt layout and
text budgets, but makes the experiment auditable: each user is written to a
JSONL checkpoint and malformed rankings are explicitly retried.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation import average, ranking_metrics


def _clean(value: object) -> str:
    return re.sub(r"<.*?>", "", str(value))


def _tail(value: object, size: int = 200) -> str:
    return _clean(value)[-size:]


@dataclass(frozen=True)
class OfficialExample:
    row_index: int
    user_id: str
    instruction: str
    titles: list[str]
    descriptions: list[str]
    reviews: list[str]
    item_ids: list[int]
    target_id: int
    candidates: list[int]


def load_official_examples(data_dir: Path, *, start: int = 0, limit: int | None = None) -> tuple[list[OfficialExample], dict[int, tuple[str, str]]]:
    """Load the released Books DataFrame without regenerating instructions/slates."""
    with (data_dir / "booksAll_recagent.pkl").open("rb") as stream:
        frame = pickle.load(stream)
    mapping: dict[int, tuple[str, str]] = {}
    # Amazon descriptions can exceed csv's conservative 128 KiB default.
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit //= 10
    with (data_dir / "combined_books_asin_mapping.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            mapping[int(row["index"])] = (str(row.get("title", "")), str(row.get("description", "")))
    stop = None if limit is None else start + limit
    records: list[OfficialExample] = []
    for row_index, row in frame.iloc[start:stop].iterrows():
        ids = [int(item) for item in list(row["asin"])]
        slate = [int(item) for item in row["ranked_lists"].tolist()]
        if len(ids) < 2 or len(slate) != 10 or ids[-1] not in slate or any(item not in mapping for item in slate):
            continue
        records.append(OfficialExample(
            int(row_index), str(row["reviewerID"]), str(row["instruction"]),
            [str(value) for value in list(row["title"])[:-1]],
            [str(value) for value in list(row["description"])[:-1]],
            [str(value) for value in list(row["reviewText"])[:-1]], ids[:-1], ids[-1], slate,
        ))
    return records, mapping


class OfficialProtocolAgent:
    """Runnable iAgent/i²Agent protocol with provider-selectable JSON transport.

    ``strict`` preserves the authors' public prompt text and response fields.
    ``compact`` is retained exclusively for the earlier local-Qwen baseline.
    """

    def __init__(self, *, model: str, base_url: str, api_key: str, agent_type: str, rng: random.Random,
                 protocol_mode: str = "strict", json_mode: str = "json_object",
                 disable_thinking: bool = False, bounded_explanations: bool = False):
        from openai import OpenAI

        # A bounded client timeout prevents a single malformed generation from
        # blocking a multi-day checkpointed run indefinitely.
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
        if protocol_mode not in {"strict", "compact"}:
            raise ValueError("protocol_mode must be 'strict' or 'compact'")
        if json_mode not in {"json_object", "json_schema"}:
            raise ValueError("json_mode must be 'json_object' or 'json_schema'")
        self.model, self.agent_type, self.rng = model, agent_type, rng
        self.protocol_mode, self.json_mode = protocol_mode, json_mode
        self.disable_thinking = disable_thinking
        self.bounded_explanations = bounded_explanations

    @staticmethod
    def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {"type": "json_schema", "json_schema": {"name": "iagent_response", "strict": True,
                "schema": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}}}

    def _ask(self, messages: list[dict[str, str]], properties: dict[str, Any], required: list[str],
             max_tokens: int = 512) -> dict[str, Any]:
        last_error: Exception | None = None
        request_messages = messages
        if self.json_mode == "json_object":
            # DeepSeek rejects json_object calls unless a request message
            # explicitly mentions JSON and shows its required fields. This is
            # transport metadata only: the authors' task prompts, fields, and
            # protocol stay intact.
            fields = ", ".join(required)
            request_messages = [{"role": "system", "content":
                                 f"Return one valid JSON object only. Required keys: {fields}. "
                                 "Do not use Markdown or add any other keys."}, *messages]
        for attempt in range(3):
            try:
                response_format: dict[str, Any]
                if self.json_mode == "json_schema":
                    response_format = self._schema(properties, required)
                else:
                    response_format = {"type": "json_object"}
                extra_body: dict[str, Any] | None = None
                if self.json_mode == "json_object":
                    extra_body = {"thinking": {"type": "disabled"}}
                elif self.disable_thinking:
                    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
                request: dict[str, Any] = dict(model=self.model, messages=request_messages, temperature=0,
                                               response_format=response_format, max_tokens=max_tokens)
                if extra_body is not None:
                    request["extra_body"] = extra_body
                response = self.client.chat.completions.create(**request)
                payload = json.loads(response.choices[0].message.content or "")
                missing = [field for field in required if field not in payload]
                if missing:
                    raise ValueError(f"JSON response missing required keys: {missing}")
                return payload
            except Exception as exc:  # provider errors and invalid JSON share the same retry policy
                last_error = exc
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after 3 attempts: {last_error}")

    @staticmethod
    def _static_memory(example: OfficialExample) -> str:
        return "".join(
            f"user historical information, item title:{title},item description:{_tail(description)} ;"
            for title, description in zip(example.titles, example.descriptions)
        )

    @staticmethod
    def _candidate_text(example: OfficialExample, mapping: dict[int, tuple[str, str]]) -> str:
        return "".join(
            f"item id:{item}, corresponding title:{mapping[item][0][:50]}, description:{_clean(mapping[item][1])[:20]} ;"
            for item in example.candidates
        )

    @staticmethod
    def _rank_schema() -> tuple[dict[str, Any], list[str]]:
        return ({"rerank_list": {"type": "array", "items": {"type": "integer"}}}, ["rerank_list"])

    def _response_schema(self) -> tuple[dict[str, Any], list[str]]:
        if self.protocol_mode == "compact":
            return self._rank_schema()
        rerank_list: dict[str, Any] = {"type": "array", "items": {"type": "integer"}}
        explanation_item: dict[str, Any] = {"type": "string"}
        explanation: dict[str, Any] = {"type": "array", "items": explanation_item}
        if self.bounded_explanations:
            # This Qwen/vLLM transport guard is deliberately limited to the
            # non-metric explanation field.  At 10 short strings (100 Unicode
            # characters each), even a worst-case JSON encoding fits below the
            # 2K completion budget; the original prompts and ranking contract
            # are unaffected.
            rerank_list.update({"minItems": 10, "maxItems": 10})
            explanation_item["maxLength"] = 100
            explanation["maxItems"] = 10
        return ({"rerank_list": rerank_list, "explanation": explanation},
                ["rerank_list", "explanation"])

    def _rerank(self, messages: list[dict[str, str]], prompt: str, candidates: list[int]) -> list[int]:
        properties, required = self._response_schema()
        for retry in range(4):
            # The authors' strict response includes free-form explanations for
            # every candidate.  A 1K completion cap can cut a valid JSON
            # string mid-token on local models, even when the ranking itself
            # is ready.  This only raises the transport limit; prompts,
            # schema, candidates, and metrics stay unchanged.
            response = self._ask(messages + [{"role": "assistant", "content": prompt}], properties, required,
                                 max_tokens=2048 if self.protocol_mode == "strict" else 128)
            ranked = [int(item) for item in response["rerank_list"]]
            if len(ranked) == len(candidates) and set(ranked) == set(candidates):
                return ranked
            if self.protocol_mode == "strict":
                prompt = ("Rerank list is out of the order, you should rerank the item from the pure ranking list. "
                          f"The previous list:{ranked}. Therefore, try it again according the following information. "
                          "\n Don’t use numerical numbering for the generated content; you can use bullet points instead. \n "
                          f"{prompt} Please generate the reranked list from Pure Ranking List:{candidates}. "
                          f"The length of the reranked list should be {len(candidates)}.")
            else:
                prompt = ("Your previous rerank list was invalid. Return each and only each ID from the pure ranking "
                          f"list exactly once. Pure Ranking List:{candidates}")
        raise RuntimeError("self-reflection failed to return a permutation of the candidate slate")

    def static(self, example: OfficialExample, mapping: dict[int, tuple[str, str]]) -> list[int]:
        knowledge_prompt = ("Based on the following instruction, assist me in generating relevant knowledge. "
                            "Please specify the types of descriptions that the recommended items should include. "
                            "Do not directly recommend specific items. \n. Don’t use numerical numbering for the "
                            "generated content; you can use bullet points instead. \n "
                            f"Instruction:{example.instruction}")
        messages = [{"role": "assistant", "content": knowledge_prompt}]
        knowledge = self._ask(messages, {"knowledge": {"type": "string"}}, ["knowledge"],
                              max_tokens=1024 if self.protocol_mode == "strict" else 96)["knowledge"]
        prompt = ("Based on the information, give recommendations for the user based on the constraints. .\n "
                  "Don’t use numerical numbering for the generated content; you can use bullet points instead. \n "
                  f"Candidate ranking list:{self._candidate_text(example, mapping)},Knowledge:{knowledge},"
                  f"Static Interest:{self._static_memory(example)}, Pure Ranking List:{example.candidates}")
        return self._rerank(messages, prompt, example.candidates)

    def dynamic(self, example: OfficialExample, mapping: dict[int, tuple[str, str]]) -> list[int]:
        # The authors' released code uses the last 15 interactions and one feedback round.
        titles, descriptions, reviews, ids = (example.titles[-15:], example.descriptions[-15:],
                                              example.reviews[-15:], example.item_ids[-15:])
        if len(titles) < 2:
            raise ValueError("i2Agent requires at least two history interactions")
        negatives = [item for item in mapping if item not in set(ids) | {example.target_id}]
        negative_title, negative_description = mapping[self.rng.choice(negatives)]
        profile_messages: list[dict[str, str]] = []
        first = ("Here is the background of one user. \n. Please recommend one item for her. "
                 f"The first one title:{titles[-2]}, descrition:{_tail(descriptions[-2])}. "
                 f"The second one title:{negative_title}, description:{_tail(negative_description)}. ")
        profile_messages.append({"role": "assistant", "content": first})
        recommendation = self._ask(profile_messages, {"recommend_content": {"type": "string"}},
                                   ["recommend_content"], max_tokens=256)["recommend_content"]
        profile_messages.append({"role": "assistant", "content": f"The recommend content: {recommendation} \n. "})
        second = (f"\n. Great! Actually, this user choose the item with title:{titles[-2]} and review:{_tail(reviews[-2])}. "
                  "Can you generate the profile of this user background? Please make a detailed profile. "
                  "Don’t use numerical numbering for the generated content; you can use bullet points instead.")
        profile = self._ask(profile_messages + [{"role": "assistant", "content": second}],
                            {"profile": {"type": "string"}}, ["profile"], max_tokens=512)["profile"]
        knowledge_messages = [{"role": "assistant", "content":
            "Based on the following instruction, assist me in generating relevant knowledge. Please specify the types "
            "of descriptions that the recommended items should include. Do not directly recommend specific items. \n. "
            "Don’t use numerical numbering for the generated content; you can use bullet points instead. Use no more "
            "than 40 words. \n "
            f"Instruction:{example.instruction}"}]
        knowledge = self._ask(knowledge_messages, {"knowledge": {"type": "string"}}, ["knowledge"],
                              max_tokens=1024 if self.protocol_mode == "strict" else 96)["knowledge"]
        memory = "".join(f"user historical information, item title:{title},item description:{_tail(description)} ;"
                         for title, description in zip(titles, descriptions))
        dynamic_prompt = ("Based on the generated knowledge and the instruction, extract some dynamic interest information "
                          "from the static memory. Moreover, based on the profile and the instruction, extract some dynamic "
                          "profile information. . Don’t use numerical numbering for the generated content; you can use bullet "
                          f"points instead. \n Generated Knowledge:{knowledge} Instruction:{example.instruction} "
                          f"Historical Information:{memory} Profile:{profile}")
        dynamic = self._ask(knowledge_messages + [{"role": "assistant", "content": dynamic_prompt}],
                            {"dynamic_interest": {"type": "string"}, "dynamic_profile": {"type": "string"}},
                            ["dynamic_interest", "dynamic_profile"], max_tokens=512)
        final = ("Based on the information, give recommendations for the user based on the constrains. .\n "
                 "Don’t use numerical numbering for the generated content; you can use bullet points instead. \n "
                 f"Candidate ranking list:{self._candidate_text(example, mapping)},Knowledge:{knowledge},"
                 f"Dynamic Interest:{dynamic['dynamic_interest']},Static Interest:{memory}, Static User Profile:{profile}, "
                 f"Dynamic User Profile:{dynamic['dynamic_profile']}, Pure Ranking List:{example.candidates}")
        return self._rerank(knowledge_messages + [{"role": "assistant", "content": dynamic_prompt}], final, example.candidates)

    def rank(self, example: OfficialExample, mapping: dict[int, tuple[str, str]]) -> list[int]:
        return self.static(example, mapping) if self.agent_type == "static" else self.dynamic(example, mapping)


def completed_rows(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("status") == "ok":
                done.add(int(row["row_index"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return done


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def summarize(path: Path) -> dict[str, float | int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = [row["metrics"] for row in rows if row.get("status") == "ok"]
    return {"attempted": len(rows), "successful": len(metrics), **({key: value * 100 for key, value in average(metrics).items()} if metrics else {})}

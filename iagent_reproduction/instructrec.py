"""Exact adapter for the authors' released InstructRec DataFrame files."""

from __future__ import annotations

import csv
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

from .core import Interaction, Item

DOMAIN_FILES = {"books": "booksAll_recagent.pkl", "movietv": "movietvAll_recagent.pkl",
                "reads": "readsAll_recagent.pkl", "yelp": "yelpAll_recagent.pkl"}


@dataclass(frozen=True)
class InstructRecExample:
    user_id: str
    history: list[Interaction]
    instruction: str
    target_id: str
    candidates: list[Item]
    negative_pool: list[Item]


def _item(item_id: object, title: object, description: object) -> Item:
    return Item(str(item_id), str(title), str(description))


def _mapping(path: Path) -> dict[str, Item]:
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate-text map: {path.name}")
    # Amazon descriptions can exceed csv's conservative 128 KiB default.
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit //= 10
    result: dict[str, Item] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            result[str(row["index"])] = _item(row["index"], row.get("title", ""), row.get("description", ""))
    return result


def load_examples(data_dir: str | Path, domain: str, limit: int | None = None) -> list[InstructRecExample]:
    """Preserve released instruction, platform slate and held-out final item.

    Authors' `main_iagent_mp.py` defines `asin[:-1]` as history, `asin[-1]`
    as answer, and `ranked_lists` as the platform slate. This adapter mirrors
    precisely those choices.
    """
    if domain not in DOMAIN_FILES:
        raise ValueError(f"domain must be one of {sorted(DOMAIN_FILES)}")
    root = Path(data_dir)
    map_path = root / f"combined_{domain}_asin_mapping.csv"
    lookup = _mapping(map_path)
    with (root / DOMAIN_FILES[domain]).open("rb") as stream:
        frame = pickle.load(stream)
    examples: list[InstructRecExample] = []
    for _, row in frame.iloc[:limit].iterrows():
        item_ids, titles, descriptions = list(row["asin"]), list(row["title"]), list(row["description"])
        reviews = list(row["reviewText"])
        if len(item_ids) < 2:
            continue
        user = str(row["reviewerID"])
        history = [Interaction(user, _item(item_id, title, desc), step, str(review))
                   for step, (item_id, title, desc, review) in enumerate(zip(item_ids[:-1], titles[:-1], descriptions[:-1], reviews[:-1]))]
        target = str(item_ids[-1])
        candidate_ids = [str(item_id) for item_id in row["ranked_lists"].tolist()]
        if target not in candidate_ids or any(item_id not in lookup for item_id in candidate_ids):
            continue
        candidates = [lookup[item_id] for item_id in candidate_ids]
        seen = {interaction.item.item_id for interaction in history} | {target}
        negative_pool = [item for item_id, item in lookup.items() if item_id not in seen][:20]
        examples.append(InstructRecExample(user, history, str(row["instruction"]), target, candidates, negative_pool))
    return examples

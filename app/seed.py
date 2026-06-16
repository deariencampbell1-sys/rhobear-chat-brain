from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from app.cache import SemanticCache
from app.embeddings import Embedder


def question_hash(question: str) -> str:
    normalized = question.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class SeedPair:
    question: str
    answer: str
    thought: str | None = None


def parse_seed_line(line: str) -> SeedPair | None:
    line = line.strip()
    if not line:
        return None
    data = json.loads(line)
    return SeedPair(
        question=data["q"],
        answer=data["a"],
        thought=data.get("thought"),
    )


def load_seed_file(path: Path) -> list[SeedPair]:
    if not path.exists():
        return []
    pairs: list[SeedPair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            pair = parse_seed_line(line)
            if pair is not None:
                pairs.append(pair)
    return pairs


def seed_cache(
    cache: SemanticCache,
    embedder: Embedder,
    pairs: list[SeedPair],
) -> dict[str, int]:
    inserted = 0
    skipped = 0
    for pair in pairs:
        q_hash = question_hash(pair.question)
        if cache.has_hash(q_hash):
            skipped += 1
            continue
        embedding = embedder.embed(pair.question)
        if cache.insert_entry(
            q_hash,
            pair.question,
            pair.answer,
            pair.thought,
            embedding,
        ):
            inserted += 1
        else:
            skipped += 1
    return {"inserted": inserted, "skipped": skipped}
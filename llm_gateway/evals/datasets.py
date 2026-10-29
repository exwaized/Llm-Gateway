from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalItem:
    id: str
    prompt: str
    reference: str | None = None
    task: str = "default"  # groups items for per-task reporting
    difficulty: str | None = None  # optional stratification label
    meta: dict = field(default_factory=dict)


def load_jsonl(path: str | Path) -> list[EvalItem]:
    """Load items from a JSONL file; blank lines are skipped, duplicate ids rejected."""
    items: list[EvalItem] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            item = EvalItem(
                id=str(row["id"]),
                prompt=row["prompt"],
                reference=row.get("reference"),
                task=row.get("task", "default"),
                difficulty=row.get("difficulty"),
                meta=row.get("meta", {}),
            )
        except KeyError as exc:
            raise ValueError(f"{path}:{lineno}: missing field {exc}") from exc
        if item.id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {item.id!r}")
        seen.add(item.id)
        items.append(item)
    return items

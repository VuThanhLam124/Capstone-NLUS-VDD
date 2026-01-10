from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Set

DEFAULT_FULL_SCHEMA = ""


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def load_full_schema(path: Optional[Path] = None) -> str:
    """Load full schema text for prompt context."""
    if path is None:
        path = Path(__file__).with_name("full_schema.txt")
    content = _read_text_file(path).strip()
    return content if content else DEFAULT_FULL_SCHEMA


def _filter_examples(examples: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        question = ex.get("question")
        sql = ex.get("sql")
        if not question or not sql:
            continue
        cleaned.append({"question": question, "sql": sql})
    return cleaned


def load_few_shot_examples(
    key: str = "default",
    path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Load few-shot examples for prompts."""
    if path is None:
        path = Path(__file__).with_name("few_shot_examples.json")
    raw = _read_text_file(path).strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(data, list):
        return _filter_examples(data)
    if isinstance(data, dict):
        if key in data:
            return _filter_examples(data[key])
        if "default" in data:
            return _filter_examples(data["default"])
    return []


def load_valid_tables(path: Optional[Path] = None) -> Set[str]:
    """Load valid table names for SQL validation."""
    if path is None:
        path = Path(__file__).with_name("valid_tables.txt")
    raw = _read_text_file(path)
    if not raw:
        return set()

    tables: Set[str] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        tables.add(line.lower())
    return tables

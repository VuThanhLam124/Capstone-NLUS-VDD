from __future__ import annotations

from pathlib import Path
from typing import Optional

DEFAULT_RULES = """=== CRITICAL RULES ===
1. Output ONLY the SQL query, no explanation.
2. Do NOT hallucinate tables or columns.
3. Do NOT add filters unless explicitly requested.
"""


def load_business_rules(path: Optional[Path] = None) -> str:
    """Load business rules text for prompt injection."""
    if path is None:
        path = Path(__file__).with_name("business_rule.txt")

    fallback_path = Path(__file__).with_name("bussiness_rule.txt")
    for candidate in (path, fallback_path):
        try:
            if candidate.exists():
                content = candidate.read_text(encoding="utf-8").strip()
                if content:
                    return content
        except OSError:
            continue

    return DEFAULT_RULES

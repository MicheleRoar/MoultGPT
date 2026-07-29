# llm/config/trait_schema.py
"""
MoultDB trait schema — the 55 character/trait fields the original locally
fine-tuned Mistral model was trained to extract from a paper (see
llm/finetuning/modules/example_generator.py: the "ALL traits" training
example is exactly `"\\n".join(f"{k}: {', '.join(v)}" for k, v in traits.items())`,
where `traits` covers every non-metadata column of
`llm/finetuning/MoultDB character annotations.xlsx`).

A generic remote model has never seen that schema, so asking it a single
narrow question (the old default behaviour) produces far sparser output
than the fine-tuned model used to. Injecting this schema into the prompt —
field names plus, where the annotators used a controlled vocabulary, the
allowed values — lets a zero-shot remote model approximate the same rich,
multi-field extraction without any fine-tuning.

Data source: llm/data/moultdb_trait_schema.json (generated once from the
Excel — see the extraction snippet in this module's docstring history /
git log if it ever needs regenerating after the Excel changes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "data" / "moultdb_trait_schema.json"

with open(_SCHEMA_PATH, "r", encoding="utf-8") as _f:
    _SCHEMA = json.load(_f)

TRAIT_COLUMNS: List[str] = _SCHEMA["trait_columns"]
ENUM_HINTS: Dict[str, List[str]] = _SCHEMA["enums"]
EXCLUDED_METADATA_COLUMNS: List[str] = _SCHEMA["excluded_metadata_columns"]


def build_trait_schema_block(max_enum_values: int = 12) -> str:
    """
    Render the trait schema as a prompt-ready text block: one line per
    field, with a short controlled-vocabulary hint where the annotators
    used one. Long enum lists are truncated (full list is still valid
    input; this just keeps the prompt from ballooning).
    """
    lines = []
    for col in TRAIT_COLUMNS:
        values = ENUM_HINTS.get(col)
        if values:
            shown = values[:max_enum_values]
            more = "…" if len(values) > max_enum_values else ""
            hint = f" (typical values: {', '.join(shown)}{more})"
        else:
            hint = ""
        lines.append(f"- {col}{hint}")
    return "\n".join(lines)

# llm/pipeline/prompting.py
"""
Shared prompt construction for MoultGPT's remote LLM calls.

Used by both llm/backend/app.py (the live API) and llm/eval/compare_models.py
(the benchmarking script), same reasoning as pipeline/domain_pipeline.py:
keeping this in one place means the live API and the comparison script can
never quietly drift apart on what they actually ask the model to do.

Two modes:
    full_traits (default) — mirrors the "Extract all moulting-related
        traits from the following article" examples the original local
        LoRA fine-tune was trained on (see
        llm/finetuning/modules/example_generator.py). The MoultDB field
        schema (llm/config/trait_schema.py, 55 fields) is injected directly
        into the prompt so a generic remote model — which has never seen
        that schema — can approximate the same rich, multi-field output the
        fine-tuned model produced natively.
    single_trait — the original narrow behaviour: answer exactly the one
        question in `prompt`, ignore everything else. Useful for a quick
        targeted follow-up, but on its own is NOT equivalent to what the
        fine-tuned model did — that's why it is no longer the default.
"""

from __future__ import annotations

from config.trait_schema import build_trait_schema_block

MODE_SINGLE_TRAIT = "single_trait"
MODE_FULL_TRAITS = "full_traits"
DEFAULT_MODE = MODE_FULL_TRAITS
VALID_MODES = (MODE_SINGLE_TRAIT, MODE_FULL_TRAITS)

_TRAIT_SCHEMA_BLOCK = build_trait_schema_block()


def build_single_trait_system_prompt() -> str:
    return (
        "You are a scientific assistant specialized in arthropod moulting.\n"
        "You receive:\n"
        "  1) A set of sentences extracted from a scientific paper, already\n"
        "     filtered to focus on moulting-related content.\n"
        "  2) A user query describing which biological trait to extract.\n\n"
        "Your task:\n"
        "- Extract ONLY the trait requested in the user query.\n"
        "- Ignore all other information.\n"
        "- If the trait is not mentioned or cannot be inferred, say so clearly.\n"
        "- Return the answer as CLEAN YAML only, with no extra prose,\n"
        "  no explanations, and no surrounding text.\n"
    )


def build_full_traits_system_prompt() -> str:
    return (
        "You are a scientific assistant specialized in extracting arthropod "
        "moulting traits for the MoultDB database.\n"
        "You receive sentences extracted from a scientific paper, already "
        "filtered to focus on moulting-related content.\n\n"
        "Extract a value for every one of the following MoultDB trait fields "
        "that the text gives clear evidence for. Use the exact field names "
        "below as keys. Skip a field entirely if the text does not support "
        "it — never guess, infer beyond the text, or invent a value.\n\n"
        f"Fields:\n{_TRAIT_SCHEMA_BLOCK}\n\n"
        "Output format: exactly one line per field found, formatted as\n"
        "Field Name: value1, value2\n"
        "No YAML, no bullet points, no headers, no extra prose, no "
        "explanations — only the field lines, one per field with evidence."
    )


def build_system_prompt(mode: str) -> str:
    return build_single_trait_system_prompt() if mode == MODE_SINGLE_TRAIT else build_full_traits_system_prompt()


def build_user_content(summary: str, user_prompt: str, mode: str) -> str:
    if mode == MODE_SINGLE_TRAIT:
        return f"Context:\n{summary.strip()}\n\nUser query:\n{user_prompt.strip()}"
    return (
        f"Context (moulting-related sentences extracted from the paper):\n{summary.strip()}\n\n"
        f"Optional user focus (does not replace the field list in the system "
        f"prompt, only prioritizes among it): {user_prompt.strip()}"
    )

# llm/config/models.py
"""
Central registry of candidate LLM models for MoultGPT.

This is the single source of truth for "which remote models can MoultGPT
call". It is used by:

  - llm/backend/app.py          -> GET /models (populates the frontend dropdown)
                                    and /query (validates the requested model_id)
  - llm/backend/providers.py    -> maps provider name -> API adapter
  - llm/eval/compare_models.py  -> runs the same paper/query pairs across
                                    every model in this list for benchmarking

Adding or removing a model to compare = editing this list. No other file
should need to change.

IMPORTANT — pinning versions:
    For anything that will end up in a publication, replace "latest"-style
    aliases with the exact dated model id from the provider's docs at the
    time you run the comparison (e.g. "mistral-small-2506" instead of
    "mistral-small-latest"). Aliases can silently point to a different
    checkpoint over time, which breaks reproducibility.

IMPORTANT — OpenRouter's free catalog churns fast:
    On 2026-07-23 a live request to https://openrouter.ai/api/v1/models
    showed that 3 of the 4 OpenRouter entries originally in this registry
    (Llama 3.3 70B, Qwen3 Next 80B, Gemma 4 31B, GPT-OSS 20B) no longer
    exist AT ALL under those ids (not just their ":free" variant — gone
    entirely), confirmed first-hand by a 404 from the API. Free-tier model
    ids are not a stable interface. Before relying on an OpenRouter entry
    here (especially for a publication run), re-verify it's still live:
        curl -s https://openrouter.ai/api/v1/models | grep -o '"id":"[^"]*:free"'
    or check https://openrouter.ai/models?max_price=0 in a browser.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Fields
# ------
# id           : exact model identifier sent to the provider's API.
# provider     : one of "openrouter", "mistral", "gemini"
#                (must have a matching adapter in llm/backend/providers.py)
# label        : human-readable name shown in the UI dropdown.
# family       : lab/organization, used to group models in comparisons.
# free         : whether this id is expected to be free-tier as of the date
#                below. Free-tier catalogs change; verify before relying on it.
# verified_on  : date this entry was last checked against the provider's
#                live model list / pricing page.
# notes        : short justification for including this model.

MODEL_REGISTRY: List[Dict[str, Any]] = [
    {
        "id": "mistral-small-latest",
        "provider": "mistral",
        "label": "Mistral Small",
        "family": "Mistral",
        "free": True,
        "verified_on": "2026-07-23",
        "notes": (
            "Continuity with the original MoultGPT design (Mistral-7B-Instruct "
            "was the local baseline model). Replace with the exact dated id "
            "before running a comparison meant for publication."
        ),
    },
    {
        "id": "mistral-medium-latest",
        "provider": "mistral",
        "label": "Mistral Medium",
        "family": "Mistral",
        "free": True,
        "verified_on": "2026-07-27",
        "notes": (
            "Added specifically for llm/eval/trait_extraction/'s model comparison. "
            "Mistral's free \"Experiment\" tier on La Plateforme gives rate-limited "
            "access to the full model catalog (Small/Medium/Large/Codestral), not "
            "just Small (see https://mistral.ai/news/september-24-release/ and "
            "the account's Admin Console -> Limits for the current rate cap) — "
            "this was previously left out of the registry by omission, not because "
            "it isn't accessible. Not used as the /query default; comparison-only."
        ),
    },
    {
        "id": "mistral-large-latest",
        "provider": "mistral",
        "label": "Mistral Large",
        "family": "Mistral",
        "free": True,
        "verified_on": "2026-07-27",
        "notes": (
            "Same free \"Experiment\" tier access as mistral-medium-latest above. "
            "Included as the top of the Mistral size tier for the trait-extraction "
            "model comparison, and used as the default LLM-as-judge model in "
            "llm/eval/trait_extraction/run_model_comparison.py (largest available "
            "model reviewing the others' disagreements, on the assumption that a "
            "larger model is more reliable as a judge than as the extractor being "
            "judged is a reasonable but untested assumption, stated as a limitation "
            "in that module's README)."
        ),
    },
    {
        "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "provider": "openrouter",
        "label": "Nemotron 3 Ultra 550B",
        "family": "NVIDIA",
        "free": True,
        "verified_on": "2026-07-23",
        "notes": (
            "Confirmed live via https://openrouter.ai/api/v1/models on 2026-07-23. "
            "Large general-purpose MoE, 1M context — replaces the now-removed Llama 3.3 70B slot."
        ),
    },
    {
        "id": "openrouter/free",
        "provider": "openrouter",
        "label": "OpenRouter (auto-selected free model)",
        "family": "OpenRouter auto-router",
        "free": True,
        "verified_on": "2026-07-23",
        "notes": (
            "Pinned free-tier ids on OpenRouter kept 404ing within minutes of being "
            "verified live (confirmed twice on 2026-07-23 — tencent/hy3:free among "
            "others). This special id asks OpenRouter to auto-pick whichever free "
            "model is currently available, so it doesn't break the same way. The "
            "actual model that answered is reported back per-request as "
            "`resolved_model_id` in the API response / eval CSV — check that column, "
            "since it varies. Less reproducible than a pinned id: for a publication "
            "comparison, prefer the other (pinned) entries and treat this one as a "
            "best-effort fallback."
        ),
    },
    {
        "id": "cohere/north-mini-code:free",
        "provider": "openrouter",
        "label": "Cohere North Mini Code",
        "family": "Cohere",
        "free": True,
        "verified_on": "2026-07-23",
        "notes": (
            "Confirmed live via https://openrouter.ai/api/v1/models on 2026-07-23. "
            "Name suggests a code focus — worth a quick quality check on trait-extraction "
            "output before relying on it; swap out if it underperforms on prose."
        ),
    },
    {
        "id": "gemini-2.5-flash",
        "provider": "gemini",
        "label": "Gemini 2.5 Flash",
        "family": "Google (closed)",
        "free": True,
        "verified_on": "2026-07-23",
        "notes": (
            "Proprietary model for an open-vs-closed comparison point. "
            "Generous free tier (check current daily request limit before a full run)."
        ),
    },
]

DEFAULT_MODEL_ID = "mistral-small-latest"


def get_model(model_id: str) -> Optional[Dict[str, Any]]:
    """Look up a single registry entry by id. Returns None if unknown."""
    for entry in MODEL_REGISTRY:
        if entry["id"] == model_id:
            return entry
    return None


def list_models() -> List[Dict[str, Any]]:
    """Return the full registry, e.g. for a GET /models endpoint."""
    return MODEL_REGISTRY


def is_known_model(model_id: str) -> bool:
    return get_model(model_id) is not None

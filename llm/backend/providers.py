# llm/backend/providers.py
"""
Provider-agnostic LLM calling layer for MoultGPT.

Replaces the old local Mistral-7B (transformers + torch + optional LoRA)
inference path with HTTP calls to remote free-tier APIs. This means:

  - no GPU / no local model weights required to run the LLM backend,
  - the same paper summary + user query can be sent to several different
    models (see llm/config/models.py) for side-by-side comparison,
  - swapping or adding a model is a config change, not a code change.

Usage
-----
    from backend.providers import generate

    result = generate(
        system_prompt="You are a scientific assistant...",
        user_content="Context:\\n...\\n\\nUser query:\\n...",
        model_id="mistral-small-latest",
    )
    result["text"]               # the model's reply (string)
    result["provider"]           # "mistral"
    result["model_id"]           # the id you asked for, e.g. "mistral-small-latest"
    result["resolved_model_id"]  # what actually answered (matters for auto-routers
                                  # like "openrouter/free"; equals model_id otherwise)
    result["latency_sec"]

All adapters are called with temperature=0 (or the closest equivalent) to
keep outputs as deterministic/reproducible as possible across runs, which
matters when comparing models for a publication.

Required environment variables (only the ones you actually use):
    OPENROUTER_API_KEY
    MISTRAL_API_KEY
    GEMINI_API_KEY
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests

from config.models import get_model

DEFAULT_TIMEOUT_SEC = 60
DEFAULT_MAX_NEW_TOKENS = 512


class ProviderError(RuntimeError):
    """Raised when a remote provider call fails or returns something unusable."""


# ---------------------------------------------------------------------------
# Adapters — one function per provider.
# Each returns the raw text produced by the model (str).
# ---------------------------------------------------------------------------


def _call_openrouter(
    system_prompt: str,
    user_content: str,
    model_id: str,
    max_new_tokens: int,
) -> tuple[str, Optional[str]]:
    """
    Returns (text, resolved_model_id). resolved_model_id matters most for
    "openrouter/free" (the auto-router that picks whichever free model is
    currently live) — OpenRouter's response echoes back which underlying
    model actually answered, in the standard OpenAI-compatible "model"
    field. For a pinned model_id this will normally just repeat model_id.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ProviderError("OPENROUTER_API_KEY is not set.")

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "max_tokens": max_new_tokens,
        },
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    if resp.status_code != 200:
        raise ProviderError(f"OpenRouter error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ProviderError(f"Unexpected OpenRouter response shape: {data}") from e
    return text, data.get("model")


def _call_mistral(
    system_prompt: str,
    user_content: str,
    model_id: str,
    max_new_tokens: int,
) -> tuple[str, Optional[str]]:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ProviderError("MISTRAL_API_KEY is not set.")

    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "max_tokens": max_new_tokens,
        },
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    if resp.status_code != 200:
        raise ProviderError(f"Mistral error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ProviderError(f"Unexpected Mistral response shape: {data}") from e
    return text, data.get("model")


def _call_gemini(
    system_prompt: str,
    user_content: str,
    model_id: str,
    max_new_tokens: int,
) -> tuple[str, Optional[str]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ProviderError("GEMINI_API_KEY is not set.")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={api_key}"
    )
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_content}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": max_new_tokens,
            },
        },
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    if resp.status_code != 200:
        raise ProviderError(f"Gemini error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError) as e:
        raise ProviderError(f"Unexpected Gemini response shape: {data}") from e
    return text, data.get("modelVersion")


_ADAPTERS = {
    "openrouter": _call_openrouter,
    "mistral": _call_mistral,
    "gemini": _call_gemini,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate(
    system_prompt: str,
    user_content: str,
    model_id: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """
    Call the given model (looked up in llm/config/models.py) with a
    system prompt + user content, and return a structured result.

    Raises:
        ProviderError: unknown model_id, missing API key, or a failed/
                        malformed response from the provider.
    """
    entry = get_model(model_id)
    if entry is None:
        raise ProviderError(
            f"Unknown model_id '{model_id}'. "
            "Add it to llm/config/models.py first."
        )

    provider = entry["provider"]
    adapter = _ADAPTERS.get(provider)
    if adapter is None:
        raise ProviderError(f"No adapter registered for provider '{provider}'.")

    t0 = time.time()
    text, resolved_model_id = adapter(system_prompt, user_content, model_id, max_new_tokens)
    latency_sec = time.time() - t0

    return {
        "text": text.strip() if text else "",
        "provider": provider,
        "model_id": model_id,
        # For a pinned model this normally just repeats model_id. For an
        # auto-router entry (e.g. "openrouter/free") this is the actual
        # underlying model that answered — record it, otherwise a row in a
        # comparison table credited to "openrouter/free" is meaningless.
        "resolved_model_id": resolved_model_id or model_id,
        "model_label": entry.get("label"),
        "latency_sec": latency_sec,
    }

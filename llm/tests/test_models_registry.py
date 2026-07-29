# llm/tests/test_models_registry.py
"""
Unit tests for config/models.py — the single source of truth for which
remote models the backend and eval/compare_models.py are allowed to call.

These tests check structural invariants (every entry has the fields the
rest of the codebase relies on, ids are unique, the default model is
actually in the registry) rather than which specific models are listed —
the registry's contents churn on purpose as providers add/remove free-tier
models (see that file's header comment), so pinning exact model ids here
would make this test suite fight the registry instead of protecting it.
"""

from config.models import DEFAULT_MODEL_ID, MODEL_REGISTRY, get_model, is_known_model, list_models

REQUIRED_FIELDS = {"id", "provider", "label", "family", "free", "verified_on", "notes"}
KNOWN_PROVIDERS = {"openrouter", "mistral", "gemini"}


def test_registry_is_non_empty():
    assert len(MODEL_REGISTRY) > 0


def test_every_entry_has_required_fields():
    for entry in MODEL_REGISTRY:
        missing = REQUIRED_FIELDS - entry.keys()
        assert not missing, f"entry {entry.get('id')} missing fields: {missing}"


def test_every_entry_has_a_known_provider():
    for entry in MODEL_REGISTRY:
        assert entry["provider"] in KNOWN_PROVIDERS, entry["id"]


def test_model_ids_are_unique():
    ids = [entry["id"] for entry in MODEL_REGISTRY]
    assert len(ids) == len(set(ids))


def test_default_model_id_is_actually_in_the_registry():
    assert is_known_model(DEFAULT_MODEL_ID)


def test_get_model_returns_none_for_unknown_id():
    assert get_model("not-a-real-model-id") is None


def test_get_model_returns_the_matching_entry():
    first = MODEL_REGISTRY[0]
    assert get_model(first["id"]) == first


def test_list_models_returns_the_full_registry():
    assert list_models() == MODEL_REGISTRY

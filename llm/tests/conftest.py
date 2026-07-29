# llm/tests/conftest.py
"""
Shared pytest setup for the llm/ test suite.

These tests exercise the pure-Python decision/prompt-building logic in
llm/ (domain gating, prompt construction, the model registry) without
requiring GPU access, network calls, or the OWL ontology / taxonomy data
files — they run anywhere with `pip install pytest` and nothing else,
which is the point: fast, dependency-free regression coverage for the
logic that's easiest to silently break while iterating on the pipeline.
"""

import sys
from pathlib import Path

LLM_ROOT = Path(__file__).resolve().parents[1]
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

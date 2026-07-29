#!/usr/bin/env python3
# llm/finetuning/feedback_to_preferences.py
"""
Turns `llm/backend/feedback/feedback.jsonl` — the 👍/👎 + comment data the
unified demo's feedback UI writes via `POST /feedback` — into preference
pairs `{"prompt": ..., "chosen": ..., "rejected": ...}` for `train_dpo.py`.

This is the "close the loop" piece: the feedback UI wasn't added as an
isolated nice-to-have, it's the data-collection half of a preference-
optimization pipeline. Whether that pipeline has ENOUGH real data yet is a
separate question this script answers honestly (see the summary it prints
and `--min_pairs`) rather than pretending — DPO needs pairs where the same
prompt got both a 👍 and a 👎 response, and a demo that just started
collecting feedback will usually not have many of those yet.

How pairs are built
--------------------
1. Group feedback entries by (normalized) prompt text.
2. Within a group, entries with rating > 0 are "chosen" candidates and
   rating < 0 are "rejected" candidates (rating == 0 / missing is dropped
   — DPO needs a signed preference, not a neutral one).
3. Every (chosen, rejected) combination within a group becomes one pair,
   skipping identical chosen==rejected text, capped at
   `--max_pairs_per_prompt` so one prompt with a long feedback history
   can't dominate the dataset.

Usage
-----
    python feedback_to_preferences.py \\
        --feedback ../backend/feedback/feedback.jsonl \\
        --out output/dpo_pairs.jsonl \\
        --min_pairs 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def normalize_prompt(prompt: str) -> str:
    """Grouping key only — collapse whitespace/case so trivial retyping
    of the same question still groups together. The pair's `prompt` field
    in the output keeps the original (first-seen) text, not this."""
    return re.sub(r"\s+", " ", prompt.strip().lower())


def load_feedback(path: Path) -> List[Dict[str, Any]]:
    """
    Reads one JSON object per line, skipping malformed lines with a
    warning rather than failing the whole load — feedback.jsonl is
    real, hand-generated-over-time log data (appended to by
    backend/app.py's /feedback endpoint), not a controlled fixture, so a
    single bad line shouldn't take down the rest.
    """
    entries: List[Dict[str, Any]] = []
    if not path.exists():
        return entries
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{lineno}: skipping malformed JSON line ({e})")
    return entries


def build_preference_pairs(
    entries: List[Dict[str, Any]],
    max_pairs_per_prompt: int = 5,
) -> List[Dict[str, str]]:
    """
    Core logic, kept filesystem-free and importable so it's directly unit
    -testable (see llm/tests/test_feedback_to_preferences.py) without
    needing a real feedback.jsonl on disk.
    """
    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"prompt": None, "chosen": [], "rejected": []})

    for entry in entries:
        prompt = (entry.get("query") or "").strip()
        response = (entry.get("response") or "").strip()
        rating = entry.get("rating")

        if not prompt or not response or rating in (None, 0):
            continue
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            continue

        key = normalize_prompt(prompt)
        group = groups[key]
        if group["prompt"] is None:
            group["prompt"] = prompt  # keep first-seen original casing/text
        if rating > 0:
            group["chosen"].append(response)
        elif rating < 0:
            group["rejected"].append(response)

    pairs: List[Dict[str, str]] = []
    for group in groups.values():
        if not group["chosen"] or not group["rejected"]:
            continue
        seen_in_group = 0
        for chosen in group["chosen"]:
            for rejected in group["rejected"]:
                if chosen == rejected:
                    continue
                if seen_in_group >= max_pairs_per_prompt:
                    break
                pairs.append({"prompt": group["prompt"], "chosen": chosen, "rejected": rejected})
                seen_in_group += 1
            if seen_in_group >= max_pairs_per_prompt:
                break

    return pairs


def summarize(entries: List[Dict[str, Any]], pairs: List[Dict[str, str]]) -> str:
    n_rated = sum(1 for e in entries if e.get("rating") not in (None, 0))
    prompts = {normalize_prompt(e.get("query", "")) for e in entries if e.get("query")}
    one_sided = 0
    groups: Dict[str, Dict[str, int]] = defaultdict(lambda: {"pos": 0, "neg": 0})
    for e in entries:
        if not e.get("query") or e.get("rating") in (None, 0):
            continue
        key = normalize_prompt(e["query"])
        if e["rating"] and float(e["rating"]) > 0:
            groups[key]["pos"] += 1
        elif e["rating"] and float(e["rating"]) < 0:
            groups[key]["neg"] += 1
    for g in groups.values():
        if (g["pos"] > 0) != (g["neg"] > 0):  # exactly one side present
            one_sided += 1

    return (
        f"Read {len(entries)} feedback entries ({n_rated} rated) across {len(prompts)} distinct prompts.\n"
        f"{one_sided} prompt(s) have feedback on only one side (all 👍 or all 👎) — usable as future SFT "
        f"signal, but not as a DPO pair without an opposite-rated response for the same prompt.\n"
        f"Built {len(pairs)} preference pair(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feedback", type=str, default="../backend/feedback/feedback.jsonl")
    parser.add_argument("--out", type=str, default="output/dpo_pairs.jsonl")
    parser.add_argument("--max_pairs_per_prompt", type=int, default=5)
    parser.add_argument("--min_pairs", type=int, default=0,
                         help="Exit with a non-zero status (but still write whatever was found) if fewer "
                              "pairs than this were built. 0 (default) just warns — train_dpo.py enforces "
                              "its own floor before actually spending GPU time on too little data.")
    args = parser.parse_args()

    feedback_path = Path(args.feedback)
    entries = load_feedback(feedback_path)
    if not entries:
        print(f"[WARN] No feedback entries found at {feedback_path} — nothing to convert yet. "
              f"(The unified demo's 👍/👎 UI writes here as it's used.)")

    pairs = build_preference_pairs(entries, max_pairs_per_prompt=args.max_pairs_per_prompt)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(summarize(entries, pairs))
    print(f"Wrote {len(pairs)} pair(s) to {out_path}")

    if len(pairs) < args.min_pairs:
        print(f"[WARN] Fewer than --min_pairs={args.min_pairs} pairs available — "
              f"collect more feedback (via the unified demo) before running train_dpo.py on this.")
        sys.exit(1)


if __name__ == "__main__":
    main()

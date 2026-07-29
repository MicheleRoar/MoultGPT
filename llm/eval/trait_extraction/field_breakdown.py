#!/usr/bin/env python3
# llm/eval/trait_extraction/field_breakdown.py
"""
Per-field (per-trait) breakdown of the main-condition trait-extraction
results, analogous to the "Quality by features" chart from the earlier
LoRA pilot (correct/wrong/ambiguous/unknown per trait across test papers).

Unlike that pilot, this reads directly from an existing results_scored.csv
produced by run_model_comparison.py (no new API calls) -- every count here
traces to the real 500-item run. Single-field and combo items are both
included: combo items already contribute one row per sub-field in
results_scored.csv, so grouping by `field` merges both naturally.

Only single+combo items (real gold value) are counted here -- negative
(should-abstain) items are a separate hallucination-rate question, not a
per-field "quality" question, and are reported elsewhere (see report.md's
hallucination-check tables).

Usage
-----
    cd llm
    python eval/trait_extraction/field_breakdown.py \
        --scored_csv /path/to/results_scored.csv \
        --models mistral-small-latest mistral-medium-latest \
        --out eval/trait_extraction/results/field_breakdown.md
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_csv", type=str, required=True)
    ap.add_argument("--models", type=str, nargs="+",
                     default=["mistral-small-latest", "mistral-medium-latest"])
    ap.add_argument("--min_n_flag", type=int, default=3,
                     help="Fields with fewer than this many items get an explicit small-sample flag.")
    ap.add_argument("--out", type=str, default=str(EVAL_ROOT / "results" / "field_breakdown.md"))
    args = ap.parse_args()

    rows = []
    with open(args.scored_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("condition") != "main":
                continue
            if r.get("item_type") not in ("single", "combo"):
                continue
            if r.get("model_id") not in args.models:
                continue
            rows.append(r)

    if not rows:
        raise SystemExit(f"No main-condition single/combo rows found for models {args.models} in {args.scored_csv}. "
                          f"Check --models matches the model_id values actually in the CSV.")

    # per_model[model][field] -> counts
    per_model: dict[str, dict[str, dict[str, int]]] = {
        m: defaultdict(lambda: {"n": 0, "correct": 0, "ambiguous": 0, "incorrect": 0, "abstained": 0, "other": 0})
        for m in args.models
    }

    for r in rows:
        model = r["model_id"]
        field = r["field"]
        c = per_model[model][field]
        c["n"] += 1
        label = r["label"]
        if label == "correct":
            c["correct"] += 1
        elif label == "abstained":
            c["abstained"] += 1
        elif label == "disagreement":
            verdict = r.get("judge_verdict", "")
            if verdict == "incorrect":
                c["incorrect"] += 1
            elif verdict == "ambiguous":
                c["ambiguous"] += 1
            else:
                c["other"] += 1  # unjudged/unclassified disagreement
        else:
            c["other"] += 1  # gated_out / error / parse_failed etc, shouldn't normally appear in "main"+single/combo

    lines = []
    lines.append("# Per-field trait-extraction breakdown (real 500-item run)\n")
    lines.append(
        f"Source: `{Path(args.scored_csv).name}`, main condition, single+combo items only (negative/hallucination "
        f"items are a separate question, see report.md). Models: {', '.join(args.models)}. Combo items already "
        f"contribute one row per sub-field in the source CSV, so per-field counts here merge single- and "
        f"combo-derived rows for the same field naturally.\n"
    )
    lines.append(
        f"**Read the `n` column before the percentages.** Many fields have very few sampled items (some as low "
        f"as 1) -- a field flagged with n < {args.min_n_flag} is not a statistically meaningful per-field estimate, "
        f"just a single (or handful of) real data point(s). Fields are sorted by correct rate (descending) within "
        f"each model, matching the earlier pilot's presentation style.\n"
    )

    overall_summary = {}
    for model in args.models:
        fields = per_model[model]
        lines.append(f"\n## {model}\n")
        lines.append("| Field | n | correct | ambiguous | incorrect | abstained | other | small sample? |")
        lines.append("|---|---|---|---|---|---|---|---|")

        ordered = sorted(fields.items(), key=lambda kv: (-(kv[1]["correct"] / kv[1]["n"]) if kv[1]["n"] else 0,
                                                          -kv[1]["n"]))
        n_total = correct_total = ambiguous_total = incorrect_total = abstained_total = other_total = 0
        for field, c in ordered:
            flag = "yes" if c["n"] < args.min_n_flag else ""
            lines.append(f"| {field} | {c['n']} | {c['correct']} | {c['ambiguous']} | {c['incorrect']} | "
                         f"{c['abstained']} | {c['other']} | {flag} |")
            n_total += c["n"]; correct_total += c["correct"]; ambiguous_total += c["ambiguous"]
            incorrect_total += c["incorrect"]; abstained_total += c["abstained"]; other_total += c["other"]

        n_fields = len(fields)
        n_fields_maj_correct = sum(1 for _, c in fields.items() if c["n"] > 0 and c["correct"] / c["n"] >= 0.5)
        n_fields_zero_correct = sum(1 for _, c in fields.items() if c["correct"] == 0)
        overall_summary[model] = {
            "n_fields": n_fields, "n_total": n_total, "correct_total": correct_total,
            "ambiguous_total": ambiguous_total, "incorrect_total": incorrect_total,
            "abstained_total": abstained_total, "other_total": other_total,
            "n_fields_maj_correct": n_fields_maj_correct, "n_fields_zero_correct": n_fields_zero_correct,
        }

    lines.append("\n## Overall (all fields combined, per model)\n")
    lines.append("| Model | fields | items | correct | ambiguous | incorrect | abstained | "
                 "fields >=50% correct | fields with 0 correct |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for model in args.models:
        s = overall_summary[model]
        pct_correct = f"{100.0 * s['correct_total'] / s['n_total']:.1f}%" if s["n_total"] else "n/a"
        lines.append(f"| {model} | {s['n_fields']} | {s['n_total']} | {s['correct_total']} ({pct_correct}) | "
                     f"{s['ambiguous_total']} | {s['incorrect_total']} | {s['abstained_total']} | "
                     f"{s['n_fields_maj_correct']}/{s['n_fields']} | {s['n_fields_zero_correct']}/{s['n_fields']} |")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] Wrote {out_path}")
    for model in args.models:
        s = overall_summary[model]
        print(f"  {model}: {s['n_total']} items across {s['n_fields']} fields, "
              f"{s['correct_total']} correct ({100.0*s['correct_total']/s['n_total']:.1f}%), "
              f"{s['n_fields_maj_correct']}/{s['n_fields']} fields >=50% correct, "
              f"{s['n_fields_zero_correct']}/{s['n_fields']} fields with 0 correct")


if __name__ == "__main__":
    main()

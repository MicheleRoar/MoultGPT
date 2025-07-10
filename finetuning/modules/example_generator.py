import json
import random

def generate_examples(summary: str, traits: dict, metadata: dict, trait_columns: list,
                      generate_trait_question_templates,
                      generate_combined_templates,
                      generate_negative_templates,
                      negative_outputs) -> list[dict]:

    examples = []
    total_positive, total_negative = 0, 0

    # General full-trait query
    examples.append({
        "instruction": random.choice(metadata["positive_templates"]),
        "input": summary,
        "output": "\n".join([f"{k}: {', '.join(v)}" for k, v in traits.items()]),
        "metadata": {**metadata, "column": "ALL", "category": "general"}
    })

    # Combined traits
    trait_keys = list(traits.keys())
    for _ in range(min(5, len(trait_keys) // 2)):
        subset = random.sample(trait_keys, k=min(3, len(trait_keys)))
        instr = random.choice(generate_combined_templates(subset))
        out = "\n".join([f"{col}: {', '.join(traits[col])}" for col in subset])
        examples.append({
            "instruction": instr,
            "input": summary,
            "output": out,
            "metadata": {**metadata, "column": subset, "category": "combined"}
        })

    # One trait per column
    for col, vals in traits.items():
        for _ in range(2):
            instr = random.choice(generate_trait_question_templates(col))
            examples.append({
                "instruction": instr,
                "input": summary,
                "output": f"{col}: {', '.join(vals)}",
                "metadata": {**metadata, "column": col, "category": "trait"}
            })

    # Negatives
    all_traits = set(trait_columns)
    present_traits = set(traits.keys())
    absent_traits = list(all_traits - present_traits)
    for trait in random.sample(absent_traits, min(3, len(absent_traits))):
        instr = random.choice(generate_negative_templates(trait))
        out = random.choice(negative_outputs)
        examples.append({
            "instruction": instr,
            "input": summary,
            "output": out,
            "metadata": {**metadata, "column": trait, "category": "negative"}
        })

    return examples
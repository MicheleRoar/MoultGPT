from pathlib import Path
import pandas as pd
import json

from data_loader import load_excel_annotations, get_trait_columns
from summarizer import pdf_to_summary
from template_generator import (
    generate_trait_question_templates,
    generate_combined_templates,
    generate_negative_templates,
    positive_templates,
    negative_outputs
)
from example_generator import generate_examples

EXCEL_PATH = "../file/MoultDB character annotations.xlsx"
PAPERS_DIR = Path("../papers")
OUT_JSONL = Path("../output/finetune_full.jsonl")
SUMMARY_LOG = Path("../output/summary_log.csv")

df_data = load_excel_annotations(EXCEL_PATH)
trait_columns = get_trait_columns(df_data)

jsonl_records = []
log_records = []

for paper_id in sorted(df_data["Paper ID"].unique()):
    paper_rows = df_data[df_data["Paper ID"] == paper_id]
    paper_file = PAPERS_DIR / f"{int(paper_id)}.pdf"
    if not paper_file.exists():
        continue

    try:
        summary = pdf_to_summary(paper_file)
        if not summary.strip():
            continue

        traits = {}
        for col in trait_columns:
            vals = paper_rows[col].dropna().unique()
            vals = [str(v).strip() for v in vals if v and str(v).strip().lower() not in ["?", "nan"]]
            if vals:
                traits[col] = vals

        if not traits:
            continue

        metadata = {
            "paper_id": int(paper_id),
            "taxon": paper_rows.iloc[0]["Taxon"],
            "order": paper_rows.iloc[0]["Order"],
            "accession": paper_rows.iloc[0]["Published reference: accession"],
            "source": "merged",
            "positive_templates": positive_templates
        }

        examples = generate_examples(
            summary, traits, metadata, trait_columns,
            generate_trait_question_templates,
            generate_combined_templates,
            generate_negative_templates,
            negative_outputs
        )
        jsonl_records.extend(examples)

        log_records.append({
            "paper_id": paper_id,
            "n_rows": len(paper_rows),
            "n_traits": len(traits),
            "total_examples": len(examples)
        })

    except Exception as e:
        print(f"[ERROR] Paper {paper_id}: {e}")

with open(OUT_JSONL, "w", encoding="utf-8") as fout:
    for r in jsonl_records:
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")

pd.DataFrame(log_records).to_csv(SUMMARY_LOG, index=False)
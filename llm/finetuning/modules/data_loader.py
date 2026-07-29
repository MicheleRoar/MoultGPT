import pandas as pd

def load_excel_annotations(excel_path: str, sheet_name: str = "data") -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df.dropna(subset=["Paper ID"])
    df["Paper ID"] = df["Paper ID"].astype(int)
    return df

def get_trait_columns(df: pd.DataFrame) -> list:
    excluded = [
        "Paper ID", "Taxon", "Order", "Location: name",
        "Published reference: citation (APA style)",
        "Published reference: accession"
    ]
    return [col for col in df.columns if col not in excluded]
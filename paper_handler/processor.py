import os
import sys

# Add relative import for paper_handler/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from downloader import download_pdf_from_doi
from parser import parse_pdf_with_grobid, tei_to_text


def input_to_text(doi: str = None, pdf_path: str = None, email: str = "your@email.com") -> str:
    """
    Main processor function that handles input from either a DOI or a local PDF path.

    Returns extracted plain text from the article, or empty string if failed.
    """
    if doi:
        print(f"[DEBUG] Processing DOI: {doi}")
        pdf_path = download_pdf_from_doi(doi, email=email)
        if not pdf_path:
            print(f"[!] Failed to download PDF for DOI: {doi}")
            return ""
    elif pdf_path:
        print(f"[DEBUG] Processing local PDF: {pdf_path}")
        if not os.path.exists(pdf_path):
            print(f"[!] Provided PDF path does not exist: {pdf_path}")
            return ""
    else:
        print(f"[ERROR] No DOI or PDF path provided.")
        return ""

    tei_path = parse_pdf_with_grobid(pdf_path)
    if not tei_path:
        print(f"[!] GROBID failed for file: {pdf_path}")
        return ""

    text = tei_to_text(tei_path)
    print(f"[DEBUG] Extracted text length: {len(text)}")
    return text

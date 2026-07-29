import os

# NOTE: this module is meant to be imported as part of the `pipeline` package
# (e.g. `from pipeline.processor import input_to_text`, as done in
# llm/backend/app.py and llm/debug_preprocess.py, both of which add llm/ to
# sys.path before importing). Using package-relative imports here (instead of
# the previous mix of a bare `from downloader import ...` plus a sys.path
# hack) makes that the only supported way to run this file, which avoids
# import errors when the working directory differs from llm/pipeline/.
from .downloader import download_pdf_from_doi
from .parser import parse_pdf_with_grobid, tei_to_text

def input_to_text(doi: str = None, pdf_path: str = None, email: str = None) -> str:
    """
    Main processor function that handles input from either a DOI or a local PDF path.

    `email` defaults to the UNPAYWALL_EMAIL environment variable (see
    llm/.env) rather than a hardcoded placeholder. Previously this signature
    defaulted to the literal string "your@email.com", and llm/backend/app.py
    called input_to_text(doi=doi) without an email argument -- so every DOI
    lookup silently used that placeholder regardless of what UNPAYWALL_EMAIL
    was set to in .env. Not confirmed to be the cause of any specific
    extraction failure (Unpaywall does not verify email deliverability), but
    it was genuinely disconnected wiring and worth fixing regardless.

    Returns extracted plain text from the article, or empty string if failed.
    """
    if email is None:
        email = os.getenv("UNPAYWALL_EMAIL", "your@email.com")

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

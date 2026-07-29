import os
from pathlib import Path

import requests
from bs4 import BeautifulSoup

GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070/api/processFulltextDocument")
# GROBID exposes a lightweight health-check endpoint on the same host/port
# as the processing endpoint above; derive it instead of hardcoding it twice.
GROBID_ISALIVE_URL = GROBID_URL.split("/api/")[0] + "/api/isalive"

# Resolve relative to this file (llm/pipeline/parser.py -> llm/data/papers_txt)
# instead of the process's current working directory, so this works the same
# whether the caller runs from llm/, llm/backend/, or anywhere else.
_DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "data" / "papers_txt")


def grobid_is_alive(timeout: float = 2.0) -> bool:
    """
    Quick reachability check for the GROBID service (its own /api/isalive
    endpoint). Used to fail fast with a clear message instead of silently
    returning an empty extracted text when GROBID isn't running — GROBID is
    a separate long-running Java service (see llm/tools/grobid), not
    something this backend starts on its own.
    """
    try:
        resp = requests.get(GROBID_ISALIVE_URL, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def parse_pdf_with_grobid(pdf_path: str, output_dir: str = _DEFAULT_OUTPUT_DIR) -> str:
    """
    Uses GROBID service via HTTP to convert a PDF into TEI XML.
    Returns the path to the saved .tei.xml file, or None on failure
    (including when GROBID is unreachable — check grobid_is_alive() first
    for a clearer diagnosis than a generic failure).
    """
    if not grobid_is_alive():
        print(
            f"[ERROR] GROBID is not reachable at {GROBID_ISALIVE_URL}. "
            f"Start it separately: cd llm/tools/grobid && ./gradlew run"
        )
        return None

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".tei.xml"
    output_path = os.path.join(output_dir, filename)

    try:
        with open(pdf_path, "rb") as f:
            response = requests.post(
                GROBID_URL,
                files={"input": f},
                data={"consolidateHeader": 1}
            )

        if response.status_code == 200:
            with open(output_path, "w", encoding="utf-8") as out:
                out.write(response.text)
            return output_path
        else:
            print(f"[ERROR] GROBID HTTP error: {response.status_code}")
            return None
    except Exception as e:
        print(f"[EXCEPTION] Failed to call GROBID: {e}")
        return None

def tei_to_text(tei_path: str) -> str:
    """
    Extracts plain text from a TEI XML file produced by GROBID.
    Joins paragraph contents into a single text block.
    """
    try:
        with open(tei_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "xml")
        paragraphs = soup.find_all("p")
        return "\n".join(p.get_text().strip() for p in paragraphs if p.get_text(strip=True))
    except Exception as e:
        print(f"[!] TEI parse error: {e}")
        return ""

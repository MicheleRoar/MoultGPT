import os
import requests
from bs4 import BeautifulSoup

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

def parse_pdf_with_grobid(pdf_path: str, output_dir="data/papers_txt") -> str:
    """
    Uses GROBID service via HTTP to convert a PDF into TEI XML.
    Returns the path to the saved .tei.xml file.
    """
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

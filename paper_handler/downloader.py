import os
import requests

def download_pdf_from_doi(doi: str, email: str, output_dir: str = "data/papers_pdf") -> str:
    """
    Downloads a PDF from a given DOI using the Unpaywall API.
    Returns the local file path if successful, or None if failed.
    """
    os.makedirs(output_dir, exist_ok=True)
    api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code != 200:
            print(f"[!] Failed to contact Unpaywall API for DOI: {doi}")
            return None

        data = response.json()
        pdf_url = data.get("best_oa_location", {}).get("url_for_pdf")
        if not pdf_url:
            print(f"[!] No PDF URL found for DOI: {doi}")
            return None

        pdf_response = requests.get(pdf_url, headers=headers)
        if pdf_response.status_code == 200:
            filename = doi.replace("/", "_") + ".pdf"
            path = os.path.join(output_dir, filename)
            with open(path, "wb") as f:
                f.write(pdf_response.content)
            return path
        else:
            print(f"[!] Failed to download PDF from: {pdf_url}")
            return None

    except Exception as e:
        print(f"[!] Exception while downloading PDF: {e}")
        return None

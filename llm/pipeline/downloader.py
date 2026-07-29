import os
import re
import time
import requests

# ---------------------------------------------------------------------
# Default email for Unpaywall
# ---------------------------------------------------------------------

UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "moultgpt@unil.ch")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_filename(s: str) -> str:
    """Sanitize DOI into a valid filename."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def _http_get(url: str, headers: dict, retries: int = 2, sleep: float = 1.0):
    """HTTP GET with retries."""
    for i in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                return resp
            else:
                print(f"[UNPAYWALL] HTTP {resp.status_code} for {url}")
        except Exception as e:
            print(f"[UNPAYWALL] Exception during GET: {e}")

        time.sleep(sleep)

    return None


# ---------------------------------------------------------------------
# Main downloader
# ---------------------------------------------------------------------

def download_pdf_from_doi(
    doi: str,
    email: str = None,
    output_dir: str = "data/papers_pdf"
) -> str | None:
    """
    Download a PDF using Unpaywall's OA API.

    Returns:
        local path to PDF, or None if download failed.
    """
    # Validate email
    if email is None:
        email = UNPAYWALL_EMAIL

    if not email or "@" not in email:
        print("[UNPAYWALL] ERROR: invalid email. Set UNPAYWALL_EMAIL env variable.")
        return None

    # Prepare
    os.makedirs(output_dir, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}

    # Query Unpaywall API
    api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    print(f"[UNPAYWALL] Querying: {api_url}")

    resp = _http_get(api_url, headers)
    if resp is None:
        print(f"[UNPAYWALL] ERROR: Could not contact Unpaywall for DOI: {doi}")
        return None

    data = resp.json()

    # Extract PDF URL from best_oalocation
    best = data.get("best_oa_location") or {}

    pdf_url = (
        best.get("url_for_pdf")
        or best.get("pdf_url")
        or best.get("url")  # fallback (publisher URL)
    )

    if not pdf_url:
        print(f"[UNPAYWALL] No PDF link found for DOI: {doi}")
        return None

    print(f"[UNPAYWALL] Found PDF URL: {pdf_url}")

    # Download PDF
    pdf_resp = _http_get(pdf_url, headers)
    if pdf_resp is None:
        print(f"[UNPAYWALL] ERROR: Failed to download PDF from: {pdf_url}")
        return None

    # Save file
    fname = _safe_filename(doi) + ".pdf"
    path = os.path.join(output_dir, fname)

    try:
        with open(path, "wb") as f:
            f.write(pdf_resp.content)
        print(f"[UNPAYWALL] Saved PDF → {path}")
        return path
    except Exception as e:
        print(f"[UNPAYWALL] ERROR saving PDF: {e}")
        return None

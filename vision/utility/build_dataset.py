import csv
import time
import random
from pathlib import Path
import requests
from tqdm import tqdm

# ===================== SETTINGS =====================
PROJECT_ID = 200497
API_ENDPOINT = "https://api.inaturalist.org/v1/observations"

# Licenze ammesse (aggiungi 'cc-by-sa' se lo accetti)
VALID_LICENSES = {"cc0", "cc-by", "cc-by-nc"}

# Scarica quante più osservazioni/foto possibile
PER_PAGE = 100
MAX_RESULTS = 100000
MAX_PHOTOS_PER_OBS = 5

# Rate limiting & retry
REQUEST_TIMEOUT = 15
REQUEST_SLEEP = 0.15
RETRY = 3

# Output (niente split: solo per stage)
OUTPUT_DIR = Path("../data/inat_raw")
CSV_PATH = OUTPUT_DIR / "inat_dataset_minimal.csv"

# Stage da tenere (escludiamo pre-moult)
KEEP_STAGES = {"moulting", "exuviae", "post-moult"}


# ===================== HELPERS =====================
def get_taxon_group(ancestor_ids):
    """Mappa macro-gruppo tassonomico custom."""
    if not ancestor_ids:
        return "Unknown"
    if 245097 in ancestor_ids:
        return "Chelicerata"
    elif 144128 in ancestor_ids:
        return "Myriapoda"
    elif 85493 in ancestor_ids:
        return "Crustacea"
    elif 372739 in ancestor_ids:
        return "Hexapoda"
    else:
        return "Unknown"


def best_photo_url(photo: dict) -> str:
    """Preferisci original > large > medium (partendo da url)."""
    url = photo.get("url") or ""
    if not url:
        return ""
    # iNat: …/square.jpg, …/small.jpg, …/medium.jpg, …/large.jpg, …/original.jpg
    if "original" in url:
        return url
    if "large" in url:
        return url
    return url.replace("square", "large")


def fetch_observations(project_id, per_page=PER_PAGE, max_results=MAX_RESULTS):
    results = []
    page = 1
    with tqdm(total=max_results, desc="Fetching observations") as pbar:
        while len(results) < max_results:
            params = {
                "project_id": project_id,
                "per_page": per_page,
                "page": page,
                "license": ",".join(VALID_LICENSES),
                "photo_license": ",".join(VALID_LICENSES),
                "order": "desc",
                "order_by": "created_at",
            }
            data = {}
            for attempt in range(RETRY):
                try:
                    r = requests.get(API_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception:
                    if attempt + 1 == RETRY:
                        data = {}
                    time.sleep(1.0 * (attempt + 1))

            obs = data.get("results", [])
            if not obs:
                break
            results.extend(obs)
            pbar.update(len(obs))
            if len(obs) < per_page:
                break
            page += 1
            time.sleep(REQUEST_SLEEP)
    return results[:max_results]


def download(url: str, dest: Path) -> bool:
    for attempt in range(RETRY):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and resp.content:
                dest.write_bytes(resp.content)
                return True
        except Exception:
            pass
        time.sleep(0.5 * (attempt + 1))
    return False


# ===================== MAIN =====================
def build_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV minimal: una riga per foto (campi richiesti)
    header = [
        "observation_id",
        "photo_id",
        "stage",
        "taxon_id",
        "taxon_name",
        "taxon_group",
        "photo_license",
    ]
    observations = fetch_observations(PROJECT_ID)

    counts = {k: 0 for k in KEEP_STAGES}
    skipped_stage = 0

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=header)
        writer.writeheader()

        for obs in tqdm(observations, desc="Downloading photos"):
            # 1) Stage dai observation field values (OFVs)
            stage = None
            for ofv in obs.get("ofvs", []):
                if ofv.get("name_ci", "").lower() == "moulting stage" and ofv.get("value_ci"):
                    stage = ofv["value_ci"].strip().lower()
                    break
            if stage not in KEEP_STAGES:
                skipped_stage += 1
                continue

            photos = [p for p in obs.get("photos", []) if p.get("license_code") in VALID_LICENSES]
            if not photos:
                continue

            taxon = obs.get("taxon", {}) or {}
            taxon_id = taxon.get("id")
            taxon_name = taxon.get("name", "Unknown")
            taxon_group = get_taxon_group(taxon.get("ancestor_ids", []))

            # cartella per stage (niente split ora)
            stage_dir = OUTPUT_DIR / stage
            stage_dir.mkdir(parents=True, exist_ok=True)

            # salva fino a N foto dell'osservazione
            for p in photos[:MAX_PHOTOS_PER_OBS]:
                photo_id = p.get("id") or random.randint(0, 10**9)
                url = best_photo_url(p)
                if not url:
                    continue

                filename = f"{obs['id']}_{photo_id}.jpg"
                dest = stage_dir / filename
                ok = download(url, dest)
                time.sleep(REQUEST_SLEEP)
                if not ok:
                    continue

                counts[stage] += 1

                writer.writerow({
                    "observation_id": obs.get("id"),
                    "photo_id": photo_id,
                    "stage": stage,
                    "taxon_id": taxon_id,
                    "taxon_name": taxon_name,
                    "taxon_group": taxon_group,
                    "photo_license": p.get("license_code"),
                })

    print("\nCounts per stage:")
    for k, v in counts.items():
        print(f"  - {k}: {v}")
    print(f"Skipped observations (stage not in {KEEP_STAGES}): {skipped_stage}")
    print(f"CSV saved to: {CSV_PATH.resolve()}")
    print(f"Images saved under: {OUTPUT_DIR.resolve()}/*")


if __name__ == "__main__":
    build_dataset()

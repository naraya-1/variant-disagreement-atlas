"""Download ProteinGym DMS substitution benchmark zero-shot scores.

Data source: marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/
Scores file: ~4.4 GB unzipped, contains one CSV per DMS assay.
Reference metadata: GitHub OATML-Markslab/ProteinGym reference_files/
"""

import sys
import zipfile
from pathlib import Path

import requests
import urllib3
from tqdm import tqdm

PROTEINGYM_VERSION = "v1.3"
BASE_URL = f"https://marks.hms.harvard.edu/proteingym/ProteinGym_{PROTEINGYM_VERSION}"
SCORES_FILENAME = "zero_shot_substitutions_scores.zip"
SCORES_URL = f"{BASE_URL}/{SCORES_FILENAME}"
REF_URL = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym"
    "/main/reference_files/DMS_substitutions.csv"
)
REF_FILENAME = "DMS_substitutions.csv"
DATA_DIR = Path("data/raw")


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Download url to dest with a tqdm progress bar, resuming if partial."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    def _get(verify: bool) -> requests.Response:
        return requests.get(url, headers=headers, stream=True, timeout=60, verify=verify)

    try:
        resp = _get(verify=True)
    except requests.exceptions.SSLError:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        print("  SSL verification failed — retrying without verification.")
        resp = _get(verify=False)

    if resp.status_code == 416:
        print(f"  {dest.name}: already complete, skipping.")
        return

    resp.raise_for_status()

    total = int(resp.headers.get("Content-Length", 0))
    if resp.status_code == 206:
        total += existing

    mode = "ab" if existing and resp.status_code == 206 else "wb"

    with open(dest, mode) as fh, tqdm(
        total=total or None,
        initial=existing,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest.name,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            bar.update(len(chunk))


def _remote_size(url: str) -> int:
    """Return Content-Length from a HEAD request, or 0 on failure."""
    try:
        r = requests.head(url, timeout=15, allow_redirects=True)
        return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract all members of zip_path into extract_to with a progress bar."""
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, extract_to)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Reference metadata (small, from GitHub) ---
    ref_dest = DATA_DIR / REF_FILENAME
    if ref_dest.exists():
        print(f"Reference file already present: {ref_dest}")
    else:
        print("Downloading DMS reference metadata...")
        download_file(REF_URL, ref_dest)
        print(f"  Saved to {ref_dest}")

    # --- Zero-shot scores zip ---
    scores_zip = DATA_DIR / SCORES_FILENAME

    # Detect the extraction target; the zip is expected to contain a top-level
    # directory called zero_shot_substitutions_scores/.
    extract_dir = DATA_DIR / "zero_shot_substitutions_scores"

    if extract_dir.exists() and any(extract_dir.glob("*.csv")):
        print(f"Zero-shot scores already extracted: {extract_dir}")
        return

    # Decide whether to (re)download
    need_download = True
    if scores_zip.exists():
        local_size = scores_zip.stat().st_size
        remote_size = _remote_size(SCORES_URL)
        if remote_size and local_size >= remote_size:
            print(f"Zip already fully downloaded ({local_size / 1e9:.2f} GB): {scores_zip}")
            need_download = False
        elif local_size:
            print(
                f"Resuming download ({local_size / 1e6:.0f} MB already on disk)..."
            )

    if need_download:
        print(f"Downloading zero-shot substitution scores (~4.4 GB unzipped).")
        print(f"URL: {SCORES_URL}")
        download_file(SCORES_URL, scores_zip)

    print(f"\nExtracting {scores_zip.name}...")
    extract_zip(scores_zip, DATA_DIR)
    print(f"\nDone. Data available in {DATA_DIR}/")


if __name__ == "__main__":
    main()

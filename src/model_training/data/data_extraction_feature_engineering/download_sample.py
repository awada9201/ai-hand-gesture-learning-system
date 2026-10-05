"""Download and unzip a pre-made HaGRID sample from Hugging Face.
 
Default: cj-mills/hagrid-sample-30k-384p (31,833 images, downscaled to 384p).
Public dataset - no token or API key is needed.
 
Requires: pip install huggingface_hub
The repo id and zip filename come from the dataset page; verify them if the
download fails (the author may have renamed or moved the files).
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path
from typing import Optional, Sequence

from huggingface_hub import hf_hub_download

LOGGER = logging.getLogger(__name__)
DEFAULT_REPO = "cj-mills/hagrid-sample-30k-384p"
DEFAULT_FILE = "hagrid-sample-30k-384p.zip"

def download_and_extract(repo_id: str, filename: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    try:
        zip_path = Path(
            hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=str(dest))
        )
    except Exception as exc:
        raise RuntimeError(f"Download failed for {repo_id}/{filename}: {exc}") from exc
    
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(dest)
    except(zipfile.BadZipFile, OSError) as exc:
        raise RuntimeError(f"Could not extract: {zip_path}: {exc}") from exc
    return dest

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--filename", default=DEFAULT_FILE)
    parser.add_argument("--dest", type=Path, default=Path("data/hagrid_sample"))
    
    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    
    try:
        out = download_and_extract(args.repo_id, args.filename, args.dest)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("Extracted to %s", out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
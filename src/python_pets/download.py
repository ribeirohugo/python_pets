"""Locate the dataset on disk, fetching it from DATASET_URL when datasets/ is empty."""
from __future__ import annotations

import re
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import kagglehub

from python_pets import config

SOURCE_DIRS = ("images", "annotations")  # layout inside a Kaggle dataset or archive
KAGGLE_URL = re.compile(r"kaggle\.com/datasets/([^/?#]+/[^/?#]+)")


def is_source_dataset(p: Path) -> bool:
    return all((p / name).is_dir() for name in SOURCE_DIRS)


def has_files(p: Path) -> bool:
    """True if `p` holds anything besides dotfiles such as .gitkeep."""
    return p.is_dir() and any(not f.name.startswith(".") for f in p.iterdir())


def extract_archive(url: str, workdir: Path) -> Path:
    """Download a .zip/.tar[.gz] from a direct URL and return the folder holding the dataset."""
    archive = workdir / "download"
    with urllib.request.urlopen(url) as response, archive.open("wb") as out:
        shutil.copyfileobj(response, out)

    extracted = workdir / "extracted"
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(extracted)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(extracted, filter="data")
    else:
        raise SystemExit(f"DATASET_URL did not return a .zip or .tar archive: {url}")

    for candidate in [extracted, *extracted.rglob("*")]:
        if candidate.is_dir() and is_source_dataset(candidate):
            return candidate
    raise SystemExit("The archive has no folder containing both images/ and annotations/.")


def install(source: Path, destination: Path) -> None:
    """Rename `source` onto the empty `destination` folder, keeping its .gitkeep."""
    if destination.is_dir():
        (destination / ".gitkeep").unlink(missing_ok=True)
        destination.rmdir()
    source.rename(destination)
    (destination / ".gitkeep").touch()


def fetch_dataset(url: str) -> None:
    """Fetch `url` into datasets/. Kaggle pages use kagglehub, anything else is a direct archive."""
    for folder in (config.RAW_DIR, config.ANNOTATIONS_DIR):
        if has_files(folder):
            raise SystemExit(f"{folder} has files but the dataset is incomplete. "
                             "Move or delete them, then run again.")

    config.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    # Staged inside datasets/ so the final renames are instant and an interrupted
    # download can't leave a half-filled dataset that looks complete.
    with tempfile.TemporaryDirectory(dir=config.DATASETS_DIR) as tmp:
        staging = Path(tmp)
        kaggle = KAGGLE_URL.search(url)
        if kaggle:
            source = Path(kagglehub.dataset_download(kaggle.group(1)))
            if not is_source_dataset(source):
                raise SystemExit(f"The Kaggle dataset has no images/ and annotations/ folders: {url}")
            for name in SOURCE_DIRS:
                shutil.copytree(source / name, staging / name)
            source = staging
        else:
            source = extract_archive(url, staging)
        install(source / "images", config.RAW_DIR)
        install(source / "annotations", config.ANNOTATIONS_DIR)


def resolve_dataset(arg: str | None) -> tuple[Path, Path]:
    """Return (images folder, annotations folder).

    Uses --data if given, else datasets/, fetching it from DATASET_URL if empty.
    """
    if arg:
        return Path(arg), Path(arg)
    if not (has_files(config.RAW_DIR) and has_files(config.ANNOTATIONS_DIR)):
        url = config.dataset_url()
        if not url:
            raise SystemExit("No dataset in datasets/ and DATASET_URL is not set. "
                             "Copy .env.example to .env, or pass --data.")
        print(f"Dataset not found in {config.DATASETS_DIR} - fetching {url}")
        fetch_dataset(url)
    return config.RAW_DIR, config.ANNOTATIONS_DIR

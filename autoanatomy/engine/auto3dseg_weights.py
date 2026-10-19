"""Pinned MONAI whole-head checkpoint, downloaded separately from the app."""
import hashlib
from pathlib import Path
import shutil
import tempfile
import zipfile

import requests

from autoanatomy.engine.config import get_weights_dir

MODEL_ID = "whole-head-05mm-v1.0.1"
MODEL_FOLDER = "Auto3DSeg_" + MODEL_ID
MODEL_URL = (
    "https://github.com/lassoan/SlicerMONAIAuto3DSeg/releases/download/Models/"
    + MODEL_ID + ".zip"
)
ARCHIVE_SHA256 = "2d82b035dc90d9ae7b95a4c6f296da8d758776f374219e1ec3c052bf3cc50f85"
CHECKPOINT_SHA256 = "facb15e72d5c3b1d70c2d58e6a0d0583ae8a50edb1f816ae4bdb34f0442a0784"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_directory():
    return get_weights_dir() / MODEL_FOLDER


def weights_cached():
    folder = model_directory()
    return all((folder / name).is_file() for name in ("model.pt", "labels.csv"))


def install_archive(archive_path, destination=None):
    """Verify a downloaded archive and install only its two expected members.

    Never extract archive-provided paths. Temporary files share the destination
    filesystem so replacing the checkpoint cannot leave a partial model file.
    """
    archive_path = Path(archive_path)
    if _sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError("Auto3DSeg download checksum mismatch; the model was not installed.")
    destination = Path(destination) if destination is not None else model_directory()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="auto3dseg-install-", dir=destination.parent) as tmp:
        staging = Path(tmp)
        with zipfile.ZipFile(archive_path) as archive:
            if sorted(archive.namelist()) != ["labels.csv", "model.pt"]:
                raise ValueError("Unexpected files in the Auto3DSeg model archive.")
            for name in ("model.pt", "labels.csv"):
                with archive.open(name) as source, (staging / name).open("wb") as target:
                    shutil.copyfileobj(source, target)
        if _sha256(staging / "model.pt") != CHECKPOINT_SHA256:
            raise ValueError("Auto3DSeg checkpoint checksum mismatch.")
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("labels.csv", "model.pt"):
            (staging / name).replace(destination / name)
    return destination / "model.pt"


def download_weights(quiet=False):
    destination = model_directory()
    if weights_cached():
        if _sha256(destination / "model.pt") != CHECKPOINT_SHA256:
            raise ValueError(
                f"Cached Auto3DSeg checkpoint checksum mismatch. Remove {destination} "
                "and download the model again."
            )
        return destination / "model.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print("Downloading Auto3DSeg Whole Head Segmentation 0.5 mm (323 MB)...")
    with tempfile.TemporaryDirectory(prefix="auto3dseg-download-", dir=destination.parent) as tmp:
        archive_path = Path(tmp) / "model.zip"
        with requests.get(MODEL_URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with archive_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    output.write(chunk)
        return install_archive(archive_path, destination)

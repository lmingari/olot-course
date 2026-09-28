from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve
import warnings


def download_file(url: str, folder: str | Path = "data") -> Path:
    """Download a file and optionally verify its SHA-256 checksum."""

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    filename = Path(urlparse(url).path).name
    filepath = folder / filename
    checksum_file = folder / f"{filename}.sha256"

    if not filepath.exists():
        print(f"Downloading {filename}...")
        urlretrieve(url, filepath)

    if not checksum_file.exists():
        warnings.warn(
            f"No checksum file found for {filename}; "
            "the file integrity could not be verified."
        )
        return filepath

    expected = checksum_file.read_text().split()[0]

    digest = sha256()
    with filepath.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    if digest.hexdigest() != expected:
        raise RuntimeError(f"Checksum verification failed: {filepath}")

    return filepath

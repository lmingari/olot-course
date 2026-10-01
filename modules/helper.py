import shutil
import warnings
from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlopen


def _filename_from_response(response, url: str) -> str:
    """Get the filename from Content-Disposition, falling back to the URL path."""
    # Parses both filename="x" and RFC 5987 filename*=UTF-8''x forms
    name = response.headers.get_filename()

    if not name:
        # Final URL after redirects, then the original URL
        for candidate in (response.geturl(), url):
            name = Path(unquote(urlparse(candidate).path)).name
            if name:
                break

    # Strip any directory components (protects against "../../x" names)
    name = Path(name or "").name
    if not name:
        raise ValueError(f"Could not determine a filename for {url}")
    return name


def download_file(url: str, folder: str | Path = "data") -> Path:
    """Download a file and optionally verify its SHA-256 checksum."""

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    with urlopen(url) as response:
        filename = _filename_from_response(response, url)
        filepath = folder / filename

        if not filepath.exists():
            print(f"Downloading {filename}...")
            partial = filepath.with_name(filename + ".part")
            with partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            partial.replace(filepath)  # avoid leaving a half-downloaded file

    checksum_file = folder / f"{filename}.sha256"

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

def get_decision_regions(model, transform):
    import numpy as np
    import xarray as xr
    import torch
    # Define the grid
    lat_min, lat_max = 28.4, 28.9
    lon_min, lon_max = -18.1, -17.65
    n_lat, n_lon = 220, 220

    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)

    lat_grid, lon_grid = np.meshgrid(
        lats, lons, indexing="ij"
    )

    # Prepare input coordinates
    X = np.column_stack([
        lat_grid.ravel(),
        lon_grid.ravel(),
    ])

    # Apply the same standardisation used for training
    X = transform(torch.from_numpy(X).float())

    # Predict impact class
    model.eval()
    with torch.no_grad():
        logits = model(X)
        impact = logits.argmax(dim=1)

    # Restore grid shape
    impact = impact.numpy().reshape(n_lat, n_lon)

    # Return as DataArray
    return xr.DataArray(
        impact,
        dims=("lat", "lon"),
        coords={
            "lat": lats,
            "lon": lons,
        },
        name="impact",
    )

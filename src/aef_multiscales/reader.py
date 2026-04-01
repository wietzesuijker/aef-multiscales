"""Read tiles from AEF mosaic Zarr arrays."""

import logging
import time

import zarr
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_STORE = "s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/"

# Full-res shard spatial size
SHARD_SPATIAL = 4096


def open_store(path: str = DEFAULT_STORE) -> zarr.Group:
    if path.startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})
        store = s3fs.S3Map(root=path.replace("s3://", ""), s3=fs)
        return zarr.open_group(store, mode="r")
    return zarr.open_group(path, mode="r")


def read_tile(
    arr: zarr.Array,
    time_idx: int,
    tile_row: int,
    tile_col: int,
    tile_size: int,
    retries: int = 3,
    backoff: float = 1.0,
) -> np.ndarray:
    """Read one tile with retry. Returns (bands, tile_h, tile_w) int8 array."""
    y0 = tile_row * tile_size
    x0 = tile_col * tile_size
    y1 = min(y0 + tile_size, arr.shape[2])
    x1 = min(x0 + tile_size, arr.shape[3])

    for attempt in range(retries):
        try:
            return arr[time_idx, :, y0:y1, x0:x1]
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            logger.warning("Read failed (t=%d, r=%d, c=%d), attempt %d/%d, retrying in %.1fs: %s",
                           time_idx, tile_row, tile_col, attempt + 1, retries, wait, e)
            time.sleep(wait)


def tile_grid(arr: zarr.Array, tile_size: int) -> tuple[int, int]:
    """Return (n_tile_rows, n_tile_cols) for a given array and tile size."""
    _, _, h, w = arr.shape
    rows = (h + tile_size - 1) // tile_size
    cols = (w + tile_size - 1) // tile_size
    return rows, cols

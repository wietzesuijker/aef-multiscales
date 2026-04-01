"""Create and write overview arrays using zarr-python with zarrs codec plugin."""

import zarr
import zarr.codecs
import numpy as np

from aef_multiscales.naming import array_name, coord_names

FULL_SHAPE = (9, 64, 1859584, 4009984)
PIXEL_SIZE = 0.00009
BBOX = (-180.0, -83.36, 180.22, 83.69)

# Threshold: shard if compressed size would exceed ~500 GB
_SHARD_THRESHOLD_PIXELS = 500_000_000_000  # ~500 GB at 1 byte/pixel before compression


def _configure_zarrs():
    """Enable zarrs Rust codec pipeline if available."""
    try:
        import zarrs
        zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    except ImportError:
        pass


_configure_zarrs()


def create_overview_array(
    store: zarr.abc.store.Store,
    factor: int,
    compression_level: int = 3,
) -> zarr.Array:
    """Create the overview array with appropriate chunk/shard config.

    Small overviews (128x, ~78 GB): chunk-only, (1, 64, 256, 256).
    Large overviews (16x, ~5 TB): sharded, matching the full-res config.
    """
    time, bands, h, w = FULL_SHAPE
    oh, ow = h // factor, w // factor
    name = array_name(factor)

    total_pixels = time * bands * oh * ow
    needs_sharding = total_pixels > _SHARD_THRESHOLD_PIXELS

    chunk_h = min(256, oh)
    chunk_w = min(256, ow)

    if needs_sharding:
        shard_h = min(4096 // factor, oh)
        shard_w = min(4096 // factor, ow)
        # Ensure shard is a multiple of chunk
        shard_h = max(chunk_h, (shard_h // chunk_h) * chunk_h)
        shard_w = max(chunk_w, (shard_w // chunk_w) * chunk_w)

        arr = zarr.create_array(
            store,
            name=name,
            shape=(time, bands, oh, ow),
            dtype="int8",
            chunks=(1, 64, chunk_h, chunk_w),
            shards=(1, 64, shard_h, shard_w),
            fill_value=-128,
            compressors=[zarr.codecs.ZstdCodec(level=compression_level)],
            zarr_format=3,
            overwrite=True,
        )
    else:
        arr = zarr.create_array(
            store,
            name=name,
            shape=(time, bands, oh, ow),
            dtype="int8",
            chunks=(1, 64, chunk_h, chunk_w),
            fill_value=-128,
            compressors=[zarr.codecs.ZstdCodec(level=compression_level)],
            zarr_format=3,
            overwrite=True,
        )

    return arr


def write_overview_region(
    arr: zarr.Array,
    data: np.ndarray,
    time_idx: int,
    y_start: int,
    x_start: int,
) -> None:
    """Write a (bands, h, w) int8 block to the overview array."""
    _, h, w = data.shape
    arr[time_idx, :, y_start:y_start + h, x_start:x_start + w] = data


def create_coord_arrays(
    store: zarr.abc.store.Store,
    factor: int,
) -> None:
    """Write x and y coordinate arrays for the overview level."""
    _, _, h, w = FULL_SHAPE
    oh, ow = h // factor, w // factor
    pixel = PIXEL_SIZE * factor
    lon_min, _, _, lat_max = BBOX

    x_name, y_name = coord_names(factor)
    x_coords = lon_min + (np.arange(ow) + 0.5) * pixel
    y_coords = lat_max - (np.arange(oh) + 0.5) * pixel

    zarr.create_array(store, name=x_name, data=x_coords, dtype="float64", zarr_format=3, overwrite=True)
    zarr.create_array(store, name=y_name, data=y_coords, dtype="float64", zarr_format=3, overwrite=True)

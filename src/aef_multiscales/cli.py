"""CLI for generating AEF mosaic overview layers."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import numpy as np
import zarr

from aef_multiscales.aggregate import aggregate_shard
from aef_multiscales.checkpoint import Checkpoint, NoCheckpoint
from aef_multiscales.naming import array_name, resolution_label
from aef_multiscales.reader import open_store, read_tile, tile_grid, SHARD_SPATIAL
from aef_multiscales.writer import (
    create_overview_array,
    write_overview_region,
    create_coord_arrays,
)
from aef_multiscales.metadata import update_root_metadata

logger = logging.getLogger(__name__)

_LAT_MAX = 83.69
_LON_MIN = -180.0
_PIXEL = 0.00009
_SHARD_DEG = SHARD_SPATIAL * _PIXEL


def _latlon_to_shard(lat: float, lon: float) -> tuple[int, int]:
    """Convert lat/lon to shard row/col indices."""
    row = int((_LAT_MAX - lat) / _SHARD_DEG)
    col = int((lon - _LON_MIN) / _SHARD_DEG)
    return row, col


def _bbox_to_shard_range(
    west: float, south: float, east: float, north: float,
) -> tuple[int, int, int, int]:
    """Convert bbox to inclusive shard row/col ranges."""
    row_min = int((_LAT_MAX - north) / _SHARD_DEG)
    row_max = int((_LAT_MAX - south) / _SHARD_DEG)
    col_min = int((west - _LON_MIN) / _SHARD_DEG)
    col_max = int((east - _LON_MIN) / _SHARD_DEG)
    return row_min, row_max, col_min, col_max


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Debug logging")
def main(verbose):
    """Generate overview layers for the AEF mosaic."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@main.command()
@click.option("--level", type=int, required=True, help="Downsampling factor (e.g., 128)")
@click.option("--bbox", type=float, nargs=4, default=None, help="west south east north")
@click.option("--lat", type=float, default=None, help="Latitude (single shard)")
@click.option("--lon", type=float, default=None, help="Longitude (single shard)")
@click.option("--shard-row", type=int, default=None, help="Shard row index")
@click.option("--shard-col", type=int, default=None, help="Shard column index")
@click.option("--time-idx", type=int, default=0, help="Time index (default: 0 = year 2017)")
@click.option("--input", "input_path", default=None, help="Store URL (default: public S3)")
@click.option("--output", "output_path", default="./validation", help="Local output path")
def validate(level, bbox, lat, lon, shard_row, shard_col, time_idx, input_path, output_path):
    """Validate aggregation against float64 reference.

    Specify region with --bbox (west south east north), --lat/--lon (single point),
    or --shard-row/--shard-col (direct indices).
    """
    if bbox is not None:
        west, south, east, north = bbox
        row_min, row_max, col_min, col_max = _bbox_to_shard_range(west, south, east, north)
        n_shards = (row_max - row_min + 1) * (col_max - col_min + 1)
        click.echo(f"bbox ({west}, {south}, {east}, {north}) -> {n_shards} shards "
                    f"(rows {row_min}-{row_max}, cols {col_min}-{col_max})")
        group = open_store(input_path) if input_path else open_store()
        arr = group["embeddings"]
        for r in range(row_min, row_max + 1):
            for c in range(col_min, col_max + 1):
                click.echo(f"\n--- Shard ({r}, {c}) ---")
                shard_data = read_tile(arr, time_idx, r, c, SHARD_SPATIAL)
                result = aggregate_shard(shard_data, factor=level)
                reference = _reference_aggregate_f64(shard_data, level)
                _compare(result, reference)
        click.echo("\nAll shards validated.")
        return

    if lat is not None and lon is not None:
        shard_row, shard_col = _latlon_to_shard(lat, lon)
        click.echo(f"({lat}, {lon}) -> shard ({shard_row}, {shard_col})")
    elif shard_row is None or shard_col is None:
        raise click.UsageError("Provide --bbox, --lat/--lon, or --shard-row/--shard-col")

    click.echo(f"Reading shard ({shard_row}, {shard_col}) at time={time_idx}...")
    group = open_store(input_path) if input_path else open_store()
    arr = group["embeddings"]
    shard_data = read_tile(arr, time_idx, shard_row, shard_col, SHARD_SPATIAL)
    click.echo(f"  Shape: {shard_data.shape}, dtype: {shard_data.dtype}")

    click.echo(f"Aggregating at {level}x ({resolution_label(level)})...")
    t0 = time.monotonic()
    result = aggregate_shard(shard_data, factor=level)
    elapsed = time.monotonic() - t0
    click.echo(f"  Output shape: {result.shape}, took {elapsed:.2f}s")

    click.echo("Running float64 reference...")
    reference = _reference_aggregate_f64(shard_data, level)

    click.echo("Comparing...")
    _compare(result, reference)

    click.echo(f"Writing output to {output_path}...")
    out_store = zarr.storage.LocalStore(output_path)
    zarr.open_group(out_store, mode="w", zarr_format=3)
    zarr.create_array(out_store, name="result", data=result, zarr_format=3, overwrite=True)
    zarr.create_array(out_store, name="reference", data=reference, zarr_format=3, overwrite=True)
    click.echo("Done.")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def _process_tile(source_arr, t, r, c, source_tile_size, local_factor):
    """Read one tile and aggregate. Pure function for thread pool."""
    tile_data = read_tile(source_arr, t, r, c, source_tile_size)
    return t, r, c, aggregate_shard(tile_data, factor=local_factor)


@main.command()
@click.option("--levels", type=int, multiple=True, required=True, help="Downsampling factors")
@click.option("--bbox", type=float, nargs=4, default=None, help="west south east north")
@click.option("--input", "input_path", default=None, help="Store URL (default: public S3)")
@click.option("--output", "output_path", required=True, help="Output store URL or path")
@click.option("--compression-level", type=int, default=3, help="Zstd compression level")
@click.option("--workers", type=int, default=8, help="Concurrent read/aggregate workers")
@click.option("--checkpoint-dir", type=str, default=None, help="Directory for checkpoint files (enables resume)")
def generate(levels, bbox, input_path, output_path, compression_level, workers, checkpoint_dir):
    """Generate overview layers for the mosaic.

    Output arrays are always full global extent. --bbox filters which tiles
    to compute (rest stays nodata). Run again with a different --bbox to
    fill in more regions incrementally.

    Use --checkpoint-dir to enable resume on crash. Use --workers to
    control concurrency (default: 8 parallel reads).
    """
    factors = sorted(levels)
    labels = [f"{f}x ({resolution_label(f)})" for f in factors]
    logger.info("Generating overview levels: %s", ", ".join(labels))

    input_group = open_store(input_path) if input_path else open_store()
    if output_path.startswith("s3://"):
        out_store = _s3_store(output_path)
    else:
        out_store = zarr.storage.LocalStore(output_path)
    time_steps = input_group["embeddings"].shape[0]

    prev_source_name = "embeddings"
    prev_factor = 1
    prev_source_group = input_group

    for factor in factors:
        name = array_name(factor)
        local_factor = factor // prev_factor
        assert local_factor >= 1, f"Factors must be ascending: {factor} not > {prev_factor}"

        logger.info("Level: %s (from %s, %dx local)", name, prev_source_name, local_factor)

        # Checkpoint per level
        if checkpoint_dir:
            ckpt = Checkpoint(f"{checkpoint_dir}/{name}.json")
            logger.info("Checkpoint: %d tiles already completed", len(ckpt))
        else:
            ckpt = NoCheckpoint()

        overview = create_overview_array(out_store, factor, compression_level)

        source_arr = prev_source_group[prev_source_name]
        source_tile_size = SHARD_SPATIAL if prev_factor == 1 else SHARD_SPATIAL // prev_factor
        out_tile_size = source_tile_size // local_factor
        assert out_tile_size >= 1, (
            f"Tile size underflow: source_tile={source_tile_size}, local_factor={local_factor}"
        )

        n_rows, n_cols = tile_grid(source_arr, source_tile_size)

        if bbox is not None:
            west, south, east, north = bbox
            r_min = max(0, int((_LAT_MAX - north) / (source_tile_size * _PIXEL)))
            r_max = min(n_rows - 1, int((_LAT_MAX - south) / (source_tile_size * _PIXEL)))
            c_min = max(0, int((west - _LON_MIN) / (source_tile_size * _PIXEL)))
            c_max = min(n_cols - 1, int((east - _LON_MIN) / (source_tile_size * _PIXEL)))
            logger.info("bbox filter: rows %d-%d, cols %d-%d", r_min, r_max, c_min, c_max)
        else:
            r_min, r_max = 0, n_rows - 1
            c_min, c_max = 0, n_cols - 1

        # Build work list, skipping checkpointed tiles
        work = []
        for t in range(time_steps):
            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    if not ckpt.is_done(t, r, c):
                        work.append((t, r, c))

        total = len(work) + len(ckpt)
        done = len(ckpt)
        skipped = total - len(work)
        if skipped:
            logger.info("Skipping %d already-completed tiles, %d remaining", skipped, len(work))

        t0 = time.monotonic()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_tile, source_arr, t, r, c, source_tile_size, local_factor): (t, r, c)
                for t, r, c in work
            }
            for future in as_completed(futures):
                t_idx, r_idx, c_idx, agg = future.result()
                y_start = r_idx * out_tile_size
                x_start = c_idx * out_tile_size
                write_overview_region(overview, agg, t_idx, y_start, x_start)
                ckpt.mark_done(t_idx, r_idx, c_idx)
                done += 1
                if done % 100 == 0:
                    elapsed = time.monotonic() - t0
                    rate = (done - skipped) / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    logger.info("%d/%d tiles (%.1f tiles/s, ETA %.0fm)", done, total, rate, eta / 60)

        ckpt.flush()
        create_coord_arrays(out_store, factor)
        elapsed = time.monotonic() - t0
        logger.info("Completed %s: %s in %.0fs", name, overview.shape, elapsed)

        prev_source_name = name
        prev_factor = factor
        prev_source_group = zarr.open_group(out_store, mode="r", zarr_format=3)

    logger.info("Updating root metadata...")
    out_group = zarr.open_group(out_store, mode="r+", zarr_format=3)
    update_root_metadata(out_group, factors)
    logger.info("Done.")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _reference_aggregate_f64(data: np.ndarray, factor: int) -> np.ndarray:
    """Brute-force float64 reference for validation."""
    bands, h, w = data.shape
    oh, ow = h // factor, w // factor
    out = np.full((bands, oh, ow), -128, dtype=np.int8)

    for iy in range(oh):
        for ix in range(ow):
            block = data[:, iy * factor:(iy + 1) * factor, ix * factor:(ix + 1) * factor]
            valid = block[0] != -128
            if not valid.any():
                continue
            fblock = block[:, valid].astype(np.float64)
            fblock = (fblock / 127.5) ** 2 * np.sign(fblock)
            mean_vec = fblock.mean(axis=1)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            raw = np.sign(mean_vec) * np.sqrt(np.abs(mean_vec)) * 127.5
            out[:, iy, ix] = np.clip(np.round(raw), -127, 127).astype(np.int8)

    return out


def _compare(result: np.ndarray, reference: np.ndarray) -> None:
    """Compare aggregated result against reference, print stats."""
    valid = result[0] != -128
    if not valid.any():
        click.echo("  No valid pixels to compare.")
        return

    def _deq(x):
        f = x.astype(np.float64)
        return (f / 127.5) ** 2 * np.sign(f)

    r = _deq(result[:, valid].T)
    ref = _deq(reference[:, valid].T)

    r_norm = r / np.linalg.norm(r, axis=1, keepdims=True)
    ref_norm = ref / np.linalg.norm(ref, axis=1, keepdims=True)

    dots = np.clip(np.sum(r_norm * ref_norm, axis=1), -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))

    click.echo(f"  Valid pixels: {valid.sum()}")
    click.echo(f"  Angular error -- max: {angles.max():.3f}, mean: {angles.mean():.3f}, median: {np.median(angles):.3f}")
    exact = (result[:, valid] == reference[:, valid]).all(axis=0).sum()
    click.echo(f"  Exact matches: {exact}/{valid.sum()} ({100*exact/valid.sum():.1f}%)")

    assert angles.max() < 1.0, f"FAIL: max angular error {angles.max():.3f} >= 1.0"
    assert angles.mean() < 0.5, f"FAIL: mean angular error {angles.mean():.3f} >= 0.5"
    click.echo("  PASS: within error bounds.")


def _s3_store(path: str):
    """Create an S3 store from a URL."""
    import s3fs
    fs = s3fs.S3FileSystem(anon=False)
    return zarr.storage.FsspecStore(fs, path=path.replace("s3://", ""))

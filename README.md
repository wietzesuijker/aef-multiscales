# aef-multiscales

Add overview layers to the [AEF mosaic](https://source.coop/tge-labs/aef-mosaic) Zarr store, following the [multiscales convention](https://github.com/zarr-conventions/multiscales).

The AEF mosaic is a global ~10m embedding dataset. It currently has no overview layers, so any consumer that doesn't need full resolution still has to read and downsample the full-res array. This tool pre-computes overview layers as sibling arrays in the same store.

### Aggregation

The mosaic stores float embeddings as int8 using a nonlinear encoding: `float = (int8 / 127.5)² * sign(int8)`. This encoding gives more int8 resolution to small float values and less to large ones. Think of a ruler where the markings get closer together at the top: a float value of 0.25 maps to int8 64, but a float value of 1.0 only maps to int8 127. The top 75% of the float range is squeezed into the top 50% of int8 values.

This matters for spatial averaging. If you average two int8 values (say 0 and 127), you get 63, which decodes to 0.24. But the actual float values were 0.0 and 1.0, whose true average is 0.50. The int8 average is biased toward zero because the encoding compresses high values together.

For homogeneous regions (all pixels are similar), this bias is small. For mixed regions, it depends on how spread out the values are. Either way, this tool decodes to float before averaging, L2-normalizes the result, and re-encodes. `tests/test_demo.py` has more examples.

## Install

```bash
pip install -e .
```

## Usage

### `validate`: check the math on a single shard

Reads a shard from the public S3 store, runs the aggregation, and compares against a brute-force float64 reference.

```bash
aef-multiscales validate --level 128 --lat 35 --lon -100
aef-multiscales validate --level 128 --bbox -105 34 -95 36
aef-multiscales validate --level 128 --shard-row 132 --shard-col 217
```

### `generate`: write overview arrays into the store

Overview arrays are written as siblings to the existing `embeddings` array in the same Zarr store:

```
s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/
├── zarr.json          # existing; multiscales convention appended
├── embeddings         # existing; untouched
├── embeddings_1km     # new: 128x spatial downsample
├── x, y, time         # existing
└── x_1km, y_1km       # new: coordinate arrays for overview
```

```bash
# ~1km overview for the full mosaic
aef-multiscales generate --levels 128 \
  --output s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/

# Subregion only (e.g., CONUS)
# Output array is always full global extent; --bbox filters which tiles to compute.
# Run again with a different --bbox to fill in more regions incrementally.
aef-multiscales generate --levels 128 --bbox -125 24 -66 50 \
  --output s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/

# Multiple levels with cascade (128x reads from 16x, cutting read I/O)
aef-multiscales generate --levels 16 128 \
  --output s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/
```

`--levels` follows [`gdaladdo`](https://gdal.org/en/stable/programs/gdaladdo.html) conventions. Levels are processed ascending; each reads from the previous when cascading. `--input` defaults to the public AEF mosaic on Source Cooperative.

### Production flags

```bash
aef-multiscales generate --levels 128 \
  --output s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/ \
  --workers 16 \
  --checkpoint-dir ./checkpoints
```

- `--workers N`: concurrent S3 reads + aggregation (default: 8). S3 is the bottleneck, not CPU.
- `--checkpoint-dir`: enables resume on crash. Each level gets a checkpoint file tracking completed tiles. Restart the same command and it picks up where it left off.
- `-v`: debug logging (per-tile timings, retry warnings).

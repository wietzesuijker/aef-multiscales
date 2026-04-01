"""Embedding-aware spatial aggregation for int8 quantized AEF data."""

import numpy as np

NODATA: int = -128


def _dequantize(x: np.ndarray) -> np.ndarray:
    """int8 array -> float32 via signed-square: (x/127.5)^2 * sign(x)."""
    f = x.astype(np.float32)
    return (f / 127.5) ** 2 * np.sign(f)


def _quantize(x: np.ndarray) -> np.ndarray:
    """float32 array -> int8 via inverse signed-square."""
    raw = np.sign(x) * np.sqrt(np.abs(x)) * 127.5
    return np.clip(np.round(raw), -127, 127).astype(np.int8)


def aggregate_shard(data: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a (bands, H, W) int8 shard by `factor` with correct aggregation.

    Steps per spatial window:
    1. Dequantize int8 -> float32
    2. Mask nodata (-128)
    3. Mean over valid pixels
    4. L2-normalize the result
    5. Re-quantize to int8

    Nodata contract: a pixel is nodata iff band 0 == -128. This is unambiguous
    because _quantize clips to [-127, 127], so -128 can only come from fill value.

    Returns (bands, H//factor, W//factor) int8 array.
    """
    bands, h, w = data.shape
    oh, ow = h // factor, w // factor

    blocks = data[:, :oh * factor, :ow * factor].reshape(bands, oh, factor, ow, factor)

    # Nodata: band 0 == -128 is unambiguous (quantize clips to [-127, 127])
    pixel_nodata = blocks[0] == NODATA
    valid_count = (~pixel_nodata).sum(axis=(1, 3))
    all_nodata = valid_count == 0

    fblocks = _dequantize(blocks)

    nodata_expanded = np.broadcast_to(pixel_nodata[np.newaxis], fblocks.shape)
    fblocks = np.where(nodata_expanded, 0.0, fblocks)

    block_sum = fblocks.sum(axis=(2, 4))

    safe_count = np.where(all_nodata, 1, valid_count).astype(np.float32)
    block_mean = block_sum / safe_count[np.newaxis]

    # Use epsilon threshold (not exact == 0) to catch near-cancellation
    norms = np.linalg.norm(block_mean, axis=0, keepdims=True)
    norms[norms < 1e-7] = 1.0
    block_norm = block_mean / norms

    result = _quantize(block_norm)
    result[:, all_nodata] = NODATA

    return result

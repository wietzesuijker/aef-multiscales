import numpy as np
from aef_multiscales.aggregate import aggregate_shard

NODATA = -128
BANDS = 64


def _dequantize(x: np.ndarray) -> np.ndarray:
    f = x.astype(np.float32)
    return (f / 127.5) ** 2 * np.sign(f)


def _reference_aggregate(data: np.ndarray, factor: int) -> np.ndarray:
    """Brute-force float64 reference implementation with explicit loops."""
    bands, h, w = data.shape
    oh, ow = h // factor, w // factor
    out = np.full((bands, oh, ow), NODATA, dtype=np.int8)

    for iy in range(oh):
        for ix in range(ow):
            block = data[:, iy * factor:(iy + 1) * factor, ix * factor:(ix + 1) * factor]
            valid = block[0] != NODATA
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


def test_aggregate_matches_reference(synthetic_shard):
    result = aggregate_shard(synthetic_shard, factor=128)
    reference = _reference_aggregate(synthetic_shard, factor=128)

    assert result.shape == reference.shape == (BANDS, 4, 4)

    valid = result[0] != NODATA
    assert valid.any(), "Should have valid output pixels"

    result_f = _dequantize(result[:, valid].T).astype(np.float64)
    ref_f = _dequantize(reference[:, valid].T).astype(np.float64)

    # Normalize to unit vectors before computing cosine similarity
    result_f = result_f / np.linalg.norm(result_f, axis=1, keepdims=True)
    ref_f = ref_f / np.linalg.norm(ref_f, axis=1, keepdims=True)

    dots = np.sum(result_f * ref_f, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dots))

    assert angles_deg.max() < 1.0, f"Max angular error {angles_deg.max():.2f} >= 1.0"
    assert angles_deg.mean() < 0.5, f"Mean angular error {angles_deg.mean():.2f} >= 0.5"


def test_aggregate_all_nodata(all_nodata_block):
    data = np.full((BANDS, 128, 128), NODATA, dtype=np.int8)
    result = aggregate_shard(data, factor=128)
    assert result.shape == (BANDS, 1, 1)
    assert (result == NODATA).all()


def test_aggregate_mixed_nodata_block():
    """A block with some valid and some nodata pixels produces a valid output."""
    rng = np.random.default_rng(99)
    data = np.full((BANDS, 128, 128), NODATA, dtype=np.int8)
    # Fill the left half with valid embeddings
    raw = rng.standard_normal((BANDS, 128, 64)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    raw = raw / norms
    quantized = np.sign(raw) * np.sqrt(np.abs(raw)) * 127.5
    data[:, :, :64] = np.clip(np.round(quantized), -127, 127).astype(np.int8)

    result = aggregate_shard(data, factor=128)
    assert result.shape == (BANDS, 1, 1)
    # Half valid pixels -> valid output (not nodata)
    assert result[0, 0, 0] != NODATA


def test_aggregate_nodata_output_for_full_nodata_block():
    """A block that is entirely nodata produces nodata output."""
    data = np.full((BANDS, 128, 128), NODATA, dtype=np.int8)
    result = aggregate_shard(data, factor=128)
    assert (result == NODATA).all()


def test_aggregate_output_dtype(synthetic_shard):
    result = aggregate_shard(synthetic_shard, factor=128)
    assert result.dtype == np.int8
    assert result.min() >= NODATA
    assert result[result != NODATA].max() <= 127
    assert result[result != NODATA].min() >= -127

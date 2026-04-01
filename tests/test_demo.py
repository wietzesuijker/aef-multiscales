"""Walkthrough tests using small, human-readable data.

Each test is small enough to verify by hand and demonstrates a specific
aspect of the aggregation pipeline.
"""

import numpy as np

from aef_multiscales.aggregate import aggregate_shard, _dequantize, _quantize, NODATA


# ---------------------------------------------------------------------------
# 1. Nonlinear quantization makes naive averaging incorrect
# ---------------------------------------------------------------------------

def test_naive_averaging_gives_wrong_answer():
    """Demonstrate why you can't just average the raw int8 values.

    The AEF mosaic uses signed-square quantization:
        float = (int8 / 127.5)^2 * sign(int8)

    This is nonlinear: small int8 values are spread apart, large values
    are packed together. Averaging in int8 space gives a biased result.

    Example: int8 values 0 and 127.
      Naive int8 average:  (0 + 127) / 2 = 63  -> decodes to 0.24
      Correct float average: decode(0)=0.0, decode(127)=1.0 -> mean=0.5

    That's a 2x error on a single pair.
    """
    # Two pixels: one near zero, one near max
    pixel_a = np.int8(0)    # decodes to 0.0
    pixel_b = np.int8(127)  # decodes to ~1.0

    # Naive: average the raw bytes, then decode
    naive_avg = np.int8((int(pixel_a) + int(pixel_b)) // 2)  # = 63
    naive_decoded = _dequantize(np.array([naive_avg]))[0]     # ≈ 0.24

    # Correct: decode first, then average
    decoded_a = _dequantize(np.array([pixel_a]))[0]  # = 0.0
    decoded_b = _dequantize(np.array([pixel_b]))[0]  # ≈ 0.993
    correct_avg = (decoded_a + decoded_b) / 2         # ≈ 0.497

    # The naive approach underestimates by ~2x
    assert abs(naive_decoded - 0.24) < 0.02, f"Naive should decode to ~0.24, got {naive_decoded}"
    assert abs(correct_avg - 0.50) < 0.02, f"Correct should be ~0.50, got {correct_avg}"
    assert correct_avg > naive_decoded * 1.5, (
        f"Correct ({correct_avg:.3f}) should be much larger than naive ({naive_decoded:.3f})"
    )


# ---------------------------------------------------------------------------
# 2. Aggregation on small data (verifiable by hand)
# ---------------------------------------------------------------------------

def test_tiny_mosaic_by_hand():
    """A 4-band, 4x4 pixel mosaic downsampled 2x. Small enough to verify by hand.

    Layout (each cell is a pixel, all 4 bands have the same pattern):

        band 0:  [ 127,  127,   64,   64 ]    Block (0,0): top-left 2x2
                 [ 127,  127,   64,   64 ]    Block (0,1): top-right 2x2
                 [   0,    0, -128, -128 ]    Block (1,0): bottom-left 2x2
                 [   0,    0, -128, -128 ]    Block (1,1): all nodata

    Expected output (2x2):
      (0,0): 4 pixels of int8=127 -> decode -> mean -> normalize -> encode
      (0,1): 4 pixels of int8=64  -> decode -> mean -> normalize -> encode
      (1,0): 4 pixels of int8=0   -> decode -> all bands 0.0 -> near-zero norm
      (1,1): all nodata -> output nodata (-128)
    """
    bands, h, w, factor = 4, 4, 4, 2

    data = np.full((bands, h, w), 0, dtype=np.int8)

    # Block (0,0): strong signal (int8=127 in all bands)
    data[:, 0:2, 0:2] = 127

    # Block (0,1): moderate signal (int8=64 in all bands)
    data[:, 0:2, 2:4] = 64

    # Block (1,0): zero signal (int8=0 in all bands)
    data[:, 2:4, 0:2] = 0

    # Block (1,1): nodata
    data[:, 2:4, 2:4] = NODATA

    result = aggregate_shard(data, factor=factor)
    assert result.shape == (bands, 2, 2)

    # Block (0,0): all bands identical -> after L2-norm, each component = 1/sqrt(4) = 0.5
    # Quantize(0.5) = sign(0.5) * sqrt(0.5) * 127.5 ≈ 90
    assert result[0, 0, 0] == result[1, 0, 0] == result[2, 0, 0] == result[3, 0, 0]
    assert 85 <= result[0, 0, 0] <= 95, f"Expected ~90 for strong uniform signal, got {result[0, 0, 0]}"

    # Block (0,1): same pattern at int8=64, but after decode+normalize, same direction
    # L2-norm of uniform vector is the same regardless of magnitude -> same output
    assert result[0, 0, 1] == result[0, 0, 0], (
        "Uniform vectors at different magnitudes should normalize to the same direction"
    )

    # Block (1,1): all nodata
    assert (result[:, 1, 1] == NODATA).all(), "All-nodata block must produce nodata output"


def test_aggregate_preserves_embedding_direction():
    """When all pixels in a block point the same direction, the overview
    should point the same direction (modulo quantization noise).

    This is the key semantic property: spatial averaging of embeddings
    should preserve the "meaning" (direction in embedding space).
    """
    bands, factor = 8, 4

    # Create a known unit vector
    direction = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)  # axis-aligned
    quantized = _quantize(direction)  # int8 representation

    # Fill a 4x4 block with this vector (all pixels identical)
    data = np.zeros((bands, factor, factor), dtype=np.int8)
    for b in range(bands):
        data[b, :, :] = quantized[b]

    result = aggregate_shard(data, factor=factor)
    assert result.shape == (bands, 1, 1)

    # Decode the result and check it points in the same direction
    result_float = _dequantize(result[:, 0, 0])
    result_dir = result_float / np.linalg.norm(result_float)

    # Should be very close to [1, 0, 0, ...] after normalization
    cosine_sim = np.dot(result_dir, direction)
    assert cosine_sim > 0.99, f"Direction not preserved: cosine similarity = {cosine_sim:.4f}"


# ---------------------------------------------------------------------------
# 3. Nodata handling
# ---------------------------------------------------------------------------

def test_nodata_pixels_excluded_from_average():
    """Nodata pixels should not contribute to the average.

    If a 2x2 block has 3 nodata pixels and 1 valid pixel, the output
    should equal the valid pixel (not be diluted by zeros from nodata).
    """
    bands, factor = 4, 2

    data = np.full((bands, factor, factor), NODATA, dtype=np.int8)

    # One valid pixel at (0,0) with a known embedding
    valid_vec = np.array([127, 64, -64, 0], dtype=np.int8)
    data[:, 0, 0] = valid_vec

    result = aggregate_shard(data, factor=factor)
    assert result.shape == (bands, 1, 1)

    # The output should match the single valid pixel (after normalize + requantize)
    decoded = _dequantize(valid_vec.astype(np.float32))
    normalized = decoded / np.linalg.norm(decoded)
    expected = _quantize(normalized)

    assert (result[:, 0, 0] == expected).all(), (
        f"Single valid pixel should pass through: expected {expected}, got {result[:, 0, 0]}"
    )


def test_fully_nodata_block_stays_nodata():
    """A block where every pixel is nodata must produce nodata output."""
    data = np.full((4, 2, 2), NODATA, dtype=np.int8)
    result = aggregate_shard(data, factor=2)
    assert (result == NODATA).all()


# ---------------------------------------------------------------------------
# 4. Quantization roundtrip error budget
# ---------------------------------------------------------------------------

def test_quantization_roundtrip_error():
    """The signed-square quantization roundtrip introduces ~0.7° angular error.

    This test verifies that encoding -> decoding a unit vector gives back
    approximately the same direction.
    """
    rng = np.random.default_rng(42)

    # Generate 100 random unit vectors with 64 dimensions (like AEF embeddings)
    vecs = rng.standard_normal((100, 64)).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    # Roundtrip: float -> int8 -> float
    quantized = np.array([_quantize(v) for v in vecs])
    decoded = np.array([_dequantize(q) for q in quantized]).astype(np.float64)
    decoded = decoded / np.linalg.norm(decoded, axis=1, keepdims=True)

    # Angular error
    dots = np.clip(np.sum(vecs.astype(np.float64) * decoded, axis=1), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dots))

    assert angles_deg.mean() < 1.0, f"Mean roundtrip error {angles_deg.mean():.2f}° should be < 1°"
    assert angles_deg.max() < 2.0, f"Max roundtrip error {angles_deg.max():.2f}° should be < 2°"

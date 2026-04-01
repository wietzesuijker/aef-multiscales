import numpy as np
import pytest

NODATA = -128
FACTOR = 128
SHARD_SIZE = 512
BANDS = 64


def _dequantize(x: np.ndarray) -> np.ndarray:
    """int8 -> float32 via signed-square quantization."""
    f = x.astype(np.float32)
    return (f / 127.5) ** 2 * np.sign(f)


def _quantize(x: np.ndarray) -> np.ndarray:
    """float32 -> int8 via inverse of signed-square quantization."""
    raw = np.sign(x) * np.sqrt(np.abs(x)) * 127.5
    return np.clip(np.round(raw), -127, 127).astype(np.int8)


@pytest.fixture
def synthetic_shard():
    """(64, 512, 512) int8 shard with known embedding vectors and some nodata."""
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((BANDS, SHARD_SIZE, SHARD_SIZE)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    raw = raw / norms
    data = _quantize(raw)
    nodata_mask = rng.random((SHARD_SIZE, SHARD_SIZE)) < 0.05
    data[:, nodata_mask] = NODATA
    return data


@pytest.fixture
def all_nodata_block():
    """(64, 128, 128) block where every pixel is nodata."""
    return np.full((BANDS, FACTOR, FACTOR), NODATA, dtype=np.int8)

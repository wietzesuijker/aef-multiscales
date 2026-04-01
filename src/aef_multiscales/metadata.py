"""Generate multiscales convention metadata for the overview store."""

import zarr

from aef_multiscales.naming import array_name

PIXEL_SIZE = 0.00009
BBOX = (-180.0, -83.36, 180.22, 83.69)
FULL_SHAPE_YX = (1859584, 4009984)

MULTISCALES_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v1/README.md",
    "uuid": "d35379db-88df-4056-af3a-620245f8e347",
    "name": "multiscales",
    "description": "Multiscale layout of zarr datasets",
}


def _layout_entry(
    asset: str,
    factor: int,
    derived_from: str | None = None,
    relative_scale: float = 1.0,
) -> dict:
    """Build one multiscales layout entry."""
    h, w = FULL_SHAPE_YX
    oh, ow = (h // factor, w // factor) if factor > 1 else (h, w)
    pixel = PIXEL_SIZE * factor
    lon_min, _, _, lat_max = BBOX

    entry: dict = {
        "asset": asset,
        "transform": {
            "scale": [relative_scale, relative_scale],
            "translation": [0.0, 0.0],
        },
        "spatial:transform": [pixel, 0.0, lon_min, 0.0, -pixel, lat_max],
        "spatial:shape": [oh, ow],
    }
    if derived_from is not None:
        entry["derived_from"] = derived_from
        entry["resampling_method"] = "mean"
    return entry


def build_multiscales_metadata(factors: list[int]) -> dict:
    """Build the multiscales attribute for the root group."""
    layout = [_layout_entry("embeddings", factor=1)]

    prev_asset = "embeddings"
    prev_factor = 1
    for f in sorted(factors):
        asset = array_name(f)
        relative_scale = f / prev_factor
        layout.append(
            _layout_entry(asset, f, derived_from=prev_asset, relative_scale=relative_scale)
        )
        prev_asset = asset
        prev_factor = f

    return {
        "layout": layout,
        "resampling_method": "mean",
    }


def update_root_metadata(group: zarr.Group, factors: list[int]) -> None:
    """Add multiscales convention and layout to root group attributes."""
    attrs = dict(group.attrs)

    conventions = list(attrs.get("zarr_conventions", []))
    if not any(c.get("name") == "multiscales" for c in conventions):
        conventions.append(MULTISCALES_CONVENTION)
    attrs["zarr_conventions"] = conventions

    attrs["multiscales"] = build_multiscales_metadata(factors)

    group.attrs.update(attrs)

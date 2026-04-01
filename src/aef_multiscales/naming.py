"""Factor -> resolution label and array name conventions."""

_LABELS: dict[int, str] = {
    4: "40m",
    16: "160m",
    128: "1km",
    1024: "10km",
}


def resolution_label(factor: int) -> str:
    return _LABELS.get(factor, f"{factor}x")


def array_name(factor: int) -> str:
    return f"embeddings_{resolution_label(factor)}"


def coord_names(factor: int) -> tuple[str, str]:
    label = resolution_label(factor)
    return f"x_{label}", f"y_{label}"

from aef_multiscales.naming import array_name, coord_names, resolution_label


def test_resolution_label_128():
    assert resolution_label(128) == "1km"


def test_resolution_label_16():
    assert resolution_label(16) == "160m"


def test_resolution_label_1024():
    assert resolution_label(1024) == "10km"


def test_resolution_label_unknown():
    assert resolution_label(7) == "7x"


def test_array_name():
    assert array_name(128) == "embeddings_1km"
    assert array_name(16) == "embeddings_160m"


def test_coord_names():
    assert coord_names(128) == ("x_1km", "y_1km")

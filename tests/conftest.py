import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def dice_image():
    return np.array(Image.open("assets/demo_cube.png"))


@pytest.fixture
def equirec_image():
    return np.array(Image.open("assets/demo_equirec.png"))


@pytest.fixture
def horizon_image():
    return np.array(Image.open("assets/demo_horizon.png"))


@pytest.fixture
def list_image(horizon_image):
    return [
        horizon_image[:, 0:256],
        horizon_image[:, 256:512],
        horizon_image[:, 512:768],
        horizon_image[:, 768:1024],
        horizon_image[:, 1024:1280],
        horizon_image[:, 1280:1536],
    ]


@pytest.fixture
def dict_image(list_image):
    return {
        "F": list_image[0],
        "R": list_image[1],
        "B": list_image[2],
        "L": list_image[3],
        "U": list_image[4],
        "D": list_image[5],
    }

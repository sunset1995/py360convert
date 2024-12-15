import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def dice_image():
    return np.array(Image.open("assets/demo_cube.png"))

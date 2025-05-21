import numpy as np
import pytest
from scipy.ndimage import map_coordinates as scipy_map_coordinates

import py360convert
from py360convert.utils import map_coordinates


def test_dice2h_h2dice(dice_image):
    cube_h = py360convert.cube_dice2h(dice_image)  # the inverse is cube_h2dice
    assert cube_h.dtype == np.uint8
    assert cube_h.shape == (256, 1536, 3)

    dice_actual = py360convert.cube_h2dice(cube_h)

    # Round trip should result in the same image.
    np.testing.assert_allclose(dice_image, dice_actual)


@pytest.mark.parametrize("order", [0, 1, 3])
def test_map_coordinates_vs_scipy(order):
    if order == 0:
        img = np.arange(16).reshape(4, 4)
        coords = np.array([[1.2, 2.7], [0.4, 3.6]])  # y, x
        result = map_coordinates(img, coords, order=order)
        expected = scipy_map_coordinates(img, coords, order=order, mode="nearest")
        np.testing.assert_allclose(result, expected)
    else:
        img = np.arange(16).reshape(4, 4).astype(float)
        coords = np.array([[1.5, 2.2], [0.5, 3.1]])  # y, x
        result = map_coordinates(img, coords, order=order)
        expected = scipy_map_coordinates(img, coords, order=order, mode="nearest")
        rtol = 1e-5 if order == 1 else 1e-4
        atol = 1e-5 if order == 1 else 1e-4
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("order", [0, 1, 3])
def test_map_coordinates_cube_faces_vs_scipy(order):
    # 6 faces, 4x4 pixels
    img = np.arange(6 * 4 * 4).reshape(6, 4, 4).astype(float)
    tp = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([1.2, 2.7, 0.5, 3.1, 2.2, 1.5])
    x = np.array([0.4, 3.6, 2.1, 1.9, 0.0, 3.0])
    coords = (tp, y, x)
    result = map_coordinates(img, coords, order=order)

    # 각 face별로 2D 슬라이스와 좌표를 꺼내서 scipy와 비교
    for i in range(len(tp)):
        face = int(tp[i])
        y_i = np.array([y[i]])
        x_i = np.array([x[i]])
        expected = scipy_map_coordinates(img[face], np.vstack([y_i, x_i]), order=order, mode="nearest")
        np.testing.assert_allclose(result[i], expected, rtol=1e-4, atol=1e-4)

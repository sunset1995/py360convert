import numpy as np

import py360convert


def test_dice2h_h2dice(dice_image):
    cube_h = py360convert.cube_dice2h(dice_image)  # the inverse is cube_h2dice
    assert cube_h.dtype == np.uint8
    assert cube_h.shape == (256, 1536, 3)

    dice_actual = py360convert.cube_h2dice(cube_h)

    # Round trip should result in the same image.
    np.testing.assert_allclose(dice_image, dice_actual)

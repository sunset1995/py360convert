import numpy as np

from . import utils


def c2e(cubemap, h, w, cube_format='horizon', k=4):
    if cube_format == 'list':
        cubemap = utils.cube_list2h(cubemap)
    elif cube_format == 'dict':
        cubemap = utils.cube_dict2h(cubemap)
    elif cube_format == 'dice':
        cubemap = utils.cube_dice2h(cubemap)
    assert len(cubemap.shape) == 3
    assert cubemap.shape[0] * 6 == cubemap.shape[1]
    k = k if k > 1 else [1]
    face_w = cubemap.shape[0]
    channel = cubemap.shape[2]

    uv = utils.equirect_uvgrid(h, w)
    u, v = np.split(uv, 2, axis=-1)

    # Assign face id to each pixel
    # 0F 1R 2B 3L 4U 5D
    deg45 = np.pi / 4
    nonUD = (-deg45 < v) & (v < deg45)
    tp = np.zeros((h, w, 1), dtype=np.int32) - 1
    tp[(-deg45 < u) & (u < deg45) & nonUD] = 0  # Front face
    tp[(-deg45 < u) & (u < deg45) & nonUD] = 1
    tp[~nonUD & (v > 0)] = 4
    tp[~nonUD & (v < 0)] = 5

    from PIL import Image
    color = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [0, 0, 0]
    ], np.uint8)
    cb = color[tp][:, :, 0, :]
    print(cb.shape)
    Image.fromarray(cb).save('assert/demo.jpg', 'JPEG', quality=80)


import numpy as np

from . import utils


def c2e(cubemap, h, w, mode="bilinear", cube_format="dice"):
    if mode == "bilinear":
        order = 1
    elif mode == "nearest":
        order = 0
    else:
        raise ValueError(f'Unknown mode "{mode}".')

    if cube_format == "horizon":
        pass
    elif cube_format == "list":
        cubemap = utils.cube_list2h(cubemap)
    elif cube_format == "dict":
        cubemap = utils.cube_dict2h(cubemap)
    elif cube_format == "dice":
        cubemap = utils.cube_dice2h(cubemap)
    else:
        raise ValueError('Unknown cube_format "{cube_format}".')

    if cubemap.ndim != 3:
        raise ValueError(f"Cubemap must have 3 dimensions; got {cubemap.ndim}.")
    if cubemap.shape[0] * 6 != cubemap.shape[1]:
        raise ValueError("Cubemap's width must by 6x it's height.")
    if w % 8 != 0:
        raise ValueError("w must be a multiple of 8.")
    face_w = cubemap.shape[0]

    uv = utils.equirect_uvgrid(h, w)
    u, v = np.split(uv, 2, axis=-1)
    u = u[..., 0]
    v = v[..., 0]
    cube_faces = np.stack(np.split(cubemap, 6, 1), 0)

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = utils.equirect_facetype(h, w)
    coor_x = np.zeros((h, w))
    coor_y = np.zeros((h, w))

    for i in range(4):
        mask = tp == i
        coor_x[mask] = 0.5 * np.tan(u[mask] - np.pi * i / 2)
        coor_y[mask] = -0.5 * np.tan(v[mask]) / np.cos(u[mask] - np.pi * i / 2)

    mask = tp == 4
    c = 0.5 * np.tan(np.pi / 2 - v[mask])
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = c * np.cos(u[mask])

    mask = tp == 5
    c = 0.5 * np.tan(np.pi / 2 - np.abs(v[mask]))
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = -c * np.cos(u[mask])

    # Final renormalize
    coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
    coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

    equirec = np.stack(
        [
            utils.sample_cubefaces(cube_faces[..., i], tp, coor_y, coor_x, order=order)
            for i in range(cube_faces.shape[3])
        ],
        axis=-1,
    )

    return equirec

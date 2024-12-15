from typing import Literal, Union, overload

import numpy as np
from numpy.typing import NDArray

from .utils import (
    CubeFormat,
    DType,
    InterpolationMode,
    cube_dice2h,
    cube_dict2h,
    cube_list2h,
    equirect_facetype,
    equirect_uvgrid,
    sample_cubefaces,
)


@overload
def c2e(
    cubemap: NDArray[DType],
    h: int,
    w: int,
    mode: InterpolationMode = "bilinear",
    cube_format: Literal["horizon", "dice"] = "dice",
) -> NDArray[DType]: ...


@overload
def c2e(
    cubemap: list[NDArray[DType]],
    h: int,
    w: int,
    mode: InterpolationMode = "bilinear",
    cube_format: Literal["list"] = "list",
) -> NDArray[DType]: ...


@overload
def c2e(
    cubemap: dict[str, NDArray[DType]],
    h: int,
    w: int,
    mode: InterpolationMode = "bilinear",
    cube_format: Literal["dict"] = "dict",
) -> NDArray[DType]: ...


def c2e(
    cubemap: Union[NDArray[DType], list[NDArray[DType]], dict[str, NDArray[DType]]],
    h: int,
    w: int,
    mode: InterpolationMode = "bilinear",
    cube_format: CubeFormat = "dice",
) -> NDArray:
    """Convert the cubemap to equirectangular.

    Parameters
    ----------
    cubemap: Union[NDArray, list[NDArray], dict[str, NDArray]]
    h: int
        Output equirectangular height.
    w: int
        Output equirectangular width.
    mode: Literal["bilinear", "nearest"]
        Interpolation mode.
    cube_format: Literal["horizon", "list", "dict", "dice"]
        Format of input cubemap.

    Returns
    -------
    np.ndarray
        Equirectangular image.
    """
    if mode == "bilinear":
        order = 1
    elif mode == "nearest":
        order = 0
    else:
        raise ValueError(f'Unknown mode "{mode}".')

    if cube_format == "horizon":
        if not isinstance(cubemap, np.ndarray):
            raise TypeError('cubemap must be a numpy array for cube_format="horizon"')
    elif cube_format == "list":
        if not isinstance(cubemap, list):
            raise TypeError('cubemap must be a list for cube_format="list"')
        cubemap = cube_list2h(cubemap)
    elif cube_format == "dict":
        if not isinstance(cubemap, dict):
            raise TypeError('cubemap must be a dict for cube_format="dict"')
        cubemap = cube_dict2h(cubemap)
    elif cube_format == "dice":
        if not isinstance(cubemap, np.ndarray):
            raise TypeError('cubemap must be a numpy array for cube_format="dice"')
        cubemap = cube_dice2h(cubemap)
    else:
        raise ValueError('Unknown cube_format "{cube_format}".')

    if cubemap.ndim not in (2, 3):
        raise ValueError(f"Cubemap must have 2 or 3 dimensions; got {cubemap.ndim}.")

    if cubemap.ndim == 2:
        cubemap = cubemap[..., None]
        squeeze = True
    else:
        squeeze = False

    if cubemap.shape[0] * 6 != cubemap.shape[1]:
        raise ValueError("Cubemap's width must by 6x its height.")
    if w % 8 != 0:
        raise ValueError("w must be a multiple of 8.")
    face_w = cubemap.shape[0]

    uv = equirect_uvgrid(h, w)
    u, v = np.split(uv, 2, axis=-1)
    u = u[..., 0]
    v = v[..., 0]
    cube_faces = np.stack(np.split(cubemap, 6, 1), 0)

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = equirect_facetype(h, w)
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
    coor_x_norm = (np.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
    coor_y_norm = (np.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

    equirec = np.stack(
        [
            sample_cubefaces(cube_faces[..., i], tp, coor_y_norm, coor_x_norm, order=order)
            for i in range(cube_faces.shape[3])
        ],
        axis=-1,
    )

    return equirec[..., 0] if squeeze else equirec

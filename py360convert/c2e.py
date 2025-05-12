from typing import Literal, Union, overload

import numpy as np
from numpy.typing import NDArray

from .utils import (
    CubeFaceSampler,
    CubeFormat,
    DType,
    InterpolationMode,
    cube_dice2list,
    cube_dict2list,
    cube_h2list,
    mode_to_order,
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
    order = mode_to_order(mode)
    if w % 8 != 0:
        raise ValueError("w must be a multiple of 8.")

    if cube_format == "horizon":
        if not isinstance(cubemap, np.ndarray):
            raise TypeError('cubemap must be a numpy array for cube_format="horizon"')
        if cubemap.ndim == 2:
            cubemap = cubemap[..., None]
            squeeze = True
        else:
            squeeze = False
        cube_faces = cube_h2list(cubemap)
    elif cube_format == "list":
        if not isinstance(cubemap, list):
            raise TypeError('cubemap must be a list for cube_format="list"')
        if len({x.shape for x in cubemap}) != 1:
            raise ValueError("All cubemap elements must have same shape")
        if cubemap[0].ndim == 2:
            cube_faces = [x[..., None] for x in cubemap]
            squeeze = True
        else:
            cube_faces = cubemap
            squeeze = False
    elif cube_format == "dict":
        if not isinstance(cubemap, dict):
            raise TypeError('cubemap must be a dict for cube_format="dict"')
        if len({x.shape for x in cubemap.values()}) != 1:
            raise ValueError("All cubemap elements must have same shape")
        if cubemap["F"].ndim == 2:
            cubemap = {k: v[..., None] for k, v in cubemap.items()}
            squeeze = True
        else:
            squeeze = False
        cube_faces = cube_dict2list(cubemap)
    elif cube_format == "dice":
        if not isinstance(cubemap, np.ndarray):
            raise TypeError('cubemap must be a numpy array for cube_format="dice"')
        if cubemap.ndim == 2:
            cubemap = cubemap[..., None]
            squeeze = True
        else:
            squeeze = False
        cube_faces = cube_dice2list(cubemap)
    else:
        raise ValueError(f'Unknown cube_format "{cube_format}".')

    cube_faces = np.stack(cube_faces)

    if cube_faces.shape[1] != cube_faces.shape[2]:
        raise ValueError("Cubemap faces must be square.")
    face_w = cube_faces.shape[2]

    sampler = CubeFaceSampler.from_equirec(face_w, h, w, order)

    equirec = np.empty((h, w, cube_faces.shape[3]), dtype=cube_faces[0].dtype)
    for i in range(cube_faces.shape[3]):
        equirec[..., i] = sampler(cube_faces[..., i])

    return equirec[..., 0] if squeeze else equirec

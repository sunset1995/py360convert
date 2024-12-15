from typing import Literal, Union, overload

import numpy as np
from numpy.typing import NDArray

from .utils import (
    CubeFormat,
    DType,
    InterpolationMode,
    cube_h2dice,
    cube_h2dict,
    cube_h2list,
    sample_equirec,
    uv2coor,
    xyz2uv,
    xyzcube,
)


@overload
def e2c(
    e_img: NDArray[DType],
    face_w: int,
    mode: InterpolationMode,
    cube_format: Literal["horizon", "dice"],
) -> NDArray[DType]: ...


@overload
def e2c(
    e_img: NDArray[DType],
    face_w: int,
    mode: InterpolationMode,
    cube_format: Literal["list"],
) -> list[NDArray[DType]]: ...


@overload
def e2c(
    e_img: NDArray[DType],
    face_w: int,
    mode: InterpolationMode,
    cube_format: Literal["dict"],
) -> dict[str, NDArray[DType]]: ...


def e2c(
    e_img: NDArray[DType],
    face_w: int = 256,
    mode: InterpolationMode = "bilinear",
    cube_format: CubeFormat = "dice",
) -> Union[NDArray[DType], list[NDArray[DType]], dict[str, NDArray[DType]]]:
    """Convert equirectangular image to cubemap.

    Parameters
    ----------
    e_img: ndarray
        Equirectangular image in shape of [H, W, *].
    face_w: int
        Length of each face of the cubemap
    mode: Literal["bilinear", "nearest"]
        Interpolation mode.
    cube_format: Literal["horizon", "list", "dict", "dice"]
        Format to return cubemap in.

    Returns
    -------
    Union[NDArray, list[NDArray], dict[str, NDArray]]
        Cubemap in format specified by `cube_format`.
    """
    if e_img.ndim != 3:
        raise ValueError("e_img must have 3 dimensions.")
    h, w = e_img.shape[:2]
    if mode == "bilinear":
        order = 1
    elif mode == "nearest":
        order = 0
    else:
        raise ValueError(f'Unknown mode: "{mode}".')

    xyz = xyzcube(face_w)
    uv = xyz2uv(xyz)
    coor_xy = uv2coor(uv, h, w)

    cubemap = np.stack(
        [sample_equirec(e_img[..., i], coor_xy, order=order) for i in range(e_img.shape[2])],
        axis=-1,
        dtype=e_img.dtype,
    )

    if cube_format == "horizon":
        pass
    elif cube_format == "list":
        cubemap = cube_h2list(cubemap)
    elif cube_format == "dict":
        cubemap = cube_h2dict(cubemap)
    elif cube_format == "dice":
        cubemap = cube_h2dice(cubemap)
    else:
        raise NotImplementedError

    return cubemap

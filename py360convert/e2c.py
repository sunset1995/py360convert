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
    mode_to_order,
    sample_equirec,
    uv2coor,
    xyz2uv,
    xyzcube,
)


@overload
def e2c(  # pyright: ignore[reportOverlappingOverload]
    e_img: NDArray[DType],
    face_w: int = 256,
    mode: InterpolationMode = "bilinear",
    cube_format: Literal["horizon", "dice"] = "dice",
) -> NDArray[DType]: ...


@overload
def e2c(
    e_img: NDArray[DType],
    face_w: int = 256,
    mode: InterpolationMode = "bilinear",
    cube_format: Literal["list"] = "list",
) -> list[NDArray[DType]]: ...


@overload
def e2c(
    e_img: NDArray[DType],
    face_w: int = 256,
    mode: InterpolationMode = "bilinear",
    cube_format: Literal["dict"] = "dict",
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
        Equirectangular image in shape of [H,W] or [H, W, *].
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
    if e_img.ndim not in (2, 3):
        raise ValueError("e_img must have 2 or 3 dimensions.")
    if e_img.ndim == 2:
        e_img = e_img[..., None]
        squeeze = True
    else:
        squeeze = False

    h, w = e_img.shape[:2]
    order = mode_to_order(mode)

    xyz = xyzcube(face_w)
    uv = xyz2uv(xyz)
    coor_xy = uv2coor(uv, h, w)

    cubemap = np.stack(
        [sample_equirec(e_img[..., i], coor_xy, order=order) for i in range(e_img.shape[2])],
        axis=-1,
        dtype=e_img.dtype,
    )

    if cube_format == "horizon":
        if squeeze:
            cubemap = cubemap[..., 0]
    elif cube_format == "list":
        cubemap = cube_h2list(cubemap)
        if squeeze:
            cubemap = [x[..., 0] for x in cubemap]
    elif cube_format == "dict":
        cubemap = cube_h2dict(cubemap)
        if squeeze:
            cubemap = {k: v[..., 0] for k, v in cubemap.items()}
    elif cube_format == "dice":
        cubemap = cube_h2dice(cubemap)
        if squeeze:
            cubemap = cubemap[..., 0]
    else:
        raise NotImplementedError

    return cubemap

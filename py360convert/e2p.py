from numbers import Real
from typing import Union

import numpy as np
from numpy.typing import NDArray

from .utils import (
    DType,
    InterpolationMode,
    sample_equirec,
    uv2coor,
    xyz2uv,
    xyzpers,
)


def e2p(
    e_img: NDArray[DType],
    fov_deg: Union[Real, tuple[float, float]],
    u_deg: float,
    v_deg: float,
    out_hw: tuple[int, int],
    in_rot_deg: float = 0,
    mode: InterpolationMode = "bilinear",
) -> NDArray[DType]:
    """Convert equirectangular image to perspective.

    Parameters
    ----------
    e_img: ndarray
        Equirectangular image in shape of [H, W, *].
    fov_deg: scalar or (scalar, scalar) field of view in degree
        Field of view given in float or tuple (h_fov_deg, v_fov_deg).
    u_deg:   horizon viewing angle in range [-180, 180]
        Horizontal viewing angle in range [-pi, pi]. (- Left / + Right).
    v_deg:   vertical viewing angle in range [-90, 90]
        Vertical viewing angle in range [-pi/2, pi/2]. (- Down/ + Up).
    out_hw: tuple[int, int]
        Size of output perspective image.
    in_rot_deg: float
        Inplane rotation.
    mode: Literal["bilinear", "nearest"]
        Interpolation mode.

    Returns
    -------
    np.ndarray
        Perspective image.
    """
    if e_img.ndim != 3:
        raise ValueError("e_img must have 3 dimensions.")
    h, w = e_img.shape[:2]

    if isinstance(fov_deg, Real):
        h_fov = v_fov = np.deg2rad(float(fov_deg))
    else:
        h_fov, v_fov = map(np.deg2rad, fov_deg)

    in_rot = in_rot_deg * np.pi / 180

    if mode == "bilinear":
        order = 1
    elif mode == "nearest":
        order = 0
    else:
        raise ValueError(f'Unknown mode: "{mode}".')

    u = -u_deg * np.pi / 180
    v = v_deg * np.pi / 180
    xyz = xyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
    uv = xyz2uv(xyz)
    coor_xy = uv2coor(uv, h, w)

    pers_img = np.stack([sample_equirec(e_img[..., i], coor_xy, order=order) for i in range(e_img.shape[2])], axis=-1)

    return pers_img

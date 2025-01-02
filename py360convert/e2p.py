from numbers import Real
from typing import Union

import numpy as np
from numpy.typing import NDArray

from .utils import (
    DType,
    EquirecSampler,
    InterpolationMode,
    mode_to_order,
)


def e2p(
    e_img: NDArray[DType],
    fov_deg: Union[float, int, tuple[float | int, float | int]],
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
        Equirectangular image in shape of [H,W] or [H, W, *].
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
    if e_img.ndim not in (2, 3):
        raise ValueError("e_img must have 2 or 3 dimensions.")
    if e_img.ndim == 2:
        e_img = e_img[..., None]
        squeeze = True
    else:
        squeeze = False

    h, w = e_img.shape[:2]

    if isinstance(fov_deg, (int, float, Real)):
        h_fov = v_fov = float(np.deg2rad(float(fov_deg)))
    else:
        h_fov, v_fov = float(np.deg2rad(fov_deg[0])), float(np.deg2rad(fov_deg[1]))

    order = mode_to_order(mode)

    u = -float(np.deg2rad(u_deg))
    v = float(np.deg2rad(v_deg))
    in_rot = float(np.deg2rad(in_rot_deg))
    sampler = EquirecSampler.from_perspective(h_fov, v_fov, u, v, in_rot, h, w, order)
    pers_img = np.stack([sampler(e_img[..., i]) for i in range(e_img.shape[2])], axis=-1)

    return pers_img[..., 0] if squeeze else pers_img

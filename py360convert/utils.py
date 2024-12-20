from collections.abc import Sequence
from typing import Any, Literal, Optional, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

_mode_to_order = {
    "nearest": 0,
    "linear": 1,
    "bilinear": 1,
    "biquadratic": 2,
    "quadratic": 2,
    "quad": 2,
    "bicubic": 3,
    "cubic": 3,
    "biquartic": 4,
    "quartic": 4,
    "biquintic": 5,
    "quintic": 5,
}

CubeFormat = Literal["horizon", "list", "dict", "dice"]
InterpolationMode = Literal[
    "nearest",
    "linear",
    "bilinear",
    "biquadratic",
    "quadratic",
    "quad",
    "bicubic",
    "cubic",
    "biquartic",
    "quartic",
    "biquintic",
    "quintic",
]
DType = TypeVar("DType", bound=np.generic, covariant=True)


def mode_to_order(mode: InterpolationMode) -> int:
    """Convert a human-friendly interpolation string to integer equivalent.

    Parameters
    ----------
    mode: str
        Human-friendly interpolation string.

    Returns
    -------
    The order of the spline interpolation
    """
    try:
        return _mode_to_order[mode.lower()]
    except KeyError:
        raise ValueError(f'Unknown mode "{mode}".') from None


def xyzcube(face_w: int) -> NDArray[np.float32]:
    """
    Return the xyz coordinates of the unit cube in [F R B L U D] format.

    Parameters
    ----------
        face_w: int
            Specify the length of each face of the cubemap.

    Returns
    -------
        out: ndarray
            An array object with dimension (face_w, face_w * 6, 3)
            which store the each face of numalized cube coordinates.
            The cube is centered at the origin so that each face k
            in out has range [-0.5, 0.5] x [-0.5, 0.5].

    """
    out = np.zeros((face_w, face_w * 6, 3), np.float32)
    rng = np.linspace(-0.5, 0.5, num=face_w, dtype=np.float32)
    grid = np.stack(np.meshgrid(rng, -rng), -1)

    # Front face (z = 0.5)
    out[:, 0 * face_w : 1 * face_w, [0, 1]] = grid
    out[:, 0 * face_w : 1 * face_w, 2] = 0.5

    # Right face (x = 0.5)
    out[:, 1 * face_w : 2 * face_w, [2, 1]] = grid
    out[:, 1 * face_w : 2 * face_w, [2, 1]] = np.flip(grid, axis=1)
    out[:, 1 * face_w : 2 * face_w, 0] = 0.5

    # Back face (z = -0.5)
    out[:, 2 * face_w : 3 * face_w, [0, 1]] = np.flip(grid, axis=1)
    out[:, 2 * face_w : 3 * face_w, 2] = -0.5

    # Left face (x = -0.5)
    out[:, 3 * face_w : 4 * face_w, [2, 1]] = grid
    out[:, 3 * face_w : 4 * face_w, 0] = -0.5

    # Up face (y = 0.5)
    out[:, 4 * face_w : 5 * face_w, [0, 2]] = np.flip(grid, axis=0)
    out[:, 4 * face_w : 5 * face_w, 1] = 0.5

    # Down face (y = -0.5)
    out[:, 5 * face_w : 6 * face_w, [0, 2]] = grid
    out[:, 5 * face_w : 6 * face_w, 1] = -0.5

    return out


def equirect_uvgrid(h: int, w: int) -> NDArray[np.float32]:
    u = np.linspace(-np.pi, np.pi, num=w, dtype=np.float32)
    v = np.linspace(np.pi, -np.pi, num=h, dtype=np.float32) / 2

    return np.stack(np.meshgrid(u, v), axis=-1)


def equirect_facetype(h: int, w: int) -> NDArray[np.int32]:
    """Generate a 2D equirectangular segmentation image for each facetype.

    The generated segmentation image has lookup:

    * 0 - front
    * 1 - right
    * 2 - back
    * 3 - left
    * 4 - up
    * 5 - down

    Example:

        >>> equirect_facetype(8, 12)
            array([[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                   [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                   [2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2],
                   [2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2],
                   [2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2],
                   [2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2],
                   [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                   [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]], dtype=int32)

    Parameters
    ----------
    h: int
        Desired output height.
    w: int
        Desired output width. Must be a multiple of 4.

    Returns
    -------
    ndarray
        2D numpy equirectangular segmentation image for the 6 face types.
    """
    if w % 4:
        raise ValueError(f"w must be a multiple of 4. Got {w}.")

    tp = np.roll(np.arange(4).repeat(w // 4)[None, :].repeat(h, 0), 3 * w // 8, 1)

    # Prepare ceil mask
    mask = np.zeros((h, w // 4), np.bool_)
    idx = np.linspace(-np.pi, np.pi, w // 4) / 4
    idx = np.round(h / 2 - np.arctan(np.cos(idx)) * h / np.pi).astype(np.int32)
    for i, j in enumerate(idx):
        mask[:j, i] = True
    mask = np.roll(np.concatenate([mask] * 4, 1), 3 * w // 8, 1)

    tp[mask] = 4
    tp[np.flip(mask, 0)] = 5

    return tp.astype(np.int32)


def xyzpers(h_fov: float, v_fov: float, u: float, v: float, out_hw: tuple[int, int], in_rot: float) -> NDArray:
    out = np.ones((*out_hw, 3), np.float32)

    x_max = np.tan(h_fov / 2)
    y_max = np.tan(v_fov / 2)
    x_rng = np.linspace(-x_max, x_max, num=out_hw[1], dtype=np.float32)
    y_rng = np.linspace(-y_max, y_max, num=out_hw[0], dtype=np.float32)
    out[..., :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)
    Rx = rotation_matrix(v, [1, 0, 0])
    Ry = rotation_matrix(u, [0, 1, 0])
    Ri = rotation_matrix(in_rot, np.array([0, 0, 1.0]).dot(Rx).dot(Ry))

    return out.dot(Rx).dot(Ry).dot(Ri)


def xyz2uv(xyz: NDArray[DType]) -> NDArray[DType]:
    """Transform cartesian (x,y,z) to spherical(r, u, v), and only outputs (u, v).

    Parameters
    ----------
    xyz: ndarray
        An array object in shape of [..., 3].

    Returns
    -------
    out: ndarray
        An array object in shape of [..., 2],
        any point i of this array is in [-pi, pi].

    Notes
    -----
    In this project, e2c calls utils.xyz2uv(xyz) where:

        * xyz is in [-0.5, 0.5] x [-0.5, 0.5] x [-0.5, 0.5]
        * u is in [-pi, pi]
        * v is in [-pi/2, pi/2]
        * any point i of output array is in [-pi, pi] x [-pi/2, pi/2].
    """
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(np.square(x) + np.square(z))
    v = np.arctan2(y, c)
    out = np.concatenate([u, v], axis=-1, dtype=xyz.dtype)
    return out


def uv2unitxyz(uv: NDArray[DType]) -> NDArray[DType]:
    u, v = np.split(uv, 2, axis=-1)
    y = np.sin(v)
    c = np.cos(v)
    x = c * np.sin(u)
    z = c * np.cos(u)
    return np.concatenate([x, y, z], axis=-1, dtype=uv.dtype)


def uv2coor(uv: NDArray[DType], h: int, w: int) -> NDArray[DType]:
    """Transform spherical(r, u, v) into equirectangular(x, y).

    Assume that u has range 2pi and v has range pi.
    The coordinate of the equirectangular is from (0.5, 0.5) to (h-0.5, w-0.5).

    Parameters
    ----------
    uv: ndarray
        An array object in shape of [..., 2].
    h: int
        Height of the equirectangular image.
    w: int
        Width of the equirectangular image.

    Returns
    -------
    out: ndarray
        An array object in shape of [..., 2].

    Notes
    -----
    In this project, e2c calls utils.uv2coor(uv, h, w) where:

        * uv is in [-pi, pi] x [-pi/2, pi/2]
        * coor_x is in [-0.5, w-0.5]
        * coor_y is in [-0.5, h-0.5]
    """
    u, v = np.split(uv, 2, axis=-1)
    coor_x = (u / (2 * np.pi) + 0.5) * w - 0.5  # pyright: ignore[reportOperatorIssue]
    coor_y = (-v / np.pi + 0.5) * h - 0.5  # pyright: ignore[reportOperatorIssue]
    out = np.concatenate([coor_x, coor_y], axis=-1, dtype=uv.dtype)
    return out


def coor2uv(coorxy: NDArray[DType], h: int, w: int) -> NDArray[DType]:
    coor_x, coor_y = np.split(coorxy, 2, axis=-1)
    u = ((coor_x + 0.5) / w - 0.5) * 2 * np.pi  # pyright: ignore[reportOperatorIssue]
    v = -((coor_y + 0.5) / h - 0.5) * np.pi  # pyright: ignore[reportOperatorIssue]
    return np.concatenate([u, v], axis=-1, dtype=coorxy.dtype)


def sample_equirec(e_img: NDArray[DType], coor_xy: NDArray, order: int) -> NDArray[DType]:
    w = e_img.shape[1]
    coor_x, coor_y = np.split(coor_xy, 2, axis=-1)
    pad_u = np.roll(e_img[[0]], w // 2, 1)
    pad_d = np.roll(e_img[[-1]], w // 2, 1)
    e_img = np.concatenate([e_img, pad_d, pad_u], 0, dtype=e_img.dtype)
    return map_coordinates(e_img, [coor_y, coor_x], order=order, mode="wrap")[..., 0]  # pyright: ignore[reportReturnType]


def sample_cubefaces(
    cube_faces: NDArray[DType], tp: NDArray, coor_y: NDArray, coor_x: NDArray, order: int
) -> NDArray[DType]:
    cube_faces = cube_faces.copy()
    # cube_faces[1] = np.flip(cube_faces[1], 1)
    # cube_faces[2] = np.flip(cube_faces[2], 1)
    # cube_faces[4] = np.flip(cube_faces[4], 0)

    # Pad up down
    pad_ud = np.zeros((6, 2, cube_faces.shape[2]), dtype=cube_faces.dtype)
    pad_ud[0, 0] = cube_faces[5, 0, :]
    pad_ud[0, 1] = cube_faces[4, -1, :]
    pad_ud[1, 0] = cube_faces[5, :, -1]
    pad_ud[1, 1] = cube_faces[4, ::-1, -1]
    pad_ud[2, 0] = cube_faces[5, -1, ::-1]
    pad_ud[2, 1] = cube_faces[4, 0, ::-1]
    pad_ud[3, 0] = cube_faces[5, ::-1, 0]
    pad_ud[3, 1] = cube_faces[4, :, 0]
    pad_ud[4, 0] = cube_faces[0, 0, :]
    pad_ud[4, 1] = cube_faces[2, 0, ::-1]
    pad_ud[5, 0] = cube_faces[2, -1, ::-1]
    pad_ud[5, 1] = cube_faces[0, -1, :]
    cube_faces = np.concatenate([cube_faces, pad_ud], 1, dtype=cube_faces.dtype)

    # Pad left right
    pad_lr = np.zeros((6, cube_faces.shape[1], 2), dtype=cube_faces.dtype)
    pad_lr[0, :, 0] = cube_faces[1, :, 0]
    pad_lr[0, :, 1] = cube_faces[3, :, -1]
    pad_lr[1, :, 0] = cube_faces[2, :, 0]
    pad_lr[1, :, 1] = cube_faces[0, :, -1]
    pad_lr[2, :, 0] = cube_faces[3, :, 0]
    pad_lr[2, :, 1] = cube_faces[1, :, -1]
    pad_lr[3, :, 0] = cube_faces[0, :, 0]
    pad_lr[3, :, 1] = cube_faces[2, :, -1]
    pad_lr[4, 1:-1, 0] = cube_faces[1, 0, ::-1]
    pad_lr[4, 1:-1, 1] = cube_faces[3, 0, :]
    pad_lr[5, 1:-1, 0] = cube_faces[1, -2, :]
    pad_lr[5, 1:-1, 1] = cube_faces[3, -2, ::-1]
    cube_faces = np.concatenate([cube_faces, pad_lr], 2, dtype=cube_faces.dtype)

    return map_coordinates(cube_faces, [tp, coor_y, coor_x], order=order, mode="wrap")  # pyright: ignore[reportReturnType]


def cube_h2list(cube_h: NDArray[DType]) -> list[NDArray[DType]]:
    """Split an image into a list of 6 faces."""
    if cube_h.shape[0] * 6 != cube_h.shape[1]:
        raise ValueError("Cubemap's width must by 6x its height.")
    return np.split(cube_h, 6, axis=1)


def cube_list2h(cube_list: list[NDArray[DType]]) -> NDArray[DType]:
    """Concatenate a list of 6 face images side-by-side."""
    if len(cube_list) != 6:
        raise ValueError(f"6 elements must be provided to construct a cube; got {len(cube_list)}.")
    for i, face in enumerate(cube_list):
        if face.shape != cube_list[0].shape:
            raise ValueError(
                f"Face {i}'s shape {face.shape} doesn't match the first face's shape {cube_list[0].shape}."
            )
        if face.dtype != cube_list[0].dtype:
            raise ValueError(
                f"Face {i}'s dtype {face.dtype} doesn't match the first face's shape {cube_list[0].dtype}."
            )

    return np.concatenate(cube_list, axis=1, dtype=cube_list[0].dtype)


def cube_h2dict(cube_h: NDArray[DType]) -> dict[str, NDArray[DType]]:
    return dict(zip("FRBLUD", cube_h2list(cube_h)))


def cube_dict2h(cube_dict: dict[Any, NDArray[DType]], face_k: Optional[Sequence] = None) -> NDArray[DType]:
    face_k = face_k or "FRBLUD"
    if len(face_k) != 6:
        raise ValueError(f"6 face_k keys must be provided to construct a cube; got {len(face_k)}.")
    return cube_list2h([cube_dict[k] for k in face_k])


def cube_h2dice(cube_h: NDArray[DType]) -> NDArray[DType]:
    if cube_h.shape[0] * 6 != cube_h.shape[1]:
        raise ValueError("Cubemap's width must by 6x its height.")
    w = cube_h.shape[0]
    cube_dice = np.zeros((w * 3, w * 4, cube_h.shape[2]), dtype=cube_h.dtype)
    cube_list = cube_h2list(cube_h)
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        face = cube_list[i]
        cube_dice[sy * w : (sy + 1) * w, sx * w : (sx + 1) * w] = face
    return cube_dice


def cube_dice2h(cube_dice: NDArray[DType]) -> NDArray[DType]:
    if cube_dice.shape[0] % 3 != 0:
        raise ValueError("Dice image height must be a multiple of 3.")
    w = cube_dice.shape[0] // 3
    if cube_dice.shape[1] != w * 4:
        raise ValueError(f'Dice width must be 4 "faces" (4x{w}={4*w}) wide.')
    cube_h = np.zeros((w, w * 6, cube_dice.shape[2]), dtype=cube_dice.dtype)
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        face = cube_dice[sy * w : (sy + 1) * w, sx * w : (sx + 1) * w]
        cube_h[:, i * w : (i + 1) * w] = face
    return cube_h


def rotation_matrix(rad: float, ax: Union[NDArray, Sequence]):
    ax = np.array(ax)
    if ax.shape != (3,):
        raise ValueError(f"ax must be shape (3,); got {ax.shape}")
    ax = ax / np.sqrt((ax**2).sum())
    R = np.diag([np.cos(rad)] * 3)
    R = R + np.outer(ax, ax) * (1.0 - np.cos(rad))

    ax = ax * np.sin(rad)
    R = R + np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])

    return R

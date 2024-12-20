from collections.abc import Sequence
from enum import IntEnum
from typing import Any, Literal, Optional, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

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


class Face(IntEnum):
    """Face type indexing for numpy vectorization."""

    FRONT = 0
    RIGHT = 1
    BACK = 2
    LEFT = 3
    UP = 4
    DOWN = 5


class Dim(IntEnum):
    X = 0
    Y = 1
    Z = 2


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


def slice_chunk(index: int, width: int):
    start = index * width
    return slice(start, start + width)


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

    def face_slice(index):
        return slice_chunk(index, face_w)

    # Front face (z = 0.5)
    out[:, face_slice(Face.FRONT), [Dim.X, Dim.Y]] = grid
    out[:, face_slice(Face.FRONT), Dim.Z] = 0.5

    # Right face (x = 0.5)
    out[:, face_slice(Face.RIGHT), [Dim.Z, Dim.Y]] = np.flip(grid, axis=1)
    out[:, face_slice(Face.RIGHT), Dim.X] = 0.5

    # Back face (z = -0.5)
    out[:, face_slice(Face.BACK), [Dim.X, Dim.Y]] = np.flip(grid, axis=1)
    out[:, face_slice(Face.BACK), Dim.Z] = -0.5

    # Left face (x = -0.5)
    out[:, face_slice(Face.LEFT), [Dim.Z, Dim.Y]] = grid
    out[:, face_slice(Face.LEFT), Dim.X] = -0.5

    # Up face (y = 0.5)
    out[:, face_slice(Face.UP), [Dim.X, Dim.Z]] = np.flip(grid, axis=0)
    out[:, face_slice(Face.UP), Dim.Y] = 0.5

    # Down face (y = -0.5)
    out[:, face_slice(Face.DOWN), [Dim.X, Dim.Z]] = grid
    out[:, face_slice(Face.DOWN), Dim.Y] = -0.5

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

    See ``Face``.

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

    row_idx = np.arange(len(mask))[:, None]
    mask[row_idx < idx[None, :]] = True

    mask = np.roll(np.tile(mask, (1, 4)), 3 * w // 8, 1)

    tp[mask] = Face.UP
    tp[np.flip(mask, 0)] = Face.DOWN

    return tp.astype(np.int32)


def xyzpers(h_fov: float, v_fov: float, u: float, v: float, out_hw: tuple[int, int], in_rot: float) -> NDArray:
    out = np.ones((*out_hw, 3), np.float32)

    x_max = np.tan(h_fov / 2)
    y_max = np.tan(v_fov / 2)
    x_rng = np.linspace(-x_max, x_max, num=out_hw[1], dtype=np.float32)
    y_rng = np.linspace(-y_max, y_max, num=out_hw[0], dtype=np.float32)
    out[..., :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)
    Rx = rotation_matrix(v, Dim.X)
    Ry = rotation_matrix(u, Dim.Y)
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
    """Sample cube faces.

    Parameters
    ----------
    cube_faces: numpy.ndarray
        (6, H, W) Cube faces.
    tp: numpy.ndarray
        (H, W) facetype image from ``equirect_facetype``.
    coor_y: numpy.ndarray
        (H, W) Y coordinates to sample.
    coor_x: numpy.ndarray
        (H, W) X coordinates to sample.
    order: int
        The order of the spline interpolation. See ``scipy.ndimage.map_coordinates``.
    """
    ABOVE = (0, slice(None))
    BELOW = (-1, slice(None))
    LEFT = (slice(None), 0)
    RIGHT = (slice(None), -1)
    padded = np.pad(cube_faces, ((0, 0), (1, 1), (1, 1)), mode="constant")

    # Pad above/below
    padded[Face.FRONT][ABOVE] = padded[Face.UP, -2, :]
    padded[Face.FRONT][BELOW] = padded[Face.DOWN, 1, :]
    padded[Face.RIGHT][ABOVE] = padded[Face.UP, ::-1, -2]
    padded[Face.RIGHT][BELOW] = padded[Face.DOWN, :, -2]
    padded[Face.BACK][ABOVE] = padded[Face.UP, 1, ::-1]
    padded[Face.BACK][BELOW] = padded[Face.DOWN, -2, ::-1]
    padded[Face.LEFT][ABOVE] = padded[Face.UP, :, 1]
    padded[Face.LEFT][BELOW] = padded[Face.DOWN, ::-1, 1]
    padded[Face.UP][ABOVE] = padded[Face.BACK, 1, ::-1]
    padded[Face.UP][BELOW] = padded[Face.FRONT, 1, :]
    padded[Face.DOWN][ABOVE] = padded[Face.FRONT, -2, :]
    padded[Face.DOWN][BELOW] = padded[Face.BACK, -2, ::-1]

    # Pad left/right
    padded[Face.FRONT][LEFT] = padded[Face.LEFT, :, -2]
    padded[Face.FRONT][RIGHT] = padded[Face.RIGHT, :, 1]
    padded[Face.RIGHT][LEFT] = padded[Face.FRONT, :, -2]
    padded[Face.RIGHT][RIGHT] = padded[Face.BACK, :, 1]
    padded[Face.BACK][LEFT] = padded[Face.RIGHT, :, -2]
    padded[Face.BACK][RIGHT] = padded[Face.LEFT, :, 1]
    padded[Face.LEFT][LEFT] = padded[Face.BACK, :, -2]
    padded[Face.LEFT][RIGHT] = padded[Face.FRONT, :, 1]
    padded[Face.UP][LEFT] = padded[Face.LEFT, 1, :]
    padded[Face.UP][RIGHT] = padded[Face.RIGHT, 1, ::-1]
    padded[Face.DOWN][LEFT] = padded[Face.LEFT, -2, ::-1]
    padded[Face.DOWN][RIGHT] = padded[Face.RIGHT, -2, :]

    return map_coordinates(padded, [tp, coor_y + 1, coor_x + 1], order=order)  # pyright: ignore[reportReturnType]


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
    #        ┌────┐
    #        │ U  │
    #   ┌────┼────┼────┬────┐
    #   │ L  │ F  │ R  │ B  │
    #   └────┼────┼────┴────┘
    #        │ D  │
    #        └────┘
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for (sx, sy), face in zip(sxy, cube_list):
        cube_dice[slice_chunk(sy, w), slice_chunk(sx, w)] = face
    return cube_dice


def cube_dice2h(cube_dice: NDArray[DType]) -> NDArray[DType]:
    if cube_dice.shape[0] % 3 != 0:
        raise ValueError("Dice image height must be a multiple of 3.")
    w = cube_dice.shape[0] // 3
    if cube_dice.shape[1] != w * 4:
        raise ValueError(f'Dice width must be 4 "faces" (4x{w}={4*w}) wide.')
    cube_h = np.zeros((w, w * 6, cube_dice.shape[2]), dtype=cube_dice.dtype)
    # Order: F R B L U D
    #        ┌────┐
    #        │ U  │
    #   ┌────┼────┼────┬────┐
    #   │ L  │ F  │ R  │ B  │
    #   └────┼────┼────┴────┘
    #        │ D  │
    #        └────┘
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        cube_h[:, slice_chunk(i, w)] = cube_dice[slice_chunk(sy, w), slice_chunk(sx, w)]
    return cube_h


def rotation_matrix(rad: float, ax: Union[int, NDArray, Sequence]):
    if isinstance(ax, int):
        ax = (np.arange(3) == ax).astype(float)
    ax = np.array(ax)
    if ax.shape != (3,):
        raise ValueError(f"ax must be shape (3,); got {ax.shape}")
    R = Rotation.from_rotvec(rad * ax).as_matrix()
    return R

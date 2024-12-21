from collections.abc import Sequence
from enum import IntEnum
from typing import Any, Literal, Optional, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

try:
    import cv2  # pyright: ignore[reportMissingImports]
except ImportError:
    cv2 = None

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


def slice_chunk(index: int, width: int, offset=0):
    start = index * width + offset
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
    out = np.empty((face_w, face_w * 6, 3), np.float32)

    # Create coordinates once and reuse
    rng = np.linspace(-0.5, 0.5, num=face_w, dtype=np.float32)
    x, y = np.meshgrid(rng, -rng)

    # Pre-compute flips
    x_flip = np.flip(x, 1)
    y_flip = np.flip(y, 0)

    def face_slice(index):
        return slice_chunk(index, face_w)

    # Front face (z = 0.5)
    out[:, face_slice(Face.FRONT), Dim.X] = x
    out[:, face_slice(Face.FRONT), Dim.Y] = y
    out[:, face_slice(Face.FRONT), Dim.Z] = 0.5

    # Right face (x = 0.5)
    out[:, face_slice(Face.RIGHT), Dim.X] = 0.5
    out[:, face_slice(Face.RIGHT), Dim.Y] = y
    out[:, face_slice(Face.RIGHT), Dim.Z] = x_flip

    # Back face (z = -0.5)
    out[:, face_slice(Face.BACK), Dim.X] = x_flip
    out[:, face_slice(Face.BACK), Dim.Y] = y
    out[:, face_slice(Face.BACK), Dim.Z] = -0.5

    # Left face (x = -0.5)
    out[:, face_slice(Face.LEFT), Dim.X] = -0.5
    out[:, face_slice(Face.LEFT), Dim.Y] = y
    out[:, face_slice(Face.LEFT), Dim.Z] = x

    # Up face (y = 0.5)
    out[:, face_slice(Face.UP), Dim.X] = x
    out[:, face_slice(Face.UP), Dim.Y] = 0.5
    out[:, face_slice(Face.UP), Dim.Z] = y_flip

    # Down face (y = -0.5)
    out[:, face_slice(Face.DOWN), Dim.X] = x
    out[:, face_slice(Face.DOWN), Dim.Y] = -0.5
    out[:, face_slice(Face.DOWN), Dim.Z] = y

    return out


def equirect_uvgrid(h: int, w: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    u = np.linspace(-np.pi, np.pi, num=w, dtype=np.float32)
    v = np.linspace(np.pi / 2, -np.pi / 2, num=h, dtype=np.float32)
    return np.meshgrid(u, v)  # pyright: ignore[reportReturnType]


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

    # Create the pattern [2,3,3,0,0,1,1,2]
    w4 = w // 4
    w8 = w // 8
    h3 = h // 3
    tp = np.empty((h, w), dtype=np.int32)
    tp[:, :w8] = 2
    tp[:, w8 : w8 + w4] = 3
    tp[:, w8 + w4 : w8 + 2 * w4] = 0
    tp[:, w8 + 2 * w4 : w8 + 3 * w4] = 1
    tp[:, w8 + 3 * w4 :] = 2

    # Prepare ceil mask
    idx = np.linspace(-np.pi, np.pi, w4) / 4
    idx = np.round(h / 2 - np.arctan(np.cos(idx)) * h / np.pi).astype(np.int32)
    # It'll never go past a third of the image, so only process that for optimization
    mask = np.empty((h3, w4), np.bool_)
    row_idx = np.arange(h3, dtype=np.int32)[:, None]
    np.less(row_idx, idx[None], out=mask)

    flip_mask = np.flip(mask, 0)
    tp[:h3, :w8][mask[:, w8:]] = Face.UP
    tp[-h3:, :w8][flip_mask[:, w8:]] = Face.DOWN
    for i in range(3):
        s = slice_chunk(i, w4, w8)
        tp[:h3, s][mask] = Face.UP
        tp[-h3:, s][flip_mask] = Face.DOWN
    remainder = w - s.stop  # pyright: ignore[reportPossiblyUnboundVariable]
    tp[:h3, s.stop :][mask[:, :remainder]] = Face.UP  # pyright: ignore[reportPossiblyUnboundVariable]
    tp[-h3:, s.stop :][flip_mask[:, :remainder]] = Face.DOWN  # pyright: ignore[reportPossiblyUnboundVariable]

    return tp


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


def xyz2uv(xyz: NDArray[DType]) -> tuple[NDArray[DType], NDArray[DType]]:
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
    x = xyz[..., 0:1]  # Keep dimensions but avoid copy
    y = xyz[..., 1:2]
    z = xyz[..., 2:3]
    u = np.arctan2(x, z)
    c = np.hypot(x, z)
    v = np.arctan2(y, c)
    return u, v


def uv2unitxyz(uv: NDArray[DType]) -> NDArray[DType]:
    u, v = np.split(uv, 2, axis=-1)
    y = np.sin(v)
    c = np.cos(v)
    x = c * np.sin(u)
    z = c * np.cos(u)
    return np.concatenate([x, y, z], axis=-1, dtype=uv.dtype)


def uv2coor(u: NDArray[DType], v: NDArray[DType], h: int, w: int) -> tuple[NDArray[DType], NDArray[DType]]:
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
    coor_x = (u / (2 * np.pi) + 0.5) * w - 0.5  # pyright: ignore[reportOperatorIssue]
    coor_y = (-v / np.pi + 0.5) * h - 0.5  # pyright: ignore[reportOperatorIssue]
    return coor_x, coor_y


def coor2uv(coorxy: NDArray[DType], h: int, w: int) -> NDArray[DType]:
    coor_x, coor_y = np.split(coorxy, 2, axis=-1)
    u = ((coor_x + 0.5) / w - 0.5) * 2 * np.pi  # pyright: ignore[reportOperatorIssue]
    v = -((coor_y + 0.5) / h - 0.5) * np.pi  # pyright: ignore[reportOperatorIssue]
    return np.concatenate([u, v], axis=-1, dtype=coorxy.dtype)


class EquirecSampler:
    def __init__(
        self,
        coor_x: NDArray,
        coor_y: NDArray,
        order: int,
    ):
        if cv2 and order in (0, 1, 3):
            self._use_cv2 = True
            if order == 0:
                self._order = cv2.INTER_NEAREST
                nninterpolation = True
            elif order == 1:
                self._order = cv2.INTER_LINEAR
                nninterpolation = False
            elif order == 3:
                self._order = cv2.INTER_CUBIC
                nninterpolation = False
            else:
                raise NotImplementedError

            # TODO: I think coor_y has an off-by-one due to the 1 pixel padding?
            self._coor_x, self._coor_y = cv2.convertMaps(
                coor_x,
                coor_y,
                cv2.CV_16SC2,
                nninterpolation=nninterpolation,
            )
        else:
            self._use_cv2 = False
            self._coor_x = coor_x
            self._coor_y = coor_y
            self._order = order

    def __call__(self, img: NDArray[DType]) -> NDArray[DType]:
        padded = self._pad(img)
        if self._use_cv2:
            out = cv2.remap(padded, self._coor_x, self._coor_y, interpolation=self._order)  # pyright: ignore
        else:
            out = map_coordinates(
                padded,
                (self._coor_y, self._coor_x),
                order=self._order,
                mode="wrap",
            )[..., 0]

        return out  # pyright: ignore[reportReturnType]

    def _pad(self, img: NDArray[DType]) -> NDArray[DType]:
        """Adds 1 pixel of padding above/below image."""
        w = img.shape[1]
        pad_u = np.roll(img[[0]], w // 2, 1)
        pad_d = np.roll(img[[-1]], w // 2, 1)
        img = np.concatenate([img, pad_d, pad_u], 0, dtype=img.dtype)
        return img


class CubeFaceSampler:
    """Arranged as a class so coordinate computations can be re-used across multiple image interpolations."""

    def __init__(
        self,
        tp: NDArray,
        coor_x: NDArray,
        coor_y: NDArray,
        order: int,
        h: int,
        w: int,
    ):
        """Initializes sampler and performs pre-computations.

        Parameters
        ----------
        tp: numpy.ndarray
            (H, W) facetype image from ``equirect_facetype``.
        coor_x: numpy.ndarray
            (H, W) X coordinates to sample.
        coor_y: numpy.ndarray
            (H, W) Y coordinates to sample.
        order: int
            The order of the spline interpolation. See ``scipy.ndimage.map_coordinates``.
        h: int
            Expected input image height.
        w: int
            Expected input image width.
        """
        # Add 1 to compensate for 1-pixel-surround padding.
        coor_x = coor_x + 1  # Not done inplace on purpose.
        coor_y = coor_y + 1  # Not done inplace on purpose.

        self._tp = tp
        self._h = h
        self._w = w
        if cv2 and order in (0, 1, 3):
            self._use_cv2 = True
            if order == 0:
                self._order = cv2.INTER_NEAREST
                nninterpolation = True
            elif order == 1:
                self._order = cv2.INTER_LINEAR
                nninterpolation = False
            elif order == 3:
                self._order = cv2.INTER_CUBIC
                nninterpolation = False
            else:
                raise NotImplementedError

            # The +2 comes from padding from self._pad.
            coor_y += np.multiply(tp, h + 2, dtype=np.float32)
            self._coor_x, self._coor_y = cv2.convertMaps(
                coor_x,
                coor_y,
                cv2.CV_16SC2,
                nninterpolation=nninterpolation,
            )
        else:
            self._use_cv2 = False
            self._coor_x = coor_x
            self._coor_y = coor_y
            self._order = order

    def __call__(self, cube_faces: NDArray[DType]) -> NDArray[DType]:
        """Sample cube faces.

        Parameters
        ----------
        cube_faces: numpy.ndarray
            (6, S, S) Cube faces.

        Returns
        -------
        numpy.ndarray
            (H, W) Sampled image.
        """
        h, w = cube_faces.shape[-2:]
        if h != self._h:
            raise ValueError("Input height {h} doesn't match expected height {self._h}.")
        if w != self._w:
            raise ValueError("Input width {w} doesn't match expected height {self._w}.")

        padded = self._pad(cube_faces)
        if self._use_cv2:
            w = padded.shape[-1]
            v_img = padded.reshape(-1, w)
            out = cv2.remap(v_img, self._coor_x, self._coor_y, interpolation=self._order)  # pyright: ignore
        else:
            out = map_coordinates(padded, (self._tp, self._coor_y, self._coor_x), order=self._order)
        return out  # pyright: ignore[reportReturnType]

    def _pad(self, cube_faces: NDArray[DType]) -> NDArray[DType]:
        """Adds 1 pixel of padding around each cube face."""
        ABOVE = (0, slice(None))
        BELOW = (-1, slice(None))
        LEFT = (slice(None), 0)
        RIGHT = (slice(None), -1)
        padded = np.pad(cube_faces, ((0, 0), (1, 1), (1, 1)), mode="empty")

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

        return padded


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


def cube_dict2list(cube_dict: dict[Any, NDArray[DType]], face_k: Optional[Sequence] = None) -> list[NDArray[DType]]:
    face_k = face_k or "FRBLUD"
    if len(face_k) != 6:
        raise ValueError(f"6 face_k keys must be provided to construct a cube; got {len(face_k)}.")
    return [cube_dict[k] for k in face_k]


def cube_dict2h(cube_dict: dict[Any, NDArray[DType]], face_k: Optional[Sequence] = None) -> NDArray[DType]:
    return cube_list2h(cube_dict2list(cube_dict, face_k))


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


def cube_dice2list(cube_dice: NDArray[DType]) -> list[NDArray[DType]]:
    if cube_dice.shape[0] % 3 != 0:
        raise ValueError("Dice image height must be a multiple of 3.")
    w = cube_dice.shape[0] // 3
    if cube_dice.shape[1] != w * 4:
        raise ValueError(f'Dice width must be 4 "faces" (4x{w}={4*w}) wide.')
    # Order: F R B L U D
    #        ┌────┐
    #        │ U  │
    #   ┌────┼────┼────┬────┐
    #   │ L  │ F  │ R  │ B  │
    #   └────┼────┼────┴────┘
    #        │ D  │
    #        └────┘
    out = []
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for sx, sy in sxy:
        out.append(cube_dice[slice_chunk(sy, w), slice_chunk(sx, w)])
    return out


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

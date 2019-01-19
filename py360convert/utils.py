import numpy as np
from scipy.ndimage import map_coordinates


def xyzcube(out_wh):
    '''
    Return the xyz cordinates of the unit cube in [F R B L U D] format.
    '''
    out = np.zeros((out_wh, out_wh * 6, 3), np.float32)
    rng = np.linspace(-0.5, 0.5, num=out_wh, dtype=np.float32)
    grid = np.stack(np.meshgrid(rng, -rng), -1)

    # Front face (z = 0.5)
    out[:, 0*out_wh:1*out_wh, [0, 1]] = grid
    out[:, 0*out_wh:1*out_wh, 2] = 0.5

    # Right face (x = 0.5)
    out[:, 1*out_wh:2*out_wh, [2, 1]] = grid
    out[:, 1*out_wh:2*out_wh, 0] = 0.5

    # Back face (z = -0.5)
    out[:, 2*out_wh:3*out_wh, [0, 1]] = grid
    out[:, 2*out_wh:3*out_wh, 2] = -0.5

    # Left face (x = -0.5)
    out[:, 3*out_wh:4*out_wh, [2, 1]] = grid
    out[:, 3*out_wh:4*out_wh, 0] = -0.5

    # Up face (y = 0.5)
    out[:, 4*out_wh:5*out_wh, [0, 2]] = grid
    out[:, 4*out_wh:5*out_wh, 1] = 0.5

    # Down face (y = -0.5)
    out[:, 5*out_wh:6*out_wh, [0, 2]] = grid
    out[:, 5*out_wh:6*out_wh, 1] = -0.5

    return out


def xyz2uv(xyz):
    '''
    xyz: ndarray in shape of [..., 3]
    '''
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(z, x)
    c = np.sqrt(z**2 + x**2)
    v = np.arctan2(y, c)

    return np.concatenate([u, v], axis=-1)


def uv2coor(uv, h, w):
    '''
    uv: ndarray in shape of [..., 2]
    h: int, height of the equirectangular image
    w: int, width of the equirectangular image
    '''
    u, v = np.split(uv, 2, axis=-1)
    coor_x = (-(u - np.pi / 2) % (2 * np.pi) / (2 * np.pi) + 0.5) * w
    coor_y = (-v / np.pi + 0.5) * h

    return np.concatenate([coor_x, coor_y], axis=-1)


def sample_equirec(e_img, coor_xy, order):
    w = e_img.shape[1]
    coor_x, coor_y = np.split(coor_xy, 2, axis=-1)
    pad_u = np.roll(e_img[[0]], w // 2, 1)
    pad_d = np.roll(e_img[[-1]], w // 2, 1)
    e_img = np.concatenate([e_img, pad_d, pad_u], 0)
    return map_coordinates(e_img, [coor_y, coor_x],
                           order=order, mode='wrap')[..., 0]


def cube_h2list(cube_h):
    assert cube_h.shape[0] * 6 == cube_h.shape[1]
    return np.split(cube_h, 6, axis=1)


def cube_list2h(cube_list):
    assert len(cube_list) == 6
    assert sum(face.shape == cube_list[0].shape for face in cube_list) == 6
    return np.concatenate(cube_list, axis=1)


def cube_h2dice(cube_h):
    assert cube_h.shape[0] * 6 == cube_h.shape[1]
    w = cube_h.shape[0]
    cube_dice = np.zeros((w * 3, w * 4, cube_h.shape[2]), dtype=cube_h.dtype)
    cube_list = cube_h2list(cube_h)
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        face = cube_list[i]
        if i in [1, 2]:
            face = np.flip(face, axis=1)
        if i == 4:
            face = np.flip(face, axis=0)
        cube_dice[sy*w:(sy+1)*w, sx*w:(sx+1)*w] = face
    return cube_dice


def cube_dice2h(cube_dice):
    w = cube_dice.shape[0] // 3
    assert cube_dice.shape[0] == w * 3 and cube_dice.shape[1] == w * 4
    cube_h = np.zeros((w, w * 6, cube_dice.shape[2]), dtype=cube_dice.dtype)
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        face = cube_dice[sy*w:(sy+1)*w, sx*w:(sx+1)*w]
        if i in [1, 2]:
            face = np.flip(face, axis=1)
        if i == 4:
            face = np.flip(face, axis=0)
        cube_h[:, i*w:(i+1)*w] = face
    return cube_h

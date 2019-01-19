import numpy as np

from . import utils


def e2c(e_img, out_wh=256, mode='bilinear', cube_format='horizon'):
    '''
    e_img:  ndarray in shape of [H, W, *]
    out_wh: int, the length of each face of the cubemap
    '''
    assert len(e_img.shape) == 3
    h, w = e_img.shape[:2]
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    xyz = utils.xyzcube(out_wh)
    uv = utils.xyz2uv(xyz)
    coor_xy = utils.uv2coor(uv, h, w)

    cubemap = np.stack([
        utils.sample_equirec(e_img[..., i], coor_xy, order=order)
        for i in range(e_img.shape[2])
    ], axis=-1)

    if cube_format == 'horizon':
        pass
    elif cube_format == 'list':
        cubemap = utils.cube_h2list(cubemap)
    elif cube_format == 'dice':
        cubemap = utils.cube_h2dice(cubemap)

    return cubemap

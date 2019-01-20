import numpy as np
from scipy.spatial import cKDTree

from . import utils


def c2e(cubemap, h, w, cube_format='horizon', k=4):
    if cube_format == 'list':
        cubemap = utils.cube_list2h(cubemap)
    elif cube_format == 'dict':
        cubemap = utils.cube_dict2h(cubemap)
    elif cube_format == 'dice':
        cubemap = utils.cube_dice2h(cubemap)
    assert len(cubemap.shape) == 3
    assert cubemap.shape[0] * 6 == cubemap.shape[1]
    k = k if k > 1 else [1]
    face_w = cubemap.shape[0]
    channel = cubemap.shape[2]
    cubemap = cubemap.reshape(-1, channel)

    cube_xyz = utils.uv2unitxyz(utils.xyz2uv(utils.xyzcube(face_w))).reshape(-1, 3)
    equirec_xyz = utils.uv2unitxyz(utils.equirect_uvgrid(h, w)).reshape(-1, 3)

    tree = cKDTree(cube_xyz, balanced_tree=False)
    dist, idx = tree.query(equirec_xyz, k=k)
    dist = dist.sum(1, keepdims=True) - dist
    p = (dist / dist.sum(1, keepdims=True))[..., None]
    equirec = (cubemap[idx] * p).sum(1).reshape(h, w, channel)
    equirec = np.clip(equirec, cubemap.min(), cubemap.max())

    return equirec

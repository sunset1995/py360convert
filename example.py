import time
import numpy as np
from PIL import Image
from py360convert import e2c, e2p, c2e

img = np.array(Image.open('assert/example_input.jpg'))

# Test e2c
s_time = time.time()
cube_h = e2c(img)
print('e2c (horizon): %.4f sec' % (time.time() - s_time))
Image.fromarray(cube_h.astype(np.uint8))\
     .save('assert/example_results/e2c_cube_h.jpg', 'JPEG', quality=80)

s_time = time.time()
cube_list = e2c(img, cube_format='list')
print('e2c (list): %.4f sec' % (time.time() - s_time))
for i, j in enumerate(['F', 'R', 'B', 'L', 'U', 'D']):
    Image.fromarray(cube_list[i].astype(np.uint8))\
         .save('assert/example_results/e2c_%s.jpg' % j, 'JPEG', quality=80)

s_time = time.time()
cube_dict = e2c(img, cube_format='dict')
print('e2c (dict): %.4f sec' % (time.time() - s_time))
for i, k in enumerate(['F', 'R', 'B', 'L', 'U', 'D']):
    assert np.allclose(cube_dict[k], cube_list[i])

s_time = time.time()
cube_dice = e2c(img, cube_format='dice')
print('e2c (dice): %.4f sec' % (time.time() - s_time))
Image.fromarray(cube_dice.astype(np.uint8))\
     .save('assert/example_results/e2c_cube_dice.jpg', 'JPEG', quality=80)

# Test e2p
s_time = time.time()
pers = e2p(img, fov_deg=120, u_deg=-90, v_deg=30, out_hw=(512, 512), in_rot_deg=90)
print('e2p: %.4f sec' % (time.time() - s_time))
Image.fromarray(pers.astype(np.uint8))\
     .save('assert/example_results/e2p.jpg', 'JPEG', quality=80)

# Test c2e
s_time = time.time()
equirec = c2e(cube_h, img.shape[0], img.shape[1])
diff = np.abs(equirec - img)
print('c2e: %.4f sec' % (time.time() - s_time))
print('Reconstruction L2: %.6f' % np.sqrt(np.sqrt((diff**2).mean(-1)).mean()))
Image.fromarray(equirec.astype(np.uint8))\
     .save('assert/example_results/c2e.jpg', 'JPEG', quality=80)

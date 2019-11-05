#!/usr/bin/env python

import numpy as np
from PIL import Image
import py360convert


#fill the required variable fo the function that you are calling
img_path = "py360convert/images/white_kitten.jpg"
output_img_path = "py360convert/converted_images/1.jpg"
convert = "e2c"  #choices=['c2e', 'e2c', 'e2p']
face_w=256
cube_format = "dice" #choices=["dice","dict","list"]
h = None #define the height of the image
w = None #define the width of the image
h_fov = None
v_fov = None
u_deg = None
v_deg = None
in_rot_deg = None
mode ="bilinear" #choices=["nearest","bilinear"]

# Read image
img = np.array(Image.open(img_path))
if len(img.shape) == 2:
    img = img[..., None]

if convert == 'c2e':
    out = py360convert.c2e(img, h=h, w=w, mode=mode)
elif convert == 'e2c':
    out = py360convert.e2c(img, face_w=face_w, mode=mode, cube_format=cube_format)
elif convert == 'e2p':
    out = py360convert.e2p(img, fov_deg=(h_fov, v_fov), u_deg=u_deg, v_deg=v_deg,
                           out_hw=(h, w), in_rot_deg=in_rot_deg, mode=mode)
else:
    raise NotImplementedError('Unknown convertion')

# Output image
Image.fromarray(out.astype(np.uint8)).save(output_img_path)

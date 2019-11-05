#!/usr/bin/env python
"""
mode: bilinear

mode: nearest

"""
import numpy as np
from PIL import Image

from py360convert import py360convert


def read_img(img_path):
    """

    :param img_path: path to the input image
    :return: converted image array list
    """
    img = np.array(Image.open(img_path))
    if len(img.shape) == 2:
        img = img[..., None]
    return img


def get_filename(img_path):
    """

    :param img_path: path to the input image
    :return: file name without the extension
    """
    fname = str(img_path).split("/")[-1]
    return fname.split(".")[0]


def convert_to_dice(img_path,output_folder_path):
    """

    :param img_path:  path to the input image
    :param output_folder_path:  path to the output folder to store the image
    :return: image converted to dice shape
    """
    f_name = get_filename(img_path)
    img = read_img(img_path)
    converted_img = py360convert.e2c(img, face_w=256, mode='bilinear', cube_format='dice')

    Image.fromarray(converted_img.astype(np.uint8)).save(output_folder_path+f_name+"_convert_to_dice"+".jpg")


def convert_to_dict(img_path,output_folder_path):
    """

    :param img_path: path to the input image
    :param output_folder_path: path to the output folder to store the list of image
    :return: list of image saved to output folder with dict keys['F', 'R', 'B', 'L', 'U', 'D'] names at the end of the file.
    """
    f_name = get_filename(img_path)
    img = read_img(img_path)
    converted_img = py360convert.e2c(img, face_w=256, mode='bilinear', cube_format='dict')

    # for saving dict type images
    for k, i in converted_img.items():
        Image.fromarray(i.astype(np.uint8)).save(
            output_folder_path + f_name+"_converted_to_dict"+str(k) + ".jpg")


def convert_to_list(img_path,output_folder_path):
    """

    :param img_path: path to the input image
    :param output_folder_path: path to the output folder to store the list of image
    :return: list of image saved to output folder with [0.....n] names at the end of the file.
    """
    f_name = get_filename(img_path)
    img = read_img(img_path)
    converted_img = py360convert.e2c(img, face_w=256, mode='bilinear', cube_format='list')

    # for saving list of images
    for i in range(len(converted_img)):
        Image.fromarray(converted_img[i].astype(np.uint8)).save(
            output_folder_path+ f_name+"_converted_to_list"+str(i) + ".jpg")

#PROJECT_PATH = "py360convert"
# convert_to_dice(PROJECT_PATH+"/images/white_kitten.jpg",PROJECT_PATH+"py360convert/converted_images/")
# convert_to_list(PROJECT_PATH+"/images/white_kitten.jpg",PROJECT_PATH+"py360convert/converted_images/")
# convert_to_dict(PROJECT_PATH+"/images/white_kitten.jpg",PROJECT_PATH+"py360convert/converted_images/")
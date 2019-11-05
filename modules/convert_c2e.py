#!/usr/bin/env python
"""
mode: bilinear

mode: nearest

"""
import numpy as np
from PIL import Image

from py360convert import py360convert


def read_img(img_path):
    img = np.array(Image.open(img_path))
    if len(img.shape) == 2:
        img = img[..., None]
    return img


def get_filename(img_path):
    fname = str(img_path).split("/")[-1]
    return fname.split(".")[0]


def convert_dice_to_img(cubemap_path,output_folder_path):
    f_name = get_filename(cubemap_path)
    img = read_img(cubemap_path)
    converted_img = py360convert.c2e(img, h=724, w=320, mode='bilinear', cube_format='dice')

    Image.fromarray(converted_img.astype(np.uint8)).save(output_folder_path+f_name+"_converted_dice_to_img"+".jpg")


def convert_dict_to_img(cubemap_list_path,output_folder_path):
    img_dict = {}
    f_name = None
    for i in cubemap_list_path:
        f_name = get_filename(i)
        if "B.jpg" in i:
            img_dict["B"] = read_img(i)
        elif "D.jpg" in i:
            img_dict["D"] = read_img(i)
        elif "F.jpg" in i:
            img_dict["F"] = read_img(i)
        elif "L.jpg" in i:
            img_dict["L"] = read_img(i)
        elif "R.jpg" in i:
            img_dict["R"] = read_img(i)
        elif "U.jpg" in i:
            img_dict["U"] = read_img(i)
    converted_img = py360convert.c2e(img_dict, h=724, w=320, mode='bilinear', cube_format='dict')
    Image.fromarray(converted_img.astype(np.uint8)).save(output_folder_path+f_name+"_converted_dict_to_img"+".jpg")


def convert_list_to_img(cubemap_list_path,output_folder_path):
    img_array_list = []
    f_name = None
    for i in cubemap_list_path:
        img = read_img(i)
        img_array_list.append(img)
        f_name = get_filename(i)

    converted_img = py360convert.c2e(img_array_list, h=724, w=320, mode='bilinear', cube_format='list')
    Image.fromarray(converted_img.astype(np.uint8)).save(output_folder_path+f_name+"_converted_list_to_img"+".jpg")


PROJECT_PATH = "py360convert"
input_img_foler = PROJECT_PATH + "/converted_images/"

# convert_dice_to_img(PROJECT_PATH+"/converted_images/white_kitten_convert_to_dice.jpg",PROJECT_PATH+"/converted_images/")
# convert_list_to_img([input_img_foler+"white_kitten_converted_to_list0.jpg",
#                     input_img_foler+"white_kitten_converted_to_list1.jpg",
#                     input_img_foler+"white_kitten_converted_to_list2.jpg",
#                     input_img_foler+"white_kitten_converted_to_list3.jpg",
#                     input_img_foler+"white_kitten_converted_to_list4.jpg",
#                     input_img_foler+"white_kitten_converted_to_list5.jpg"],
#                     PROJECT_PATH+"/converted_images/")

convert_dict_to_img([input_img_foler+"white_kitten_converted_to_dictB.jpg",
                    input_img_foler+"white_kitten_converted_to_dictD.jpg",
                    input_img_foler+"white_kitten_converted_to_dictF.jpg",
                    input_img_foler+"white_kitten_converted_to_dictL.jpg",
                    input_img_foler+"white_kitten_converted_to_dictR.jpg",
                    input_img_foler+"white_kitten_converted_to_dictU.jpg"],
                    PROJECT_PATH+"/converted_images/")




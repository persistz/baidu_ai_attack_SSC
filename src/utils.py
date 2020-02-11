#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import time
import sys
import math
import numpy as np
import argparse
import functools
import distutils.util
import six

from PIL import Image, ImageOps
#绘图函数
import matplotlib
#服务器环境设置
import matplotlib.pyplot as plt


#去除batch_norm的影响
def init_prog(prog):
    for op in prog.block(0).ops:
        #print("op type is {}".format(op.type))
        if op.type in ["batch_norm"]:
            # 兼容旧版本 paddle
            if hasattr(op, 'set_attr'):
                op.set_attr('is_test', False)
                op.set_attr('use_global_stats', True)
            else:
                op._set_attr('is_test', False)
                op._set_attr('use_global_stats', True)
                op.desc.check_attrs()

def img2tensor(img,image_shape=[3,224,224]):
    
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225] 
      
    img = cv2.resize(img,(image_shape[1],image_shape[2]))

    #RGB img [224,224,3]->[3,224,224]
    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
     
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    img=img.astype('float32')
    img=np.expand_dims(img, axis=0)
    
    return img

def crop_image(img, target_size, center):
    """ crop_image """
    height, width = img.shape[:2]
    size = target_size
    if center == True:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img

def process_img(img_path="",image_shape=[3,224,224]):
    
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
      
    img = cv2.imread(img_path)
    img = cv2.resize(img,(image_shape[1],image_shape[2]))
    #img = cv2.resize(img,(256,256))
    #img = crop_image(img, image_shape[1], True)
    
    # RBG img [224,224,3]->[3,224,224]
    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
    # img = img.astype('float32').transpose((2, 0, 1)) / 255
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    img=img.astype('float32')
    img=np.expand_dims(img, axis=0)
    
    return img

def tensor2img(tensor):
    
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225] 
    
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    
    img = tensor.copy()
      
    img *= img_std
    img += img_mean
    
    img = np.round(img*255) 
    img = np.clip(img,0,255)

    img = img[0].astype(np.uint8)
        
    img = img.transpose(1, 2, 0)
    img = img[:, :, ::-1]
    
    return img

def save_adv_image(img, output_path):
    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return

def calc_mse(org_img, adv_img):
    org_img = org_img.astype(np.float32)
    adv_img = adv_img.astype(np.float32)
    diff = org_img.reshape((-1, 3)) - adv_img.reshape((-1, 3))
    distance = np.mean(np.sqrt(np.sum((diff ** 2), axis=1)))
    return distance

def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-------------  Configuration Arguments -------------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%25s : %s" % (arg, value))
    print("----------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)

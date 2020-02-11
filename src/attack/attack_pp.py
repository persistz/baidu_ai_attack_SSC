#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import math
import numpy as np
import argparse
import functools
import logging

import paddle
import paddle.fluid as fluid
from utils import *
import six


#实现linf约束 输入格式都是tensor 返回也是tensor [1,3,224,224]
def linf_img_tenosr(o,adv,epsilon=16.0/256):
    o_img = tensor2img(o)
    adv_img = tensor2img(adv)

    clip_max = np.clip(o_img * (1.0 + epsilon), 0, 255)
    clip_min = np.clip(o_img * (1.0 - epsilon), 0, 255)

    # print(np.min(adv_img), np.max(adv_img))
    adv_img = np.clip(adv_img, clip_min, clip_max)

    adv_img = img2tensor(adv_img)
    
    return adv_img


def l2_img_tenosr(adv):
    adv_img = tensor2img(adv)
    adv_img = img2tensor(adv_img)
    return adv_img

"""
Explaining and Harnessing Adversarial Examples, I. Goodfellow et al., ICLR 2015
实现了FGSM 支持定向和非定向攻击的单步FGSM


input_layer:输入层
output_layer:输出层
step_size:攻击步长
adv_program：生成对抗样本的prog 
eval_program:预测用的prog
isTarget：是否定向攻击
target_label：定向攻击标签
epsilon:约束linf大小
o:原始数据
use_gpu：是否使用GPU

返回：
生成的对抗样本
"""
def FGSM(adv_program,eval_program,gradients,o,input_layer,output_layer,step_size=16.0/256,epsilon=16.0/256,isTarget=False,target_label=0,use_gpu=False):
    
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
   
    result = exe.run(eval_program,
                     fetch_list=[output_layer],
                     feed={ input_layer.name:o })
    result = result[0][0]
   
    o_label = np.argsort(result)[::-1][:1][0]
    
    if not isTarget:
        #无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label=o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label,o_label))
        
        
    target_label=np.array([target_label]).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    
    #计算梯度
    g = exe.run(adv_program,
                     fetch_list=[gradients],
                     feed={ input_layer.name:o,'label': target_label  }
               )
    g = g[0][0]
    
    #print(g)
    
    if isTarget:
        adv=o-np.sign(g)*step_size
    else:
        adv=o+np.sign(g)*step_size
    
    #实施linf约束
    adv=linf_img_tenosr(o,adv,epsilon)
    
    return adv


"""
Towards deep learning models resistant to adversarial attacks, A. Madry, A. Makelov, L. Schmidt, D. Tsipras, 
and A. Vladu, ICLR 2018
实现了PGD 支持定向和非定向攻击的PGD


input_layer:输入层
output_layer:输出层
step_size:攻击步长
adv_program：生成对抗样本的prog 
eval_program:预测用的prog
isTarget：是否定向攻击
target_label：定向攻击标签
epsilon:约束linf大小
o:原始数据
use_gpu：是否使用GPU

返回：
生成的对抗样本
"""
def PGD(adv_program,eval_program,loss,gradients,o,input_layer,output_layer,input_layer_eval,output_layer_eval,
        step_size=2.0/256,epsilon=16.0/256,iteration=20,isTarget=False,
        target_label=0,src_label=0,use_gpu=True, discrete=False, step_adjust=False, random_init=False):
    
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
   
    result = exe.run(eval_program, fetch_list=[output_layer_eval],
                     feed={input_layer_eval.name:o})
    result = result[0][0]
   
    o_label = np.argsort(result)[::-1][:1][0]
    
    if not isTarget:
        #无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label={}, o_label={}".format(src_label, o_label))
        # target_label=o_label
        target_label=src_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label,o_label))
        
        
    target_label=np.array([target_label]).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    
    adv=o.copy()

    # fix step size
    # mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # mean = np.array(mean).reshape((3, 1, 1))
    std = np.array(std).reshape((3, 1, 1))

    # add random init
    if random_init:
        rand_minmax = 10 # 当设置为epsilon时，MSE大大增加
        eta = np.random.random_integers(-rand_minmax, rand_minmax, adv.shape)
        eta = eta / std
        adv = adv + eta
        adv = adv.astype('float32')

    for _ in range(iteration):
        # step adjust
        if step_adjust:
            step_size *= np.random.random_integers(1, 2)

        #计算梯度
        g, loss_, out_logits_ = exe.run(adv_program, fetch_list=[gradients, loss, output_layer],
                    feed={input_layer.name:adv, 'label': target_label})
        g = g[0][0]
        if _%50 == 0:
            try:
                print('iter: [%d]/[%d], loss: %.2f'%(_, iteration, loss_[0]))
            except:
                print('iter: [%d]/[%d], loss: %.2f'%(_, iteration, loss_[0]))
            # print(out_logits_[0])
            print(out_logits_.max() - out_logits_[0, src_label])
            sys.stdout.flush()
        if isTarget:
            if discrete:
                real_step_size = np.sign(g) * step_size
                real_step_size = real_step_size / std
                adv = adv - real_step_size
                adv = adv.astype('float32')
            else:
                adv = adv - np.sign(g) * step_size
        else:
            adv = adv + np.sign(g) * step_size

        # move linf in
        # 每一步都需要约束
        # 对于不限制整数步长的方法，不能这样约束，
        # 因为每一次约束都会转换为整数域，导致扰动丢失
        if discrete:
            adv = linf_img_tenosr(o, adv, epsilon)
    
    # 实施linf约束
    adv = linf_img_tenosr(o, adv, epsilon)
    
    return adv


def MPGD(adv_program, eval_program, gradients, o, input_layer,
         output_layer, step_size=2.0 / 256, epsilon=16.0 / 256,
        iteration=20, isTarget=False, target_label=0, use_gpu=True):

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    result = exe.run(eval_program, fetch_list=[output_layer],
                     feed={input_layer.name: o})
    result = result[0][0]

    o_label = np.argsort(result)[::-1][:1][0]

    if not isTarget:
        # 无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label = o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label, o_label))

    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)

    adv = o.copy()

    # add random init
    # rand_minmax = step_size  # 当设置为epsilon时，MSE大大增加
    # eta = np.random.uniform(-rand_minmax, rand_minmax, adv.shape)
    # adv = adv + eta
    # adv = adv.astype('float32')

    # add momentum
    momentum = np.zeros_like(o)
    decay_factor = 0.5

    # fix step size
    # mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # mean = np.array(mean).reshape((3, 1, 1))
    std = np.array(std).reshape((3, 1, 1))

    for _ in range(iteration):

        # 计算梯度
        g = exe.run(adv_program, fetch_list=[gradients],
                    feed={input_layer.name: adv, 'label': target_label})
        g = g[0][0]
        momentum = decay_factor * momentum + g

        if isTarget:
            real_step_size = np.sign(momentum) * step_size
            # real_step_size = (real_step_size - mean) / std
            real_step_size = real_step_size / std
            adv = adv - real_step_size
            adv = adv.astype('float32')
            # adv = adv - np.sign(momentum) * step_size

        else:
            real_step_size = np.sign(momentum) * step_size
            real_step_size = real_step_size / std
            adv = adv + real_step_size
            adv = adv.astype('float32')

        # move linf in
        # 每一步都需要约束
        # adv = linf_img_tenosr(o, adv, epsilon)

    # 实施linf约束
    adv = linf_img_tenosr(o, adv, epsilon)

    return adv


def MPGD_mix(adv_program_1, adv_program_2, eval_program_1, eval_program_2,
             gradients_1, gradients_2, input_layer_1, input_layer_2, output_layer_1, output_layer_2,
             o, step_size=2.0 / 256, epsilon=16.0 / 256,
             iteration=20, isTarget=False, target_label=0, use_gpu=True):

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    result_1 = exe.run(eval_program_1, fetch_list=[output_layer_1],
                     feed={input_layer_1.name: o})
    result_1 = result_1[0][0]

    o_label = np.argsort(result_1)[::-1][:1][0]

    if not isTarget:
        # 无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label = o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label, o_label))

    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)

    adv = o.copy()

    # add random init
    rand_minmax = step_size  # 当设置为epsilon时，MSE大大增加
    eta = np.random.uniform(-rand_minmax, rand_minmax, adv.shape)
    adv = adv + eta
    adv = adv.astype('float32')

    # add momentum
    momentum = np.zeros_like(o)
    decay_factor = 0.9

    for _ in range(iteration):

        # 计算梯度
        g_1 = exe.run(adv_program_1, fetch_list=[gradients_1],
                    feed={input_layer_1.name: adv, 'label': target_label})
        g_2 = exe.run(adv_program_2, fetch_list=[gradients_2],
                    feed={input_layer_2.name: adv, 'label': target_label})

        g = g_1[0][0] + g_2[0][0]
        momentum = decay_factor * momentum + g

        if isTarget:
            adv = adv - np.sign(momentum) * step_size
        else:
            adv = adv + np.sign(momentum) * step_size

        # move linf in
        # 每一步都需要约束
        adv = linf_img_tenosr(o, adv, epsilon)

    # 实施linf约束
    adv = linf_img_tenosr(o, adv, epsilon)

    return adv


def CW():
    # build cw attack compute graph within attack programs
    with fluid.program_guard(main_program=self.attack_main_program,
                             startup_program=self.attack_startup_program):
        img_0_1_placehold = fluid.layers.data(name='img_data_scaled', shape=self._shape, dtype="float32")
        target_placehold = fluid.layers.data(name='target', shape=[self._dim], dtype="float32")
        c_placehold = fluid.layers.data(name='c', shape=[1], dtype="float32")
        # add this perturbation
        self.ad_perturbation = fluid.layers.create_parameter(name='parameter',
                                                             shape=self._shape,
                                                             dtype='float32',
                                                             is_bias=False)
        # add clip_min and clip_max for normalization
        self.clip_min = fluid.layers.create_parameter(name='clip_min',
                                                      shape=self.clip_shape,
                                                      dtype='float32',
                                                      is_bias=False)
        self.clip_max = fluid.layers.create_parameter(name='clip_max',
                                                      shape=self.clip_shape,
                                                      dtype='float32',
                                                      is_bias=False)

        # construct graph with perturbation and cnn model
        constrained, dis_L2 = self._constrain_cwb(img_0_1_placehold)
        loss, _, _ = self._loss_cwb(target_placehold, constrained, dis_L2, c_placehold)

        # Adam optimizer as suggested in paper
        optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
        optimizer.minimize(loss, parameter_list=['parameter'])

    # initial variables and parameters every time before attack
    self.exe.run(self.attack_startup_program)

    ad_min = fluid.global_scope().find_var("clip_min").get_tensor()
    ad_min.set(self.pa_clip_min.astype('float32'), self.place)
    ad_max = fluid.global_scope().find_var("clip_max").get_tensor()
    ad_max.set(self.pa_clip_max.astype('float32'), self.place)


def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    # TODO: move this function to utils
    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    assert isinstance(p, float) or isinstance(p, int)
    x = x.reshape(-1, 224)
    norm = np.linalg.norm(x, ord=2)
    norm = max(norm, small_constant)
    x_norm = 1. / norm * x
    return x_norm.reshape(3, 224, 224)

def PGDL2(adv_program, eval_program, loss, gradients, o, input_layer, output_layer, input_layer_eval, output_layer_eval,
        step_size=2.0 / 256, epsilon=16.0 / 256, iteration=20, isTarget=False,
        target_label=0, src_label=0, use_gpu=True, discrete=False, step_adjust=False, random_init=False, confidence=1000):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    result = exe.run(eval_program, fetch_list=[output_layer_eval],
                     feed={input_layer_eval.name: o})
    result = result[0][0]

    o_label = np.argsort(result)[::-1][:1][0]
    print('istarget: ', isTarget)
    if not isTarget:
        # 无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label={}, o_label={}".format(src_label, o_label))
        # target_label=o_label
        target_label = src_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label, o_label))

    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)

    adv = o.copy()

    # fix step size
    # mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # mean = np.array(mean).reshape((3, 1, 1))
    std = np.array(std).reshape((3, 1, 1))

    # add random init
    if random_init:
        rand_minmax = 10  # 当设置为epsilon时，MSE大大增加
        eta = np.random.random_integers(-rand_minmax, rand_minmax, adv.shape)
        eta = eta / std
        adv = adv + eta
        adv = adv.astype('float32')

    for _ in range(iteration):
        # step adjust
        if step_adjust:
            step_size *= np.random.random_integers(1, 2)

        # 计算梯度
        g, loss_, out_logits_ = exe.run(adv_program, fetch_list=[gradients, loss, output_layer],
                                        feed={input_layer.name: adv, 'label': target_label})
        g = g[0]

        logit_diff = out_logits_.max() - out_logits_[0, src_label]
        if _ % 10 == 0:
            print('iter: [%d]/[%d], loss: %.2f, logits diff: %.2f' % (_, iteration, loss_[0], logit_diff))
            sys.stdout.flush()
        # early_stop
        if logit_diff > confidence:
            break
        if isTarget:
            if discrete:
                real_step_size = np.sign(g) * step_size
                real_step_size = real_step_size / std
                adv = adv - real_step_size
                adv = adv.astype('float32')
            else:
                # adv = adv - np.sign(g) * step_size
                adv = adv - step_size * normalize_by_pnorm(g, p=2)
        else:
            # adv = adv + np.sign(g) * step_size
            adv = adv + step_size * normalize_by_pnorm(g, p=2)

        # move linf in
        # 每一步都需要约束
        # 对于不限制整数步长的方法，不能这样约束，
        # 因为每一次约束都会转换为整数域，导致扰动丢失
        # if discrete:
        #     adv = linf_img_tenosr(o, adv, epsilon)

    # 实施linf约束
    adv = l2_img_tenosr(adv)

    return adv

topk_idx = int(224*224*3*0.00000001)
def PGD_patch(adv_program, eval_program, loss, gradients, o, input_layer, output_layer, input_layer_eval, output_layer_eval,
        step_size=2.0 / 256, epsilon=16.0 / 256, iteration=20, isTarget=False,
        target_label=0, src_label=0, use_gpu=True, discrete=False, step_adjust=False, random_init=False, confidence=1000):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    result = exe.run(eval_program, fetch_list=[output_layer_eval],
                     feed={input_layer_eval.name: o})
    result = result[0][0]

    o_label = np.argsort(result)[::-1][:1][0]

    if not isTarget:
        # 无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label={}, o_label={}".format(src_label, o_label))
        # target_label=o_label
        target_label = src_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label, o_label))

    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)

    adv = o.copy()

    # fix step size
    # mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # mean = np.array(mean).reshape((3, 1, 1))
    std = np.array(std).reshape((3, 1, 1))


    # mask = 1
    # add random init
    if random_init:
        rand_minmax = 10  # 当设置为epsilon时，MSE大大增加
        eta = np.random.random_integers(-rand_minmax, rand_minmax, adv.shape)
        eta = eta / std
        adv = adv + eta
        adv = adv.astype('float32')

    for _ in range(iteration):
        # step adjust
        if step_adjust:
            step_size *= np.random.random_integers(1, 2)

        # 计算梯度
        g, loss_, out_logits_ = exe.run(adv_program, fetch_list=[gradients, loss, output_layer],
                                        feed={input_layer.name: adv, 'label': target_label})

        g = g[0]

        logit_diff = out_logits_.max() - out_logits_[0, src_label]
        if _ % 10 == 0:
            try:
                print('iter: [%d]/[%d], loss: %.2f' % (_, iteration, loss_[0]))
            except:
                print('iter: [%d]/[%d], loss: %.2f' % (_, iteration, loss_[0]))
            # print(out_logits_[0])
            print(logit_diff)
            sys.stdout.flush()
        # early_stop
        if logit_diff > confidence:
            break
        if isTarget:
            if discrete:
                real_step_size = np.sign(g) * step_size
                real_step_size = real_step_size / std
                adv = adv - real_step_size
                adv = adv.astype('float32')
            else:
                adv = adv - np.sign(g) * step_size
        else:
            # adv = adv + np.sign(g) * step_size
            # adv = adv + step_size * normalize_by_pnorm(g, p=2)
            abs_g = abs(g)
            mask = np.zeros_like(g)

            topk = np.partition(abs_g.flatten(), topk_idx)[-topk_idx]
            mask[abs_g >= topk] = 1
            print(mask.sum(), topk, g.shape, abs_g.max(), abs_g.min())
            adv = adv + np.sign(g) * step_size * mask

        # move linf in
        # 每一步都需要约束
        # 对于不限制整数步长的方法，不能这样约束，
        # 因为每一次约束都会转换为整数域，导致扰动丢失
        # if discrete:
        #     adv = linf_img_tenosr(o, adv, epsilon)

    # 实施linf约束
    adv = l2_img_tenosr(adv)

    return adv
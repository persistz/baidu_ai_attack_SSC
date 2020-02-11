#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import numpy as np
import paddle.fluid as fluid
import os
import sys

#加载自定义文件
import models
from attack.attack_pp import FGSM, PGD, PGDL2
from utils import init_prog, save_adv_image, process_img, tensor2img, calc_mse, add_arguments, print_arguments

#######parse parameters
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# attack second time
add_arg('class_dim',        int,   121,                  "Class number.")
add_arg('start_idx',        int,   0,                    "start_idx.")
add_arg('end_idx',          int,   120,                  "end_idx.")
add_arg('num_steps',        int,   250,                  "num pgd steps.")
add_arg('eps',              float, 32.,                  "eps")
add_arg('step_size',        float, 0.01,                 "step size")
add_arg('noise_scale',      float, 0.6,                  "noise scale")
add_arg('confidence',       float, 45.,                  "early stop logit confidence")
add_arg('num_samples',      int,   5,                    "EOT samples")
add_arg('is_targeted',      int,   0,                    "0: means untargeted, 1: means targeted")
add_arg('shape',            str,   "3,224,224",          "output image shape")
add_arg('input',            str,   "./input_image/",     "Input directory with images")
add_arg('output',           str,   "./pgdl2_output_image_resnet_eot/",    "Output directory with images")
add_arg('model_name',       str,   "MobileNetV2",    "model name")
add_arg('subfix',           str,   ".jpg",    "sub fix")

args = parser.parse_args()
print_arguments(args)
# "./output_image_mobile_eot/"
# "./output_image_mobile_resnet_eot/"
######Init args
image_shape = [int(m) for m in args.shape.split(",")]
class_dim=args.class_dim
input_dir = args.input
output_dir = args.output
# if os.path.exists(output_dir) is False:
#     os.mkdir(output_dir)
model_name=args.model_name
pretrained_model="./models_parameters/" + model_name
# model_name = "MobileNetV2"
# pretrained_model="./models_parameters/MobileNetV2"
val_list = 'val_list.txt'
use_gpu=True

if args.is_targeted == 0:
    IsTarget = False
elif args.is_targeted == 1:
    IsTarget = True


######Attack graph
adv_program=fluid.Program()
NUM_SAMPLES = args.num_samples
model = models.__dict__[model_name]()

#完成初始化
with fluid.program_guard(adv_program):
    # model definition
    input_layer = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name="label", shape=[1], dtype='int64')
    #设置为可以计算梯度
    input_layer.stop_gradient=False

    # add different transformations
    loss = 0
    for i in range(NUM_SAMPLES):
        if i == 0:
            scale_ratio = 1
            random_noise = fluid.layers.uniform_random(shape=[1, 3, 224, 224], min=-0.0001,
                                                       max=0.0001)
        else:
            # scale_ratio = np.random.uniform(low=1, high=9, size=1)[0]
            scale_ratio = i*1.0
            print(scale_ratio)
        # 1. random crop
        # cropped_img = fluid.layers.random_crop(input_layer, shape=[3, 170, 170])
        # 2. random noise
            random_noise = fluid.layers.uniform_random(shape=[1, 3, 224, 224], min=-args.noise_scale, max=args.noise_scale)
        noised_img = fluid.layers.elementwise_add(input_layer, random_noise)
        # 3. random scaling
        scale_down = fluid.layers.image_resize(noised_img, scale=scale_ratio, name='scale_down_%d'%i, resample='BILINEAR')
        scale_back = fluid.layers.image_resize(scale_down, out_shape=(224, 224), name='scale_back_%d'%i, resample='BILINEAR')

        out_logits = model.net(input=scale_back, class_dim=class_dim)
        out = fluid.layers.softmax(out_logits)
        tmp_loss = fluid.layers.cross_entropy(input=out, label=label)
        loss += tmp_loss

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    # print(fluid.default_startup_program())
    #记载模型参数
    fluid.io.load_persistables(exe, pretrained_model)

#设置adv_program的BN层状态
init_prog(adv_program)

evala_program = fluid.Program()
with fluid.program_guard(evala_program):
    input_layer_eval = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    out_logits_eval = model.net(input=input_layer_eval, class_dim=class_dim)
    out_eval = fluid.layers.softmax(out_logits_eval)

#创建测试用评估模式
eval_program = evala_program.clone(for_test=True)
init_prog(eval_program)

#定义梯度
with fluid.program_guard(adv_program):
    gradients = fluid.backward.gradients(targets=loss, inputs=[input_layer])[0]

######Inference
def inference(img):
    fetch_list = [out_eval.name]

    result = exe.run(eval_program,
                     fetch_list=fetch_list,
                     feed={ 'image':img })
    result = result[0][0]
    pred_label = np.argmax(result)

    result_least = result[1:121]
    least_label = np.argmin(result_least) + 1
    pred_score = result[pred_label].copy()

    return pred_label, pred_score, least_label

######FGSM attack
#untarget attack
def attack_nontarget_by_FGSM(img, src_label, target):
    pred_label = src_label

    step = float(args.step_size)
    eps = float(args.eps)
    while pred_label == src_label:
        #生成对抗样本
        adv = PGDL2(adv_program=adv_program, eval_program=eval_program,loss=loss,
        gradients=gradients,o=img, input_layer=input_layer,output_layer=out_logits,
                  input_layer_eval=input_layer_eval,output_layer_eval=out_eval,
        step_size=step,epsilon=eps,iteration=int(args.eps/step),isTarget=IsTarget,target_label=target,src_label=src_label,
                  use_gpu=True, discrete=False, confidence=args.confidence)

        pred_label, pred_score, _ = inference(adv)
        step *= 2
        if step > eps:
            break

    print("Test-score: {0}, class {1}".format(pred_score, pred_label))

    adv_img=tensor2img(adv)
    return adv_img

####### Main #######
def get_original_file(filepath):
    with open(filepath, 'r') as cfile:
        full_lines = [line.strip() for line in cfile]
    cfile.close()
    original_files = []
    for line in full_lines:
        label, file_name = line.split()
        original_files.append([file_name, int(label)])
    return original_files

def gen_adv():
    mse = 0
    original_files = get_original_file('./input_image/' + val_list)

    target_label_list = [76, 18, 104, 36, 72, 72, 47, 92, 113, 5, 84, 74, 82, 34, 42, 84, 70, 98, 29, 87, 104, 94, 103,
                         61, 21, 83, 108, 104, 26, 112, 84, 107, 104, 45, 72, 19, 72, 75, 55, 104, 54, 104, 72, 74, 91,
                         25, 68, 107, 91,
                         41, 116, 21, 104, 56, 102, 51, 46, 87, 113, 19, 113, 85, 24, 93, 110, 102, 24, 84, 27, 38, 48,
                         43, 10, 32,
                         68, 87, 54, 12, 84, 29, 3, 13, 26, 2, 3, 106, 105, 34, 118, 66, 19, 74, 63, 42, 9, 113, 21, 6,
                         40, 40, 21, 104,
                         86, 23, 40, 12, 37, 20, 40, 12, 79, 15, 9, 48, 74, 51, 91, 79, 46, 80]
    # hard examples need use targeted attack
    for filename, label in original_files[args.start_idx:args.end_idx]:
        img_path = input_dir + filename.split('.')[0] + args.subfix
        print("Image: {0} ".format(img_path))
        img=process_img(img_path)

        target = target_label_list[label - 1]
        if IsTarget:
            print('target class', target)
        adv_img = attack_nontarget_by_FGSM(img, label, target)

        # adv_img = attack_nontarget_by_FGSM(img, label)
        image_name, image_ext = filename.split('.')

        ##Save adversarial image(.png)
        save_adv_image(adv_img, output_dir+image_name+'.png')

        org_img = tensor2img(img)
        score = calc_mse(org_img, adv_img)
        mse += score
        print('MSE %.2f'%(score))
        sys.stdout.flush()
    print("ADV {} files, AVG MSE: {} ".format(len(original_files), mse/len(original_files)))

def main():
    gen_adv()


if __name__ == '__main__':
    main()

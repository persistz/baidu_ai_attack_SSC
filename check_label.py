# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import numpy as np
import paddle.fluid as fluid
from tqdm import tqdm
# 加载自定义文件
import models
from attack.attack_pp import FGSM, PGD
from utils import init_prog, save_adv_image, process_img, tensor2img, calc_mse, add_arguments, print_arguments

#######parse parameters
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('class_dim', int, 121, "Class number.")
add_arg('model_type', int, 1, "model name type")
add_arg('noise', int, 1, "large than 0 use random noise")
add_arg('shape', str, "3,224,224", "output image shape")
add_arg('input', str, "./input_image/", "Input directory with images")
add_arg('output', str, "./pgdl2_output_image_mobile_resnet_eot_l2_15_early_stop_30_mobile_l2_5/", "Output directory with images")
# "output_image_resnet_eot_eps_32"
# "./input_image/"
args = parser.parse_args()
print_arguments(args)

######Init args
image_shape = [int(m) for m in args.shape.split(",")]
class_dim = args.class_dim
input_dir = args.input
output_dir = args.output
model_name_list = ["ResNeXt50_32x4d",
				   "MobileNetV2",
				   "EfficientNetB0",
				   "Res2Net50_26w_4s",
				   "SE_ResNet34_vd",
				   "ShuffleNetV2_x2_0",
				   "MobileNetV2_x2_0",
				   "ResNet50",
				   "SE_ResNet50_vd",
				   "ResNeXt50_vd_32x4d",
				   "MobileNetV1",
				   "ShuffleNetV2",
				   "ResNet34",
				   "VGG19",
				   "VGG16",
				   "SqueezeNet1_1",
				   "DPN92",
				   ]
# model_name = "ResNeXt50_32x4d"
# model_name = "MobileNetV2"
# model_name = "EfficientNetB0"
# model_name = "Res2Net50_26w_4s"
model_name = model_name_list[args.model_type]
print('model name: ', model_name)
pretrained_model="./models_parameters/" + model_name
val_list = 'val_list.txt'
use_gpu = True

######Attack graph
adv_program = fluid.Program()

# 完成初始化
with fluid.program_guard(adv_program):
	input_layer = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
	# 设置为可以计算梯度
	input_layer.stop_gradient = False

	# model definition
	model = models.__dict__[model_name]()
	if args.noise >= 10:
		print('uniform noise')
		random_noise = fluid.layers.uniform_random(shape=[1, 3, 224, 224], min=-.6, max=.6, seed=0)
		noised_img = fluid.layers.elementwise_add(input_layer, random_noise)
		out_logits = model.net(input=noised_img, class_dim=class_dim)
	elif args.noise >= 1:
		print('gaussian noise')
		gaussian_noise = fluid.layers.gaussian_random_batch_size_like(input_layer, shape=[-1, 3, 224, 224], mean=0., std=0.45)
		noised_img = fluid.layers.elementwise_add(input_layer, gaussian_noise)
		out_logits = model.net(input=noised_img, class_dim=class_dim)
	else:
		out_logits = model.net(input=input_layer, class_dim=class_dim)
	out = fluid.layers.softmax(out_logits)

	place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
	exe = fluid.Executor(place)
	exe.run(fluid.default_startup_program())

	# 记载模型参数
	fluid.io.load_persistables(exe, pretrained_model)

# 设置adv_program的BN层状态
init_prog(adv_program)

# 创建测试用评估模式
eval_program = adv_program.clone(for_test=True)

# 定义梯度
with fluid.program_guard(adv_program):
	label = fluid.layers.data(name="label", shape=[1], dtype='int64')
	loss = fluid.layers.cross_entropy(input=out, label=label)
	gradients = fluid.backward.gradients(targets=loss, inputs=[input_layer])[0]


######Inference
'''
def inference(img):
	fetch_list = [out.name]

	result = exe.run(eval_program,
					 fetch_list=fetch_list,
					 feed={ 'image':img })
	result = result[0][0]
	pred_label = np.argmax(result)
	pred_score = result[pred_label].copy()

	# add by s3l
	least_class = np.argmin(result)
	return pred_label, pred_score, least_class
'''

# inference for ResNet
def inference(img):
	fetch_list = [out.name]

	result = exe.run(eval_program,
					 fetch_list=fetch_list,
					 feed={ 'image':img })
	result = result[0][0]
	pred_label = np.argmax(result)

	result_least = result[1:121]
	least_label = np.argmin(result_least) + 1
	pred_score = result[pred_label].copy()

	return pred_label, pred_score, least_label


# output logits
def inference_logits(img):
	fetch_list = [out_logits.name]

	result = exe.run(eval_program,
					 fetch_list=fetch_list,
					 feed={ 'image':img })

	logits_value = result[0][0]

	return logits_value


######FGSM attack
# untarget attack
def attack_nontarget_by_FGSM(img, src_label):
	pred_label = src_label

	step = 8.0 / 256.0
	eps = 32.0 / 256.0
	while pred_label == src_label:
		# 生成对抗样本
		# adv=FGSM(adv_program=adv_program,eval_program=eval_program,gradients=gradients,o=img,
		#          input_layer=input_layer,output_layer=out,step_size=step,epsilon=eps,
		#          isTarget=False,target_label=0,use_gpu=use_gpu)

		adv = PGD(adv_program=adv_program, eval_program=eval_program,
				  gradients=gradients, o=img, input_layer=input_layer, output_layer=out,
				  step_size=2.0 / 256, epsilon=16.0 / 256, iteration=20, isTarget=True, target_label=100, use_gpu=True)

		pred_label, pred_score = inference(adv)
		step *= 2
		if step > eps:
			break

	print("Test-score: {0}, class {1}".format(pred_score, pred_label))

	adv_img = tensor2img(adv)
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
	print('gen adv')

	mse = 0
	adv_files = get_original_file(input_dir + val_list)
	print("read original files", len(adv_files))

	target_label_list = [76, 18, 104, 36, 72, 72, 47, 92, 113, 5, 84, 74, 82, 34, 42, 84, 70,
						 98, 29, 87, 104, 94, 103, 61, 21, 83, 108, 104, 26, 112, 84, 107, 104,
						 45, 72, 19, 72, 75, 55, 104, 54, 104, 72, 74, 91, 25, 68, 107, 91,
						 41, 116, 21, 104, 56, 102, 51, 46, 87, 113, 19, 113, 85, 24, 93, 110,
						 102, 24, 84, 27, 38, 48, 43, 10, 32, 68, 87, 54, 12, 84, 29, 3, 13, 26,
						 2, 3, 106, 105, 34, 118, 66, 19, 74, 63, 42, 9, 113, 21, 6, 40, 40, 21, 104,
						 86, 23, 40, 12, 37, 20, 40, 12, 79, 15, 9, 48, 74, 51, 91, 79, 46, 80]

	least_list = []
	counter = 0
	unt_counter = 0
	# 记录logits
	logits_list = []
	mse_list = []
	# 记录max和second logit的差值
	logits_diff = []
	count = 0
	for filename, label in tqdm(adv_files):
		if args.output == 'input_image/':
			img_path = output_dir + filename.split('.')[0] + '.jpg'
		else:
			img_path = output_dir + filename.split('.')[0] + '.png'

		# print("Image: {0} ".format(img_path))

		# !ssize.empty() in function 'resize'
		img = process_img(img_path)

		# print('Image range', np.min(img), np.max(img))

		pred_label, pred_score, least_class = inference(img)
		logits = inference_logits(img)
		# print(logits)
		return_logits = np.sort(logits)[::-1]
		logits_list.append(return_logits)
		# logits_diff.append(return_logits[0] - return_logits[1])
		logits_diff.append(return_logits[0] - logits[label])

		# print("Test-score: {0}, class {1}".format(pred_score, pred_label))

		# if pred_label == target_label_list[label - 1]:
		# 	counter += 1
		if pred_label != label:
			# print("Failed target image: {0} ".format(img_path))
			unt_counter += 1
		else:
			# print("Serious!!!Failed untarget image: {0} ".format(img_path))
			pass

		least_list.append(least_class)

		# adv_img = attack_nontarget_by_FGSM(img, label)
		# image_name, image_ext = filename.split('.')
		## Save adversarial image(.png)
		# save_adv_image(adv_img, output_dir + image_name + '.png')

		## check MSE
		# org_img = tensor2img(img)
		org_filename = filename.split('.')[0] + '.jpg'
		org_img_path = input_dir + org_filename
		org_img = process_img(org_img_path)
		score = calc_mse(tensor2img(org_img), tensor2img(img))
		mse_list.append(score)
		mse += score
		count += 1


	# print('Least likely list', least_list)
	# print('logits', logits_list)
	print('logits diff: ')
	for i, logit in enumerate(logits_diff):
		if logit < 0.001:
			print('id: %d, mse: %.10f, diff logits: %.2f, ******************'%(i+1, mse_list[i], logit))
		else:
			print('id: %d, mse: %.10f, diff logits: %.2f'%(i+1, mse_list[i], logit))
	print('logits diff', np.mean(logits_diff))
	print('The LL success number is', counter)
	print('The untargeted success number is', unt_counter)
	print("AVG MSE: {} ", mse / count)


def main():
	print('Run...')
	gen_adv()

if __name__ == '__main__':
	main()
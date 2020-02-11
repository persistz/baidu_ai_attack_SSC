export FLAGS_fraction_of_gpu_memory_to_use=0.9

model_name_list=("ResNeXt50_32x4d"
                 "MobileNetV2"
                 "EfficientNetB0"
                 "Res2Net50_26w_4s"
                 "SE_ResNet34_vd"
                 "ShuffleNetV2_x2_0"
                 "MobileNetV2_x2_0"
                 "SE_ResNet50_vd"
                 "ResNet50"
                 "SqueezeNet1_1"
                 "ResNeXt50_vd_32x4d"
                 "ShuffleNetV2"
                 "ResNet34"
                 "VGG19"
                 "VGG16"
                 "MobileNetV1"
                 )
fold_prefix=img_pgdl2_without_eot
log_prefix=log_pgdl2_without_eot

num=${#model_name_list[@]}
GPU_ID=0
for id in `seq 0 $((num-1))`
    do
        mkdir ${fold_prefix}_m${id}
        echo ${id}
        echo ${model_name_list[$id]}
    done


# gray box model is ResNeXt50, so we only apply EoT on ResNeXt50_32x4d model
CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python attack_pgd_l2.py \
    --input="./input_image/" \
    --output=./${fold_prefix}_m0/\
    --model_name=ResNeXt50_32x4d \
    --subfix=.jpg \
    --eps=20.0 \
    --confidence=20 \
    --step_size=0.01 \
    --is_targeted=0 \
    --num_samples=5 \
    --noise_scale=0.6 \
    --start_idx=0 \
    --end_idx=120 > ${log_prefix}_0_${GPU_ID}.txt &

wait

for m in `seq 1 $((num-1))`
    do
        wait
        CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python attack_pgd_l2.py \
            --input=./${fold_prefix}_m$((m-1))/ \
            --output=./${fold_prefix}_m${m}/ \
            --model_name=${model_name_list[$m]} \
            --subfix=.png \
            --eps=20.0 \
            --confidence=15 \
            --step_size=0.01 \
            --is_targeted=0 \
            --num_samples=1 \
            --noise_scale=0.00001 \
            --start_idx=0 \
            --end_idx=120 > ${log_prefix}_${m}_${GPU_ID}.txt &
    done

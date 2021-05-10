#!/bin/bash
#YBATCH -r am_8
#SBATCH -N 1
#SBATCH -J vit_finetuning_224_fake_1k_to_CIFAR10
#SBATCH --output output/%j.out

. /etc/profile.d/modules.sh
module load openmpi/3.1.6 cuda/11.1 cudnn/cuda-11.1/8.0

echo 'Hello World'

export NUM_PROC=8
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py ./ \
    --pretrained \
    --pretrained-path ./train_result/PreTraining_vit_deit_base_patch16_224_fake_1k/model_best.pth.tar \
    --dataset CIFAR10 \
    --num-classes 10 \
    --model vit_deit_base_patch16_224 \
    --input-size 3 224 224 \
    --opt sgd \
    --batch-size 96 \
    --epochs 1000 \
    --cooldown-epochs 0 \
    --lr 0.01 \
    --sched cosine \
    --warmup-epochs 5 \
    --weight-decay 0.0001 \
    --smoothing 0.1 \
    --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug \
    --mixup 0.8 \
    --cutmix 1.0 \
    --log-wandb \
    --output train_result \
    --experiment finetuning_vit_deit_base_patch16_224_fake_1k_to_CIFAR10 \
    -j 8

echo 'Hello World'
#!/bin/bash
#YBATCH -r am_8
#SBATCH -N 1
#SBATCH -J vit_finetuning
#SBATCH --output output/%j.out

. /etc/profile.d/modules.sh
module load openmpi/3.1.6 cuda/11.1 cudnn/cuda-11.1/8.0

echo 'Hello world'

export NUM_PROC=8
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py /mnt/nfs/datasets/ILSVRC2012/train \
    --model vit_deit_tiny_patch16_224 \
    --opt adamw \
    --batch-size 128 \
    --epochs 300 \
    --cooldown-epochs 0 \
    --lr 0.001 \
    --sched cosine \
    --warmup-epochs 5 \
    --weight-decay 0.05 \
    -j 8 \
    --amp \
    --seed 1234 \
    --log-wandb \
    --output train_result \
    --experiment FineTuning_vit_deit_tiny_patch16_224_1k \
    --pretrained

echo 'Hello world'
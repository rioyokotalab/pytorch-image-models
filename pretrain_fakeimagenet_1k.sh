#!/bin/bash
#YBATCH -r am_8
#SBATCH -N 1
#SBATCH -J vit_pretraining_fake
#SBATCH --output output/%j.out

. /etc/profile.d/modules.sh
module load openmpi/3.1.6 cuda/11.1 cudnn/cuda-11.1/8.0

export NUM_PROC=8
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py /mnt/nfs/datasets/FakeImageNet1k_v1 \
    --model vit_deit_base_patch16_224 \
    --opt adamw \
    --batch-size 128 \
    --epochs 300 \
    --cooldown-epochs 0 \
    --lr 0.001 \
    --sched cosine \
    --warmup-epochs 5 \
    --weight-decay 0.05 \
    --smoothing 0.1 \
    --drop-path 0.1 \
    --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug \
    --mixup 0.8 \
    --cutmix 1.0 \
    --reprob 0.25 \
    --log-wandb \
    --output train_result \
    --experiment PreTraining_vit_deit_base_patch16_224_fake_1k \
    -j 8

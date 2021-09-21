#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=04:00:00
#$ -j y
#$ -o output/o.$JOB_ID

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load cuda/10.2/10.2.89 cudnn openmpi nccl/2.7/2.7.8-1

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export NGPUS=16
export NPERNODE=4
mpirun -npernode $NPERNODE -np $NGPUS \
python train.py ./ --dataset CIFAR10 \
    --model deit_small_patch16_224 --experiment finetune_deit_small_cifar10_fractal21k_50epoch \
    --input-size 3 224 224 --num-classes 10 \
    --sched cosine_iter --epochs 1000 --lr 0.01 --weight-decay 0.0001 \
    --batch-size 48 --opt sgd \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 \
    -j 4 \
    --pretrained-path /groups/gcc50533/acc12016yi/pytorch-image-models/output/train/pretrain_deit_small_fractal21k_128gpus_cooldown_50/last.pth.tar \
    --output /groups/gcc50533/acc12016yi/pytorch-image-models/output/train \
    --log-wandb

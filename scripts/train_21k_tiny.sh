#!/bin/bash
#$ -cwd
#$ -l rt_F=32
#$ -l h_rt=23:30:00
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

export NGPUS=128
export NUM_PROC=4
mpirun -npernode $NUM_PROC -np $NGPUS \
python train_without_eval.py /groups/gcc50533/imnet/ImageNet21k \
    --model deit_tiny_patch16_224 --experiment pretrain_deit_tiny_imagenet21k_128gpus_300epochs \
    --sched cosine_iter --epochs 300 --lr 8.0e-3 --weight-decay 0.05 \
    --batch-size 64 --opt adamw --num-classes 21841 \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    -j 64 --eval-metric loss \
    --hold-epochs 50 100 150 200 250 \
    --output /groups/gcc50533/acc12016yi/pytorch-image-models/output/train \
    --log-wandb
#    --resume /groups/gcc50533/acc12016yi/pytorch-image-models/output/train/pretrain_deit_base_imagenet21k_128gpus_300epochs/last.pth.tar \
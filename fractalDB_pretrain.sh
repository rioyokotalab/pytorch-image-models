#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/o.$JOB_ID

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load openmpi/3.1.6 cuda/11.1 cudnn/8.0 nccl/2.7

# export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | cut -d " " -f 6 | cut -d "/" -f 1)

export NGPUS=16
# batch-size = 1024 / NGPUS
export NUM_PROC=4
mpirun -npernode $NUM_PROC -np $NGPUS \
python fractalDB_pretrain.py /groups/gcd50691/datasets/FractalDB-1k-Gray \
    --model vit_deit_tiny_patch16_224 \
    --opt adamw \
    --batch-size 64 \
    --epochs 300 \
    --cooldown-epochs 0 \
    --lr 6.0e-4 \
    --sched cosine \
    --warmup-epochs 10 \
    --weight-decay 0.05 \
    --smoothing 0.1 \
    --drop-path 0.1 \
    --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug \
    --mixup 0.8 \
    --cutmix 1.0 \
    --reprob 0.25 \
    --remode pixel \
    --interpolation bicubic \
    --hflip 0.0 \
    --eval-metric loss \
    --log-wandb \
    --output train_result \
    --experiment PreTraining_vit_deit_tiny_patch16_224_fractalDB_1k_gray \
    -j 8

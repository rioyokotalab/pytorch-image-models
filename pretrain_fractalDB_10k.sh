#!/bin/bash
#$ -cwd
#$ -l rt_F=32
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

export NGPUS=128
export NUM_PROC=4
mpirun -npernode $NUM_PROC -np $NGPUS \
python fractalDB_pretrain.py /groups/gcd50691/datasets/FractalDB-10k-Color \
    --model vit_deit_tiny_patch16_224 \
    --num-classes 10000 \
    --opt adamw \
    --batch-size 64 \
    --epochs 30 \
    --cooldown-epochs 0 \
    --lr 8.0e-3 \
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
    --remode pixel \
    --interpolation bicubic \
    --hflip 0.0 \
    --eval-metric loss \
    --log-wandb \
    --output /groups1/gcc50533/acc12015ij/train_result \
    --experiment PreTraining_vit_deit_tiny_patch16_224_fractalDB_10k_color_bs=64_128_epochs=30 \
    -j 4

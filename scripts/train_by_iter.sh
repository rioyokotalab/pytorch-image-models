#!/bin/bash
#$ -cwd
#$ -l rt_F=2
#$ -l h_rt=30:00:00
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

export NGPUS=8
export NUM_PROC=4
mpirun -npernode $NUM_PROC -np $NGPUS \
python train_by_iter.py /groups/gcc50533/imnet/ILSVRC2012 \
    --model deit_tiny_patch16_224 --experiment pretrain_deit_tiny_imnet_test \
    --input-size 3 224 224 --num-classes 1000 \
    --sched cosine_iter --epochs 3 --lr 0.001 --weight-decay 0.05 \
    --batch-size 32 --opt adamw \
    --warmup-iter 1000 --cooldown-epochs 0 \
    --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 \
    --drop-path 0.1 --reprob 0.25 \
    -j 8 --eval-metric loss \
    --recovery-interval 100 \
    --output /groups/gcc50533/acc12016yi/pytorch-image-models/output/train \
    --log-wandb
#    --resume /groups/gcc50533/acc12016yi/pytorch-image-models/output/train/pretrain_deit_tiny_imnet_test/recovery-1-599-iter.pth.tar \
#    --log-wandb

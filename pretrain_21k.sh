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
# global_mini-batch_size = batch-size * NGPUS
export NUM_PROC=4
mpirun -npernode $NUM_PROC -np $NGPUS \
python pretrain_21k.py /groups/gca50014/imnet/ImageNet21k \
    --num-classes 21841 \
    --model vit_deit_tiny_patch16_224 \
    --opt adamw \
    --batch-size 64 \
    --epochs 30 \
    --cooldown-epochs 0 \
    --lr 0.008 \
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
    --eval-metric loss \
    --log-wandb \
    --output /groups1/gcc50533/acc12015ij/train_result \
    --experiment PreTraining_vit_deit_tiny_patch16_224_21k_bs=64_128_epochs=30 \
    -j 8

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
python finetune_with_separate.py /groups/gca50014/imnet/ILSVRC2012 \
    --model vit_deit_tiny_patch16_224 \
    --initial-checkpoint train_result/Pretrain_vit_deit_tiny_patch16_224_fake1k_label_rate=0.0/model_best.pth.tar \
    --opt adamw \
    --num-classes 1000 \
    --batch-size 64 \
    --epochs 100 \
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
    --experiment Finetuning_with_imnet1k_from_vit_deit_tiny_patch16_224_fake1k_label_rate=0.0 \
    -j 4

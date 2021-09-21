#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=24:00:00
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
python train_with_fake.py /groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs --val-split val_50 \
    --model vit_deit_tiny_patch16_224 --experiment pretrain_deit_tiny_with_fake_imagenet_0.5fake \
    --sched cosine --epochs 300 --lr 0.001 --weight-decay 0.05 \
    --batch-size 32 --opt adamw --num-classes 2000 \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    -j 8 --amp --split-loss --weight 0.5 \
    --resume output/train/pretrain_deit_tiny_with_fake_imagenet_0.5fake/last.pth.tar \
    --log-wandb

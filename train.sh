#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=01:00:00
#$ -j y
#$ -o output/o.$JOB_ID

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load openmpi cuda/11.1 cudnn nccl/2.7/2.7.8-1

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export NUM_PROC=4
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py /groups/gca50014/imnet/ILSVRC2012 \
    --model vit_base_patch16_224 --experiment finetune_vit_base \
    --sched cosine --epochs 8 --lr 0.01 --weight-decay 0 \
    --batch-size 64 --opt sgd --clip-grad 1 --cooldown-epochs 0 \
    --warmup-epochs 0 -j 4 --amp \
    --pretrained --num-classes 1000
#    --log-wandb
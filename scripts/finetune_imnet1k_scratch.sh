#!/bin/bash
#$ -cwd
#$ -l rt_F=8
#$ -l h_rt=60:00:00
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

export MODEL=base
export OUT_DIR=/groups/gcc50533/check_points/${MODEL}/scratch

export NGPUS=32
export NPERNODE=4
mpirun -npernode $NPERNODE -np $NGPUS \
python train.py /groups/gcc50533/imnet/ILSVRC2012 \
    --model deit_${MODEL}_patch16_224 --experiment finetune_deit_${MODEL}_imnet1k_scratch_seed \
    --input-size 3 224 224 --num-classes 1000 \
    --sched cosine_iter --epochs 300 --lr 0.001 --weight-decay 0.05 \
    --batch-size 32 --opt adamw \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 \
    --drop-path 0.1 --reprob 0.25 -j 16 \
    --output ${OUT_DIR} \
    --log-wandb \
    --seed 20
#    --resume ${OUT_DIR}/finetune_deit_${MODEL}_imnet1k_${DATA}${CLASSES}k/last.pth.tar

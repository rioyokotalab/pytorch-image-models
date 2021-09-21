#!/bin/bash
#$ -cwd
#$ -l rt_F=64
#$ -l h_rt=13:20:00
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

export NGPUS=256
export NUM_PROC=4
mpirun -npernode $NUM_PROC -np $NGPUS \
python train_with_wds.py /groups/gca50014/Fractal/edRender/x256/FractalDB-1000_PATCHGR \
    --model deit_base_patch16_224 --experiment pretrain_deit_base_fractal100k_256gpus_shards \
    --sched cosine_iter --epochs 20 --lr 1.0e-3 --weight-decay 0.05 \
    --batch-size 32 --opt adamw --num-classes 100000 \
    --warmup-iter 30000 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    -j 64 --eval-metric loss \
    -w --trainshards "/bb/grandchallenge/gad50725/dataset/shardTest_100k/fdb_100k_shards/FractalDB_{000000..000999}.tar" \
    --trainsize 100000 --no-prefetcher \
    --output /groups/gcc50533/acc12016yi/pytorch-image-models/output/train \
    --recovery-interval 2000 --log-wandb

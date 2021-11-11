#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/pretrain_tiny_mvf10k_1e-3_$JOB_ID.out

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load cuda/10.2/10.2.89 cudnn openmpi nccl/2.7/2.7.8-1

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export MODEL=tiny
export LR=1.0e-3
export CLS=10
export EPOCHS=300
export OUT_DIR=/groups/gcc50533/check_points/${MODEL}/mvf${CLS}k/pre_training

export NGPUS=4
export NUM_PROC=4
mpirun -npernode $NUM_PROC -np $NGPUS \
python train_with_wds.py /NOT/WORKING \
    -w --trainshards "/groups/gcd50691/datasets/MVFractal_Shards_10k/mvf_10k-train-{000000..000999}.tar" \
    --model deit_${MODEL}_patch16_224 --experiment pretrain_deit_${MODEL}_MVfractal${CLS}k_${LR}_shards_bs256 \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --batch-size 64 --opt adamw --num-classes ${CLS}000 \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    -j 1 --eval-metric loss --no-prefetcher \
    --hold-epochs 10 --output ${OUT_DIR}
    # --log-wandb

#!/bin/bash
#$ -cwd
#$ -l rt_F=32
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/pretrain_swin_base_rc21k_1e-3_$JOB_ID.out

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load cuda/10.2/10.2.89 cudnn openmpi nccl/2.7/2.7.8-1

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export MODEL=base
export LR=1.0e-3
export CLS=21
export EPOCHS=90
export OUT_DIR=/groups/gcc50533/check_points/swin_${MODEL}/rc${CLS}k/pre_training

export NGPUS=128
export NUM_PROC=4
mpirun -npernode $NUM_PROC -np $NGPUS \
python train_with_wds.py /NOT/WORKING \
    -w --trainshards "/groups/gcd50691/datasets/RCDB_Shards_21k/rcdb_21k-train-{000000..002099}.tar" \
    --model swin_${MODEL}_patch4_window7_224 --experiment pretrain_deit_${MODEL}_RCDB${CLS}k_${LR}_shards_bs4096 \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.01 \
    --batch-size 32 --opt adamw --num-classes ${CLS}000 \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    -j 1 --eval-metric loss --no-prefetcher \
    --hold-epochs 10 --output ${OUT_DIR} \
    --log-wandb \
    --resume /groups/gcc50533/check_points/swin_base/rc21k/pre_training/pretrain_deit_base_RCDB21k_1.0e-3_shards_bs4096/last.pth.tar

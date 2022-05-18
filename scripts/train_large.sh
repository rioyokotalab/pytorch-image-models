#!/bin/bash
#$ -cwd
#$ -l rt_F=128
#$ -l h_rt=23:50:00
#$ -j y
#$ -o output/pretrain_large_mvf21k_5e-4_8192_$JOB_ID.out

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load openmpi cuda/10.2/10.2.89 cudnn nccl/2.7/2.7.8-1

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export MODEL=large

# CHANGE -o output/pretrain_large_{DSET}_5e-4_8192_$JOB_ID.out
# CHANGE DSET, CLS, EPOCHS

export LR=5.0e-4
export DSET=mvf # {i, mvf, rc}
export CLS=21 # {21, 50, 100, 300}
export EPOCHS=90 # {90, 40, 20, 7}
export OUT_DIR=/groups/gcc50533/check_points/${MODEL}/${DSET}${CLS}k/pre_training

# CHANGE --trainshards
# /groups/gcd50691/datasets/MVFractal_Shards_21k/mvf_21k-train-{000000..002099}.tar
# /groups/gcd50691/datasets/MVFractal_Shards_50k/mvf_50k-train-{000000..004999}.tar
# /groups/gcd50691/datasets/MVFractal_Shards_100k/mvf_100k-train-{000000..009999}.tar
# /groups/gcd50691/datasets/MVFractal_Shards_300k/mvf_300k-train-{000000..029999}.tar
# /groups/gcd50691/datasets/RCDB_Shards_21k/rcdb_21k-train-{000000..002099}.tar
# /groups/gcd50691/datasets/ImageNet_Shards_21k/imnet_21k-train-{000000..001419}.tar

export NGPUS=512
export NUM_PROC=4
mpirun -npernode $NUM_PROC -np $NGPUS \
python train_with_wds.py /NOT/WORKING \
    -w --trainshards "/groups/gcd50691/datasets/MVFractal_Shards_21k/mvf_21k-train-{000000..002099}.tar" \
    --model vit_${MODEL}_patch16_224 --experiment pretrain_vit_${MODEL}_${DSET}${CLS}k_${LR}_bs8192_shards \
    --sched linear --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --batch-size 16 --opt adamw --num-classes ${CLS}000 \
    --warmup-iter 10000 --warmup-lr 1e-6 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    -j 1 --eval-metric loss --no-prefetcher \
    --hold-epochs 10 --output ${OUT_DIR} \
    --log-wandb
    
#     --resume ${OUT_DIR}/pretrain_deit_${MODEL}_MVfractal${CLS}k_${LR}_shards/last.pth.tar

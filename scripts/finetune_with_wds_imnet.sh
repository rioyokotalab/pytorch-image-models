#!/bin/bash
#$ -cwd
#$ -l rt_F=32
#$ -l h_rt=02:50:00
#$ -j y
#$ -o output/o.$JOB_ID

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load openmpi cuda/10.2/10.2.89 cudnn nccl/2.7/2.7.8-1

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export MODEL=base
export PRE_CLS=21
export PRE_LR=1.0e-3
export PRE_EPOCHS=90
export PRE_BATCH=8192

# export CP_DIR=/groups/gcc50533/check_points/${MODEL}/mvf21k/pre_training/pretrain_deit_base_MVfractal21k_shards/model_best.pth.tar
export CP_DIR=/groups/gcd50691/yokota_check_points/base/mvf21k/pre_training/model_best.pth.tar
# export OUT_DIR=/bb/grandchallenge/gae50854/yokota_check_points/fine_tuning
# export OUT_DIR=/groups/gcc50533/check_points/${MODEL}/mvf${PRE_CLS}k/fine_tuning
export OUT_DIR=/groups/gcd50691/yokota_check_points/base/mvf21k/fine_tuning

export NGPUS=128
export NPERNODE=4
export BATCH_SIZE=2048
export LOCAL_BATCH_SIZE=16
export LR=2.4e-1
export RES=224
export CHUNK=None
export EPOCHS=150

mpirun -npernode $NPERNODE -np $NGPUS \
python train_with_wds_eval.py /groups/gcd50691/datasets/ImageNet \
    --model deit_${MODEL}_patch16_224 --experiment finetune_vit_base_i1k_${RES}_from_MVfractal${PRE_CLS}k_lr${PRE_LR}_epochs${PRE_EPOCHS}_bs${PRE_BATCH}_shards_ftlr${LR}_ftbs${BATCH_SIZE}_ftlbs${LOCAL_BATCH_SIZE}_ftep${EPOCHS}_without_mu_shard \
    --input-size 3 224 224 --num-classes 1000 \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.00005 \
    --batch-size ${LOCAL_BATCH_SIZE} --opt sgd --momentum 0.9 \
    --clip-grad 1 \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.0 --cutmix 1.0 \
    --drop-path 0.1 --reprob 0.25 \
    -j 1 --no-prefetcher \
    -w --trainshards "/groups/gcd50691/datasets/ImageNet_shuffled_1GB/imagenet-train-{000000..000146}.tar" \
    --output ${OUT_DIR} \
    --pretrained-path ${CP_DIR} \
    --log-wandb

    # --gradient-ckp ${CHUNK} \
    # --resume  ${OUT_DIR}/finetune_deit_with_vit_large_i1k_${RES}_from_ImageNet${PRE_CLS}k_lr${PRE_LR}_epochs${PRE_EPOCHS}_bs${PRE_BATCH}_shards/last.pth.tar
    # --weight-decay 0.00005 \
    # --warmup-epochs 5 --cooldown-epochs 0 \
    # --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
    # --repeated-aug --mixup 0.8 --cutmix 1.0 \
    # --drop-path 0.1 --reprob 0.25 \

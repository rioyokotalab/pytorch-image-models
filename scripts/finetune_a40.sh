#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=24:00:00
#$ -j y
#$ -o output/finetune_tiny_a40_mvf21k_$JOB_ID.out

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load cuda/10.2/10.2.89 cudnn openmpi nccl/2.7/2.7.8-1

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export MODEL=tiny
# choose from {RCDB, MVfractal, fractal, imagenet}
export DATA=MVfractal
export CLASSES=21
export PRE_LR=8.0e-3

if [ ${DATA} = "fractal" ]; then
  export CP_DIR=/groups/gcc50533/check_points/${MODEL}/${CLASSES}k/pre_training/pretrain_deit_${MODEL}_${DATA}${CLASSES}000/model_best.pth.tar
  export OUT_DIR=/groups/gcc50533/check_points/${MODEL}/${CLASSES}k/fine_tuning
elif [ ${DATA} = "MVfractal" ]; then
  export CP_DIR=/groups/gcc50533/check_points/${MODEL}/mvf${CLASSES}k/pre_training/pretrain_deit_${MODEL}_${DATA}${CLASSES}k_${PRE_LR}_shards/model_best.pth.tar
  export OUT_DIR=/groups/gcc50533/check_points/${MODEL}/mvf${CLASSES}k/fine_tuning
elif [ ${DATA} = "RCDB" ]; then
  export CP_DIR=/groups/gcc50533/check_points/${MODEL}/rc${CLASSES}k/pre_training/pretrain_deit_${MODEL}_${DATA}${CLASSES}k_${PRE_LR}_shards/model_best.pth.tar
  export OUT_DIR=/groups/gcc50533/check_points/${MODEL}/rc${CLASSES}k/fine_tuning
else
  export CP_DIR=/groups/gcc50533/check_points/${MODEL}/i${CLASSES}k/pre_training/pretrain_deit_${MODEL}_${DATA}${CLASSES}k/model_best.pth.tar
  export OUT_DIR=/groups/gcc50533/check_points/${MODEL}/i${CLASSES}k/fine_tuning
fi

export NGPUS=16
export NPERNODE=4
mpirun -npernode $NPERNODE -np $NGPUS \
python train.py /groups/gcd50691/datasets/Action40 \
    --model deit_${MODEL}_patch16_224 --experiment finetune_deit_${MODEL}_a40_${DATA}${CLASSES}k \
    --input-size 3 224 224 --num-classes 40 \
    --sched cosine_iter --epochs 1000 --lr 0.01 --weight-decay 0.0001 \
    --batch-size 48 --opt sgd \
    --warmup-epochs 10 --cooldown-epochs 0 \
    --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 \
    -j 16 \
    --output $OUT_DIR \
    --log-wandb \
    --pretrained-path $CP_DIR
#    --resume ${OUT_DIR}/finetune_deit_${MODEL}_cifar10_${DATA}${CLASSES}k/last.pth.tar

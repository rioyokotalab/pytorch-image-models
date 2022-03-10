#!/bin/bash
#$ -cwd
#$ -l rt_F=32
#$ -l h_rt=72:00:00
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

export MODEL=tiny
export LR=8.0e-3
export DSET=rc21k
if [ ${DSET} = "imnet21k" ]; then
  export DATA_DIR=/groups/gcd50691/datasets/ImageNet21k
  export CLASSES=21841
elif [ ${DSET} = "mvf21k" ]; then
  export DATA_DIR=/groups/gcd50691/datasets/MV-FractalDB/var0.05/MVFractaDB21k
  export CLASSES=21000
elif [ ${DSET} = "rc21k" ]; then
  export DATA_DIR=/groups/gcd50691/datasets/RadialContourDB/RCDB-21k_hayamizu/image
  export CLASSES=21000
fi

export NGPUS=128
export NPERNODE=4
mpirun -npernode $NPERNODE -np $NGPUS \
python train_without_eval.py $DATA_DIR \
    --model autoformer_${MODEL}_patch16_224 --experiment pretrain_autoformer_${MODEL}_${DSET}_qkvbias \
    --input-size 3 224 224 --num-classes $CLASSES \
    --sched cosine_iter --epochs 90 --lr $LR --weight-decay 0.05 \
    --batch-size 64 --opt adamw \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 \
    --drop-path 0.1 --reprob 0.25 \
    -j 64 --eval-metric loss \
    --output /groups/gcc50533/acc12016yi/AutoFormer/output \
    --log-wandb --project-name AutoFormer --wandb-group pretrain \
    --resume /groups/gcc50533/acc12016yi/AutoFormer/output/pretrain_autoformer_${MODEL}_${DSET}_qkvbias/last.pth.tar \
    --remode pixel --interpolation bicubic --hflip 0.0
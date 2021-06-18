#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/o.$JOB_ID

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load openmpi/3.1.6 cuda/11.1 cudnn/8.0 nccl/2.7

# export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | cut -d " " -f 6 | cut -d "/" -f 1)

export NGPUS=16
# batch-size = 256 / NGPUS
export NUM_PROC=4
mpirun -npernode $NUM_PROC -np $NGPUS \
python train_sam.py ./ \
    --model wide_resnet28_10 \
    --dataset CIFAR10 \
    --num-classes 10 \
    --input-size 3 32 32 \
    --opt sam \
    --sync-bn \
    --batch-size 16 \
    --momentum 0.9 \
    --epochs 200 \
    --cooldown-epochs 0 \
    --lr 0.1 \
    --sched cosine \
    --weight-decay 5e-4 \
    --log-wandb \
    --output /groups1/gcc50533/acc12015ij/trained_models/Vit_deit_224_16_SAM \
    --experiment SAM_vit_deit_tiny_patch16_224_1k_cos_without_optional \
    --nbs 0.05 \
    -j 8

    # --warmup-epochs 5 \
    # --smoothing 0.1 \
    # --aa v0 \
    # --repeated-aug \
    # --mixup 0.8 \
    # --cutmix 1.0 \
    # --reprob 0.25 \

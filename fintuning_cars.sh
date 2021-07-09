#!/bin/bash
#$ -cwd
#$ -l rt_F=1
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

export NGPUS=4
# batch-size = 768 / NGPUS
export NUM_PROC=4
mpirun -npernode $NUM_PROC -np $NGPUS \
python train.py ./ \
    --pretrained \
    --pretrained-path ./train_result/PreTraining_vit_deit_tiny_patch16_224_fractalDB_1k_gray/model_best.pth.tar \
    --dataset CARS \
    --val-split val \
    --num-classes 196 \
    --model vit_deit_tiny_patch16_224 \
    --input-size 3 224 224 \
    --opt sgd \
    --batch-size 192 \
    --epochs 1000 \
    --cooldown-epochs 0 \
    --lr 0.01 \
    --sched cosine \
    --warmup-epochs 5 \
    --weight-decay 0.0001 \
    --smoothing 0.1 \
    --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug \
    --mixup 0.8 \
    --cutmix 1.0 \
    --log-wandb \
    --output train_result \
    --experiment finetuning_vit_deit_tiny_patch16_224_fractalDB_1k_gray_to_Cars \
    -j 4

    # --pretrained-path ./train_result/PreTraining_vit_deit_tiny_patch16_224_fractalDB_1k_gray/model_best.pth.tar \
    # --experiment finetuning_vit_deit_tiny_patch16_224_fractalDB_1k_gray_to_CIFAR10 \

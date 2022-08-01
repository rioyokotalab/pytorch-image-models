#!/bin/bash
#$ -cwd
#$ -l rt_F=16
#$ -l h_rt=24:00:00
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

export MODEL=tiny
export OUT_DIR=/groups/gcc50533/acc12016yi/pytorch-image-models-result/output

export NGPUS=64
export NPERNODE=4
mpirun -npernode $NPERNODE -np $NGPUS \
python train_without_eval.py /groups/gcd50691/datasets/ImageNet21k \
    --model deit_${MODEL}_patch16_LS --experiment pretrain_deit3_${MODEL}_imnet21k \
    --input-size 3 224 224 --num-classes 21841 \
    --sched cosine_iter --epochs 90 --lr 0.003 --weight-decay 0.02 \
    --batch-size 32 --opt lamb \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --use-3aug --src \
    --smoothing 0.1 --color-jitter 0.3 \
    --clip-grad 1.0 \
    --mixup 0.0 --cutmix 1.0 \
    -j 16 --eval-metric loss \
    --hold-epochs-interval 10 \
    --log-wandb --output ${OUT_DIR} \
    --amp \
    --resume ${OUT_DIR}/pretrain_deit3_${MODEL}_imnet21k/last.pth.tar

#    --gradient-ckp 6
#    --resume ${OUT_DIR}/finetune_deit_${MODEL}_imnet1k_${DATA}${CLASSES}k/last.pth.tar

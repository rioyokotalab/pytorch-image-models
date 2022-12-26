#!/bin/bash
#$ -cwd
#$ -l rt_F=32
#$ -l h_rt=48:00:00
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
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

export NGPUS=128
export NPERNODE=4
mpirun -npernode $NPERNODE -np $NGPUS \
python train.py /groups/gcd50691/datasets/ImageNet \
    --model auto_metaformer_async --experiment autometaformer_async_dist_100ep \
    --input-size 3 224 224 --num-classes 1000 \
    --sched cosine_iter --epochs 100 --lr 2e-3 --weight-decay 0.05 \
    --batch-size 16 --opt adamw \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
    --mixup 0.8 --cutmix 1.0 \
    --drop-path 0.1 --reprob 0.25 -j 16 \
    --output /groups/gcc50533/acc12016yi/AutoFormer/output \
    --log-wandb --project-name AutoFormer --wandb-group metaformer
    # --pause 100 \
    # --resume /groups/gcc50533/acc12016yi/AutoFormer/output/test_autometaformer_async/checkpoint-95.pth.tar

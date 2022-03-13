#!/bin/bash
#$ -cwd
#$ -l rt_F=32
#$ -l h_rt=01:00:00
#$ -j y
#$ -o deepspeed_example/o.$JOB_ID

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load cuda/10.2/10.2.89 cudnn openmpi nccl/2.7/2.7.8-1

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export NGPUS=128
export NUM_PROC=4
mpirun -npernode $NUM_PROC -np $NGPUS \
python train_with_ds.py /groups/gcc50533/imnet/ILSVRC2012 \
    --model vit_large_patch16_224 --experiment test_deepspeed \
    --input-size 3 224 224 --num-classes 1000 \
    --sched cosine_iter --epochs 300 --lr 0.001 --weight-decay 0.05 \
    --batch-size 32 --opt adamw \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --output ./deepspeed_example \
    --deepspeed_mpi --deepspeed --deepspeed_config ./deepspeed_example/ds_config.json
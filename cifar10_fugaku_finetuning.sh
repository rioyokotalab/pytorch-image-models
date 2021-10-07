#!/bin/bash
#PJM -L "node=4"
#PJM -L "rscgrp=small"
#PJM -L "elapse=1:00:00"
#PJM --mpi "proc=4"

export PATH=/home/apps/oss/PyTorch-1.7.0/bin:$PATH
export CPATH=/home/apps/oss/PyTorch-1.7.0/include:$CPATH
export LIBRARY_PATH=/home/apps/oss/PyTorch-1.7.0/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/home/apps/oss/PyTorch-1.7.0/lib:$LD_LIBRARY_PATH

export FLIB_CNTL_BARRIER_ERR=FALSE
export LD_PRELOAD=/usr/lib/FJSVtcs/ple/lib64/libpmix.so # mpi4py実行時に必要

export NUM_PROC=$PJM_MPI_PROC

mpirun -n $NUM_PROC -stdout output/%j.out -stderr output/%j.err python train_mpi.py ./ \
    --pretrained \
    --dataset CIFAR10 \
    --num-classes 10 \
    --model vit_deit_tiny_patch16_224 \
    --input-size 3 224 224 \
    --opt sgd \
    --batch-size 256 \
    --epochs 10 \
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
    --output output \
    --dist-backend mpi \
    --log-wandb \
    -j 0

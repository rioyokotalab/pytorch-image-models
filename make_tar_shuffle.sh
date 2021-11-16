#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=small"
#PJM -L "elapse=1:00:00"

export PATH=/home/apps/oss/PyTorch-1.7.0/bin:$PATH
export CPATH=/home/apps/oss/PyTorch-1.7.0/include:$CPATH
export LIBRARY_PATH=/home/apps/oss/PyTorch-1.7.0/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/home/apps/oss/PyTorch-1.7.0/lib:$LD_LIBRARY_PATH

export FLIB_CNTL_BARRIER_ERR=FALSE
export LD_PRELOAD=/usr/lib/FJSVtcs/ple/lib64/libpmix.so # mpi4py実行時に必要

python make_tar_shuffle.py \
    --data-root cifar10_data \
    --num-tars 128 \
    --shard-dir cifar10_tar_shuffle
python make_tar_shuffle.py \
    --data-root cifar10_data \
    --num-tars 128 \
    --shard-dir cifar10_tar_shuffle \
    --is-eval



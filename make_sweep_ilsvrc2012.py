from math import log
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--global-batch-size',type=int,default=256)
parser.add_argument('--model', type=str, default='deit_tiny')

def main(gbs:int, model:str)->None:
    lbs = 32
    if model=='deit_deit':
        lbs = 32
    elif model=='deit_small':
        lbs = 16
    else:
        assert False
    nodes = gbs // lbs
    if nodes > 4096:
        nodes = 4096
        lbs = gbs // nodes
    if model=='deit_tiny':
        prefix='tiny'
        end_acc = 0.72
        lr_max_per_bs = 3e-6
        lr_min_per_bs = 3e-9
    elif model=='deit_small':
        prefix='small'
        end_acc = 0.78
        lr_max_per_bs = 3e-6
        lr_min_per_bs = 3e-9
    else:
        assert False
    lr_max = log(lr_max_per_bs * gbs)
    lr_min = log(lr_min_per_bs * gbs)
    stdout_dir = 'output/%j.out'
    stderr_dir = 'output/%j.err'
    stdout_spec = '-stdout'
    stderr_spec = '-stderr'
    if nodes > 1000:
        stdout_spec = '-stdout-proc'
        stderr_spec = '-stderr-proc'
        stdout_dir = 'output/%j/%/1000r/stdout'
        stderr_dir = 'output/%j/%/1000r/stderr'
    template:str = f'''program: train_mpi_with_wds.py
command:
  - mpirun
  - -n
  - {nodes}
  - {stdout_spec}
  - {stdout_dir}
  - {stderr_spec}
  - {stderr_dir}
  - ${{interpreter}}
  - ${{program}}
  - ./
  - --pretrained-path=/home/hp190122/u01959/deit_pretrained/{prefix}_facebook.pth
  - --dataset=imagenet1k
  - --num-classes=1000
  - --model=vit_{model}_patch16_224
  - --epochs=100
  - --batch-size={lbs}
  - --log-wandb
  - --workers=0
  - --end-eval-top1-accuracy={end_acc}
  - --sched=cosine
  - --opt=sgd
  - --aa=rand-m9-mstd0.5-inc1
  - --webdataset
  - --dist-backend=mpi
  - --warmup-epochs=0
  - --trainshards=/home/hp190122/data/ILSVRC2012_shards/imagenet1k_train_{{000000..004095}}.tar
  - --evalshards=/home/hp190122/data/ILSVRC2012_shards/imagenet1k_eval_{{000000..004095}}.tar
  - ${{args}}
method: bayes
metric:
  goal: minimize
  name: end_steps
parameters:
  minus-momentum:
    distribution: log_uniform
    max: -2.302585092994046
    min: -4.605170185988091
  learning-rate-decay-epoch:
    distribution: int_uniform
    max: 100
    min: 50
  end-learning-rate-factor:
    distribution: log_uniform
    min: -4.605170185988091
    max: -1.203972804325935
  lr:
    distribution: log_uniform
    min: {lr_min}
    max: {lr_max}
  smoothing:
    distribution: categorical
    values:
    - 0.0
    - 0.1
    - 0.01
'''
    yaml_file_name = f'sweep_deit_tiny_bs{gbs}_imagenet1k.yaml'
    with open(yaml_file_name, mode='w') as f:
        f.write(template)
    t1=f'wandb sweep --project deit_{prefix}_imagenet1k_sweep_bs{gbs} --name bs{gbs}_{end_acc} {yaml_file_name}'
    with open(f'wandb_sweep_{prefix}_imagenet1k_bs{gbs}.sh', mode='w') as f:
        f.write(t1)
    rscgrp = 'small'
    elapse_limit = 72
    if nodes>=385:
        rscgrp = 'large'
        elapse_limit = 24
    t2 = f'''#!/bin/bash
#PJM -L "node={nodes}"
#PJM -L "rscgrp={rscgrp}"
#PJM -L "elapse={elapse_limit}:00:00"
#PJM --mpi "proc={nodes}"
#PJM --llio localtmp-size=10Gi
#PJM --llio sharedtmp-size=10Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0003

export PATH=$PATH:/home/hp190122/u01959/jobscheduler2slack

#TARDIR='/home/hp190122/data/ILSVRC2012_shards/'
#DATADIR='/home/hp190122/data/ILSVRC2012'
EXECFILE=train_mpi_with_wds.py
TIMM_PATH='timm/'
PYTORCH_PATH='/home/apps/oss/PyTorch-1.7.0/lib/python3.8/site-packages/'
PRETRAINED_PATH='/home/hp190122/u01959/deit_pretrained/tiny_facebook.pth'

llio_transfer $EXECFILE
llio_transfer $PRETRAINED_PATH

export PATH=/vol0003/hp190122/data/pytorch/home/apps/oss/PyTorch-1.7.0/bin:$PATH
export CPATH=/vol0003/hp190122/data/pytorch/home/apps/oss/PyTorch-1.7.0/include:$CPATH
export LIBRARY_PATH=/vol0003/hp190122/data/pytorch/home/apps/oss/PyTorch-1.7.0/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/vol0003/hp190122/data/pytorch/home/apps/oss/PyTorch-1.7.0/lib:$LD_LIBRARY_PATH

export FLIB_CNTL_BARRIER_ERR=FALSE
export LD_PRELOAD=/usr/lib/FJSVtcs/ple/lib64/libpmix.so # mpi4py実行時に必要

export OMP_NUM_THREADS=48

post_message "start sweep"
wandb agent 
post_message "end sweep"


llio_transfer --purge $EXECFILE
llio_transfer --purge $PRETRAINED_PATH
'''
    with open(f'ignore_ilsvrc2012_sweep_{prefix}_imagenet1k_bs{gbs}.sh', mode='w') as f:
        f.write(t2)


if __name__=='__main__':
    args = parser.parse_args()
    gbs = args.global_batch_size
    model = args.model
    main(gbs,model)


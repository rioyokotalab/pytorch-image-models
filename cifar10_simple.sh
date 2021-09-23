#export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | cut -d " " -f 6 | cut -d "/" -f 1)
export NGPUS=1
export NUM_PROC=1
mpirun -npernode $NUM_PROC -np $NGPUS \
python train_mpi.py ./ \
    --dataset CIFAR10 \
    --num-classes 10 \
    --model vit_deit_tiny_patch16_224 \
    --input-size 3 224 224 \
    --opt sgd \
    --batch-size 48 \
    --epochs 100 \
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
    -j 4

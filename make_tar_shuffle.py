import webdataset as wds
from torchvision import datasets
import torch.utils.data
import random
import sys
import os
import datetime
import argparse
from tqdm import tqdm
from mpi4py import MPI

comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()

def print0(s):
    if mpirank==0:
        print(s)

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root',type=str,default='cifar10_data')
    parser.add_argument('--max-count',type=int,default=None,
                        help='The maximum length of content that a tar file contain.')
    parser.add_argument('--num-tars',type=int,default=None,
                        help='Number of tar files')
    parser.add_argument('--shard-dir',type=str,default='cifar10_tar_shuffle')
    parser.add_argument('--is-eval',action='store_true')
    parser.add_argument('--dataset',type=str,choices=['cifar10','imagenet1k'],default='cifar10')
    return parser

def make_tar(
        data_root:str,
        max_count:int,
        num_tars:int,
        shard_dir:str,
        is_training:bool,
        dataset:str
    ):
    """
    data_root: root of data
    max_count: The maximum length of content that a tar file contain.
    num_tars: Number of tar files
    shard_dir: Destination path for tar files
    is_training: Whether the data set to be created is a train or not
    """
    assert (max_count is None) != (num_tars is None)
    os.makedirs(shard_dir,exist_ok=True)
    if dataset=='cifar10':
        ds = datasets.CIFAR10(
            root=data_root,
            train=is_training,
            download=False,
        )
    elif dataset=='imagenet1k':
        if is_training:
            #data_root += 'train/'
            pass
        else:
            data_root += 'val/'
        ds = datasets.ImageFolder(
            root=data_root
        )
    num_images = len(ds)
    print(f'num_images: {num_images}')
    if max_count is None:
        assert num_tars > 0
        assert num_tars <= num_images
        max_count = (num_images + num_tars + 1) // num_tars
        max_count_first_index = num_images - (max_count-1) * num_tars
        max_counts = [0]
        for i in range(max_count_first_index):
            max_counts.append((i + 1) * max_count)
        for _ in range(num_tars-max_count_first_index):
            max_counts.append(max_counts[-1] + max_count - 1)
        print(f'Generated max_count:{max_count} from num_tars:{num_tars}')
    else:
        max_counts = [0]
        for i in range(num_images // max_count + 1):
            max_counts.append((i + 1) * max_count)
        num_tars = len(max_counts)-1

    print0(f'num_tars:{num_tars}')
    print0(f'max_count:{max_count}')
    assert num_tars >= mpisize
    # recommend num_tars % mpisize == 0
    ntar_per_rank = (num_tars + mpisize - 1) // mpisize
    start_tar_ind = mpirank * ntar_per_rank
    end_tar_ind = min(num_tars, (mpirank + 1) * ntar_per_rank)
    print(f'rank: {mpirank}, tarfile_index: [{start_tar_ind}, {end_tar_ind})')

    fname = 'train' if is_training else 'eval'
    tarfile_list = [os.path.join(shard_dir,f'{dataset}_{fname}_{sink_index:06}.tar') for sink_index in range(start_tar_ind, end_tar_ind)]
    
    indexes = list(range(num_images))
    random.seed(42)
    random.shuffle(indexes)

    for ind, target_filename in enumerate(tqdm(tarfile_list)): # rank1->ind:[63,127)
        global_tar_ind = start_tar_ind + ind
        global_img_ind_begin = max_counts[global_tar_ind]
        global_img_ind_end = max_counts[global_tar_ind + 1]
        with wds.TarWriter(target_filename) as sink:
            for global_img_ind in range(global_img_ind_begin, global_img_ind_end):
                inputs, outputs = ds[global_img_ind]
                sink.write({
                    "__key__": "sample%06d" % global_img_ind,
                    "input.pyd": inputs,
                    "output.pyd": outputs,
                })
    
def is_inc_sink_index(i,max_counts, max_counts_index):
    return max_counts[max_counts_index]-1==i

def len_tar(url):
    res = 0
    ds = wds.WebDataset(url)
    for _ in ds:
        res += 1
    return res

if __name__=='__main__':
    parser = make_parse()
    args = parser.parse_args()
    #assert args.max_count is not None or args.num_tars is not None
    args.is_training = not args.is_eval
    print(args.is_training)
    make_tar(
        data_root=args.data_root,
        max_count=args.max_count,
        num_tars=args.num_tars,
        shard_dir=args.shard_dir,
        is_training=args.is_training,
        dataset=args.dataset,
    )
    
    #for i in range(128):
    #    fname = 'train' if args.is_training else 'eval'
    #    url = f'{args.shard_dir}/{args.dataset}_{fname}_{i:06}.tar'
    #    print(i,len_tar(url))

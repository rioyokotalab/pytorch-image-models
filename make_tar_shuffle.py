import webdataset as wds
from torchvision import datasets
import torch.utils.data
import random
import sys
import os
import datetime
import argparse

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root',type=str,default='cifar10_data')
    parser.add_argument('--max-count',type=int,default=None,
                        help='The maximum length of content that a tar file contain.')
    parser.add_argument('--num-tars',type=int,default=None,
                        help='Number of tar files')
    parser.add_argument('--shard-dir',type=str,default='cifar10_tar_shuffle')
    parser.add_argument('--is-eval',action='store_true')
    return parser

def make_tar(
        data_root:str,
        max_count:int,
        num_tars:int,
        shard_dir:str,
        is_training:bool
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
    
    ds = datasets.CIFAR10(
        root=data_root,
        train=is_training,
        download=False,
    )
    num_images = len(ds)
    if max_count is None:
        assert num_tars > 0
        assert num_tars <= num_images
        max_count = (num_images + num_tars + 1) // num_tars
        max_count_first_index = num_images - (max_count-1) * num_tars
        max_counts = []
        for i in range(max_count_first_index):
            max_counts.append((i + 1) * max_count)
        for _ in range(num_tars-max_count_first_index):
            max_counts.append(max_counts[-1] + max_count - 1)
        print(f'Generated max_count:{max_count} from num_tars:{num_tars}')
    else:
        max_counts = []
        for i in range(num_images // max_count + 1):
            max_counts.append((i + 1) * max_count)
    max_counts_index = 0
    indexes = list(range(num_images))
    random.shuffle(indexes)

    sink_index = 0
    fname = 'train' if is_training else 'eval'
    sink = wds.TarWriter(f'{shard_dir}/cifar10_{fname}_{sink_index:06}.tar')
    for i in range(num_images):
        is_last = i==num_images-1
        if i%10==0:
            print(f"{i:6d}\t{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", end="\r", flush=True, file=sys.stderr)
        data_index = indexes[i]
        input, output = ds[data_index]
        sink.write({
            "__key__": "sample%06d" % data_index,
            "input.pyd": input,
            "output.pyd": output,
        })
        if is_inc_sink_index(i,max_counts, max_counts_index) or is_last:
            max_counts_index += 1
            #print(i+1,end=', ',flush=True)
            sink.close()
            if not is_last:
                sink_index += 1
                sink = wds.TarWriter(f'{shard_dir}/cifar10_{fname}_{sink_index:06}.tar')
    
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
    )
    
    #for i in range(128):
    #    fname = 'train' if args.is_training else 'eval'
    #    url = f'{args.shard_dir}/cifar10_{fname}_{i:06}.tar'
    #    print(i,len_tar(url))

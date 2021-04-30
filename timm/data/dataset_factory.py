import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .dataset import IterableImageDataset, ImageDataset


def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root
    if split_name == 'validation':
        try_root = os.path.join(root, 'val')
        if os.path.exists(try_root):
            return try_root
    return root


def create_dataset(name, root, split='validation', search_split=True, is_training=False, batch_size=None, **kwargs):
    name = name.lower()
    ds = None
    if name == 'cifar10':
        root_dir = f'{root}/{split}'
        # normalization
        form = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4914, 0.4822, 0.4465],
                [0.2023, 0.1994, 0.2010]
                )
        ])
        ds = datasets.CIFAR10(
            root=root_dir,
            train=is_training,
            transform=form,
            download=True
        )
    elif name.startswith('tfds'):
        ds = IterableImageDataset(
            root, parser=name, split=split, is_training=is_training, batch_size=batch_size, **kwargs)
    else:
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        kwargs.pop('repeats', 0)  # FIXME currently only Iterable dataset support the repeat multiplier
        if search_split and os.path.isdir(root):
            root = _search_split(root, split)
        ds = ImageDataset(root, parser=name, **kwargs)
    return ds

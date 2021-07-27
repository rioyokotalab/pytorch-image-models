import os
import cv2
import torch
from scipy import io as mat_io
from skimage import io
from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .dataset import IterableImageDataset, ImageDataset


# class CarsDataset(Dataset):
#     """
#         Cars Dataset
#     """
#     def __init__(self, mode, data_dir, metas, limit=None):

#         self.data_dir = data_dir
#         self.data = []
#         self.target = []

#         self.to_tensor = transforms.ToTensor()
#         self.mode = mode

#         if not isinstance(metas, str):
#             raise Exception("Train metas must be string location !")
#         labels_meta = mat_io.loadmat(metas)

#         for idx, img_ in enumerate(labels_meta['annotations'][0]):
#             if limit:
#                 if idx > limit:
#                     break

#             # self.data.append(img_resized)
#             self.data.append(data_dir + img_[-1][0])
#             # if self.mode == 'train':
#             self.target.append(img_[-2][0][0])
#             # if not self.mode and idx < 3:
#             #     print(img_[-2][0][0])
#             #     print(img_)
#             # else:
#             #     import sys
#             #     sys.exit()
        
#         self.train_transform = transforms.Compose([transforms.ToTensor()])
#         self.val_or_test_transform = transforms.Compose([transforms.ToTensor()])

#     def __getitem__(self, idx):

#         try:
#             image = io.imread(self.data[idx])
#         except Exception as e:
#             print(f"error occured. error type : {e}")
#             print(f"file : {self.data[idx]}")

#         if len(image.shape) == 2:  # this is gray image
#             image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

#         img = Image.fromarray(image)
#         img_resized = cv2.resize(img, (-1, -1), interpolation=cv2.INTER_CUBIC)
#         if self.mode:
#             return self.train_transform(img), torch.tensor(self.target[idx]-1, dtype=torch.long)
#         else:
#             return self.val_or_test_transform(img), torch.tensor(self.target[idx]-1, dtype=torch.long)

#     def __len__(self):
#         return len(self.data)


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
    elif name == 'cifar100':
        root_dir = f'{root}/{split}'
        # normalization
        form = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [x / 255.0 for x in [129.3, 124.1, 112.4]],
                [x / 255.0 for x in [68.2, 65.4, 70.4]]
                )
        ])
        ds = datasets.CIFAR100(
            root=root_dir,
            train=is_training,
            transform=form,
            download=True
        )
    elif name == 'cars':
        root_dir = f'/groups/gcd50691/datasets/stanford_cars/train/extracted/'
        if not is_training:
            root_dir = f'/groups/gcd50691/datasets/stanford_cars/test/extracted/'
        kwargs.pop('repeats', 0)  # FIXME currently only Iterable dataset support the repeat multiplier
        ds = ImageDataset(root_dir, parser=name, **kwargs)
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

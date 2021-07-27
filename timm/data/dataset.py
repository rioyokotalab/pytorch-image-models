""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch.utils.data as data
import os
import torch
import logging
from scipy import io as mat_io
from PIL import Image

from .parsers import create_parser

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map='',
            load_bytes=False,
            transform=None,
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.tensor(-1, dtype=torch.long)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            parser=None,
            split='train',
            is_training=False,
            batch_size=None,
            class_map='',
            load_bytes=False,
            repeats=0,
            transform=None,
    ):
        assert parser is not None
        if isinstance(parser, str):
            self.parser = create_parser(
                parser, root=root, split=split, is_training=is_training, batch_size=batch_size, repeats=repeats)
        else:
            self.parser = parser
        self.transform = transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.parser:
            if self.transform is not None:
                img = self.transform(img)
            if target is None:
                target = torch.tensor(-1, dtype=torch.long)
            yield img, target

    def __len__(self):
        if hasattr(self.parser, '__len__'):
            return len(self.parser)
        else:
            return 0

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)


# class CarsDataset(data.Dataset):
#     """
#         Cars Dataset
#     """
#     def __init__(
#             self,
#             mode,
#             data_dir,
#             metas,
#             limit=None,
#             parser=None,
#             class_map='',
#             load_bytes=False,
#             transform=None,
#     ):
#         self.data_dir = data_dir
#         self.data = []
#         self.target = []
#         self.mode = mode
#         self.parser = parser
#         self.class_map = class_map
#         self.load_bytes = load_bytes
#         self._consecutive_errors = 0
#         self.transform = transform

#         if not isinstance(metas, str):
#             raise Exception("Train metas must be string location !")
#         labels_meta = mat_io.loadmat(metas)

#         for index, img_ in enumerate(labels_meta['annotations'][0]):
#             if limit:
#                 if index > limit:
#                     break

#             # self.data.append(img_resized)
#             self.data.append(data_dir + img_[-1][0])
#             # if self.mode == 'train':
#             self.target.append(img_[-2][0][0])
#             # if not self.mode and index < 3:
#             #     print(img_[-2][0][0])
#             #     print(img_)
#             # else:
#             #     import sys
#             #     sys.exit()

#     def __getitem__(self, index):

#         img, target = self.data[index], self.target[index]-1
#         try:
#             img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
#         except Exception as e:
#             _logger.warning(f'Skipped sample (index {index}, file {self.data[index]}). {str(e)}')
#             self._consecutive_errors += 1
#             if self._consecutive_errors < _ERROR_RETRY:
#                 return self.__getitem__((index + 1) % len(self.data))
#             else:
#                 raise e
#         self._consecutive_errors = 0
#         if self.transform is not None:
#             img = self.transform(img)
#         if target is None:
#             target = torch.tensor(-1, dtype=torch.long)
#         return img, target

#     def __len__(self):
#         return len(self.data)

#     def filename(self, index, basename=False, absolute=False):
#         return self.data[index]

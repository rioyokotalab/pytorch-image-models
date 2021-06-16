# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.distributed as dist
import math


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
        with repeated augmentation.
        It ensures that different each augmented version of a sample will be visible to a
        different process (GPU)
        Heavily based on torch.utils.data.DistributedSampler
        """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.rank == 0:
            print(len(self.dataset), len(indices))

        # add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(3)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        if self.rank == 0:
            print(self.total_size, len(indices))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        if self.rank == 0:
            print(self.num_samples, len(indices))
            print(self.num_selected_samples)

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class RASamplerSplit(RASampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True):
        super(RASamplerSplit, self).__init__(dataset, num_replicas, rank, shuffle)
        self.batch_size = batch_size

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        # sample only fake images
        indices = torch.randperm(len(self.dataset) // 2, generator=g).tolist()

        # add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(3)]
        indices += indices[:(self.total_size // 2 - len(indices))]
        assert len(indices) == self.total_size // 2

        # subsample
        indices = indices[self.rank:self.total_size // 2:self.num_replicas]
        assert len(indices) == self.num_samples // 2

        # implement samples with real images
        size = self.batch_size // 2
        for i in range(size, self.num_samples, self.batch_size):
            indices[i:i] = list(map(lambda x: x + (len(self.dataset) + 1) // 2, indices[i - size:i]))

        return iter(indices[:self.num_selected_samples])

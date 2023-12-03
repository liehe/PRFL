import math
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        offset_seed=0,
        subsample_percent=1,
    ):
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
        self.epoch = offset_seed

        self.dataset_length = int(len(dataset) * subsample_percent)
        self.num_samples = int(math.ceil(self.dataset_length * 1.0 / self.num_replicas))

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.offset_seed = offset_seed

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(self.dataset_length, generator=g).tolist()
        else:
            indices = list(range(self.dataset_length))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch + self.offset_seed

    def __str__(self):
        return "DistributedSampler(num_replicas={num_replicas},rank={rank},shuffle={shuffle})".format(
            num_replicas=self.num_replicas, rank=self.rank, shuffle=self.shuffle
        )


class DecentralizedSampler(DistributedSampler):
    def __init__(self, noniid_percent, *args, **kwargs):
        super(DecentralizedSampler, self).__init__(*args, **kwargs)
        self.noniid_percent = noniid_percent

    def __iter__(self):
        nlabels = len(self.dataset.classes)

        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(0)

        all_indices = torch.randperm(len(self.dataset), generator=g).tolist()

        iid_count = int((1 - self.noniid_percent) * len(all_indices))
        iid_count = iid_count - (iid_count % self.num_replicas)
        iid_indices, noniid_indices = all_indices[:iid_count], all_indices[iid_count:]

        indices = []
        for i in range(nlabels):
            indices_i = torch.nonzero(self.dataset.targets == i)
            indices_i = indices_i.flatten().tolist()
            # Find those in the noniid parts
            indices_i = set(indices_i).intersection(set(noniid_indices))
            indices += indices_i

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - iid_count - len(indices))]
        # (self.total_size - iid_count - len(indices)) is greater than len(indices)
        assert len(indices) + iid_count == self.total_size, (
            len(indices),
            iid_count,
            self.total_size,
        )

        # subsample
        num_noniid_samples_per_node = self.num_samples - iid_count // self.num_replicas
        indices = indices[
            self.rank
            * num_noniid_samples_per_node : (self.rank + 1)
            * num_noniid_samples_per_node
        ]
        # Add iid part
        indices += iid_indices[self.rank : iid_count : self.num_replicas]
        assert len(indices) == self.num_samples, (len(indices), self.num_samples)

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_idx = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in idx_idx]

        return iter(indices)


class KNNPerSampler(DistributedSampler):
    def __init__(
        self,
        val: bool,
        val_per: float,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        offset_seed=0,
        subsample_percent=1,
    ):
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
        self.epoch = offset_seed

        self.dataset_length = int(len(dataset) * subsample_percent)
        self.val = val

        assert val_per >= 0 and val_per <= 1
        # NOTE: this is the sum of both train and validation set.
        self.num_samples_per_device = int(
            math.ceil(self.dataset_length * 1.0 / self.num_replicas)
        )
        if val:
            self.num_samples = int(self.num_samples_per_device * val_per)
        else:
            self.num_samples = self.num_samples_per_device - int(
                self.num_samples_per_device * val_per
            )

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.offset_seed = offset_seed

    def __iter__(self):
        # NOTE: Fix seed=0 and
        g = torch.Generator()
        g.manual_seed(0)
        indices = torch.randperm(self.dataset_length, generator=g).tolist()

        # NOTE: choose either validation or training subset
        if self.val:
            indices = indices[-self.total_size :]
        else:
            indices = indices[: self.total_size]

        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            _shuffled = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in _shuffled]

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

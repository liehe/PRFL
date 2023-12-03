import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F

from utils import top1_accuracy
from resnet import get_resnet_model

from vgg import vgg16


def get_resnet20(use_cuda=False, gn=False):
    return get_resnet_model(
        model="resnet20", version=1, dtype="fp32", num_classes=10, use_cuda=use_cuda
    )


def cifar10(
    data_dir,
    train,
    download,
    batch_size,
    shuffle=None,
    sampler_callback=None,
    dataset_cls=datasets.CIFAR10,
    drop_last=True,
    rotation_fn=None,
    relabel_fn=None,
    **loader_kwargs
):
    if sampler_callback is not None and shuffle is not None:
        raise ValueError

    cifar10_stats = {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    }

    if train:
        applied_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_stats["mean"], cifar10_stats["std"]),
        ]
    else:
        applied_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_stats["mean"], cifar10_stats["std"]),
        ]
    if rotation_fn:
        applied_transforms.append(transforms.Lambda(rotation_fn))

    dataset = dataset_cls(
        root=data_dir,
        train=train,
        download=download,
        transform=transforms.Compose(applied_transforms),
    )

    dataset.targets = torch.LongTensor(dataset.targets)

    if relabel_fn:
        dataset.targets = relabel_fn(dataset.targets)

    sampler = sampler_callback(dataset) if sampler_callback else None
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        **loader_kwargs,
    )


def metrics():
    return {"top1": top1_accuracy}

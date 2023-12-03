import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

from utils import top1_accuracy

debug_logger = logging.getLogger("debug")


def mnist(
    data_dir,
    train,
    download,
    batch_size,
    shuffle=None,
    sampler_callback=None,
    dataset_cls=datasets.MNIST,
    drop_last=True,
    rotation_fn=None,
    relabel_fn=None,
    **loader_kwargs
):
    applied_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
    if rotation_fn:
        applied_transforms.append(transforms.Lambda(rotation_fn))

    dataset = dataset_cls(
        data_dir,
        train=train,
        download=download,
        transform=transforms.Compose(applied_transforms),
    )

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


def loss_fn(*args, **kwargs):
    return F.nll_loss(*args, **kwargs)


def metrics():
    return {"top1": top1_accuracy}


class SimpleLinear(torch.nn.Module):

    def __init__(self, h1=2048):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, h1)
        self.fc2 = torch.nn.Linear(h1, 10)

    def forward(self, x):
        # x = torch.squeeze(x, 1)
        x = torch.reshape(x, (-1, 28 * 28))
        x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output

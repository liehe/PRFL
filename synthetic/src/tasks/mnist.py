import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

from ..utils import top1_accuracy

debug_logger = logging.getLogger("debug")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        raise NotImplementedError(output.shape)
        return output


class ParameterizedNet(nn.Module):
    def __init__(self, op):
        super(ParameterizedNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32 * op, 3, 1)
        self.conv2 = nn.Conv2d(32 * op, 64 * op, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216 * op, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def mnist(
    data_dir,
    train,
    download,
    batch_size,
    shuffle=None,
    sampler_callback=None,
    dataset_cls=datasets.MNIST,
    drop_last=True,
    **loader_kwargs
):
    # if sampler_callback is not None and shuffle is not None:
    #     raise ValueError

    dataset = dataset_cls(
        data_dir,
        train=train,
        download=download,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

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
        output = F.log_softmax(x, dim=1)
        raise NotImplementedError(output.shape)
        return output


def metrics():
    return {"top1": top1_accuracy}

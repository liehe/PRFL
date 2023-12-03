import copy
import itertools
import functools
import numpy as np
import os
import torch
import torchvision.datasets as datasets

from sklearn.cluster import KMeans

# from data import metrics, SimpleLinear, loss_fn, mnist
from data_cifar100 import *

from algorithms import Train, IndividualEvaluator, KNNPerIndividualEvaluator
from sampler import DistributedSampler, KNNPerSampler, CIFAR100Sampler
from worker_server import MultiModelServer, SingleModelServer, MultiUpdateServer
from worker_server import IFCAWorker, FLWorker, LocalWorker, FCWorker, DittoWorker, KNNPerWorker


def task_subsample_percent(args, dataset_type):
    # if dataset_type == 'train':
    #     if args.data in ['rotation', 'relabel']:
    #         # We only use 50% of training samples such that local training alone is not enough
    #         return 0.5
    #     raise NotImplementedError
    # return 1
    return args.subsample_ratio 


def sampler_fn(x, args, rank, shuffle, dataset_type, offset_seed=0):
    assert args.noniid == 0
    nlabels_per_cluster = 100 // args.K_gen
    return CIFAR100Sampler(
        nlabels_per_cluster=nlabels_per_cluster,
        # noniid_percent=args.noniid,
        num_replicas=args.n,
        rank=rank,
        shuffle=shuffle,
        dataset=x,
        offset_seed=offset_seed,
        subsample_percent=task_subsample_percent(args, dataset_type))


def get_data_loader(args, sampler, rank, data_dir, dataset_type):
    if dataset_type == 'train':
        train = True
        batch_size = args.batch_size
        drop_last = False
    elif dataset_type == 'validation':
        train = True
        batch_size = args.validation_batch_size
        drop_last = False
    elif dataset_type == 'test':
        train = False
        batch_size = args.test_batch_size
        drop_last = False
    else:
        raise NotImplementedError

    if args.data == 'normal':
        rotation_fn = None
        relabel_fn = None

    elif args.data == 'rotation':
        assert args.K_gen <= 4
        num_workers_within_cluster = args.n // args.K_gen
        k = rank // num_workers_within_cluster

        def rotation_fn(img):
            return torch.rot90(img, k=k, dims=(1, 2))
        relabel_fn = None
        raise NotImplementedError

    elif args.data == 'relabel':
        rotation_fn = None

        num_workers_within_cluster = args.n // args.K_gen
        k = rank // num_workers_within_cluster

        def relabel_fn(targets):
            # For cluster 0: the labels remain the same 0, 1, 2, ..., 9
            # For cluster 1: labels are changed to 1, 2, ..., 9, 0
            return (targets + k) % 10
        raise NotImplementedError

    else:
        raise NotImplementedError

    return cifar100(
        data_dir=data_dir,
        train=train,
        download=True,
        batch_size=batch_size,
        sampler_callback=sampler,
        dataset_cls=datasets.CIFAR100,
        drop_last=drop_last,
        rotation_fn=rotation_fn,
        relabel_fn=relabel_fn,
        pin_memory=False,
    )

import json
import logging
import numpy as np
import shutil
import os
import sys
import time
import torch

from contextlib import contextmanager


class MomentumBuffer(object):
    def __init__(self, beta):
        self.beta = beta
        self.buff = None

    def update(self, grad):
        if self.buff is None:
            self.buff = grad
        else:
            self.buff = (1-self.beta) * grad + self.beta * self.buff
        return self.buff


@contextmanager
def printoptions(*args, **kwargs):
    optional_options = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**optional_options)


def initialize_logger(log_root):
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    else:
        shutil.rmtree(log_root)
        os.makedirs(log_root)

    print(f"Logging files to {log_root}")

    # Only to file; One dict per line; Easy to process
    json_logger = logging.getLogger("stats")
    json_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_root, "stats"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    json_logger.addHandler(fh)

    debug_logger = logging.getLogger("debug")
    debug_logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    debug_logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_root, "debug"))
    fh.setLevel(logging.INFO)
    debug_logger.addHandler(fh)


def setup_logs(
        args,
        LOG_DIR_PATTERN,
        script=sys.argv[0][:-3]):
    assert "script" not in args.__dict__
    assert "exp_id" not in args.__dict__
    log_dir = LOG_DIR_PATTERN.format(
        script=script.split("/")[-1],
        exp_id=args.identifier,
        # NOTE: Customize the hp
        **args.__dict__
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not args.analyze:
        initialize_logger(log_dir)
        with open(os.path.join(log_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f)

    return log_dir


class Timer(object):
    def __init__(self):
        self._time = 0
        self._counter = 0
        self.t0 = 0

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, type, value, traceback):
        self._time += time.time() - self.t0
        self._counter += 1

    @property
    def avg(self):
        return self._time / self._counter

    @property
    def counter(self):
        return self._counter


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def top1_accuracy(output, target):
    return accuracy(output, target, topk=(1,))[0].item()


def vectorize_model(model):
    state_dict = model.state_dict()
    return torch.cat([state_dict[k].data.view(-1) for k in state_dict])


def unstack_vectorized_model(model, state_dict):
    beg = 0
    for k in state_dict:
        p = state_dict[k]
        end = beg + len(p.data.view(-1))
        state_dict[k] = model[beg:end].reshape_as(p.data)
        beg = end

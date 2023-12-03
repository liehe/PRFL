import argparse
import json
import logging
import numpy as np
import os
import shutil
import sys
import torch
import time
import scipy
import functools
from contextlib import contextmanager
import multiprocessing as mp
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------- #
#                                 Define parser                                #
# ---------------------------------------------------------------------------- #


def define_parser():
    parser = argparse.ArgumentParser()

    # Running environment related arguments
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--identifier", type=str, default="debug", help="")
    parser.add_argument("--analyze", action="store_true", default=False, help="")

    # Common experiment setup
    parser.add_argument("-n", type=int, help="Number of workers")
    parser.add_argument("--epochs", type=int, default=None, help="Number of workers")

    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum.")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Train batch size of 32.",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
        help="Test batch size of 128.",
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=32,
        help="Validation batch size of 128.",
    )
    parser.add_argument(
        "--max-batch-size-per-epoch",
        type=int,
        default=999999999,
        help="Early stop of an epoch.",
    )

    # Hyper-parameters for dataset
    parser.add_argument(
        "--noniid",
        type=float,
        default=0,
        help="0 for iid and 1 for noniid",
    )

    parser.add_argument("--algorithm", type=str, default="ifca", help="")

    parser.add_argument("--K", type=int, default=None, help="The K for clustering")

    parser.add_argument(
        "--K_gen", type=int, default=None, help="The K for generating dataset"
    )

    parser.add_argument(
        "--data", type=str, default="normal", help="normal/rotation/relabel"
    )

    parser.add_argument("--model-size", type=int, default=200, help="hidden model size")

    parser.add_argument(
        "--knnper-phase",
        choices=["FedAvgVal", "kNNPerVal", "FedAvgTrain", "kNNPerTest"],
        default=None,
        help="",
    )

    parser.add_argument(
        "--force-keep-folder", action="store_true", default=False, help=""
    )

    parser.add_argument(
        "--subsample-ratio", type=float, default=1, help="Between 0 and 1."
    )

    parser.add_argument("--model-type", type=str, default="A", help="")

    return parser


def get_system_args_or_default(default_arguments=""):
    parser = define_parser()
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        args = parser.parse_args(default_arguments.split())

    # Check args
    assert args.n > 0
    assert args.epochs >= 1
    assert args.n >= args.K
    assert (args.n // args.K) * args.K == args.n
    if args.K_gen is not None:
        assert args.K >= args.K_gen
    else:
        args.K_gen = args.K

    return args


# ---------------------------------------------------------------------------- #
#                            Setup logger and seeds                            #
# ---------------------------------------------------------------------------- #


def setup_seeds(use_cuda, seed):
    if use_cuda:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


class RandomStatesController(object):
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        self.random_states = {}

    def __enter__(self) -> None:
        if self.use_cuda:
            self.random_states["torch_cuda"] = torch.cuda.get_rng_state()
        self.random_states["torch"] = torch.get_rng_state()
        self.random_states["numpy"] = np.random.get_state()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.use_cuda:
            torch.cuda.set_rng_state(self.random_states["torch_cuda"])
        torch.set_rng_state(self.random_states["torch"])
        np.random.set_state(self.random_states["numpy"])


def initialize_logger(force_delete, log_root):
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    elif force_delete:
        shutil.rmtree(log_root)
        os.makedirs(log_root)
    else:
        # In some applications, we need to run the same applications for multiple times
        # but would like to use only 1 logging folder.
        pass

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

    warning_logger = logging.getLogger("warning")
    warning_logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(message)s"))
    warning_logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_root, "warning"))
    fh.setLevel(logging.WARNING)
    warning_logger.addHandler(fh)


def setup_logs(args, fn_spec):
    assert "script" not in args.__dict__
    assert "exp_id" not in args.__dict__

    # NOTE: Customize the hp
    log_dir = fn_spec.format(
        script=sys.argv[0][:-3], exp_id=args.identifier, **args.__dict__
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not args.analyze:
        initialize_logger(not args.force_keep_folder, log_dir)
        with open(os.path.join(log_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f)

    return log_dir


# ---------------------------------------------------------------------------- #
#                               Experiment utils                               #
# ---------------------------------------------------------------------------- #
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


# ---------------------------------------------------------------------------- #
#                                ContextManagers                               #
# ---------------------------------------------------------------------------- #
class Timer(object):
    def __init__(self):
        self.t0 = None
        self.total = 0
        self.count = 0

    @contextmanager
    def timeit(self):
        try:
            start = time.time()
            yield self
        finally:
            self.total += time.time() - start
            self.count += 1

    @property
    def avg(self):
        return self.total / self.count


class TrainMeter(object):
    def __init__(self, metrics):
        self.r = {k: 0 for k in metrics}
        self.r["Loss"] = 0
        self.r["Length"] = 0
        self.metrics = metrics
        self.timer = Timer()

    def add(self, worker_results):
        self.r["Length"] += sum(res["length"] for res in worker_results)
        self.r["Loss"] += sum(res["loss"] * res["length"] for res in worker_results)
        for metric_name in self.metrics:
            self.r[metric_name] += sum(
                res["metrics"][metric_name] * res["length"] for res in worker_results
            )

    def get(self, epoch, batch):
        out = {
            "_meta": {"type": "train"},
            "E": epoch,
            "B": batch,
            "Length": self.r["Length"],
            "Loss": self.r["Loss"] / self.r["Length"],
        }
        for metric_name in self.metrics:
            out[metric_name] = self.r[metric_name] / self.r["Length"]
        return out

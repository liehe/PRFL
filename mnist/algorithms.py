import logging
import torch
from collections import defaultdict
from typing import Union, Callable, Tuple

from utils import *


class Train(object):
    """Simulate distributed programs with low memory usage.

    Functionality:
    1. randomness control: numpy, torch, torch-cuda
    2. add workers

    This base class is used by both trainer and evaluator.
    """

    def __init__(
        self,
        args,
        pre_batch_hooks: list,
        post_batch_hooks: list,
        max_batches_per_epoch: int,
        log_interval: int,
        metrics: dict,
        use_cuda: bool,
        debug: bool,
    ):
        """
        Args:
            metrics (dict): dict of metric names and their functions
            use_cuda (bool): Use cuda or not
            debug (bool):
        """
        self.args = args
        self.pre_batch_hooks = pre_batch_hooks or []
        self.post_batch_hooks = post_batch_hooks or []
        self.debug = debug
        self.max_batches_per_epoch = max_batches_per_epoch
        self.log_interval = log_interval
        self.metrics = metrics
        self.json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")
        self.warning_logger = logging.getLogger("warning")

        self.random_states_controller = RandomStatesController(use_cuda=use_cuda)

        self.server = None
        self.workers = None

    def train(self, epoch):
        self.debug_logger.info(f"Train epoch {epoch}")
        meter = TrainMeter(self.metrics)
        self.epoch_start()

        for batch_idx in range(self.max_batches_per_epoch):
            try:
                for hook in self.pre_batch_hooks:
                    hook(self, epoch, batch_idx)

                with meter.timer.timeit():
                    self.train_batch(meter, batch_idx, epoch)

                if batch_idx % self.log_interval == 0:
                    self.log(meter, batch_idx, epoch)

                for hook in self.post_batch_hooks:
                    hook(self, epoch, batch_idx)

            except StopIteration:
                break

    def epoch_start(self):
        # Prepare model and data iterators
        self.parallel_call(lambda w: w.train_epoch_start())

    def train_batch(self, meter, batch_idx, epoch):
        raise NotImplementedError

    def parallel_call(self, f) -> None:
        for w in self.workers:
            with self.random_states_controller:
                f(w)

    def parallel_get(self, f) -> list:
        results = []
        for w in self.workers:
            with self.random_states_controller:
                results.append(f(w))
        return results

    def log(self, meter, batch, epoch):
        r = meter.get(epoch, batch)

        progress = r["Length"]
        # Output to console
        total = len(self.workers[0].data_loader.dataset)
        pct = 100 * progress / total
        self.debug_logger.info(
            f"[E{r['E']:2}B{r['B']:<3}| {progress:6}/{total} ({pct:3.0f}%) "
            + f"| Avg time {meter.timer.avg:.1f}s ] Loss: {r['Loss']:.4f} "
            + " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in self.metrics)
        )

        # Output to file
        self.json_logger.info(r)

    def add_server_workers(self, server, workers):
        self.server = server
        self.workers = workers


class Evaluator(object):
    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.modules.loss._Loss,
        device: Union[torch.device, str],
        metrics: dict,
        use_cuda: bool,
        debug: bool,
        meta={"type": "validation"},
    ):
        self.metrics = metrics
        self.use_cuda = use_cuda
        self.debug = debug

        self.json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")

        self.model = model
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.device = device
        self.meta = meta

    def __str__(self):
        return f"Evaluator(type={self.meta['type']})"

    def evaluate(self, epoch):
        self.model.eval()
        r = {
            "_meta": self.meta,
            "E": epoch,
            "Length": 0,
            "Loss": 0,
        }
        for name in self.metrics:
            r[name] = 0

        with torch.no_grad():
            for _, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                r["Loss"] += self.loss_fn(output, target).item() * len(target)
                r["Length"] += len(target)

                for name, metric in self.metrics.items():
                    r[name] += metric(output, target) * len(target)

        for name in self.metrics:
            r[name] /= r["Length"]
        r["Loss"] /= r["Length"]

        # Output to file
        self.json_logger.info(r)
        self.debug_logger.info(
            f"\n=> {self.meta['type']} Eval Loss={r['Loss']:.4f} "
            + " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in self.metrics)
            + "\n"
        )


class IndividualEvaluator(object):
    """Evaluate individual worker's model on its data."""

    def __init__(
        self,
        workers: list,
        data_loaders: list,
        loss_fn: torch.nn.modules.loss._Loss,
        device: Union[torch.device, str],
        metrics: dict,
        use_cuda: bool,
        debug: bool,
    ):
        self.metrics = metrics
        self.use_cuda = use_cuda
        self.debug = debug

        self.json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")

        self.workers = workers
        self.data_loaders = data_loaders
        self.loss_fn = loss_fn
        self.device = device

    def evaluate_worker(self, m, data_loader, epoch):
        m.eval()

        r = {
            "_meta": {"type": "Individual Validation"},
            "E": epoch,
            "Length": 0,
            "Loss": 0,
        }
        for _, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = m(data)
            r["Loss"] += self.loss_fn(output, target).item() * len(target)
            r["Length"] += len(target)

            for name, metric in self.metrics.items():
                r[name] = r.get(name, 0) + metric(output, target) * len(target)

        for name in self.metrics:
            r[name] /= r["Length"]
        r["Loss"] /= r["Length"]
        self.json_logger.info(r)
        return r

    def evaluate(self, epoch):
        # ------------------------------ Compute metrics ----------------------------- #
        with torch.no_grad():
            rs = []
            for i, data_loader in enumerate(self.data_loaders):
                w = self.workers[i]
                m = w.model
                r = self.evaluate_worker(m, data_loader, epoch)
                rs.append(r)

        total_length = sum(r["Length"] for r in rs)
        total_loss = sum(r["Length"] * r["Loss"] for r in rs)
        res = {
            "_meta": {"type": "Global Validation"},
            "E": epoch,
            "Length": total_length,
            "Loss": total_loss / total_length,
        }
        for name in self.metrics:
            res[name] = sum(r[name] * r["Length"] for r in rs) / total_length

        # Output to file
        self.json_logger.info(res)
        self.debug_logger.info(
            f"\n=> Global Validation Eval Loss={res['Loss']:.4f} "
            + " ".join(
                name + "=" + "{:>8.4f}".format(res[name]) for name in self.metrics
            )
            + "\n"
        )


class KNNPerIndividualEvaluator(IndividualEvaluator):
    """Evaluate individual worker's model on its data."""

    def _fix_output(self, output):
        # Note that KNNPer uses softmax whereas the loss we used assume log_softmax (for mnist)?
        output = torch.log(output)
        return output

    def evaluate_worker(self, output, target, lambda_, rank):
        output = self._fix_output(output)

        r = {
            "_meta": {"type": "KNNPer Individual Validation"},
            "Length": 0,
            "Loss": 0,
            "Rank": rank,
            "Lambda": lambda_,
        }
        r["Loss"] += self.loss_fn(output, target).item() * len(target)
        r["Length"] += len(target)
        for name, metric in self.metrics.items():
            r[name] = r.get(name, 0) + metric(output, target) * len(target)

        for name in self.metrics:
            r[name] /= r["Length"]

        r["Loss"] /= r["Length"]
        self.json_logger.info(r)
        return r

    def evaluate_all(self, output, target, lambda_):
        output = self._fix_output(output)

        r = {
            "_meta": {"type": "KNNPer Individual Validation (Aggregated)"},
            "Length": 0,
            "Loss": 0,
            "Lambda": lambda_,
        }
        r["Loss"] += self.loss_fn(output, target).item() * len(target)
        r["Length"] += len(target)
        for name, metric in self.metrics.items():
            r[name] = r.get(name, 0) + metric(output, target) * len(target)

        for name in self.metrics:
            r[name] /= r["Length"]

        r["Loss"] /= r["Length"]
        self.json_logger.info(r)
        return r

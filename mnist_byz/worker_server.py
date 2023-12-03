import copy
from collections import defaultdict
import logging
import numpy as np
import torch
from typing import Union, Callable, Tuple
import datastore

debug_logger = logging.getLogger("debug")

# ---------------------------------------------------------------------------- #
#                              Servers and Workers                             #
# ---------------------------------------------------------------------------- #


class MultiModelServer(object):
    def __init__(self, model_fn, opt_fn, K, device, init, check_models=False):
        if init == "identical":
            model = model_fn()
            models = [copy.deepcopy(model) for _ in range(K)]
        else:
            models = [model_fn() for _ in range(K)]
        self.models = models
        for m in models:
            m.to(device)
        self.opts = [opt_fn(m) for m in self.models]

        if check_models:
            self._check_models()

    def _check_models(self):
        debug_logger.info(
            "\nCheck if server's initial models are identical or not "
            "by inspecting first few parameters."
        )
        for model in self.models:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    debug_logger.info(param.view(-1)[:5].data)
                    break

    def apply_gradient(self, k: int) -> None:
        self.opts[k].step()

    def set_gradient(self, k: int, gradient: torch.Tensor) -> None:
        beg = 0
        for group in self.opts[k].param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # for p in self.model.parameters():
                end = beg + len(p.grad.view(-1))
                x = gradient[beg:end].reshape_as(p.grad.data)
                p.grad.data = x.clone().detach()
                beg = end


class MultiUpdateServer(object):
    def __init__(self, K, model, init):
        d = self.compute_model_dimension(model)
        if init == "gaussian":
            self.updates = [torch.randn(d) for _ in range(K)]
        elif (
            init == "lazy"
        ):  # delayed to later to use the worker gradients as initialization
            self.updates = [None for _ in range(K)]
        else:
            raise NotImplementedError

    def compute_model_dimension(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _log_initial_distances(
        self,
    ):
        pass


class SingleModelServer(MultiModelServer):
    def __init__(self, model_fn, opt_fn, device):
        super().__init__(
            model_fn, opt_fn, K=1, device=device, init="identical", check_models=False
        )

    @property
    def model(self):
        return self.models[0]

    @property
    def opt(self):
        return self.opts[0]

    def set_gradient(self, gradient: torch.Tensor) -> None:
        super().set_gradient(0, gradient)

    def apply_gradient(self) -> None:
        super().apply_gradient(0)


class _BaseWorker(object):
    """Has everything except for gradient related functions."""

    def __init__(
        self,
        index: int,
        metrics: dict,
        momentum: float,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.modules.loss._Loss,
        device: Union[torch.device, str],
        lr_scheduler: None,
        rng=None,
    ):
        self.index = index
        self.momentum = momentum
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.device = device
        self.lr_scheduler = lr_scheduler

        for name in metrics:
            assert name not in ["loss", "length"]
        self.metrics = metrics

        self.train_loader_iterator = None
        # self.last_batch has attribute:
        #   - `data`: last data
        #   - `target`: last target
        self.last_batch = {}
        self.rng = rng

    def __str__(self) -> str:
        return f"_BaseWorker(index={self.index})"

    def train_epoch_start(self) -> None:
        self.train_loader_iterator = iter(self.data_loader)
        self.model.train()

    def get_data_target(self):
        data, target = self.train_loader_iterator.__next__()
        data, target = data.to(self.device), target.to(self.device)
        return data, target


class _GradientManager(object):
    """Compute a gradient/update on a model and save it to state.

    Ensure that model and optimizer are unaltered.
    """

    def __init__(self, momentum):
        self.momentum = momentum
        self.state = defaultdict(dict)

    def _compute_gradient(self, data, target, model, opt, loss_fn, update_state=True):
        # compute update but do not apply
        opt.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        if update_state:
            self._save_updates_to_state(opt)
        # opt.zero_grad()  # ensure the gradient will not be used
        return output, loss

    def _save_updates_to_state(self, opt) -> None:
        index = 0
        for group in opt.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[index]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.clone(p.grad).detach()
                else:
                    param_state["momentum_buffer"].mul_(self.momentum).add_(p.grad)
                index += 1

    def _get_buffer_by_name(self, name):
        layer_gradients = []
        for index in range(max(self.state) + 1):
            param_state = self.state[index]
            layer_gradients.append(param_state[name].data.view(-1))
        return torch.cat(layer_gradients)

    def _get_updates(self):
        return self._get_buffer_by_name("momentum_buffer")

    def _get_gradient_from_opt(self, opt):
        layer_gradients = []
        for group in opt.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                layer_gradients.append(torch.clone(p.grad).detach().data.view(-1))
        return torch.cat(layer_gradients)


class _WorkerWithGradient(_BaseWorker):
    def __init__(self, model=None, opt=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.opt = opt
        self._gradient_manager = _GradientManager(self.momentum)

    def set_model_opt(self, model, opt):
        """In some applications, the worker computes computes gradient on the incoming model"""
        self.model = model
        self.opt = opt

    def __str__(self) -> str:
        return f"_WorkerWithGradient(index={self.index})"

    def _record(self, data, target, output, loss):
        # cache
        self.last_batch["data"] = data
        self.last_batch["target"] = target

        # gather metrics
        results = {}
        results["loss"] = loss.item()
        results["length"] = len(target)
        results["metrics"] = {}
        for name, metric in self.metrics.items():
            results["metrics"][name] = metric(output, target)
        return results

    def compute_gradient(self) -> Tuple[float, int]:
        data, target = self.get_data_target()

        output, loss = self._gradient_manager._compute_gradient(
            data, target, self.model, self.opt, self.loss_fn
        )

        return self._record(data, target, output, loss)

    def get_gradient(self) -> torch.Tensor:
        return self._gradient_manager._get_updates()

    def apply_gradient(self) -> None:
        self.opt.step()

    def set_gradient(self, gradient: torch.Tensor) -> None:
        beg = 0
        for p in self.model.parameters():
            end = beg + len(p.grad.view(-1))
            x = gradient[beg:end].reshape_as(p.grad.data)
            p.grad.data = x.clone().detach()
            beg = end

    def compute_gradient_over_data(self, data, target) -> Tuple[float, int]:
        output, loss = self._gradient_manager._compute_gradient(
            data, target, self.model, self.opt, self.loss_fn, update_state=False
        )
        return self._record(data, target, output, loss)

    def get_gradient_from_opt(self) -> torch.Tensor:
        return self._gradient_manager._get_gradient_from_opt(self.opt)


class IFCAWorker(_WorkerWithGradient):
    def __init__(self, validation_data_loader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_data_loader = validation_data_loader
        self.validation_loader_iterator = None
        self.best_cluster = None

    def __str__(self) -> str:
        return f"IFCAWorker(index={self.index})"

    def train_epoch_start(self) -> None:
        self.train_loader_iterator = iter(self.data_loader)
        self.validation_loader_iterator = iter(self.validation_data_loader)

    def get_validation_data_target(self):
        try:
            data, target = self.validation_loader_iterator.__next__()
        except StopIteration:
            self.validation_loader_iterator = iter(self.validation_data_loader)
            data, target = self.validation_loader_iterator.__next__()
        data, target = data.to(self.device), target.to(self.device)
        return data, target


class FLWorker(_WorkerWithGradient):
    def train_epoch_start(self) -> None:
        self.train_loader_iterator = iter(self.data_loader)

    def __str__(self) -> str:
        return f"FLWorker(index={self.index})"


class LocalWorker(_WorkerWithGradient):
    def __str__(self) -> str:
        return f"LocalWorker(index={self.index})"


class FCWorker(_WorkerWithGradient):
    """
    A FCWorker
    """

    def __str__(self) -> str:
        return f"FCWorker(index={self.index})"

    def compute_gradient_over_data(self, data, target) -> Tuple[float, int]:
        output, loss = self._gradient_manager._compute_gradient(
            data, target, self.model, self.opt, self.loss_fn, update_state=False
        )
        return self._record(data, target, output, loss)

    def get_gradient_from_opt(self) -> torch.Tensor:
        return self._gradient_manager._get_gradient_from_opt(self.opt)


class DittoWorker(_WorkerWithGradient):
    """
    Ditto worker has a different local objective which penalize the distance
    between global model and local model. The hyper-parameter is lambda.
    """

    def compute_ditto_penalization(self, lambda_: float, server_opt):
        # Save info to gradient manager.
        index = 0
        for group_w, group_s in zip(self.opt.param_groups, server_opt.param_groups):
            for p_w, p_s in zip(group_w["params"], group_s["params"]):
                if p_w.grad is None:
                    continue
                param_state = self._gradient_manager.state[index]
                update = lambda_ * (p_w.data - p_s.data)
                param_state["ditto"] = torch.clone(update).detach()

                index += 1

    def apply_ditto_gradient(self) -> None:
        g = self.get_gradient()
        penalization = self._gradient_manager._get_buffer_by_name("ditto")
        g += penalization
        self.set_gradient(g)
        self.apply_gradient()


class KNNPerWorker(_WorkerWithGradient):
    def __init__(self, task_name, k=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.k = k
        self.features_dimension = 10

        # Cache the representations in a batch
        self.activation = {}

        self.initialize_datastore()

    def train_epoch_start(self) -> None:
        self.train_loader_iterator = iter(self.data_loader)

    def initialize_datastore(self):
        self.datastore = datastore.DataStore(
            capacity=999999999,
            strategy="random",
            dimension=self.features_dimension,
            rng=self.rng,
        )

    def fill_database(self, model):
        """
        Build the database with training samples
        """
        debug_logger.info("Start filling database.")

        def hook_fn(model, input_, output):
            self.activation["features"] = output.squeeze().cpu().numpy()

        def _prepare():
            model.eval()
            n_samples = len(self.data_loader.dataset)

            if self.task_name == "mnist":
                handle = model.fc2.register_forward_hook(hook_fn)
                self.num_classes = 10
                self.embeddings_dim = 10
            else:
                raise NotImplementedError

            embeddings = np.zeros(
                shape=(n_samples, self.embeddings_dim), dtype=np.float32
            )
            outputs = np.zeros(shape=(n_samples, self.num_classes), dtype=np.float32)
            labels = np.zeros(shape=(n_samples,), dtype=np.uint16)

            return embeddings, outputs, labels

        embeddings, outputs, labels = _prepare()

        # Filling database one by one.
        counter = 0
        for i, (data, target) in enumerate(self.data_loader):
            with torch.no_grad():
                outs = model(data)

            embeddings[counter : counter + len(target)] = self.activation["features"]
            outputs[counter : counter + len(target)] = outs
            labels[counter : counter + len(target)] = target.data.cpu().numpy()
            counter += len(target)

        self.datastore.build(embeddings, labels)

        debug_logger.info("Finish filling database.")
        return embeddings, outputs, labels

    def _knn_inference(self, model, data, scale=1.0):
        with torch.no_grad():
            # prediction with FedAvg model
            pg = model(data)

        # features / representation
        features = self.activation["features"]

        # Find k-nearest neighbors:
        #   Shapes of features, distances, indices = (batch_size, K)
        distances, indices = self.datastore.index.search(features, self.k)

        #   Shapes of similarities, neighbors_labels = (batch_size, K)
        similarities = np.exp(-distances / (self.features_dimension * scale))

        neighbors_labels = self.datastore.labels[indices]

        # Shape of masks = (num_classes, batch_size, K)
        masks = np.zeros(((self.num_classes,) + similarities.shape))

        for class_id in range(self.num_classes):
            masks[class_id] = neighbors_labels == class_id

        pk = (similarities * masks).sum(axis=2) / similarities.sum(axis=1)

        # Convert it to a tensor of (batch_size, num_classes)
        pk = torch.Tensor(pk).T
        return pk, pg

    def inference(self, model, lambda_, test_dataloader):
        model.eval()
        predictions = []
        targets = []
        for data, target in test_dataloader:
            pk, pg = self._knn_inference(model, data)

            # Combined prediction
            p = lambda_ * pk + (1 - lambda_) * pg

            predictions.append(p)
            targets.append(target)

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        return predictions, targets

    def __str__(self) -> str:
        return f"KNNPerWorker(index={self.index})"


################################ Add Byzantine workers ################################


class ByzantineWorker(_WorkerWithGradient):
    def __init__(self, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = trainer

    def compute_gradient(self) -> Tuple[float, int]:
        # Use self.simulator to get all other workers
        # Note that the byzantine worker does not modify the states directly.
        return super().compute_gradient()

    def get_gradient(self) -> torch.Tensor:
        # Use self.simulator to get all other workers
        return super().get_gradient()

    def _dummy_record(self, data, target, output, loss):
        # cache
        self.last_batch["data"] = 0
        self.last_batch["target"] = 0

        # gather metrics
        results = {}
        results["loss"] = 0
        results["length"] = 1
        results["metrics"] = {}
        for name, metric in self.metrics.items():
            results["metrics"][name] = 0
        return results


class GaussianWorker(ByzantineWorker):
    def train_epoch_start(self):
        pass

    def compute_gradient(self) -> Tuple[float, int]:
        # Do nothing
        pass

    def apply_gradient(self) -> None:
        # Do nothing
        pass

    def get_gradient(self) -> torch.Tensor:
        # Get good worker's gradient and then replace it with gaussian noise.
        gradient = self.trainer.workers[0].get_gradient()
        return torch.randn_like(gradient) * 200 + 1000000

    def compute_ditto_penalization(self, lambda_: float, server_opt):
        # Do nothing
        pass

    def apply_ditto_gradient(self):
        # Do nothing
        pass

    def compute_gradient_over_data(self, data, target) -> Tuple[float, int]:
        # Do nothing
        pass

    def get_gradient_from_opt(self) -> torch.Tensor:
        # Get good worker's gradient and then replace it with gaussian noise.
        gradient = self.trainer.workers[0].get_gradient_from_opt()
        return torch.randn_like(gradient) * 200 + 1000000


class BitFlippingWorker(ByzantineWorker):
    def train_epoch_start(self):
        pass

    def compute_gradient(self) -> Tuple[float, int]:
        # Do nothing
        pass

    def apply_gradient(self) -> None:
        # Do nothing
        pass

    def get_gradient(self) -> torch.Tensor:
        # Get good worker's gradient and then replace it with gaussian noise.
        gradient = self.trainer.workers[0].get_gradient()
        return -gradient

    def compute_ditto_penalization(self, lambda_: float, server_opt):
        # Do nothing
        pass

    def apply_ditto_gradient(self):
        # Do nothing
        pass

    def compute_gradient_over_data(self, data, target) -> Tuple[float, int]:
        # Do nothing
        pass

    def get_gradient_from_opt(self) -> torch.Tensor:
        # Get good worker's gradient and then replace it with gaussian noise.
        gradient = self.trainer.workers[0].get_gradient_from_opt()
        return -gradient

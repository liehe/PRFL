import logging
import numpy as np
import torch


class ClusterAlgorithmEvaluator(object):
    """
    This class contains the main logistics of the algorithm.
    """

    def __init__(self, dataset, cluster_fn, loss_fn, grad_fn, K, E, seed) -> None:
        self.dataset = dataset
        # cluster_fn is a callback function which takes in a matrix of size (n, d) and returns its assignments
        self.cluster_fn = cluster_fn
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.K = K
        self.E = E
        self.seed = seed

        self.history = []

    def _eval_err(self, assignments, centers):
        """Evaluate the error of (assignments, centers) pair with theta_star."""
        metric = []
        for i, client_data in zip(assignments, self.dataset.data):
            theta_star = client_data["theta_star"]
            center = centers[i]
            metric.append(torch.linalg.norm(theta_star - center))
        return torch.mean(torch.Tensor(metric))

    def run(self, init_assignments: list, init_centers: np.array):
        # initialization
        np.random.seed(self.seed)
        assignments = init_assignments.copy()
        centers = init_centers.copy()
        n = len(assignments)
        k, d = centers.shape
        assert k == self.K

        history = [
            dict(
                loss=self._eval_err(assignments, centers),
                epoch=-1,
                assignments=assignments,
            )
        ]
        for epoch in range(self.E):
            assignments, centers = self.cluster_fn(
                dataset=self.dataset,
                loss_fn=self.loss_fn,
                grad_fn=self.grad_fn,
                centers=centers,
                assignments=assignments,
                epoch=epoch,
            )
            loss = self._eval_err(assignments, centers)
            history.append(dict(loss=loss, epoch=epoch, assignments=assignments))
        self.history = history


class PersonalizationAlgorithmEvaluator(object):
    """
    This class contains the main logistics of the algorithm.
    """

    def __init__(self, dataset, cluster_fn, K, E, log_interval) -> None:
        self.dataset = dataset
        # cluster_fn is a callback function which takes in a matrix of size (n, d) and returns its assignments
        self.cluster_fn = cluster_fn
        self.K = K
        self.E = E
        self.log_interval = log_interval
        self.debug_logger = logging.getLogger("debug")
        self.json_logger = logging.getLogger("stats")

        self.history = []

    def _eval_err(self, models):
        """Evaluate the error of models pair with theta_star."""
        metric = []
        for model, client_data in zip(models, self.dataset.data):
            theta_star = client_data["theta_star"]
            metric.append(torch.linalg.norm(theta_star - model))
        return torch.mean(torch.Tensor(metric))

    def run(self, init_assignments: list, init_centers: np.array):
        # initialization
        assignments = init_assignments.copy()
        centers = init_centers.clone()
        n = len(assignments)
        k, d = centers.shape
        assert k == self.K

        models = [centers[i].clone() for i in assignments]
        history = [dict(loss=self._eval_err(models), epoch=-1)]
        for epoch in range(self.E):
            models = self.cluster_fn(models=models, epoch=epoch)
            loss = self._eval_err(models)
            history.append(dict(loss=loss, epoch=epoch))
            self.json_logger.info(
                {
                    "_meta": {"type": "hist"},
                    "E": epoch,
                    "Loss": loss.item(),
                }
            )
            if epoch % self.log_interval == 0:
                self.debug_logger.info(f"E={epoch} loss={loss:.8f}")

        self.history = history
        self.models = models

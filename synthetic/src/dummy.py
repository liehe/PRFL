from sys import get_coroutine_origin_tracking_depth
import numpy as np
import logging
import torch
from src.utils import MomentumBuffer

json_logger = logging.getLogger("stats")
debug_logger = logging.getLogger("debug")


class DummyPersonalizedOptimizer(object):

    def __init__(self, dataset, loss_fn, grad_fn, K, eta, n_local_gd_steps, n_total_devices, beta, rng):
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.eta = eta
        self.n_local_gd_steps = n_local_gd_steps
        self.momentums = [MomentumBuffer(beta) for _ in range(n_total_devices)]

    def __call__(self, models, epoch):
        for momentum, model, client_data in zip(self.momentums, models, self.dataset.data):
            for _ in range(self.n_local_gd_steps):
                grad = self.grad_fn(client_data['x'], client_data['y'], model)
                momentum.update(grad)
                model -= self.eta * momentum.buff
        return models


class DummyCentralizedOptimizer(object):

    def __init__(self, dataset, loss_fn, grad_fn, K, eta, n_local_gd_steps, n_total_devices, beta, rng):
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.eta = eta
        self.n_local_gd_steps = n_local_gd_steps
        self.momentum = MomentumBuffer(beta)

    def __call__(self, models, epoch):
        # Pack dataset together
        data_x = []
        data_y = []
        for client_data in self.dataset.data:
            data_x.append(client_data['x'])
            data_y.append(client_data['y'])

        model = models[0]
        data = {'x': torch.cat(data_x), 'y': torch.cat(data_y)}
        for _ in range(self.n_local_gd_steps):
            grad = self.grad_fn(data['x'], data['y'], model)
            self.momentum.update(grad)
            model -= self.eta * self.momentum.buff
        return [model] * len(models)


class GroundTruthOptimizer(object):

    def __init__(self, dataset, loss_fn, grad_fn, K, eta, n_local_gd_steps, n_total_devices, beta, rng):
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.n = len(dataset.data)
        self.K = K
        self.eta = eta
        self.n_local_gd_steps = n_local_gd_steps
        self.momentums = [MomentumBuffer(beta) for _ in range(K)]

        self.num_device_per_cluster = self.n // self.K

    def _FedAvg(self, models, cluster_data, momentum):
        # Pack dataset together
        data_x = []
        data_y = []
        for client_data in cluster_data:
            data_x.append(client_data['x'])
            data_y.append(client_data['y'])

        model = sum(models) / len(models)
        data = {'x': torch.cat(data_x), 'y': torch.cat(data_y)}
        for _ in range(self.n_local_gd_steps):
            grad = self.grad_fn(data['x'], data['y'], model)
            momentum.update(grad)
            model -= self.eta * momentum.buff
        return [model] * len(models)

    def __call__(self, models, epoch):
        self._run_hooks(models, epoch)
        m = self.num_device_per_cluster
        out_models = []
        for k in range(self.K):
            cluster_models = models[k*m:(k+1)*m]
            cluster_data = self.dataset.data[k*m:(k+1)*m]
            cluster_momentum = self.momentums[k]
            out_models += self._FedAvg(cluster_models,
                                       cluster_data, cluster_momentum)

        return out_models

    def _run_hooks(self, models, epoch):
        self._run_verify_assumption_hook(models, epoch)

    def _run_verify_assumption_hook(self, models, epoch):
        # Choose arbitrary model which is used to compute gradient on all data
        model = models[0]
        grads = []
        for i in range(self.n):
            data = self.dataset.data[i]
            grad = self.grad_fn(data['x'], data['y'], model)
            grads.append(grad)

        r = {
            "_meta": {"type": "Verify Assumption"},
            "E": epoch,
            # || \nabla \bar{f}_i(x) ||
            # Length K
            "ClusterCenterGradNorms": [],
            # || \nabla f_i(x) ||
            # Length n
            "GradNorms": [],
            # || \nabla f_i(x) - \nabla f_j(x) ||
            "ClusterCenterDistances": [
                [0 for _ in range(self.n)]
                for _ in range(self.n)
            ],
            # || \nabla f_i(x) - \nabla \bar{f}_i(x) ||
            "IntraClusterDistances": []
        }

        def _compute_cluster_mean(grads):
            # NOTE: that we use oracle information about clustering from get_data_loader.
            cluster_size = self.num_device_per_cluster
            groundtruth_centers = []
            for k in range(self.K):
                cluster_grads = grads[k * cluster_size:(k+1) * cluster_size]
                try:
                    groundtruth = sum(cluster_grads) / len(cluster_grads)
                except Exception as e:
                    print(k, cluster_size)
                    print(e)
                    raise
                groundtruth_centers.append(groundtruth)
            return groundtruth_centers

        centers = _compute_cluster_mean(grads)
        for g in centers:
            grad_norm = torch.linalg.norm(g)
            r["ClusterCenterGradNorms"].append(grad_norm.item())

        # Compute the norm of centers
        # debug_logger.info("=> Compute the norm of centers")
        for g in grads:
            grad_norm = torch.linalg.norm(g)
            r["GradNorms"].append(grad_norm.item())

        # Compute the distances between centers
        # debug_logger.info("=> Compute the distances between centers")
        cluster_size = self.num_device_per_cluster
        for i in range(self.n):
            k = i // cluster_size
            for j in range(i+1, self.n):
                grad_dist = torch.linalg.norm(grads[i] - grads[j]).item()
                r["ClusterCenterDistances"][i][j] = grad_dist
                r["ClusterCenterDistances"][j][i] = grad_dist

        # Compute the variance within each cluster
        # debug_logger.info("=> Compute the variance within each cluster")

        def _compute_intra_cluster_distance(centers, grads):
            # NOTE: that we use oracle information about clustering from get_data_loader.
            distances = []
            cluster_size = self.num_device_per_cluster

            for k in range(self.K):
                center = centers[k]
                cluster_grads = grads[k * cluster_size:(k+1) * cluster_size]

                for g in cluster_grads:
                    distance = torch.linalg.norm(g - center)
                    distances.append(distance.item())
            return distances

        r["IntraClusterDistances"] = _compute_intra_cluster_distance(
            centers, grads)
        # debug_logger.info(r["IntraClusterDistances"])
        json_logger.info(r)

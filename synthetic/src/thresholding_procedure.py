from ast import Not
import numpy as np
import itertools
import torch
import sklearn.cluster
import functools

from src.utils import MomentumBuffer
from src.kmeans import KMeans


class ThresholdingV1(object):
    def __init__(self, K, n_iterations, n_sample_devices, rng):
        self.n_iterations = n_iterations
        self.n_sample_devices = n_sample_devices
        self.rng = rng
        self.distance_clustering = KMeans(K=K, iterations=10)

    def __call__(self, thresholding_centers, momentums, tau):
        total_devices = len(momentums)

        output = []
        for k, v0 in enumerate(thresholding_centers):
            v = v0.copy()
            for t in range(self.n_iterations):
                indices = self.rng.choice(
                    np.arange(total_devices), size=self.n_sample_devices, replace=False
                )

                tau = self.get_clipping_radius(t, v, [momentums[i] for i in indices])
                ys = []
                for i in indices:
                    m = momentums[i]
                    y = m if torch.linalg.norm(m - v) < tau else v
                    ys.append(y)

                v = sum(ys) / len(ys)

            output.append(v)

        return output

    def get_clipping_radius(self, t, v, momentums):
        distances = [torch.linalg.norm(v - m) for m in momentums]
        distances = torch.Tensor(np.expand_dims(np.array(distances), axis=1))
        a, c = self.distance_clustering(distances)
        c = sorted(set(c.flatten()))
        while len(c) < 2:
            a, c = self.distance_clustering(distances)
            c = sorted(set(c.flatten()))
        return c[1] // 2


class HardAssignment(object):
    def __call__(self, centers, updates):
        assignments = []
        for update in updates:
            distances = []
            for center in centers:
                distances.append(torch.linalg.norm(update - center))
            i = np.argmin(distances)
            assignments.append(i)
        return assignments


class MovingAverageAssignment(object):
    def __init__(self, m, K, beta):
        self.K = K
        self.buffers = [[MomentumBuffer(beta=beta) for _ in range(K)] for _ in range(m)]

    def __call__(self, centers, updates):
        assignments = []
        for i, update in enumerate(updates):
            for k, center in enumerate(centers):
                dist = torch.linalg.norm(update - center) ** 2
                self.buffers[i][k].update(dist)
            i = np.argmin([self.buffers[i][k] for k in range(self.K)])
            assignments.append(i)
        return assignments


class ThresholdingProcedureOptimizer(object):
    """
    Non personalized --- models in the same cluster are forced to be identical
    """

    def __init__(
        self,
        dataset,
        loss_fn,
        grad_fn,
        K,
        eta,
        n_local_gd_steps,
        n_clustering_steps,
        n_total_devices,
        beta,
        thresholding_params,
        rng,
        tau,
        assignment_strategy="moving average",
    ) -> None:
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.tau = tau
        self.eta = eta
        self.n_local_gd_steps = n_local_gd_steps
        self.thresholding_params = thresholding_params

        self.clustering_oracle = ThresholdingV1(K=K, rng=rng, **thresholding_params)
        self.momentums = [MomentumBuffer(beta) for _ in range(n_total_devices)]

        if assignment_strategy == "hard":
            self.assigner = HardAssignment()
        elif assignment_strategy == "moving average":
            self.assigner = MovingAverageAssignment(m=n_total_devices, K=K, beta=beta)
        else:
            raise NotImplementedError

    def _update_model(self, model, momentum, grad_fn, client_data):
        # Initialize with assigned cluster center
        for _ in range(self.n_local_gd_steps):
            grad = grad_fn(client_data["x"], client_data["y"], model)
            momentum.update(grad)
            model -= self.eta * momentum.buff
        return model

    def _decide_thresholding_centers(self, updates, assignments, strategy="median"):
        thresholding_centers = []
        a = sorted(zip(assignments, updates), key=lambda x: x[0])
        for k, g in itertools.groupby(a, key=lambda x: x[0]):
            groups = np.array(list(v for _, v in g))
            if strategy == "median":
                thresholding_centers.append(np.median(groups, axis=0))
            else:
                raise NotImplementedError
        return thresholding_centers

    def _assign(self, centers, updates):
        assignments = []
        for update in updates:
            distances = []
            for center in centers:
                distances.append(torch.linalg.norm(update - center))
            i = np.argmin(distances)
            assignments.append(i)

        return assignments

    def __call__(self, centers, assignments, epoch):
        updates = []
        for momentum, assignment, client_data in zip(
            self.momentums, assignments, self.dataset.data
        ):
            model = centers[assignment].copy()
            model = self._update_model(model, momentum, self.grad_fn, client_data)
            updates.append((centers[assignment] - model) / self.eta)

        # Perform the clustering algorithm
        thresholding_centers = self._decide_thresholding_centers(
            updates, assignments, strategy="median"
        )

        updated_centers = self.clustering_oracle(
            thresholding_centers, np.array(updates), self.tau
        )

        assignments = self._assign(updated_centers, updates)

        output = []
        for c, u in zip(centers, updated_centers):
            output.append(c - self.eta * u)

        return assignments, output


##################### Algorithm in the paper #####################


class ThresholdingV2(object):
    def __init__(self, K, num_thresholding_steps, num_sampled_devices, strategy, rng):
        self.num_thresholding_steps = num_thresholding_steps
        self.num_sampled_devices = num_sampled_devices
        self.strategy = strategy
        self.rng = rng

        self.tau_fn = self.get_tau_fn(strategy, K)

    def get_tau_fn(self, strategy, K):
        K = 3
        if strategy == "adaptive-1":
            simple_clustering_oracle = KMeans(K=K, iterations=10)

            def fn(c, a):
                c = sorted(c)
                return (c[1] + c[2]) / 2

            return functools.partial(
                self._adaptive_radius, clustering_oracle=simple_clustering_oracle, fn=fn
            )

        if strategy == "adaptive-2":
            simple_clustering_oracle = KMeans(K=K, iterations=10)

            def fn(c, a):
                c = sorted(c)
                return c[1] * 0.99

            return functools.partial(
                self._adaptive_radius, clustering_oracle=simple_clustering_oracle, fn=fn
            )

        if strategy == "adaptive-3":
            simple_clustering_oracle = KMeans(K=K, iterations=10)

            def fn(c, a):
                sorted_c = sorted(list(zip(range(len(c)), c)), key=lambda x: x[1])
                index_c, min_c = sorted_c[0]
                s = sum(i == index_c for i in a)
                if s <= 2:
                    return (sorted_c[1][1] + sorted_c[2][1]) / 2
                else:
                    return (sorted_c[0][1] + sorted_c[1][1]) / 2

            return functools.partial(
                self._adaptive_radius, clustering_oracle=simple_clustering_oracle, fn=fn
            )

        if strategy.startswith("Q"):
            quantile = float(strategy[1:])
            assert quantile <= 1 and quantile >= 0
            return functools.partial(self._quantile_radius, quantile=quantile)

        raise NotImplementedError(strategy)

    def __call__(self, momentums, center0):
        total_devices = len(momentums)

        center = center0.clone()
        for _ in range(self.num_thresholding_steps):
            if self.num_sampled_devices == "all":
                indices = np.arange(total_devices)
            else:
                indices = self.rng.choice(
                    np.arange(total_devices),
                    size=self.num_sampled_devices,
                    replace=False,
                )
            sampled_momentum = [momentums[i] for i in indices]
            tau = self.tau_fn(center, sampled_momentum)

            ys = []
            for i in indices:
                m = momentums[i]
                y = m if torch.linalg.norm(m - center) < tau else center
                ys.append(y)
            center = sum(ys) / len(ys)

        return center

    def _adaptive_radius(self, v, momentums, clustering_oracle, fn):
        distances = torch.Tensor([torch.linalg.norm(v - m) for m in momentums])
        distances = torch.unsqueeze(distances, dim=1)
        a, c = clustering_oracle(distances)
        while len(c) < 2:
            a, c = clustering_oracle(distances)
        return fn(c, a)

    def _quantile_radius(self, v, momentums, quantile):
        distances = sorted([torch.linalg.norm(v - m) for m in momentums])
        return distances[int(len(distances) * quantile)]


class PersonalizedThresholdingProcedureOptimizer(object):
    """ """

    def __init__(
        self,
        dataset,
        loss_fn,
        grad_fn,
        K,
        eta,
        n_local_gd_steps,
        n_total_devices,
        beta,
        rng,
        thresholding_params,
        assignment_strategy="moving average",
    ) -> None:
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.eta = eta
        self.n_local_gd_steps = n_local_gd_steps
        self.thresholding_params = thresholding_params

        self.clustering_oracle = ThresholdingV2(K=K, rng=rng, **thresholding_params)
        self.momentums = [MomentumBuffer(beta) for _ in range(n_total_devices)]

        if assignment_strategy == "hard":
            self.assigner = HardAssignment()
        elif assignment_strategy == "moving average":
            self.assigner = MovingAverageAssignment(m=n_total_devices, K=K, beta=beta)
        else:
            raise NotImplementedError

    def _get_update(self, model0, grad_fn, client_data):
        model = model0.clone()
        # Initialize with assigned cluster center
        for _ in range(self.n_local_gd_steps):
            grad = grad_fn(client_data["x"], client_data["y"], model)
            model -= self.eta * grad
        return (model0 - model) / self.eta

    def __call__(self, models, epoch):
        # Step 1: compute update based on local dataset and model
        assigned_updates = []
        for (i, momentum), model in zip(enumerate(self.momentums), models):
            updates = []
            for client_data in self.dataset.data:
                update = self._get_update(model, self.grad_fn, client_data)
                updates.append(update)
            updates = torch.stack(updates, axis=0)

            # Step 2: cluster the updates and return the update-centers (of momentum)
            centered_gradient = self.clustering_oracle(updates, updates[i])
            assigned_updates.append(centered_gradient)

        # # Step 3: assign device to each cluster
        # _assigned_updates = self._assign_updates(updated_centers, updates)

        # Step 4: use the updated centers to update each device's local model
        for momentum, model, update in zip(self.momentums, models, assigned_updates):
            momentum.update(update)
            model -= self.eta * momentum.buff

        return models


class EfficientPersonalizedThresholdingProcedureOptimizer(
    PersonalizedThresholdingProcedureOptimizer
):
    """ """

    def __init__(self, frequency, subgroup_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frequency = frequency
        self.subgroup_size = subgroup_size

    def _get_update(self, model0, grad_fn, client_data):
        model = model0.clone()
        # Initialize with assigned cluster center
        for _ in range(self.n_local_gd_steps):
            grad = grad_fn(client_data["x"], client_data["y"], model)
            model -= self.eta * grad
        return (model0 - model) / self.eta

    def _local_call(self, models):
        for momentum, model, client_data in zip(
            self.momentums, models, self.dataset.data
        ):
            update = self._get_update(model, self.grad_fn, client_data)
            momentum.update(update)
            model -= self.eta * momentum.buff

    def _thresholding(self, models, momentums, data):
        # Step 1: compute update based on local dataset and model
        assigned_updates = []
        for (i, momentum), model in zip(enumerate(momentums), models):
            updates = []
            for client_data in data:
                update = self._get_update(model, self.grad_fn, client_data)
                updates.append(update)
            updates = torch.stack(updates, axis=0)

            # Step 2: cluster the updates and return the update-centers (of momentum)
            centered_gradient = self.clustering_oracle(updates, updates[i])
            assigned_updates.append(centered_gradient)

        # # Step 3: assign device to each cluster
        # _assigned_updates = self._assign_updates(updated_centers, updates)

        # Step 4: use the updated centers to update each device's local model
        for momentum, model, update in zip(momentums, models, assigned_updates):
            momentum.update(update)
            model -= self.eta * momentum.buff

    def _thresholding_call(self, models, subgroup_size):
        ids = np.arange(len(models))
        np.random.shuffle(ids)
        while len(ids) > 0:
            _ids, ids = ids[:subgroup_size], ids[subgroup_size:]
            self._thresholding(
                [models[i] for i in _ids],
                [self.momentums[i] for i in _ids],
                [self.dataset.data[i] for i in _ids],
            )

    def __call__(self, models, epoch):
        if epoch % self.frequency == 0:
            self._thresholding_call(models, self.subgroup_size)
        else:
            self._local_call(models)

        return models


class IFCA(object):
    """ """

    def __init__(
        self,
        dataset,
        loss_fn,
        grad_fn,
        K,
        eta,
        n_local_gd_steps,
        n_total_devices,
        beta,
        rng,
        strategy="gradient",
    ) -> None:
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.K = K
        self.eta = eta

        self.models = None

    def _init_models(self, models):
        if self.models is None:
            self.models = models[:: self.K]

    def _return_models(self, models):
        # stich together
        _models = []
        _per_device = len(models) // self.K
        for model in self.models:
            _models += [model] * _per_device
        return _models

    def __call__(self, models, epoch):
        self._init_models(models)

        s = []
        grads = []
        for client_data in self.dataset.data:
            losses = []
            for model in self.models:
                loss = self.loss_fn(client_data["x"], client_data["y"], model)
                losses.append(loss.mean())
            j_hat = np.argmin(losses)
            si = [int(j == j_hat) for j in range(self.K)]
            s.append(si)

            grad = self.grad_fn(client_data["x"], client_data["y"], self.models[j_hat])
            grads.append(grad)

        # Update model
        for j in range(self.K):
            mean_grad = sum(s[i][j] * g for i, g in enumerate(grads)) / len(models)
            # mean_grad = sum(s[i][j] * g for i, g in enumerate(grads)) / max(sum(
            #     s[i][j] for i, g in enumerate(grads)), 1)
            self.models[j] -= self.eta * mean_grad
            # for i in range(len(dataset.data)):
            #     if s[i][j]:
            #         models[i] -= self.eta * mean_grad

        return self._return_models(models)


class IFCA_Model(object):
    """ """

    def __init__(
        self,
        dataset,
        loss_fn,
        grad_fn,
        K,
        eta,
        n_local_gd_steps,
        n_total_devices,
        beta,
        rng,
        strategy="gradient",
    ) -> None:
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.K = K
        self.eta = eta

        self.models = None

    def _init_models(self, models):
        if self.models is None:
            self.models = models[:: self.K]

    def _return_models(self, assignments):
        # stich together
        _models = []
        _per_device = len(assignments) // self.K
        for a in assignments:
            _models.append(self.models[a])
        return _models

    def __call__(self, models, epoch):
        self._init_models(models)

        s = []
        _models = []
        assignments = []
        for client_data in self.dataset.data:
            losses = []
            for model in self.models:
                loss = self.loss_fn(client_data["x"], client_data["y"], model)
                losses.append(loss.mean())
            j_hat = np.argmin(losses)
            si = [int(j == j_hat) for j in range(self.K)]
            s.append(si)

            grad = self.grad_fn(client_data["x"], client_data["y"], self.models[j_hat])
            _models.append(self.models[j_hat] - self.eta * grad)
            assignments.append(j_hat)

        # Update model
        for j in range(self.K):
            if sum(s[i][j] for i, m in enumerate(_models)) > 0:
                self.models[j] = sum(s[i][j] * m for i, m in enumerate(_models)) / sum(
                    s[i][j] for i, m in enumerate(_models)
                )
        return self._return_models(assignments)


class ThresholdingV3(object):
    def __init__(self, K, num_thresholding_steps, num_sampled_devices, rng):
        self.num_thresholding_steps = num_thresholding_steps
        self.num_sampled_devices = num_sampled_devices
        self.rng = rng

        self.simple_clustering_oracle = KMeans(K=K, iterations=10)
        self._thresholding_centers = None

    def __call__(self, center0, momentums):
        total_devices = len(momentums)

        center = center0.copy()
        for _ in range(self.num_thresholding_steps):
            if self.num_sampled_devices == "all":
                indices = np.arange(total_devices)
            else:
                indices = self.rng.choice(
                    np.arange(total_devices),
                    size=self.num_sampled_devices,
                    replace=False,
                )
            sampled_momentum = [momentums[i] for i in indices]
            tau = self.get_clipping_radius(center, sampled_momentum)

            ys = []
            for i in indices:
                m = momentums[i]
                y = m if torch.linalg.norm(m - center) < tau else center
                ys.append(y)
            center = sum(ys) / len(ys)

        return center

    def get_clipping_radius(self, v, momentums):
        distances = [torch.linalg.norm(v - m) for m in momentums]
        a, c = self.simple_clustering_oracle(
            np.expand_dims(np.array(distances), axis=1)
        )
        c = sorted(set(c.flatten()))
        while len(c) < 2:
            a, c = self.simple_clustering_oracle(
                np.expand_dims(np.array(distances), axis=1)
            )
            c = sorted(set(c.flatten()))
        return (c[0] + c[1]) // 2


class PersonalizedThresholdingProcedureSoftOptimizer(object):
    """ """

    def __init__(
        self,
        dataset,
        loss_fn,
        grad_fn,
        K,
        eta,
        n_local_gd_steps,
        n_total_devices,
        beta,
        rng,
        thresholding_params,
    ) -> None:
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.eta = eta
        self.n_local_gd_steps = n_local_gd_steps
        self.thresholding_params = thresholding_params

        self.clustering_oracle = ThresholdingV3(K=K, rng=rng, **thresholding_params)
        self.momentums = [MomentumBuffer(beta) for _ in range(n_total_devices)]

    def _get_update(self, model0, momentum, grad_fn, client_data):
        model = model0.copy()
        # Initialize with assigned cluster center
        for _ in range(self.n_local_gd_steps):
            grad = grad_fn(client_data["x"], client_data["y"], model)
            momentum.update(grad)
            model -= self.eta * momentum.buff
        return (model0 - model) / self.eta

    def __call__(self, models, epoch):
        # Step 1: compute update based on local dataset and model
        updates = []
        for momentum, model, client_data in zip(
            self.momentums, models, self.dataset.data
        ):
            update = self._get_update(model, momentum, self.grad_fn, client_data)
            updates.append(update)
        updates = np.array(updates)

        # Step 2: cluster the updates and return the update-centers (of momentum)
        for i in range(len(self.momentums)):
            center = self.clustering_oracle(updates[i], updates)
            models[i] -= self.eta * center

        return models

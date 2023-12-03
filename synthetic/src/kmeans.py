import itertools
import numpy as np


def _kmeans_pp_initialization(vector, K):
    n_devices = len(vector)
    init_ids = np.random.choice(n_devices, 1, replace=False).tolist()
    rest_ids = set(list(range(n_devices))) - set(init_ids)
    for _ in range(K - 1):
        v = vector[init_ids[-1]]
        distances = []
        for i in list(rest_ids):
            distances.append(((v - vector[i]) ** 2).sum())
        j = list(rest_ids)[np.array(distances).argmax()]
        init_ids.append(j)
        rest_ids = rest_ids - set([j])

    init_ids = np.array(init_ids)
    return init_ids


def _kmeans_pp_initialization_improved(vector, K):
    # Pick the next initial center by distances to ALL previous centers.
    n_devices = len(vector)
    pairwise_distance = np.zeros((n_devices, n_devices))
    for i in range(n_devices):
        for j in range(i + 1, n_devices):
            dist = ((vector[j] - vector[i]) ** 2).sum()
            pairwise_distance[i][j] = dist
            pairwise_distance[j][i] = dist

    init_ids = np.random.choice(n_devices, 1, replace=False).tolist()
    rest_ids = set(list(range(n_devices))) - set(init_ids)
    for _ in range(K - 1):
        _ds = []
        for i in list(rest_ids):
            # Pick the next center while considering all previous centers
            _d = sum(pairwise_distance[i][j] for j in init_ids)
            _ds.append(_d)

        j = list(rest_ids)[np.array(_ds).argmax()]
        init_ids.append(j)
        rest_ids = rest_ids - set([j])

    init_ids = np.array(init_ids)
    return init_ids


class KMeans(object):
    def __init__(
        self, K: int, iterations: int, init_fn=_kmeans_pp_initialization_improved
    ) -> None:
        self.K = K
        self.iterations = iterations
        self.init_fn = init_fn

    def __call__(self, vector: np.array):
        init_ids = self.init_fn(vector, self.K)

        # Centers shape: (K, d)
        centers = vector[np.array(init_ids)].clone()

        for t in range(self.iterations):
            # Assignment (cluster_id, client_id)
            assignments = []
            for client_id, g in enumerate(vector):
                distances = ((centers - g) ** 2).sum(axis=1)
                cluster_id = distances.argmin()
                assignments.append((cluster_id, client_id))

            # Update centers
            for k, g in itertools.groupby(assignments, key=lambda x: x[0]):
                indices = list(map(lambda x: x[1], g))
                centers[k] = vector[indices].mean(axis=0)

        return list(map(lambda x: x[0], assignments)), centers


class KMeansOpt(object):
    """An optimizer with clustering defined for KMeans."""

    def __init__(self, K, eta, n_local_gd_steps, n_clustering_steps):
        self.eta = eta
        self.n_local_gd_steps = n_local_gd_steps
        self.clustering_oracle = KMeans(K=K, iterations=n_clustering_steps)

    def __call__(self, dataset, loss_fn, grad_fn, centers, assignments, epoch):
        models = []
        for assignment, client_data in zip(assignments, dataset.data):
            # Initialize with assigned cluster center
            model = centers[assignment].copy()
            for _ in range(self.n_local_gd_steps):
                grad = grad_fn(client_data["x"], client_data["y"], model)
                model -= self.eta * grad
            models.append(model)

        # Perform the clustering algorithm
        assignments, centers = self.clustering_oracle(np.array(models))
        return assignments, centers

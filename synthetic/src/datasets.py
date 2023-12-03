from turtle import st
import numpy as np
import torch


class SyntheticDataset(object):
    """Synthetic linear regression dataset for explorative purpose.

    Each cluster's center `theta_star` is independently drawn from {0, 1}^d.
    Devices within each cluster can either have identical dataset or just same distribution.
    The device dataset is generated through
        1) randomly draw x~N(mu=cluster_id; std=1); and
        2) compute y = x @ theta_star + noise.

    d: Dimension of features
    R: Scale the distances between true optimum by R
    K: Number of true optimum
    num_device_per_cluster: Number of devices per cluster
    n: Number of samples per device
    sigma2: Noise in data when generated from linear regression
    identical_device_within_cluster:
    """

    def __init__(
        self,
        d,
        R,
        K,
        num_device_per_cluster,
        n,
        sigma2,
        identical_device_within_cluster,
        name,
        rng,
        device,
    ):
        self.d = d
        self.R = R
        self.K = K
        self.num_device_per_cluster = num_device_per_cluster
        self.n = n
        self.sigma2 = sigma2
        self.identical_device_within_cluster = identical_device_within_cluster
        self.name = name
        self.rng = rng
        self.device = device

        self.m = num_device_per_cluster * K
        self.data = self.generate_dataset()

    def generate_dataset(self):
        data = []
        for k in range(self.K):
            # Generate cluster ground truth model
            theta_star = self.rng.integers(low=0, high=2, size=(self.d)) * self.R

            if self.identical_device_within_cluster:
                data += self._generate_cluster_with_identical_devices(k, theta_star)
            else:
                data += self._generate_cluster_with_nonidentical_devices(k, theta_star)

        return data

    def _format(self, theta_star, x, y):
        return {
            "theta_star": torch.Tensor(theta_star).to(self.device),
            "x": torch.Tensor(x).to(self.device),
            "y": torch.Tensor(y).to(self.device),
        }

    def _generate_cluster_with_identical_devices(self, k, theta_star):
        cluster_dataset = []
        # Generate feature matrix
        x = self.rng.normal(loc=k, scale=1, size=(self.n, self.d))

        # Generate noise
        epsilon = self.rng.normal(loc=0, scale=np.sqrt(self.sigma2), size=(self.n,))

        # Generate y: (n, K)
        y = x @ theta_star + epsilon
        for _ in range(self.num_device_per_cluster):
            cluster_dataset.append(self._format(theta_star, x, y))
        return cluster_dataset

    def _generate_cluster_with_nonidentical_devices(self, k, theta_star):
        cluster_dataset = []
        for _ in range(self.num_device_per_cluster):
            # Generate x
            x = self.rng.normal(loc=k, scale=1, size=(self.n, self.d))

            # Generate noise
            epsilon = self.rng.normal(loc=0, scale=np.sqrt(self.sigma2), size=(self.n,))

            # Generate y: (n, K)
            y = x @ theta_star + epsilon
            cluster_dataset.append(self._format(theta_star, x, y))
        return cluster_dataset

    def print_cluster_center_distances(self):
        cluster_centers = list(
            map(lambda x: x["theta_star"], self.data[:: self.num_device_per_cluster])
        )
        cluster_distances = np.array(
            [
                [
                    torch.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    for i in range(self.K)
                ]
                for j in range(self.K)
            ]
        )

        print("Cluster center distances")
        with np.printoptions(precision=3, suppress=True):
            print(cluster_distances)
        print()

    def initial_centers(self, strategy, noise=1):
        if strategy == "oracle":
            return self.oracle_centers(noise)
        elif strategy == "random":
            return self.random_centers(noise)
        raise NotImplementedError(strategy)

    def initial_assignment(self, strategy):
        if strategy == "oracle":
            res = []
            for k in range(self.K):
                res += [k] * self.num_device_per_cluster
            res = np.array(res)
        elif strategy == "random":
            res = self.rng.integers(low=0, high=self.K, size=(self.m,))
        else:
            raise NotImplementedError(strategy)

        print("Initial assignments")
        print(res)
        print()
        return res

    def oracle_centers(self, noise=1):
        cluster_centers = [
            self.data[k * self.num_device_per_cluster]["theta_star"]
            for k in range(self.K)
        ]
        cluster_centers = torch.stack(cluster_centers, axis=0)
        extra = self.rng.normal(loc=0, scale=noise, size=(self.K, self.d))
        return cluster_centers + torch.Tensor(extra).to(self.device)

    def random_centers(self, noise=1):
        # In each coordinate, we generate random values between [-1, 2]. (in contrast, the actual center takes value in {0, 1})
        out = self.rng.random(size=(self.K, self.d)) * 3 - 1
        return torch.Tensor(out).to(self.device)


def squared_loss(x, y, w):
    return torch.norm(x @ w - y) ** 2 / (2 * len(y))


def squared_loss_grad(x, y, w):
    return x.T @ (x @ w - y) / len(y)


# ---------------------------------------------------------------------------- #
#                               Example datasets                               #
# ---------------------------------------------------------------------------- #


class D1(SyntheticDataset):
    """
    This is suppose to be an easy dataset which all clustering algorithms perform well.

    Features
    - same dataset for devices in the same cluster
    - the dataset is generated without noise
    - overdetermined d << n
    """

    def __init__(self, device, rng):
        super().__init__(
            d=10,
            R=100,
            K=4,
            n=100,
            num_device_per_cluster=4,
            sigma2=0,
            identical_device_within_cluster=True,
            name="Dataset 1",
            rng=rng,
            device=device,
        )

    def __str__(self):
        cluster_distribution = (
            "identical dataset"
            if self.identical_device_within_cluster
            else "dataset same distribution"
        )
        return f"{self.K} clusters where each has {cluster_distribution}. \\ Over-parameterized (d={self.d}<=n={self.n})"


# Customized dataset D2
class D2(SyntheticDataset):
    """
    non-identical and non-overparameterized setting

    Features
    - different dataset for devices in the same cluster
    - the dataset is generated without noise
    - underdetermined d << n
    """

    def __init__(self, device, rng):
        super().__init__(
            d=10,
            R=100,
            K=4,
            n=100,
            num_device_per_cluster=4,
            sigma2=0,
            identical_device_within_cluster=False,
            name="Dataset 2",
            rng=rng,
            device=device,
        )


# Customized dataset D3


class D3(SyntheticDataset):
    """
    Overparameterized setting

    Features
    - different dataset for devices in the same cluster
    - the dataset is generated without noise
    - underdetermined d > n
    """

    def __init__(self, device, rng):
        super().__init__(
            d=10,
            R=100,
            K=4,
            n=9,
            num_device_per_cluster=4,
            sigma2=0,
            identical_device_within_cluster=False,
            name="Dataset 3",
            rng=rng,
            device=device,
        )


# Customized dataset D3
class DX(SyntheticDataset):
    """
    Overparameterized setting

    Features
    - different dataset for devices in the same cluster
    - the dataset is generated without noise
    - underdetermined d > n
    """

    def __init__(self, K, num_device_per_cluster, n, d, device, rng):
        super().__init__(
            d=d,
            R=100,
            K=K,
            n=n,
            num_device_per_cluster=num_device_per_cluster,
            sigma2=0,
            identical_device_within_cluster=False,
            name="Dataset X",
            rng=rng,
            device=device,
        )

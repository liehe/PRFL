"""
Aggregators which takes in weights and gradients.
"""
import torch
import logging

from src.utils import unstack_vectorized_model


class _BaseAggregator(object):
    """Base class of aggregators.

    Args:
        dist_communicator (object): A link object which can broadcast / gather, etc.
    """

    def __call__(self, inputs):
        """Aggregate the inputs and update in-place.

        Args:
            inputs (list): A list of tensors to be aggregated.

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError


class _BaseDecentralizedAggregator(object):
    """Base class of aggregators.

    Args:
        dist_communicator (object): A link object which can broadcast / gather, etc.
    """

    def __call__(self, local_inputs, neighbor_inputs):
        """Aggregate the inputs and update in-place.

        Args:
            inputs (list): A list of tensors to be aggregated.

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError


class Mean(_BaseAggregator):
    def __call__(self, inputs):
        values = torch.stack(inputs, dim=0).mean(dim=0)
        return values

    def __str__(self):
        return "Mean"


class DeMean(_BaseDecentralizedAggregator):
    def __call__(self, local_inputs, neighbor_inputs):
        inputs = [local_inputs] + neighbor_inputs
        values = torch.stack(inputs, dim=0).mean(dim=0)
        return values

    def __str__(self):
        return "DeMean"


class DeLocal(_BaseDecentralizedAggregator):
    def __call__(self, local_inputs, neighbor_inputs):
        return local_inputs

    def __str__(self):
        return "DeMean"


class DeGlobal(DeMean):
    def __init__(self, *args, **kwargs):
        self.counter = 0

    def __call__(self, local_inputs, neighbor_inputs):
        self.counter += 1
        if self.counter % 10 == 0:
            return super().__call__(local_inputs, neighbor_inputs)
        return local_inputs

    def __str__(self):
        return "DeMean"


class CM(_BaseAggregator):
    def __call__(self, inputs):
        stacked = torch.stack(inputs, dim=0)
        values_upper, _ = stacked.median(dim=0)
        values_lower, _ = (-stacked).median(dim=0)
        return (values_upper - values_lower) / 2

    def __str__(self):
        return "Coordinate-wise median"


class DeCM(_BaseAggregator):
    def __call__(self, local_inputs, neighbor_inputs):
        inputs = [local_inputs] + neighbor_inputs

        stacked = torch.stack(inputs, dim=0)
        values_upper, _ = stacked.median(dim=0)
        values_lower, _ = (-stacked).median(dim=0)
        return (values_upper - values_lower) / 2

    def __str__(self):
        return "DeCM"


class TM(_BaseAggregator):
    def __init__(self, b):
        self.b = b
        super(TM, self).__init__()

    def __call__(self, inputs):
        if len(inputs) - 2 * self.b > 0:
            b = self.b
        else:
            b = self.b
            while len(inputs) - 2 * b <= 0:
                b -= 1
            if b < 0:
                raise RuntimeError

        stacked = torch.stack(inputs, dim=0)
        largest, _ = torch.topk(stacked, b, 0)
        neg_smallest, _ = torch.topk(-stacked, b, 0)
        new_stacked = torch.cat([stacked, -largest, neg_smallest]).sum(0)
        new_stacked /= len(inputs) - 2 * b
        return new_stacked

    def __str__(self):
        return "Trimmed Mean (b={})".format(self.b)


class DeTM(TM):
    def __call__(self, local_inputs, neighbor_inputs):
        inputs = [local_inputs] + neighbor_inputs
        return super(DeTM, self).__call__(inputs)


class DecentralizedAggregator(_BaseAggregator):
    """
    This aggregator is applied to all nodes. It has access to the node information and a row of mixing matrix.
    """

    def __init__(self, node, weights):
        super().__init__()
        assert len(weights.shape) == 1
        self.node = node
        self.weights = weights

    def __call__(self, local_inputs, neighbor_inputs):
        """
        The `inputs` is a list of tensors. The first element is the weight of itself, the second to the last elements are the gradient of its neighbors.
        """
        assert len(neighbor_inputs) == len(self.node.edges)
        s = self.weights[self.node.index] * local_inputs
        for e, inp in zip(self.node.edges, neighbor_inputs):
            theothernode = e.theother(self.node)
            s += self.weights[theothernode.index] * inp
        return s

    def __str__(self):
        return "DecentralizedAggregator"


# ---------------------------------------------------------------------------- #
#                                      RFA                                     #
# ---------------------------------------------------------------------------- #
def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def smoothed_weiszfeld(weights, alphas, z, nu, T):
    m = len(weights)
    if len(alphas) != m:
        raise ValueError

    if nu < 0:
        raise ValueError

    for t in range(T):
        betas = []
        for k in range(m):
            distance = _compute_euclidean_distance(z, weights[k])
            betas.append(alphas[k] / max(distance, nu))

        z = 0
        for w, beta in zip(weights, betas):
            z += w * beta
        z /= sum(betas)
    return z


class RFA(_BaseAggregator):
    def __init__(self, T, nu=0.1):
        self.T = T
        self.nu = nu
        super(RFA, self).__init__()

    def __call__(self, inputs):
        alphas = [1 / len(inputs) for _ in inputs]
        z = torch.zeros_like(inputs[0])
        return smoothed_weiszfeld(inputs, alphas, z=z, nu=self.nu, T=self.T)

    def __str__(self):
        return "RFA(T={},nu={})".format(self.T, self.nu)


class DeRFA(RFA):
    def __call__(self, local_inputs, neighbor_inputs):
        inputs = [local_inputs] + neighbor_inputs
        return super(DeRFA, self).__call__(inputs)


# ---------------------------------------------------------------------------- #
#                                     KRUM                                     #
# ---------------------------------------------------------------------------- #
def _compute_scores(distances, i, n, f):
    """Compute scores for node i.

    Arguments:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist. i, j starts with 0.
        i {int} -- index of worker, starting from 0.
        n {int} -- total number of workers
        f {int} -- Total number of Byzantine workers.

    Returns:
        float -- krum distance score of i.
    """
    s = [distances[j][i] ** 2 for j in range(i)] + [
        distances[i][j] ** 2 for j in range(i + 1, n)
    ]
    _s = sorted(s)[: n - f - 2]
    return sum(_s)


def multi_krum(distances, n, f, m):
    """Multi_Krum algorithm

    Arguments:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist. i, j starts with 0.
        n {int} -- Total number of workers.
        f {int} -- Total number of Byzantine workers.
        m {int} -- Number of workers for aggregation.

    Returns:
        list -- A list indices of worker indices for aggregation. length <= m
    """
    if n < 1:
        raise ValueError(
            "Number of workers should be positive integer. Got {}.".format(f)
        )

    if m < 1 or m > n:
        raise ValueError(
            "Number of workers for aggregation should be >=1 and <= {}. Got {}.".format(
                m, n
            )
        )

    if 2 * f + 2 > n:
        raise ValueError("Too many Byzantine workers: 2 * {} + 2 >= {}.".format(f, n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            if distances[i][j] < 0:
                raise ValueError(
                    "The distance between node {} and {} should be non-negative: Got {}.".format(
                        i, j, distances[i][j]
                    )
                )

    scores = [(i, _compute_scores(distances, i, n, f)) for i in range(n)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    return list(map(lambda x: x[0], sorted_scores))[:m]


def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def pairwise_euclidean_distances(vectors):
    """Compute the pairwise euclidean distance.

    Arguments:
        vectors {list} -- A list of vectors.

    Returns:
        dict -- A dict of dict of distances {i:{j:distance}}
    """
    n = len(vectors)
    vectors = [v.flatten() for v in vectors]

    distances = {}
    for i in range(n - 1):
        distances[i] = {}
        for j in range(i + 1, n):
            distances[i][j] = _compute_euclidean_distance(vectors[i], vectors[j]) ** 2
    return distances


class Krum(_BaseAggregator):
    r"""
    This script implements Multi-KRUM algorithm.

    Blanchard, Peva, Rachid Guerraoui, and Julien Stainer.
    "Machine learning with adversaries: Byzantine tolerant gradient descent."
    Advances in Neural Information Processing Systems. 2017.
    """

    def __init__(self, f, m):
        self.f = f
        self.m = m
        self.top_m_indices = None
        super(Krum, self).__init__()

    def __call__(self, inputs):
        n = len(inputs)
        distances = pairwise_euclidean_distances(inputs)
        top_m_indices = multi_krum(distances, n, self.f, self.m)
        values = sum(inputs[i] for i in top_m_indices)
        self.top_m_indices = top_m_indices
        return values

    def __str__(self):
        return "Krum (m={})".format(self.m)


class DeKrum(Krum):
    def __call__(self, local_inputs, neighbor_inputs):
        inputs = [local_inputs] + neighbor_inputs
        return super().__call__(inputs)


# ---------------------------------------------------------------------------- #
#                                   Clipping                                   #
# ---------------------------------------------------------------------------- #


def clip(v, tau):
    v_norm = torch.norm(v)
    scale = min(1, tau / v_norm)
    if torch.isnan(v_norm):
        return 0
    return v * scale


class Clipping(_BaseAggregator):
    def __init__(self, tau, n_iter=1):
        self.tau = tau
        self.n_iter = n_iter
        super(Clipping, self).__init__()
        self.momentum = None

    def __call__(self, inputs):
        if self.momentum is None:
            self.momentum = inputs[0]

        for _ in range(self.n_iter):
            self.momentum = (
                sum(clip(v - self.momentum, self.tau) for v in inputs) / len(inputs)
                + self.momentum
            )
        return torch.clone(self.momentum).detach()

    def __str__(self):
        return "Clipping (tau={}, n_iter={})".format(self.tau, self.n_iter)


class DeClipping(Clipping):
    def __call__(self, local_inputs, neighbor_inputs):
        inputs = [local_inputs] + neighbor_inputs
        return super().__call__(inputs)


# ---------------------------------------------------------------------------- #
#                    Proposed Method: Self Centered Clipping                   #
# ---------------------------------------------------------------------------- #


def bucketing(inputs):
    import numpy as np

    s = 2
    indices = list(range(len(inputs)))
    np.random.shuffle(indices)
    T = int(np.ceil(len(inputs) / s))

    reshuffled_inputs = []
    for t in range(T):
        indices_slice = indices[t * s : (t + 1) * s]
        g_bar = sum(inputs[i] for i in indices_slice) / len(indices_slice)
        reshuffled_inputs.append(g_bar)
    return reshuffled_inputs


class SelfCenteredClipping(DecentralizedAggregator):
    def __init__(self, node, weights, tau, worker, delta_max):
        self.tau = tau
        self._tau = None
        self.worker = worker
        self.delta_max = delta_max
        super().__init__(node, weights)
        self.logger = logging.getLogger("debug")

    def _agg(self, local_inputs, neighbor_inputs, tau):
        zs = [
            local_inputs + clip(neighbor - local_inputs, tau)
            for neighbor in neighbor_inputs
        ]
        return super().__call__(local_inputs, zs)

    def __call__(self, local_inputs, neighbor_inputs):
        if self.tau is not None:
            return self._agg(local_inputs, neighbor_inputs, self.tau)

        weight_distances = [
            (
                self.weights[list(self.node.edges)[i].theother(self.node).index],
                (n - local_inputs).norm() ** 2,
            )
            for i, n in enumerate(neighbor_inputs)
        ]
        weight_distances = sorted(weight_distances, key=lambda x: -x[1])
        weight_count = 0
        output = 0
        w_output = 0
        for i, (w, v) in enumerate(weight_distances):
            if weight_count + w < self.delta_max:
                weight_count += w
            elif weight_count < self.delta_max:
                output += (w + weight_count - self.delta_max) * v
                w_output += w + weight_count - self.delta_max
                weight_count += w
            else:
                output += w * v
                w_output += w

        tau = (output / w_output) ** 0.5
        return self._agg(local_inputs, neighbor_inputs, tau)


    def __str__(self):
        return "SelfCenteredClipping (tau={})".format(self.tau)


class DeClipping(Clipping):
    def __call__(self, local_inputs, neighbor_inputs):
        inputs = [local_inputs] + neighbor_inputs
        return super().__call__(inputs)


def get_aggregator(args, graph, rank, worker):
    if args.agg in ["avg", "mean"]:
        return Mean() if args.graph is None else DeMean()

    if args.agg in ["local"]:
        return DeLocal()

    if args.agg in ["global"]:
        return DeGlobal()

    if args.agg in ["cm"]:
        return CM() if args.graph is None else DeCM()

    if args.agg.startswith("tm"):
        b = int(args.agg[2:])
        return TM(b) if args.graph is None else DeTM(b)

    if args.agg.startswith("rfa"):
        T = int(args.agg[3:])
        return RFA(T) if args.graph is None else DeRFA(T)

    if args.agg.startswith("krum"):
        m = int(args.agg[4:])
        return Krum(args.f, m) if args.graph is None else DeKrum(args.f, m)

    if args.agg.startswith("cp"):
        # Note that we should not scale the tau in the decentralized case
        tau = float(args.agg[2:])
        return Clipping(tau) if args.graph is None else DeClipping(tau)

    if args.agg.startswith("mozi"):
        rho, alpha = args.agg[4:].split(",")
        return Mozi(float(rho), float(alpha), lr=args.lr, worker=worker)

    if args.agg == "gossip_avg":
        node = graph.nodes[rank]
        weights = graph.metropolis_weight[rank, :]
        return DecentralizedAggregator(node, weights)

    if args.agg.startswith("scp"):
        node = graph.nodes[rank]
        weights = graph.metropolis_weight[rank, :]

        delta_max = -1
        for i in range(args.n - args.f, args.n):
            delta = 1 - graph.metropolis_weight[i][i]
            delta_max = max(delta_max, delta)

        # Note that we should not scale the tau in the decentralized case
        if args.agg == "scp":
            tau = None
        else:
            tau = float(args.agg[3:])
        return SelfCenteredClipping(node, weights, tau, worker, delta_max)

    raise NotImplementedError(f"{args.agg}")

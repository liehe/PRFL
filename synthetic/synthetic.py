import argparse
import numpy as np
import os
import torch
import functools

from src.datasets import D1, D2, D3, DX, squared_loss, squared_loss_grad
from src.dummy import (
    DummyPersonalizedOptimizer,
    DummyCentralizedOptimizer,
    GroundTruthOptimizer,
    DummyCentralizedOptimizer,
)
from src.thresholding_procedure import (
    PersonalizedThresholdingProcedureOptimizer,
    EfficientPersonalizedThresholdingProcedureOptimizer,
)
from src.thresholding_procedure import IFCA, IFCA_Model
from src.controlflow import ClusterAlgorithmEvaluator, PersonalizationAlgorithmEvaluator
from src.utils import setup_logs

# Customized dataset D1
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
DATA_DIR = ROOT_DIR + "datasets/"

EXP_PATTERN = (
    "eta{eta}_beta{beta}_local{num_local_gd_steps}_center_{initial_cluster_centers}"
    + "_{solver}_{dataset}"
)
LOG_DIR_PATTERN = ROOT_DIR + "outputs/{script}/{exp_id}/" + EXP_PATTERN + "/"


def define_parser():
    parser = argparse.ArgumentParser()

    # Running environment related arguments
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--E", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.001, help="step size")
    parser.add_argument("--beta", type=float, default=0.9, help="momentum")
    parser.add_argument("--num_local_gd_steps", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--identifier", type=str, default="debug", help="")
    parser.add_argument("--initial-cluster-centers", choices=["oracle", "random"])
    parser.add_argument("--initial-assignment", choices=["oracle", "random"])
    parser.add_argument("--solver", type=str, default=None, help="")
    parser.add_argument("--dataset", type=str, default=None, help="")
    parser.add_argument("--analyze", action="store_true", default=False)

    # parser.add_argument("--thresholding-", action="store_true", default=False)

    return parser


def parse_solver(solver: str):
    if solver == "DummyPersonalized":
        return DummyPersonalizedOptimizer, PersonalizationAlgorithmEvaluator
    if solver == "Centralized":
        return DummyCentralizedOptimizer, PersonalizationAlgorithmEvaluator
    if solver == "IFCA_Model":
        return IFCA_Model, PersonalizationAlgorithmEvaluator
    if solver == "IFCA":
        return IFCA, PersonalizationAlgorithmEvaluator
    if solver == "GT":
        return GroundTruthOptimizer, PersonalizationAlgorithmEvaluator
    if solver.startswith("Thresholding"):
        _, strategy = solver.split("=")
        return (
            functools.partial(
                PersonalizedThresholdingProcedureOptimizer,
                thresholding_params=dict(
                    num_thresholding_steps=10,
                    num_sampled_devices="all",
                    strategy=strategy,
                ),
            ),
            PersonalizationAlgorithmEvaluator,
        )
    if solver.startswith("EfficientThresholding"):
        _, strategy = solver.split("=")
        strategy, frequency, subgroup_size = strategy.split(",")
        return (
            functools.partial(
                EfficientPersonalizedThresholdingProcedureOptimizer,
                frequency=int(frequency),
                subgroup_size=int(subgroup_size),
                thresholding_params=dict(
                    num_thresholding_steps=10,
                    num_sampled_devices="all",
                    strategy=strategy,
                ),
            ),
            PersonalizationAlgorithmEvaluator,
        )

    raise NotImplementedError(solver)


def parse_dataset(dataset: str, device, rng):
    print(dataset)
    if dataset == "D1":
        return D1(device, rng)
    if dataset == "D2":
        return D2(device, rng)
    if dataset == "D3":
        return D3(device, rng)
    if dataset.startswith("DX"):
        K, num_device_per_cluster, n, d = dataset[2:].split(",")
        K, num_device_per_cluster, n, d = (
            int(K),
            int(num_device_per_cluster),
            int(n),
            int(d),
        )
        return DX(K, num_device_per_cluster, n, d, device, rng)

    raise NotImplementedError


def run_experiment(args):
    device = torch.device("cuda" if args.use_cuda else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    log_dir = setup_logs(
        args,
        LOG_DIR_PATTERN=LOG_DIR_PATTERN,
    )

    dataset = parse_dataset(args.dataset, device, rng)
    dataset.print_cluster_center_distances()

    initial_cluster_centers = dataset.initial_centers(
        args.initial_cluster_centers, noise=0.1
    )
    initial_assignment = dataset.initial_assignment(args.initial_assignment)

    cluster_fn_generator, evaluator_generator = parse_solver(args.solver)
    cluster_fn = cluster_fn_generator(
        K=dataset.K,
        eta=args.eta,
        n_local_gd_steps=args.num_local_gd_steps,
        n_total_devices=dataset.m,
        beta=args.beta,
        loss_fn=squared_loss,
        grad_fn=squared_loss_grad,
        dataset=dataset,
        rng=rng,
    )

    evaluator = evaluator_generator(
        dataset=dataset,
        cluster_fn=cluster_fn,
        K=dataset.K,
        E=args.E,
        log_interval=args.log_interval,
    )

    evaluator.run(
        init_assignments=initial_assignment, init_centers=initial_cluster_centers
    )


if __name__ == "__main__":
    parser = define_parser()
    args = parser.parse_args()
    run_experiment(args)

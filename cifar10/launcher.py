import os
import torch

from utils import *
from implementations import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
DATA_DIR = ROOT_DIR + "datasets/"

EXP_SPEC = "{algorithm}_data={data}_n={n}_K={K}_Kgen={K_gen}_lr={lr:.3e}_m={momentum:.2f}_v={validation_batch_size}_noniid={noniid}_ms={model_size}_ss={subsample_ratio}"
LOG_DIR_PATTERN = ROOT_DIR + "outputs/{script}/{exp_id}/" + EXP_SPEC + "/"

DEFAULT_ARG = """--lr 0.1 --debug -n 8 --K 4 --epochs 10 --momentum 0.9 \
--batch-size 32 --max-batch-size-per-epoch 50 --noniid 0 --data rotation --algorithm local"""


# ---------------------------------------------------------------------------- #
#                               The main entrance                              #
# ---------------------------------------------------------------------------- #


def initialize_experiment():
    args = get_system_args_or_default(DEFAULT_ARG)
    setup_seeds(args.use_cuda, args.seed)
    log_dir = setup_logs(args, fn_spec=LOG_DIR_PATTERN)

    device = torch.device("cuda" if args.use_cuda else "cpu")

    if args.algorithm.startswith('ifca'):
        run_ifca(args, device, DATA_DIR)
    elif args.algorithm.startswith('global'):
        run_global(args, device, DATA_DIR)
    elif args.algorithm.startswith('groundtruth'):
        run_groundtruth(args, device, DATA_DIR)
    elif args.algorithm.startswith('local'):
        run_local(args, device, DATA_DIR)
    elif args.algorithm.startswith('fc'):
        run_fc(args, device, DATA_DIR)
    elif args.algorithm.startswith('ditto'):
        run_ditto(args, device, DATA_DIR)
    elif args.algorithm.startswith('knnper'):
        run_knnper(args, device, log_dir, DATA_DIR)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    initialize_experiment()

from .utils import *


class DittoTrain(Train):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.algorithm.startswith("ditto-"):
            lambda_ = self.args.algorithm[len("ditto-") :]
            self.lambda_ = float(lambda_)
        else:
            raise NotImplementedError

    def epoch_start(self):
        self.parallel_call(lambda w: w.train_epoch_start())

    def train_batch(self, meter, batch_idx, epoch):
        results = self.parallel_get(lambda w: w.compute_gradient())
        meter.add(results)

        # NOTE: Store penalization term \lambda (v_k - w^t)
        # where v_k is the local model on k and w^t is the global model.
        self.parallel_call(
            lambda w: w.compute_ditto_penalization(self.lambda_, self.server.opt)
        )

        # Server aggregate model update
        avg_grad = sum(w.get_gradient() for w in self.workers) / len(self.workers)
        self.server.set_gradient(avg_grad)
        self.server.apply_gradient()

        # Local model update including ditto penalization
        self.parallel_call(lambda w: w.apply_ditto_gradient())


def run_ditto(args, device, data_dir):
    """
    Setup:
    - There is a server which hosts global model.
    - Each worker has
    - They have another hyperparameter lambda for penalization.

    Initialization:
    - Both global model and local models are identical.
    """
    model = SimpleLinear(args.model_size)

    # ------------------------------- Define server ------------------------------ #
    server = SingleModelServer(
        model_fn=lambda: copy.deepcopy(model),
        opt_fn=lambda m: torch.optim.SGD(m.parameters(), lr=args.lr),
        device=device,
    )

    # ------------------------------ Define workers ------------------------------ #
    workers = []
    for rank in range(args.n):
        sampler = functools.partial(
            sampler_fn, args=args, rank=rank, dataset_type="train", shuffle=True
        )

        train_loader = get_data_loader(
            args, sampler, rank, data_dir, dataset_type="train"
        )

        m = copy.deepcopy(model)
        opt = torch.optim.SGD(m.parameters(), lr=args.lr)
        worker = DittoWorker(
            model=m,
            opt=opt,
            index=rank,
            metrics=metrics(),
            momentum=args.momentum,
            data_loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            # IFCA never locally update the model
            lr_scheduler=None,
        )
        workers.append(worker)

    # --------------------------- Add worker to Trainer -------------------------- #
    trainer = DittoTrain(
        args,
        pre_batch_hooks=[],
        post_batch_hooks=[],
        max_batches_per_epoch=args.max_batch_size_per_epoch,
        log_interval=args.log_interval,
        metrics=metrics(),
        use_cuda=args.use_cuda,
        debug=args.debug,
    )

    trainer.add_server_workers(server, workers)

    # ----------------------------- Define evaluator ----------------------------- #
    test_data_loaders = []
    for rank in range(args.n):
        sampler = functools.partial(
            sampler_fn, args=args, rank=rank, dataset_type="test", shuffle=False
        )
        test_data_loader = get_data_loader(
            args, sampler, rank, data_dir, dataset_type="test"
        )
        test_data_loaders.append(test_data_loader)

    evaluator = IndividualEvaluator(
        trainer.workers,
        test_data_loaders,
        loss_fn,
        device,
        metrics(),
        args.use_cuda,
        debug=args.debug,
    )

    # ------------------------------ Start training ------------------------------ #
    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch)
        evaluator.evaluate(epoch)

        if hasattr(trainer.workers[0], "data_loader"):
            trainer.parallel_call(lambda w: w.data_loader.sampler.set_epoch(epoch))

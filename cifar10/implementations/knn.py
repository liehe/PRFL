from .utils import *
from .fedavg import GlobalTrain


def mnist_net_mapping(model):
    """Ret"""
    # assert isinstance(model, SimpleLinear)

    # NOTE: for the moment, let's just use model as the
    def mapping(x):
        return model(x)


class KNNPerSamplerFn(object):
    def __init__(self, args):
        _, val_per, _, _ = args.algorithm.split("-")
        self.val_per = float(val_per)
        self.args = args

    def __call__(self, x, train, rank, shuffle, dataset_type, offset_seed=0):
        phase = self.args.knnper_phase
        if phase == "FedAvgVal":
            val = False
        elif phase == "kNNPerVal" and train:
            val = False
        elif phase == "kNNPerVal" and not train:
            val = True
        elif phase in ["FedAvgTrain", "kNNPerTest"]:
            val = False
            self.val_per = 0
        else:
            raise NotImplementedError(phase, train)

        return KNNPerSampler(
            # Indicate if this is sampler uses validation set or training set.
            val=val,
            # Indicate the percentage of validation set.
            val_per=self.val_per,
            num_replicas=self.args.n,
            rank=rank,
            shuffle=shuffle,
            dataset=x,
            offset_seed=offset_seed,
            subsample_percent=task_subsample_percent(self.args, dataset_type),
        )


def run_knnper(args, device, log_dir, data_dir):
    model = vgg16().to(device)
    loss_fn = CrossEntropyLoss().to(device)

    # ------------------------------- Define server ------------------------------ #
    server = SingleModelServer(
        model_fn=lambda: model,
        opt_fn=lambda m: torch.optim.SGD(m.parameters(), lr=args.lr),
        device=device,
    )

    # ------------------------------ Define workers ------------------------------ #
    knnper_sampler_fn = KNNPerSamplerFn(args)

    workers = []
    for rank in range(args.n):
        sampler = functools.partial(
            knnper_sampler_fn, train=True, rank=rank, dataset_type="train", shuffle=True
        )

        train_loader = get_data_loader(
            args, sampler, rank, data_dir, dataset_type="train"
        )

        worker = KNNPerWorker(
            task_name="mnist",
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

    # --------------------- Add server and worker to Trainer --------------------- #
    trainer = GlobalTrain(
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
            knnper_sampler_fn,
            train=False,
            rank=rank,
            dataset_type="test",
            shuffle=False,
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

    # ----------------------- Load checkpoint and apply knn ---------------------- #
    def evaluate_knnper(lambda_, rank, knnper_evaluator):
        worker = workers[rank]
        model = server.model
        test_data_loader = test_data_loaders[rank]
        # The outputs will
        worker.initialize_datastore()
        worker.fill_database(model)
        predictions, targets = worker.inference(model, lambda_, test_data_loader)

        knnper_evaluator.evaluate_worker(predictions, targets, lambda_, rank)
        return predictions, targets

    if args.knnper_phase == "kNNPerVal":
        lambda_grid = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        knnper_evaluator = KNNPerIndividualEvaluator(
            trainer.workers,
            test_data_loaders,
            loss_fn,
            device,
            metrics(),
            args.use_cuda,
            debug=args.debug,
        )

        lambda_losses = []
        for lambda_ in lambda_grid:
            knnper_evaluator.debug_logger.info("")
            knnper_evaluator.debug_logger.info(f"====> Start {lambda_}")

            server.model.load_state_dict(
                torch.load(os.path.join(log_dir, "FedAvgVal.pt"))
            )

            ps, ts = [], []
            for rank in range(args.n):
                p, t = evaluate_knnper(lambda_, rank, knnper_evaluator)
                ps.append(p)
                ts.append(t)

            ps = torch.cat(ps)
            ts = torch.cat(ts)
            outs = knnper_evaluator.evaluate_all(ps, ts, lambda_)
            loss = outs["Loss"]
            lambda_losses.append(loss)

        # NOTE: Tune the hyper-parameter lambda_m
        print(lambda_grid, lambda_losses)
        index = np.argmin(lambda_losses)
        best_lambda, best_loss = lambda_grid[index], lambda_losses[index]
        print(best_lambda, best_loss)

        # NOTE: Dump the best hyper-parameter lambda_m
        with open(os.path.join(log_dir, "best_lambda.txt"), "w") as f:
            f.write(str(best_lambda))
        return

    if args.knnper_phase == "kNNPerTest":
        with open(os.path.join(log_dir, "best_lambda.txt"), "r") as f:
            best_lambda = float(f.readline().strip())
            print(best_lambda)

        server.model.load_state_dict(
            torch.load(os.path.join(log_dir, "FedAvgTrain.pt"))
        )

        knnper_evaluator = KNNPerIndividualEvaluator(
            trainer.workers,
            test_data_loaders,
            loss_fn,
            device,
            metrics(),
            args.use_cuda,
            debug=args.debug,
        )

        ps, ts = [], []
        for rank in range(args.n):
            p, t = evaluate_knnper(best_lambda, rank, knnper_evaluator)
            ps.append(p)
            ts.append(t)

        ps = torch.cat(ps)
        ts = torch.cat(ts)
        outs = knnper_evaluator.evaluate_all(ps, ts, best_lambda)
        return

    # ------------------------------ Start training ------------------------------ #
    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch)
        evaluator.evaluate(epoch)

        if hasattr(trainer.workers[0], "data_loader"):
            trainer.parallel_call(lambda w: w.data_loader.sampler.set_epoch(epoch))

    # -------------------------------- Checkpoint -------------------------------- #
    if args.knnper_phase == "FedAvgVal":
        torch.save(server.model.state_dict(), os.path.join(log_dir, "FedAvgVal.pt"))
    elif args.knnper_phase == "FedAvgTrain":
        torch.save(server.model.state_dict(), os.path.join(log_dir, "FedAvgTrain.pt"))

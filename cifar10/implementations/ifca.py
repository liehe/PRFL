from .utils import *


class IFCATrain(Train):
    def __init__(self, loss_fn, args, pre_batch_hooks: list, post_batch_hooks: list, max_batches_per_epoch: int, log_interval: int, metrics: dict, use_cuda: bool, debug: bool):
        super().__init__(args, pre_batch_hooks, post_batch_hooks, max_batches_per_epoch, log_interval, metrics, use_cuda, debug)
        self.loss_fn = loss_fn

    def epoch_start(self):
        for m in self.server.models:
            m.train()
        self.parallel_call(lambda w: w.train_epoch_start())

    def train_batch(self, meter, batch_idx, epoch):
        # global loss_fn
        def _find_best_server_model_on_local_validation_data(w):
            data, target = w.get_validation_data_target()
            losses = []
            for m in self.server.models:
                output = m(data)
                loss = self.loss_fn(output, target)
                losses.append(loss.item())
            best = np.argmin(losses)
            w.best_cluster = best
            return best

        def _compute_gradient(w):
            best = _find_best_server_model_on_local_validation_data(w)
            model = self.server.models[best]
            opt = self.server.opts[best]
            w.set_model_opt(model, opt)
            results = w.compute_gradient()
            return results

        if self.args.algorithm == 'ifca-grad':
            # Compute local gradients and log training information.
            results = self.parallel_get(_compute_gradient)
            meter.add(results)

            mapping = self.parallel_get(lambda w: (w.best_cluster, w))
            mapping = sorted(mapping, key=lambda x: x[0])
            for k, g in itertools.groupby(mapping, key=lambda x: x[0]):
                gradient = sum(w.get_gradient() for _, w in g) / len(mapping)

                # NOTE: update the corresponding server model
                self.server.set_gradient(k, gradient)
                self.server.apply_gradient(k)
        else:
            raise NotImplementedError


def run_ifca(args, device, data_dir):
    model = vgg16().to(device)
    loss_fn = CrossEntropyLoss().to(device)

    # ------------------------------- Define server ------------------------------ #
    server = MultiModelServer(
        model_fn=lambda: model,
        opt_fn=lambda m: torch.optim.SGD(m.parameters(), lr=args.lr),
        device=device,
        K=args.K,
        init='non-identical',
        check_models=True)

    # ------------------------------ Define workers ------------------------------ #
    workers = []
    for rank in range(args.n):
        sampler = functools.partial(
            sampler_fn, args=args, rank=rank, dataset_type='train', shuffle=True)

        train_loader = get_data_loader(
            args, sampler, rank, data_dir, dataset_type='train')

        # The offset_seed != 0 makes the sampling sequence different from training
        val_sampler = functools.partial(
            sampler_fn, args=args, rank=rank, shuffle=True, dataset_type='train', offset_seed=10000)

        validation_data_loader = get_data_loader(
            args, val_sampler, rank, data_dir, dataset_type='validation')

        worker = IFCAWorker(
            validation_data_loader=validation_data_loader,
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
    trainer = IFCATrain(
        loss_fn,
        args,
        pre_batch_hooks=[],
        post_batch_hooks=[],
        max_batches_per_epoch=args.max_batch_size_per_epoch,
        log_interval=args.log_interval,
        metrics=metrics(),
        use_cuda=args.use_cuda,
        debug=args.debug)

    trainer.add_server_workers(server, workers)

    # ----------------------------- Define evaluator ----------------------------- #
    test_data_loaders = []
    for rank in range(args.n):
        sampler = functools.partial(
            sampler_fn, args=args, rank=rank, dataset_type='test', shuffle=False)
        test_data_loader = get_data_loader(
            args, sampler, rank, data_dir, dataset_type='test')
        test_data_loaders.append(test_data_loader)

    evaluator = IndividualEvaluator(trainer.workers, test_data_loaders,
                                    loss_fn, device, metrics(), args.use_cuda, debug=args.debug)

    # ------------------------------ Start training ------------------------------ #
    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch)
        evaluator.evaluate(epoch)

        if hasattr(trainer.workers[0], "data_loader"):
            trainer.parallel_call(
                lambda w: w.data_loader.sampler.set_epoch(epoch))
            trainer.parallel_call(
                lambda w: w.validation_data_loader.sampler.set_epoch(epoch))

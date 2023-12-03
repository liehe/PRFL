from .utils import *


class LocalTrain(Train):
    def epoch_start(self):
        self.parallel_call(lambda w: w.train_epoch_start())

    def add_workers(self, workers):
        self.workers = workers

    def train_batch(self, meter, batch_idx, epoch):

        def _compute_gradient(w):
            results = w.compute_gradient()
            w.apply_gradient()
            return results

        results = self.parallel_get(_compute_gradient)
        meter.add(results)


def run_local(args, device, data_dir):
    model = SimpleLinear(args.model_size)

    # ------------------------------ Define workers ------------------------------ #
    workers = []
    for rank in range(args.n):
        sampler = functools.partial(
            sampler_fn, args=args, rank=rank, dataset_type='train', shuffle=True)

        train_loader = get_data_loader(
            args, sampler, rank, data_dir, dataset_type='train')

        m = copy.deepcopy(model)
        opt = torch.optim.SGD(m.parameters(), lr=args.lr)
        worker = LocalWorker(
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
        print(rank, args.n)


    # --------------------- Add server and worker to Trainer --------------------- #
    trainer = LocalTrain(
        args,
        pre_batch_hooks=[],
        post_batch_hooks=[],
        max_batches_per_epoch=args.max_batch_size_per_epoch,
        log_interval=args.log_interval,
        metrics=metrics(),
        use_cuda=args.use_cuda,
        debug=args.debug)

    trainer.add_workers(workers)

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

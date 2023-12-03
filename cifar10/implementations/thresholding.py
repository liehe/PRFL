from .utils import *


class STC(object):
    """Self Threshold Clustering"""

    def __init__(self, num_rounds, strategy):
        self.K = 3
        self.num_rounds = num_rounds
        self.tau_fn = self._get_tau_fn(strategy)

    def _get_tau_fn(self, strategy):
        if strategy == "cd":  # stands for cluster distances
            kmeans = KMeans(n_clusters=self.K)

            def tau_fn(ds):
                # ds = sorted(ds)
                distances = np.expand_dims(ds, axis=1)

                # Note that distances has shape (n, 1) where the feature size is 1.
                kmeans.fit(distances)

                out = sorted(kmeans.cluster_centers_)
                return (out[1] + out[2]) / 2

            return tau_fn

        if strategy.startswith("quantile"):
            quantile = float(strategy[len("quantile") :])

            def tau_fn(ds):
                ds = sorted(ds)
                index = int(len(ds) * quantile)
                tau = ds[index]
                return tau

            return tau_fn

        raise NotImplementedError(strategy)

    def __call__(self, worker_momentums: list, cluster_mean_estimate):
        v = torch.clone(cluster_mean_estimate).detach()
        for t in range(self.num_rounds):
            ds = [torch.linalg.norm(x - v).item() for x in worker_momentums]

            tau = self.tau_fn(ds)

            thresholding = 0
            count = 0
            for i, x in enumerate(worker_momentums):
                if ds[i] > tau:
                    thresholding += v
                else:
                    thresholding += x
                    count += 1
            v = thresholding / len(worker_momentums)
        return v


class FCTrain(Train):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.algorithm.startswith("fc-grad"):
            suffix = self.args.algorithm[len("fc-grad-") :]
            if len(suffix.split("-")) == 2:
                num_rounds, strategy = suffix.split("-")
                num_rounds = int(num_rounds)
                self.num_groups = None
            elif len(suffix.split("-")) == 3:
                num_rounds, strategy, num_groups = suffix.split("-")
                num_rounds = int(num_rounds)
                self.num_groups = int(num_groups)
            elif len(suffix.split("-")) == 4:
                num_rounds, strategy, num_groups, local_steps = suffix.split("-")
                num_rounds = int(num_rounds)
                self.num_groups = int(num_groups)
                self.local_steps = int(local_steps)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.tc = STC(num_rounds, strategy)

    def epoch_start(self):
        self.parallel_call(lambda w: w.train_epoch_start())

    def parallel_group_get(self, f, group) -> list:
        results = []
        for w in group:
            with self.random_states_controller:
                results.append(f(w))
        return results

    def train_batch(self, meter, batch_idx, epoch):
        if self.local_steps == 1:
            self._train_batch_fc(meter, batch_idx, epoch)
            return

        if batch_idx % 2 == 1:
            self._train_batch_fc(meter, batch_idx, epoch)
        else:
            self._train_batch_local_sgd(meter, batch_idx, epoch)

    def _train_batch_local_sgd(self, meter, batch_idx, epoch):
        def _compute_gradient(w):
            results = w.compute_gradient()
            w.apply_gradient()
            return results

        for i in range(self.local_steps - 1):
            results = self.parallel_get(_compute_gradient)
            meter.add(results)

    def _train_batch_fc(self, meter, batch_idx, epoch):
        print(f"Epoch={epoch} Batch={batch_idx}")
        if self.args.algorithm.startswith("fc-grad"):
            if self.num_groups is None:
                groups = [self.workers]
            else:
                indices = np.arange(len(self.workers))
                n = int(np.ceil(len(self.workers) / self.num_groups).item())

                np.random.shuffle(indices)
                groups = []
                while len(indices) > 0:
                    groups.append([self.workers[i] for i in indices[:n]])
                    indices = indices[n:]

            count = 0
            for i_group, group in enumerate(groups):
                grad_collections = {}
                for i, worker in enumerate(group):
                    data, target = worker.get_data_target()
                    results = self.parallel_group_get(
                        lambda w: w.compute_gradient_over_data(data, target), group
                    )

                    meter.add([results[i]])
                    # grads is gradient computed at same samples
                    grads = self.parallel_group_get(
                        lambda w: w.get_gradient_from_opt(), group
                    )

                    for j, g in enumerate(grads):
                        # grad_collections[j] is a list of gradients computed by same model but on different data
                        grad_collections[j] = grad_collections.get(j, []) + [g]

                count += len(group)

                estimates = []
                for i, worker in enumerate(group):
                    grads = grad_collections[i]

                    estimate = self.tc(grads, grads[i])
                    estimates.append(estimate)

                for w, estimate in zip(group, estimates):
                    w.set_gradient(estimate)
                    w._gradient_manager._save_updates_to_state(w.opt)
                    g = w.get_gradient()
                    w.set_gradient(g)
                    w.apply_gradient()
        else:
            raise NotImplementedError


def run_fc(args, device, data_dir):
    """
    The current Personalized Federated Clustering (PFC) algorithm, we use
    Thresholding-Clustering (TC) as the clustering algorithm.

    Setup:
    - No server.
    - Each worker has its local model.
    """
    model = vgg16().to(device)
    loss_fn = CrossEntropyLoss().to(device)

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
        worker = FCWorker(
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
    trainer = FCTrain(
        args,
        pre_batch_hooks=[],
        post_batch_hooks=[],
        max_batches_per_epoch=args.max_batch_size_per_epoch,
        log_interval=args.log_interval,
        metrics=metrics(),
        use_cuda=args.use_cuda,
        debug=args.debug,
    )

    # There is no server.
    trainer.add_server_workers(None, workers)

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

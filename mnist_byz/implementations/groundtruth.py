from .utils import *


def verify_assumption_hook(trainer, epoch, batch_idx):
    """Evaluate if our proposed cluster mean estimate initialization scheme works."""
    if trainer.args.identifier not in [
        "verify_assumption",
        "verify_assumption1",
        "find_best_model_size",
    ]:
        return

    # NOTE: here we assume the dataset is MNIST.

    if batch_idx == 0:
        args = trainer.args
        workers = trainer.workers
        model = trainer.server.models[1]
        opt = trainer.server.opts[1]

        def _compute_gradient_over_same_model(w):
            w.set_model_opt(model, opt)
            data = w.last_batch["data"]
            target = w.last_batch["target"]
            w.compute_gradient_over_data(data, target)
            grad = w.get_gradient_from_opt()
            return grad

        grads = trainer.parallel_get(_compute_gradient_over_same_model)

        r = {
            "_meta": {"type": "Verify Assumption"},
            "E": epoch,
            # || \nabla \bar{f}_i(x) ||
            # Length K
            "ClusterCenterGradNorms": [],
            # || \nabla f_i(x) ||
            # Length n
            "GradNorms": [],
            # || \nabla f_i(x) - \nabla f_j(x) ||
            "ClusterCenterDistances": [
                [0 for _ in range(args.n)] for _ in range(args.n)
            ],
            # || \nabla f_i(x) - \nabla \bar{f}_i(x) ||
            "IntraClusterDistances": [],
        }

        if args.debug:
            trainer.debug_logger.info(
                f"\nCompute gradient distances at iteration {epoch}"
            )

        def _compute_cluster_mean(args, grads):
            # NOTE: that we use oracle information about clustering from get_data_loader.
            cluster_size = args.n // args.K_gen
            groundtruth_centers = []
            for k in range(args.K_gen):
                cluster_grads = grads[k * cluster_size : (k + 1) * cluster_size]
                groundtruth = sum(cluster_grads) / len(cluster_grads)
                groundtruth_centers.append(groundtruth)
            return groundtruth_centers

        centers = _compute_cluster_mean(args, grads)
        for g in centers:
            grad_norm = torch.linalg.norm(g)
            r["ClusterCenterGradNorms"].append(grad_norm.item())

        # Compute the norm of centers
        trainer.debug_logger.info("=> Compute the norm of centers")
        for g in grads:
            grad_norm = torch.linalg.norm(g)
            r["GradNorms"].append(grad_norm.item())

        # Compute the distances between centers
        trainer.debug_logger.info("=> Compute the distances between centers")
        cluster_size = args.n // args.K_gen
        for i in range(args.n):
            k = i // cluster_size
            for j in range(i + 1, args.n):
                grad_dist = torch.linalg.norm(grads[i] - grads[j]).item()
                r["ClusterCenterDistances"][i][j] = grad_dist
                r["ClusterCenterDistances"][j][i] = grad_dist

        # Compute the variance within each cluster
        trainer.debug_logger.info("=> Compute the variance within each cluster")

        def _compute_intra_cluster_distance(args, centers, grads):
            # NOTE: that we use oracle information about clustering from get_data_loader.
            distances = []
            cluster_size = args.n // args.K_gen

            for k in range(args.K_gen):
                center = centers[k]
                cluster_grads = grads[k * cluster_size : (k + 1) * cluster_size]

                for g in cluster_grads:
                    distance = torch.linalg.norm(g - center)
                    distances.append(distance.item())
            return distances

        r["IntraClusterDistances"] = _compute_intra_cluster_distance(
            args, centers, grads
        )
        trainer.debug_logger.info(r["IntraClusterDistances"])

        trainer.json_logger.info(r)

        def reset_model_opt(w):
            k = w.gt_id
            model = trainer.server.models[k]
            opt = trainer.server.opts[k]
            w.set_model_opt(model, opt)

        trainer.parallel_call(reset_model_opt)


class GTTrain(Train):
    def epoch_start(self):
        for m in self.server.models:
            m.train()
        self.parallel_call(lambda w: w.train_epoch_start())

        for rank, worker in enumerate(self.workers):
            self.gt_assignment(worker, rank)

    def gt_assignment(self, worker, rank):
        # Map rank to the groundtruth
        if self.args.data == "rotation":
            assert self.args.K_gen <= 4
            num_workers_within_cluster = self.args.n // self.args.K_gen
            k = rank // num_workers_within_cluster

        elif self.args.data == "relabel":
            num_workers_within_cluster = self.args.n // self.args.K_gen
            k = rank // num_workers_within_cluster

        else:
            raise NotImplementedError

        worker.gt_id = k

    def train_batch(self, meter, batch_idx, epoch):
        def _compute_gradient(w):
            k = w.gt_id
            model = self.server.models[k]
            opt = self.server.opts[k]
            w.set_model_opt(model, opt)
            results = w.compute_gradient()
            return results

        results = self.parallel_get(_compute_gradient)
        meter.add(results)

        avg_grads = [0 for k in range(self.args.K_gen)]
        counts = [0 for k in range(self.args.K_gen)]
        for w in self.workers:
            avg_grads[w.gt_id] += w.get_gradient()
            counts[w.gt_id] += 1

        for k in range(self.args.K_gen):
            avg_grads[k] /= counts[k]
            self.server.set_gradient(k, avg_grads[k])
            self.server.apply_gradient(k)


def run_groundtruth(args, device, data_dir):
    # ------------------------------- Define server ------------------------------ #
    server = MultiModelServer(
        model_fn=lambda: SimpleLinear(args.model_size),
        opt_fn=lambda m: torch.optim.SGD(m.parameters(), lr=args.lr),
        K=args.K,
        device=device,
        init="identical",
    )  # Server models having same initial weights

    # ------------------------------ Define workers ------------------------------ #
    workers = []
    for rank in range(args.n):
        sampler = functools.partial(
            sampler_fn, args=args, rank=rank, dataset_type="train", shuffle=True
        )

        train_loader = get_data_loader(
            args, sampler, rank, data_dir, dataset_type="train"
        )

        worker = FLWorker(
            index=rank,
            metrics=metrics(),
            momentum=args.momentum,
            data_loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            lr_scheduler=None,
        )
        workers.append(worker)

    # --------------------- Add server and worker to Trainer --------------------- #
    trainer = GTTrain(
        args,
        pre_batch_hooks=[],
        post_batch_hooks=[verify_assumption_hook],
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

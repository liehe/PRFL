import scipy
from .utils import *


def verify_assumption_hook(trainer, epoch, batch_idx):
    """Evaluate if our proposed cluster mean estimate initialization scheme works."""
    if trainer.args.identifier != "verify_assumption":
        # raise NotImplementedError
        return

    # NOTE: here we assume the dataset is MNIST.

    if batch_idx == 0:
        args = trainer.args
        workers = trainer.workers
        grads = [w.get_gradient() for w in workers]
        r = {
            "_meta": {"type": "Verify Assumption"},
            "E": epoch,
            # || \nabla \bar{f}_i(x) ||
            "ClusterCenterGradNorms": [],
            # || \nabla \bar{f}_i(x) - \nabla \bar{f}_j(x) ||
            "ClusterCenterDistances": [
                [0 for _ in range(args.K_gen)]
                for _ in range(args.K_gen)
            ],
            # || \nabla f_i(x) - \nabla \bar{f}_i(x) ||
            "IntraClusterDistances": []
        }

        if args.debug:
            trainer.debug_logger.info(
                f"\nCompute gradient distances at iteration {epoch}")

        def _compute_cluster_mean(args, grads):
            # NOTE: that we use oracle information about clustering from get_data_loader.
            cluster_size = args.n // args.K_gen
            groundtruth_centers = []
            for k in range(args.K_gen):
                cluster_grads = grads[k * cluster_size:(k+1) * cluster_size]
                groundtruth = sum(cluster_grads) / len(cluster_grads)
                groundtruth_centers.append(groundtruth)
            return groundtruth_centers

        centers = _compute_cluster_mean(args, grads)

        # Compute the norm of centers
        trainer.debug_logger.info("=> Compute the norm of centers")
        for k in range(args.K_gen):
            grad_norm = torch.linalg.norm(centers[k])
            r["ClusterCenterGradNorms"].append(grad_norm.item())
            trainer.debug_logger.info(grad_norm.item())

        # Compute the distances between centers
        trainer.debug_logger.info("=> Compute the distances between centers")
        for k in range(args.K_gen):
            for j in range(k+1, args.K_gen):
                grad_dist = torch.linalg.norm(centers[k] - centers[j]).item()
                r["ClusterCenterDistances"][k][j] = grad_dist
                r["ClusterCenterDistances"][j][k] = grad_dist
        trainer.debug_logger.info(r["ClusterCenterDistances"])

        # Compute the variance within each cluster
        trainer.debug_logger.info(
            "=> Compute the variance within each cluster")

        def _compute_intra_cluster_distance(args, centers, grads):
            # NOTE: that we use oracle information about clustering from get_data_loader.
            distances = []
            cluster_size = args.n // args.K_gen

            for k in range(args.K_gen):
                center = centers[k]
                cluster_grads = grads[k * cluster_size:(k+1) * cluster_size]

                distance = 0
                for g in cluster_grads:
                    distance += torch.linalg.norm(g - center) ** 2
                distance /= len(cluster_grads)
                distance = (distance.item()) ** 0.5
                distances.append(distance)
            return distances

        r["IntraClusterDistances"] = _compute_intra_cluster_distance(
            args, centers, grads)
        trainer.debug_logger.info(r["IntraClusterDistances"])

        trainer.json_logger.info(r)


class GlobalTrain(Train):
    def epoch_start(self):
        self.server.model.train()
        self.parallel_call(lambda w: w.train_epoch_start())

    def train_batch(self, meter, batch_idx, epoch):

        def _compute_gradient(w):
            model = self.server.model
            opt = self.server.opt
            w.set_model_opt(model, opt)
            results = w.compute_gradient()
            return results

        results = self.parallel_get(_compute_gradient)
        meter.add(results)

        avg_grad = sum(w.get_gradient()
                       for w in self.workers) / len(self.workers)

        self.server.set_gradient(avg_grad)
        self.server.apply_gradient()


def run_global(args, device, data_dir):
    # ------------------------------- Define server ------------------------------ #
    server = SingleModelServer(
        model_fn=lambda: SimpleLinear(args.model_size),
        opt_fn=lambda m: torch.optim.SGD(m.parameters(), lr=args.lr),
        device=device)

    # ------------------------------ Define workers ------------------------------ #
    workers = []
    for rank in range(args.n):
        sampler = functools.partial(
            sampler_fn, args=args, rank=rank, dataset_type='train', shuffle=True)

        train_loader = get_data_loader(
            args, sampler, rank, data_dir, dataset_type='train')

        worker = FLWorker(
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
        post_batch_hooks=[verify_assumption_hook],
        max_batches_per_epoch=args.max_batch_size_per_epoch,
        log_interval=args.log_interval,
        metrics=metrics(),
        use_cuda=args.use_cuda,
        debug=args.debug)

    # ------------------ Add Byzantine workers if there are any ------------------ #
    attack_worker_class = get_attack_type(args)

    for i in range(args.b):
        rank  = args.n + i
        train_loader = get_data_loader(
            args, sampler, rank, data_dir, dataset_type='train')

        worker = attack_worker_class(
            trainer=trainer,
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

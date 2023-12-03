import torch


class TorchServer(object):
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def apply_gradient(self) -> None:
        self.optimizer.step()

    def set_gradient(self, gradient: torch.Tensor) -> None:
        beg = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # for p in self.model.parameters():
                end = beg + len(p.grad.view(-1))
                x = gradient[beg:end].reshape_as(p.grad.data)
                p.grad.data = x.clone().detach()
                beg = end


class ClusterServer(object):
    def __init__(self, models: list, optimizers: list):
        self.models = models
        # 1 optimizer per cluster model
        self.optimizers = optimizers

    def apply_gradients(self):
        for opt in self.optimizers:
            opt.step()

    def apply_gradient(self, k: int) -> None:
        self.optimizers[k].step()

    def set_gradients(self, gradients: list):
        for i, grad in enumerate(gradients):
            self.set_gradient(i, grad)

    def set_gradient(self, k: int, gradient: torch.Tensor) -> None:
        beg = 0
        for group in self.optimizers[k].param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # for p in self.model.parameters():
                end = beg + len(p.grad.view(-1))
                x = gradient[beg:end].reshape_as(p.grad.data)
                p.grad.data = x.clone().detach()
                beg = end

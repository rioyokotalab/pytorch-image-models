import torch
from torch.optim.optimizer import Optimizer

class SAM(Optimizer):
    r"""Implements stochastic gradient descent (optionally with Sharpness-aware Minimization).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 0.1)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum (float, optional): momentum factor (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nbs (float): Neighborhood size (default: 0)

    Example:
        >>> optimizer = sam.SAM(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9, nbs=0.1)
        >>> optimizer.zero_grad()
        >>> loss = loss_fn(model(input), target)
        >>> original_params = network.parameters()
        >>> loss.backward()
        >>> optimizer.step_calc_w_adv()
        >>> loss_sam = loss_fn(model(input), target)
        >>> loss_sam.backward()
        >>> optimizer.load_original_params_and_step(original_params)
    """

    def __init__(self, params, lr=0.1, nesterov=False, weight_decay=0,
                 momentum=0, dampening=0, nbs=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if nbs < 0.0:
            raise ValueError("Invalid neighborhood_size value: {}".format(nbs))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        nesterov=nesterov, weight_decay=weight_decay, nbs=nbs)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SAM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SAM, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step_calc_w_adv(self, closure=None):
        """Update models weight params with w_adv.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            nbs = group['nbs']
            grad_l2norm = 0

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                grad_l2norm += torch.norm(d_p)**2
            if grad_l2norm == float('inf'):
                print('!!!grad_l2norm is too large!!!')
            if torch.isnan(grad_l2norm):
                print('!!!grad_l2norm is nan!!!')
            grad_l2norm = grad_l2norm**(1/2)

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                p.data = (p.data + (nbs/grad_l2norm)*d_p).clone().detach().requires_grad_(True)

        return loss

    @torch.no_grad()
    def load_original_params_and_step(self, original_params, closure=None):
        """Load original params (w_t) without gradient information and update weights params with SAM gradient

        Arguments:
            original_params: The original model params
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, group in enumerate(self.param_groups):
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for j, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.data = (original_params[i][j].data - lr*d_p).clone().detach().requires_grad_(True)

        return loss

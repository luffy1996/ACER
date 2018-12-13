# -*- coding: utf-8 -*-
from torch import optim
import math

class SharedAdam(optim.Adam):
  def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
    super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    # State initialisation (must be done before step, else will not be shared between threads)
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'] = p.data.new().resize_(1).zero_()
        state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
        # Exponential moving average of squared gradient values
        state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
        state['max_exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

  def share_memory(self):
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'].share_memory_()
        state['exp_avg'].share_memory_()
        state['exp_avg_sq'].share_memory_()
        state['max_exp_avg_sq'].share_memory_()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        amsgrad = group['amsgrad']

        state = self.state[p]

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
          max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1
        if group['weight_decay'] != 0:
          grad.add_(group['weight_decay'], p.data)
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
          # Maintains the maximum of all 2nd moment running avg. till now
          torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
          # Use the max. for normalizing running avg. of gradient
          denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
          denom = exp_avg_sq.sqrt().add_(group['eps'])
        bias_correction1 = 1 - beta1 ** state['step'].item()
        bias_correction2 = 1 - beta2 ** state['step'].item()
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
        p.data.addcdiv_(-step_size, exp_avg, denom)
    return loss

# Non-centered RMSprop update with shared statistics (without momentum)
class SharedRMSprop(optim.RMSprop):
  def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-5, weight_decay=0):
    super(SharedRMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=0, centered=False)

    # State initialisation (must be done before step, else will not be shared between threads)
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'] = p.data.new().resize_(1).zero_()
        state['square_avg'] = p.data.new().resize_as_(p.data).zero_()

  def share_memory(self):
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'].share_memory_()
        state['square_avg'].share_memory_()

  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        state = self.state[p]

        square_avg = state['square_avg']
        alpha = group['alpha']

        state['step'] += 1

        if group['weight_decay'] != 0:
          grad = grad.add(group['weight_decay'], p.data)

        # g = αg + (1 - α)Δθ^2
        square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
        # θ ← θ - ηΔθ/√(g + ε)
        avg = square_avg.sqrt().add_(group['eps'])
        p.data.addcdiv_(-group['lr'], grad, avg)

    return loss


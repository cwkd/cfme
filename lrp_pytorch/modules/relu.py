import torch
import torch.nn as nn


class LRPRelu(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(LRPRelu, self).__init__()
        self.module = module
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']

    def forward(self, x):
        return self.autograd_fn.apply(x, self.module, self.params)


class LRPRelu_Epsilon_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        activation_map = module.forward(x)
        ctx.activation_map = activation_map
        ctx.input_ = x
        return activation_map

    @staticmethod
    def backward(ctx, grad_output):
        activation_map = ctx.activation_map
        input_ = ctx.input_
        new_grad_output = torch.where(input_ > 0,
                                      grad_output,
                                      torch.zeros_like(grad_output, device=grad_output.device))
        return new_grad_output, None
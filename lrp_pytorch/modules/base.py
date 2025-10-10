import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def safe_divide(numerator, divisor, eps0, eps):
    return numerator / (divisor + eps0 * (divisor == 0).to(divisor) + eps * divisor.sign())


def rescale_pos_neg(Z, pos_norm, neg_norm):
    Z_pos, Z_neg = Z.clamp(min=0), Z.clamp(max=0)
    Z_pos_total, Z_neg_total = Z_pos.sum(), Z_neg.sum()
    Z_pos_normed = Z_pos / Z_pos_total
    Z_neg_normed = Z_neg / Z_neg_total
    return Z_pos_normed * pos_norm + Z_neg_normed * neg_norm


def lrp_backward(input_, layer, relevance_output, eps0, eps):
    if input_.grad is not None:
        input_.grad.zero_()
    # print(relevance_output.shape[0])
    # if relevance_output.shape[0] == 1:
    #     print(relevance_output)
    relevance_output_data = relevance_output.clone().detach()
    with torch.enable_grad():
        Z = layer(input_)
    S = safe_divide(relevance_output_data, Z.clone().detach(), eps0, eps)
    Z.backward(S)
    relevance_input = input_.data * input_.grad
    layer.saved_relevance = relevance_input
    return relevance_input


def cam_backward(input_, layer, relevance_output):
    if input_.grad is not None:
        input_.grad.zero()
    relevance_output_data = relevance_output.clone().detach()
    with torch.enable_grad():
        Z = layer(input_)
    Z.backward(relevance_output_data)
    relevance_input = F.relu(input_.data * input_.grad)
    layer.saved_relevance = relevance_input
    return relevance_input


def eb_backward(input_, layer, relevance_output):
    layer: nn.Linear
    with torch.no_grad():
        try:
            layer.weight.copy_(F.relu(layer.weight))
        except:
            pass
    if input_.grad is not None:
        input_.grad.zero()
    # print(layer, flush=True)
    with torch.enable_grad():
        Z = layer(input_)  # X = W^{+}^T * A_{n}
    relevance_output_data = relevance_output.clone().detach()  # P_{n-1}
    X = Z.clone().detach()
    Y = relevance_output_data / X  # Y = P_{n-1} (/) X
    Z.backward(Y)  # Use backward pass to compute Z = W^{+} * Y
    relevance_input = input_.data * input_.grad  # P_{n} = A_{n} (*) Z
    layer.saved_relevance = relevance_input
    # print(layer, relevance_input.shape, flush=True)
    return relevance_input


def setbyname(obj, name, value):
    # print(obj, name, value)

    def iteratset(obj, components, value):
        if not hasattr(obj, components[0]):
            return False
        elif len(components) == 1:
            setattr(obj, components[0], value)
            # print('set!!', components[0])
            # exit()
            return True
        else:
            nextobj = getattr(obj, components[0])
            # print(nextobj, components)
            return iteratset(nextobj, components[1:], value)

    components = name.split('.')
    success = iteratset(obj, components, value)
    return success


def getbyname(obj, name):
    # print(obj, name, value)

    def iteratget(obj, components):
        if not hasattr(obj, components[0]):
            return False
        elif len(components) == 1:
            value = getattr(obj, components[0])
            print('found!!', components[0])
            # exit()
            return value
        else:
            nextobj = getattr(obj, components[0])
            # print(nextobj, components)
            return iteratget(nextobj, components[1:])

    components = name.split('.')
    success = iteratget(obj, components)
    return success

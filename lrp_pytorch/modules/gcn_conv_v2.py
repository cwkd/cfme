import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn
from lrp_pytorch.modules.base import safe_divide
from lrp_pytorch.modules.linear import LRPLinear, Linear_Epsilon_Autograd_Fn


class LRPGCNConv(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(LRPGCNConv, self).__init__()
        self.module = module
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']

    def forward(self, x, edge_index, edge_weight=None):
        return self.autograd_fn.apply(x, edge_index, edge_weight, self.module, self.params)


class GCNConv_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, edge_index, edge_weight, module, params):
        eps = params.get('gcn_conv', 1e-6)

        def config_values_to_tensors(module):
            if isinstance(module, geom_nn.GCNConv):
                property_names = ['in_channels', 'out_channels']
            else:
                print('Error: module not GCNConv layer')
                raise Exception

            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.lin.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.lin.weight.device)
                else:
                    print('error: property value is neither int nor tuple')
                    exit()
                values.append(value)
            return property_names, values

        property_names, values = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)

        if module.lin.bias is None:
            bias = None
        else:
            bias = module.lin.bias.data.clone()
        # print(module.lin.weight.device)
        ctx.save_for_backward(x, edge_index, edge_weight, module.lin.weight.data.clone(), bias, eps_tensor, *values)

        # print('GCNConv ctx.needs_input_grad', ctx.needs_input_grad)
        # print(x.device, edge_index.device)
        module.to(module.lin.weight.device)
        return module.forward(x, edge_index, edge_weight)

    @staticmethod
    def backward(ctx, grad_output):
        input_, edge_index, edge_weight, weight, bias, eps_tensor, *values = ctx.saved_tensors
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = ['in_channels', 'out_channels']
            params_dict = {}

            for i, property_name in enumerate(property_names):
                value = values[i]
                if value.numel == 1:
                    params_dict[property_name] = value.item()
                else:
                    value_list = value.tolist()
                    if len(value_list) == 1:
                        params_dict[property_name] = value_list[0]
                    else:
                        params_dict[property_name] = tuple(value_list)
            return params_dict

        params_dict = tensors_to_dict(values)

        if bias is None:
            module = geom_nn.GCNConv(**params_dict, bias=False, cached=True)
        else:
            module = geom_nn.GCNConv(**params_dict, bias=True, cached=True)
            module.lin.bias = nn.Parameter(bias)

        module.lin.weight = nn.Parameter(weight)

        # print('GCNConv custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        lin = LRPLinear(module.lin, Linear_Epsilon_Autograd_Fn, params={'linear_eps': eps})
        module.lin = lin
        # print(lin.module.weight.device)
        X = input_.clone().detach().requires_grad_(True)
        R = lrp_gcnconv(input_=X,
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        layer=module,
                        relevance_output=grad_output,
                        eps0=eps,
                        eps=eps)
        # print('GCNConv custom R', R.shape)
        return R, None, None, None, None


def lrp_gcnconv(input_, edge_index, edge_weight, layer, relevance_output, eps0, eps):
    if input_.grad is not None:
        input_.grad.zero()
    relevance_output_data = relevance_output.clone().detach()
    with torch.enable_grad():
        Z = layer(input_, edge_index, edge_weight)
    S = safe_divide(relevance_output_data, Z.clone().detach(), eps0, eps)
    Z.backward(S)
    relevance_input = input_.data * input_.grad
    return relevance_input
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from lrp_pytorch.modules.base import safe_divide
from lrp_pytorch.modules.linear import LRPLinear, Linear_Epsilon_Autograd_Fn, CAM_Linear_Autograd_Fn, \
    EB_Linear_Autograd_Fn


class LRPGCNConv(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(LRPGCNConv, self).__init__()
        self.module = module
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']
        self.module.saved_rels = kwargs['saved_rels']
        self.module.x_in = []

    def forward(self, x, edge_index, edge_weight=None):
        return self.autograd_fn.apply(x, edge_index, edge_weight, self.module, self.params)


class GCNConv_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, edge_index, edge_weight, module, params):
        eps = params.get('gcn_conv', 1e-6)
        skip_layer_2 = params.get('skip_layer_2', False)
        layer_1_no_self_loops = params.get('layer_1_no_self_loops', False)
        layer_2_no_self_loops = params.get('layer_2_no_self_loops', False)

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
        if module.bias is None:
            node_bias = None
        else:
            node_bias = module.bias.data.clone()
        # print(module.lin.weight.device)
        ctx.save_for_backward(x, edge_index, edge_weight, module.lin.weight.data.clone(), bias, node_bias, eps_tensor, *values)
        # ctx.node_bias = node_bias
        # print(f'node_bias device', node_bias.device)
        ctx.saved_rels = module.saved_rels
        ctx.ref_name = module.ref_name
        ctx.skip_layer_2 = skip_layer_2
        ctx.layer_1_no_self_loops = layer_1_no_self_loops
        ctx.layer_2_no_self_loops = layer_2_no_self_loops

        # print('GCNConv ctx.needs_input_grad', ctx.needs_input_grad)
        # print(x.device, edge_index.device)
        module.to(module.lin.weight.device)
        return module.forward(x, edge_index, edge_weight)

    @staticmethod
    def backward(ctx, grad_output):
        input_, edge_index, edge_weight, weight, bias, node_bias, eps_tensor, *values = ctx.saved_tensors
        # node_bias = ctx.node_bias
        saved_rels = ctx.saved_rels
        ref_name = ctx.ref_name
        skip_layer_2 = ctx.skip_layer_2
        layer_1_no_self_loops = ctx.layer_1_no_self_loops
        layer_2_no_self_loops = ctx.layer_2_no_self_loops
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
            if layer_1_no_self_loops or layer_2_no_self_loops:
                add_self_loops = False
            else:
                add_self_loops = True
            module = geom_nn.GCNConv(**params_dict, bias=False, cached=True, add_self_loops=add_self_loops)
        else:
            if layer_1_no_self_loops or layer_2_no_self_loops:
                add_self_loops = False
            else:
                add_self_loops = True
            module = geom_nn.GCNConv(**params_dict, bias=True, cached=True, add_self_loops=add_self_loops)
            module.lin.bias = nn.Parameter(bias)

        module.lin.weight = nn.Parameter(weight)

        # print()
        if node_bias is None:
            module.register_parameter('bias', None)
        else:
            module.bias = nn.Parameter(node_bias, requires_grad=True)
            module.bias.retain_grad()
        module.to(input_.device)

        # print('GCNConv custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        lin = LRPLinear(module.lin, Linear_Epsilon_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                        ref_name=ref_name+'_fc')
        module.lin = lin
        # print(lin.module.weight.device)
        X = input_.clone().detach().requires_grad_(True)
        if skip_layer_2:
            edge_index = torch.as_tensor([[], []], dtype=edge_index.dtype, device=edge_index.device)
            edge_weight = None
        # R = lrp_gcnconv(input_=X,
        #                 edge_index=edge_index,
        #                 edge_weight=edge_weight,
        #                 layer=module,
        #                 relevance_output=grad_output,
        #                 eps0=eps,
        #                 eps=eps)
        R = lrp_gcnconv(input_=X,
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        layer=module,
                        relevance_output=grad_output,
                        eps0=eps,
                        eps=0)
        # print('GCNConv custom R', R.shape)
        saved_rels[ref_name] = module.saved_relevance
        # if ref_name == 'td_conv1':
        #     print(module.saved_relevance.shape)
        # print(ref_name, [(k, v.shape) for k, v in saved_rels.items()])
        return R, None, None, None, None


class CAM_GCNConv_Autograd_Fn(torch.autograd.Function):

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
        ctx.saved_rels = module.saved_rels
        ctx.ref_name = module.ref_name

        # print('GCNConv ctx.needs_input_grad', ctx.needs_input_grad)
        # print(x.device, edge_index.device)
        module.to(module.lin.weight.device)
        return module.forward(x, edge_index, edge_weight)

    @staticmethod
    def backward(ctx, grad_output):
        input_, edge_index, edge_weight, weight, bias, eps_tensor, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        ref_name = ctx.ref_name
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

        lin = LRPLinear(module.lin, CAM_Linear_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                        ref_name=ref_name+'_fc')
        module.lin = lin
        # print(lin.module.weight.device)
        X = input_.clone().detach().requires_grad_(True)
        R = cam_gcnconv(input_=X,
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        layer=module,
                        relevance_output=grad_output)
        # print('GCNConv custom R', R.shape)
        saved_rels[ref_name] = module.saved_relevance
        # if ref_name == 'td_conv1':
        #     print(module.saved_relevance.shape)
        # print(ref_name, [(k, v.shape) for k, v in saved_rels.items()])
        return R, None, None, None, None


class EB_GCNConv_Autograd_Fn(torch.autograd.Function):

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
        ctx.saved_rels = module.saved_rels
        ctx.ref_name = module.ref_name

        # print('GCNConv ctx.needs_input_grad', ctx.needs_input_grad)
        # print(x.device, edge_index.device)
        module.to(module.lin.weight.device)
        return module.forward(x, edge_index, edge_weight)

    @staticmethod
    def backward(ctx, grad_output):
        input_, edge_index, edge_weight, weight, bias, eps_tensor, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        ref_name = ctx.ref_name
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

        lin = LRPLinear(module.lin, EB_Linear_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                        ref_name=ref_name+'_fc')
        module.lin = lin
        # print(lin.module.weight.device)
        X = input_.clone().detach().requires_grad_(True)
        R = eb_gcnconv(input_=X,
                       edge_index=edge_index,
                       edge_weight=edge_weight,
                       layer=module,
                       relevance_output=grad_output)
        # print('GCNConv custom R', R.shape, X.grad.shape)
        saved_rels[ref_name] = module.saved_relevance
        # if ref_name == 'td_conv1':
        #     print(module.saved_relevance.shape)
        # print(ref_name, [(k, v.shape) for k, v in saved_rels.items()])
        return R, None, None, None, None


class DeepLIFT_GCNConv_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, edge_index, edge_weight, module, params):
        eps = params.get('gcn_conv', 1e-6)
        skip_layer_2 = params.get('skip_layer_2', False)
        layer_1_no_self_loops = params.get('layer_1_no_self_loops', False)
        layer_2_no_self_loops = params.get('layer_2_no_self_loops', False)

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
        if module.bias is None:
            node_bias = None
        else:
            node_bias = module.bias.data.clone()
        if len(module.x_in) == 0:
            ref = None
            module.x_in.append(x.clone().detach())
        else:
            ref = module.x_in[0]
        # print(module.lin.weight.device)
        ctx.save_for_backward(ref, x, edge_index, edge_weight, module.lin.weight.data.clone(), bias, node_bias,
                              eps_tensor, *values)
        # ctx.node_bias = node_bias
        # print(f'node_bias device', node_bias.device)
        ctx.saved_rels = module.saved_rels
        ctx.ref_name = module.ref_name
        ctx.skip_layer_2 = skip_layer_2
        ctx.layer_1_no_self_loops = layer_1_no_self_loops
        ctx.layer_2_no_self_loops = layer_2_no_self_loops

        # print('GCNConv ctx.needs_input_grad', ctx.needs_input_grad)
        # print(x.device, edge_index.device)
        module.to(module.lin.weight.device)
        return module.forward(x, edge_index, edge_weight)

    @staticmethod
    def backward(ctx, grad_output):
        ref, input_, edge_index, edge_weight, weight, bias, node_bias, eps_tensor, *values = ctx.saved_tensors
        # node_bias = ctx.node_bias
        saved_rels = ctx.saved_rels
        ref_name = ctx.ref_name
        skip_layer_2 = ctx.skip_layer_2
        layer_1_no_self_loops = ctx.layer_1_no_self_loops
        layer_2_no_self_loops = ctx.layer_2_no_self_loops
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
            if layer_1_no_self_loops or layer_2_no_self_loops:
                add_self_loops = False
            else:
                add_self_loops = True
            module = geom_nn.GCNConv(**params_dict, bias=False, cached=True, add_self_loops=add_self_loops)
        else:
            if layer_1_no_self_loops or layer_2_no_self_loops:
                add_self_loops = False
            else:
                add_self_loops = True
            module = geom_nn.GCNConv(**params_dict, bias=True, cached=True, add_self_loops=add_self_loops)
            module.lin.bias = nn.Parameter(bias)

        module.lin.weight = nn.Parameter(weight)

        # print()
        if node_bias is None:
            module.register_parameter('bias', None)
        else:
            module.bias = nn.Parameter(node_bias, requires_grad=True)
            module.bias.retain_grad()
        module.to(input_.device)

        # print('GCNConv custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        lin = LRPLinear(module.lin, DeepLIFT_Linear_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                        ref_name=ref_name+'_fc')
        module.lin = lin
        # print(lin.module.weight.device)
        X = input_.clone().detach().requires_grad_(True)
        ref_copy = ref.clone().detach().requires_grad_(True)
        if skip_layer_2:
            edge_index = torch.as_tensor([[], []], dtype=edge_index.dtype, device=edge_index.device)
            edge_weight = None
        R = deeplift_gcnconv(ref_input=ref_copy,
                             input_=X,
                             edge_index=edge_index,
                             edge_weight=edge_weight,
                             layer=module,
                             relevance_output=grad_output)
        # print('GCNConv custom R', R.shape)
        saved_rels[ref_name] = module.saved_relevance
        # if ref_name == 'td_conv1':
        #     print(module.saved_relevance.shape)
        # print(ref_name, [(k, v.shape) for k, v in saved_rels.items()])
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
    layer.saved_relevance = relevance_input
    return relevance_input


def cam_gcnconv(input_, edge_index, edge_weight, layer, relevance_output):
    if input_.grad is not None:
        input_.grad.zero()
    relevance_output_data = relevance_output.clone().detach()
    with torch.enable_grad():
        Z = layer(input_, edge_index, edge_weight)
    Z.backward(relevance_output_data)
    relevance_input = F.relu(input_.data * input_.grad)
    layer.saved_relevance = relevance_input
    return relevance_input


def eb_gcnconv(input_, edge_index, edge_weight, layer, relevance_output):
    if edge_weight is not None:
        w_pos = F.relu(edge_weight)
    else:
        w_pos = edge_weight
    if input_.grad is not None:
        input_.grad.zero()
    with torch.enable_grad():
        Z = layer(input_, edge_index, w_pos)  # X = W^{+}^T * A_{n}
    relevance_output_data = relevance_output.clone().detach()  # P_{n-1}
    # print(Z.clone().detach().shape)
    X = Z.clone().detach()
    Y = relevance_output_data / X  # Y = P_{n-1} (/) X
    # print('gcnconv: ', X.shape, Y.shape)
    Z.backward(Y)  # Use backward pass to compute Z = W^{+} * Y
    relevance_input = input_.data * input_.grad  # P_{n} = A_{n} (*) Z
    layer.saved_relevance = relevance_input
    return relevance_input
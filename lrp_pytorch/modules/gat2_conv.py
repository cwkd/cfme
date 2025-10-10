import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from lrp_pytorch.modules.base import safe_divide
from lrp_pytorch.modules.linear import LRPLinear, Linear_Epsilon_Autograd_Fn, CAM_Linear_Autograd_Fn, \
    EB_Linear_Autograd_Fn


class LRPGATv2Conv(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(LRPGATv2Conv, self).__init__()
        self.module = module
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']
        self.module.saved_rels = kwargs.get('saved_rels', None)
        self.module.ref_name = kwargs.get('ref_name', None)

    def forward(self, x, edge_index, skip_layer_2=False, layer_1_no_self_loops=False, layer_2_no_self_loops=False):
        self.params['skip_layer_2'] = skip_layer_2
        self.params['layer_1_no_self_loops'] = layer_1_no_self_loops
        self.params['layer_2_no_self_loops'] = layer_2_no_self_loops

        return self.autograd_fn.apply(x, edge_index, self.module, self.params)


class GATv2Conv_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, edge_index, module, params):
        eps = params.get('gcn_conv', 1e-6)
        skip_layer_2 = params.get('skip_layer_2', False)
        layer_1_no_self_loops = params.get('layer_1_no_self_loops', False)
        layer_2_no_self_loops = params.get('layer_2_no_self_loops', False)

        if isinstance(module, geom_nn.GATv2Conv):
            is_GATv2 = True
        else:
            is_GATv2 = False

        def config_values_to_tensors(module):
            if isinstance(module, geom_nn.GATv2Conv) or isinstance(module, geom_nn.conv.gat_conv.GATConv):
                property_names = ['in_channels', 'out_channels', 'heads', 'concat', 'negative_slope', 'dropout', 'edge_dim']
            else:
                print('Error: module not GATv2Conv layer')
                raise Exception

            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int) and attr in ['in_channels', 'out_channels']:
                    value = torch.tensor([value], dtype=torch.int32, device=x.device)
                elif attr == 'concat':
                    concat = value
                    continue
                elif attr == 'edge_dim':
                    edge_dim = value
                    continue
                elif isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=x.device)
                elif isinstance(value, float):
                    value = torch.tensor([value], dtype=torch.float, device=x.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=x.device)
                elif isinstance(value, bool):
                    value = torch.tensor(value, dtype=torch.bool, device=x.device)
                else:
                    print('error: property value is neither int nor float nor tuple')
                    exit()
                values.append(value)
            return property_names, values, concat, edge_dim

        property_names, values, concat, edge_dim = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)

        if is_GATv2:
            if module.lin_l.bias is None:
                lin_l_bias = None
            else:
                lin_l_bias = module.lin_l.bias.data.clone()
            lin_l_weight = module.lin_l.weight.data.clone()
            if module.lin_r.bias is None:
                lin_r_bias = None
            else:
                lin_r_bias = module.lin_r.bias.data.clone()
            lin_r_weight = module.lin_r.weight.data.clone()
            if edge_dim is not None:
                if module.lin_edge.bias is None:
                    lin_edge_bias = None
                else:
                    lin_edge_bias = module.lin_edge.bias.data.clone()
                lin_edge_weight = module.lin_edge.weight.data.clone()
                att = module.att.data.clone()
            else:
                lin_edge_bias = None
                lin_edge_weight = None
                att = None
            if module.bias is None:
                bias = None
            else:
                bias = module.bias.data.clone()
            # print(module.lin.weight.device)
            ctx.save_for_backward(x, edge_index, att,
                                  lin_l_weight, lin_l_bias,
                                  lin_r_weight, lin_r_bias,
                                  lin_edge_weight, lin_edge_bias,
                                  bias, eps_tensor, *values)
        else:
            if module.lin.bias is None:
                lin_bias = None
            else:
                lin_bias = module.lin.bias.data.clone()
            lin_weight = module.lin.weight.data.clone()
            if edge_dim is not None:
                if module.lin_edge.bias is None:
                    lin_edge_bias = None
                else:
                    lin_edge_bias = module.lin_edge.bias.data.clone()
                lin_edge_weight = module.lin_edge.weight.data.clone()
                att_edge = module.att_edge.data.clone()
            else:
                lin_edge_bias = None
                lin_edge_weight = None
                att_edge = None
            if module.bias is None:
                bias = None
            else:
                bias = module.bias.data.clone()
            ctx.save_for_backward(x, edge_index, att_edge,
                                  lin_weight, lin_bias,
                                  lin_edge_weight, lin_edge_bias,
                                  bias, eps_tensor, *values)

        ctx.saved_rels = module.saved_rels
        ctx.ref_name = module.ref_name
        ctx.is_GATv2 = is_GATv2
        ctx.skip_layer_2 = skip_layer_2
        ctx.layer_1_no_self_loops = layer_1_no_self_loops
        ctx.layer_2_no_self_loops = layer_2_no_self_loops
        ctx.concat = concat
        ctx.edge_dim = edge_dim
        # print('GCNConv ctx.needs_input_grad', ctx.needs_input_grad)
        # print(x.device, edge_index.device)
        module.to(module.lin.weight.device)
        return module.forward(x, edge_index)

    @staticmethod
    def backward(ctx, grad_output):
        is_GATv2 = ctx.is_GATv2
        if is_GATv2:
            input_, edge_index, att, \
            lin_l_weight, lin_l_bias, \
            lin_r_weight, lin_r_bias, \
            lin_edge_weight, lin_edge_bias, \
            bias, eps_tensor, *values = ctx.saved_tensors
        else:
            input_, edge_index, att_edge, \
            lin_weight, lin_bias, \
            lin_edge_weight, lin_edge_bias, \
            bias, eps_tensor, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        ref_name = ctx.ref_name
        skip_layer_2 = ctx.skip_layer_2
        layer_1_no_self_loops = ctx.layer_1_no_self_loops
        layer_2_no_self_loops = ctx.layer_2_no_self_loops
        concat = ctx.concat
        edge_dim = ctx.edge_dim
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = ['in_channels', 'out_channels', 'heads', 'negative_slope', 'dropout']
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
        params_dict['concat'] = concat
        params_dict['edge_dim'] = edge_dim

        if is_GATv2:
            if bias is None:
                if layer_1_no_self_loops or layer_2_no_self_loops:
                    add_self_loops = False
                else:
                    add_self_loops = True
                module = geom_nn.GATv2Conv(**params_dict, bias=False, add_self_loops=add_self_loops)
                if lin_l_bias is not None:
                    module.lin_l.bias = nn.Parameter(lin_l_bias)
                if lin_r_bias is not None:
                    module.lin_r.bias = nn.Parameter(lin_r_bias)
            else:
                if layer_1_no_self_loops or layer_2_no_self_loops:
                    add_self_loops = False
                else:
                    add_self_loops = True
                module = geom_nn.GATv2Conv(**params_dict, bias=True, add_self_loops=add_self_loops)
                if lin_l_bias is not None:
                    module.lin_l.bias = nn.Parameter(lin_l_bias)
                if lin_r_bias is not None:
                    module.lin_r.bias = nn.Parameter(lin_r_bias)
            module.lin_l.weight = nn.Parameter(lin_l_weight)
            module.lin_r.weight = nn.Parameter(lin_r_weight)
            module.att = nn.Parameter(att)
        else:
            if bias is None:
                if layer_1_no_self_loops or layer_2_no_self_loops:
                    add_self_loops = False
                else:
                    add_self_loops = True
                module = geom_nn.conv.gat_conv.GATConv(**params_dict, bias=False, add_self_loops=add_self_loops)
                if lin_bias is not None:
                    module.lin.bias = nn.Parameter(lin_bias)
            else:
                if layer_1_no_self_loops or layer_2_no_self_loops:
                    add_self_loops = False
                else:
                    add_self_loops = True
                module = geom_nn.conv.gat_conv.GATConv(**params_dict, bias=True, add_self_loops=add_self_loops)
                if lin_bias is not None:
                    module.lin.bias = nn.Parameter(lin_bias)
            module.lin.weight = nn.Parameter(lin_weight)
            module.att_edge = nn.Parameter(att_edge)

        if bias is not None:
            module.bias = nn.Parameter(bias)
        if edge_dim is not None:
            module.lin_edge.weight = nn.Parameter(lin_edge_weight)

        # print('GCNConv custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        if is_GATv2:
            lin_l = LRPLinear(module.lin_l, Linear_Epsilon_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                              ref_name=ref_name + '_lin_l')
            lin_r = LRPLinear(module.lin_r, Linear_Epsilon_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                              ref_name=ref_name + '_lin_r')
            module.lin_l = lin_l
            module.lin_r = lin_r
        else:
            lin = LRPLinear(module.lin, Linear_Epsilon_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                            ref_name=ref_name + '_lin')
            module.lin = lin
        if edge_dim is not None:
            lin_edge = LRPLinear(module.lin_edge, Linear_Epsilon_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                                 ref_name=ref_name + '_lin_edge')
            module.lin_edge = lin_edge
        # print(lin.module.weight.device)
        X = input_.clone().detach().requires_grad_(True)
        if skip_layer_2:
            edge_index = torch.as_tensor([[], []], dtype=edge_index.dtype, device=edge_index.device)
        R = lrp_gatconv(input_=X,
                        edge_index=edge_index,
                        layer=module,
                        relevance_output=grad_output,
                        eps0=eps,
                        eps=eps)
        # print('GCNConv custom R', R.shape)
        return R, None, None, None, None


class CAM_GATv2Conv_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, edge_index, module, params):
        eps = params.get('gcn_conv', 1e-6)
        skip_layer_2 = params.get('skip_layer_2', False)
        layer_1_no_self_loops = params.get('layer_1_no_self_loops', False)
        layer_2_no_self_loops = params.get('layer_2_no_self_loops', False)

        if isinstance(module, geom_nn.GATv2Conv):
            is_GATv2 = True
        else:
            is_GATv2 = False

        def config_values_to_tensors(module):
            if isinstance(module, geom_nn.GATv2Conv) or isinstance(module, geom_nn.conv.gat_conv.GATConv):
                property_names = ['in_channels', 'out_channels', 'heads', 'concat', 'negative_slope', 'dropout', 'edge_dim']
            else:
                print('Error: module not GATv2Conv layer')
                raise Exception

            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int) and attr in ['in_channels', 'out_channels']:
                    value = torch.tensor([value], dtype=torch.int32, device=x.device)
                elif attr == 'concat':
                    concat = value
                    continue
                elif attr == 'edge_dim':
                    edge_dim = value
                    continue
                elif isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=x.device)
                elif isinstance(value, float):
                    value = torch.tensor([value], dtype=torch.float, device=x.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=x.device)
                elif isinstance(value, bool):
                    value = torch.tensor(value, dtype=torch.bool, device=x.device)
                else:
                    print('error: property value is neither int nor float nor tuple')
                    exit()
                values.append(value)
            return property_names, values, concat, edge_dim

        property_names, values, concat, edge_dim = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)

        if is_GATv2:
            if module.lin_l.bias is None:
                lin_l_bias = None
            else:
                lin_l_bias = module.lin_l.bias.data.clone()
            lin_l_weight = module.lin_l.weight.data.clone()
            if module.lin_r.bias is None:
                lin_r_bias = None
            else:
                lin_r_bias = module.lin_r.bias.data.clone()
            lin_r_weight = module.lin_r.weight.data.clone()
            if edge_dim is not None:
                if module.lin_edge.bias is None:
                    lin_edge_bias = None
                else:
                    lin_edge_bias = module.lin_edge.bias.data.clone()
                lin_edge_weight = module.lin_edge.weight.data.clone()
                att = module.att.data.clone()
            else:
                lin_edge_bias = None
                lin_edge_weight = None
                att = None
            if module.bias is None:
                bias = None
            else:
                bias = module.bias.data.clone()
            # print(module.lin.weight.device)
            ctx.save_for_backward(x, edge_index, att,
                                  lin_l_weight, lin_l_bias,
                                  lin_r_weight, lin_r_bias,
                                  lin_edge_weight, lin_edge_bias,
                                  bias, eps_tensor, *values)
        else:
            if module.lin.bias is None:
                lin_bias = None
            else:
                lin_bias = module.lin.bias.data.clone()
            lin_weight = module.lin.weight.data.clone()
            if edge_dim is not None:
                if module.lin_edge.bias is None:
                    lin_edge_bias = None
                else:
                    lin_edge_bias = module.lin_edge.bias.data.clone()
                lin_edge_weight = module.lin_edge.weight.data.clone()
                att_edge = module.att_edge.data.clone()
            else:
                lin_edge_bias = None
                lin_edge_weight = None
                att_edge = None
            if module.bias is None:
                bias = None
            else:
                bias = module.bias.data.clone()
            ctx.save_for_backward(x, edge_index, att_edge,
                                  lin_weight, lin_bias,
                                  lin_edge_weight, lin_edge_bias,
                                  bias, eps_tensor, *values)

        ctx.saved_rels = module.saved_rels
        ctx.ref_name = module.ref_name
        ctx.is_GATv2 = is_GATv2
        ctx.skip_layer_2 = skip_layer_2
        ctx.layer_1_no_self_loops = layer_1_no_self_loops
        ctx.layer_2_no_self_loops = layer_2_no_self_loops
        ctx.concat = concat
        ctx.edge_dim = edge_dim
        # print('GCNConv ctx.needs_input_grad', ctx.needs_input_grad)
        # print(x.device, edge_index.device)
        module.to(module.lin.weight.device)
        return module.forward(x, edge_index)

    @staticmethod
    def backward(ctx, grad_output):
        is_GATv2 = ctx.is_GATv2
        if is_GATv2:
            input_, edge_index, att, \
            lin_l_weight, lin_l_bias, \
            lin_r_weight, lin_r_bias, \
            lin_edge_weight, lin_edge_bias, \
            bias, eps_tensor, *values = ctx.saved_tensors
        else:
            input_, edge_index, att_edge, \
            lin_weight, lin_bias, \
            lin_edge_weight, lin_edge_bias, \
            bias, eps_tensor, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        ref_name = ctx.ref_name
        skip_layer_2 = ctx.skip_layer_2
        layer_1_no_self_loops = ctx.layer_1_no_self_loops
        layer_2_no_self_loops = ctx.layer_2_no_self_loops
        concat = ctx.concat
        edge_dim = ctx.edge_dim
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = ['in_channels', 'out_channels', 'heads', 'negative_slope', 'dropout']
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
        params_dict['concat'] = concat
        params_dict['edge_dim'] = edge_dim

        if is_GATv2:
            if bias is None:
                if layer_1_no_self_loops or layer_2_no_self_loops:
                    add_self_loops = False
                else:
                    add_self_loops = True
                module = geom_nn.GATv2Conv(**params_dict, bias=False, add_self_loops=add_self_loops)
                if lin_l_bias is not None:
                    module.lin_l.bias = nn.Parameter(lin_l_bias)
                if lin_r_bias is not None:
                    module.lin_r.bias = nn.Parameter(lin_r_bias)
            else:
                if layer_1_no_self_loops or layer_2_no_self_loops:
                    add_self_loops = False
                else:
                    add_self_loops = True
                module = geom_nn.GATv2Conv(**params_dict, bias=True, add_self_loops=add_self_loops)
                if lin_l_bias is not None:
                    module.lin_l.bias = nn.Parameter(lin_l_bias)
                if lin_r_bias is not None:
                    module.lin_r.bias = nn.Parameter(lin_r_bias)
            module.lin_l.weight = nn.Parameter(lin_l_weight)
            module.lin_r.weight = nn.Parameter(lin_r_weight)
            module.att = nn.Parameter(att)
        else:
            if bias is None:
                if layer_1_no_self_loops or layer_2_no_self_loops:
                    add_self_loops = False
                else:
                    add_self_loops = True
                module = geom_nn.conv.gat_conv.GATConv(**params_dict, bias=False, add_self_loops=add_self_loops)
                if lin_bias is not None:
                    module.lin.bias = nn.Parameter(lin_bias)
            else:
                if layer_1_no_self_loops or layer_2_no_self_loops:
                    add_self_loops = False
                else:
                    add_self_loops = True
                module = geom_nn.conv.gat_conv.GATConv(**params_dict, bias=True, add_self_loops=add_self_loops)
                if lin_bias is not None:
                    module.lin.bias = nn.Parameter(lin_bias)
            module.lin.weight = nn.Parameter(lin_weight)
            module.att_edge = nn.Parameter(att_edge)

        if bias is not None:
            module.bias = nn.Parameter(bias)
        if edge_dim is not None:
            module.lin_edge.weight = nn.Parameter(lin_edge_weight)

        # print('GCNConv custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        if is_GATv2:
            lin_l = LRPLinear(module.lin_l, CAM_Linear_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                              ref_name=ref_name + '_lin_l')
            lin_r = LRPLinear(module.lin_r, CAM_Linear_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                              ref_name=ref_name + '_lin_r')
            module.lin_l = lin_l
            module.lin_r = lin_r
        else:
            lin = LRPLinear(module.lin, CAM_Linear_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                            ref_name=ref_name + '_lin')
            module.lin = lin
        if edge_dim is not None:
            lin_edge = LRPLinear(module.lin_edge, CAM_Linear_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                                 ref_name=ref_name + '_lin_edge')
            module.lin_edge = lin_edge
        # print(lin.module.weight.device)
        X = input_.clone().detach().requires_grad_(True)
        if skip_layer_2:
            edge_index = torch.as_tensor([[], []], dtype=edge_index.dtype, device=edge_index.device)
        R = cam_gatconv(input_=X,
                        edge_index=edge_index,
                        layer=module,
                        relevance_output=grad_output)
        # print('GCNConv custom R', R.shape)
        return R, None, None, None, None


class EB_GATv2Conv_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, edge_index, module, params):
        eps = params.get('gcn_conv', 1e-6)
        skip_layer_2 = params.get('skip_layer_2', False)
        layer_1_no_self_loops = params.get('layer_1_no_self_loops', False)
        layer_2_no_self_loops = params.get('layer_2_no_self_loops', False)

        if isinstance(module, geom_nn.GATv2Conv):
            is_GATv2 = True
        else:
            is_GATv2 = False

        def config_values_to_tensors(module):
            if isinstance(module, geom_nn.GATv2Conv) or isinstance(module, geom_nn.conv.gat_conv.GATConv):
                property_names = ['in_channels', 'out_channels', 'heads', 'concat', 'negative_slope', 'dropout', 'edge_dim']
            else:
                print('Error: module not GATv2Conv layer')
                raise Exception

            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int) and attr in ['in_channels', 'out_channels']:
                    value = torch.tensor([value], dtype=torch.int32, device=x.device)
                elif attr == 'concat':
                    concat = value
                    continue
                elif attr == 'edge_dim':
                    edge_dim = value
                    continue
                elif isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=x.device)
                elif isinstance(value, float):
                    value = torch.tensor([value], dtype=torch.float, device=x.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=x.device)
                elif isinstance(value, bool):
                    value = torch.tensor(value, dtype=torch.bool, device=x.device)
                else:
                    print('error: property value is neither int nor float nor tuple')
                    exit()
                values.append(value)
            return property_names, values, concat, edge_dim

        property_names, values, concat, edge_dim = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)

        if is_GATv2:
            if module.lin_l.bias is None:
                lin_l_bias = None
            else:
                lin_l_bias = module.lin_l.bias.data.clone()
            lin_l_weight = module.lin_l.weight.data.clone()
            if module.lin_r.bias is None:
                lin_r_bias = None
            else:
                lin_r_bias = module.lin_r.bias.data.clone()
            lin_r_weight = module.lin_r.weight.data.clone()
            if edge_dim is not None:
                if module.lin_edge.bias is None:
                    lin_edge_bias = None
                else:
                    lin_edge_bias = module.lin_edge.bias.data.clone()
                lin_edge_weight = module.lin_edge.weight.data.clone()
                att = module.att.data.clone()
            else:
                lin_edge_bias = None
                lin_edge_weight = None
                att = None
            if module.bias is None:
                bias = None
            else:
                bias = module.bias.data.clone()
            # print(module.lin.weight.device)
            ctx.save_for_backward(x, edge_index, att,
                                  lin_l_weight, lin_l_bias,
                                  lin_r_weight, lin_r_bias,
                                  lin_edge_weight, lin_edge_bias,
                                  bias, eps_tensor, *values)
        else:
            if module.lin.bias is None:
                lin_bias = None
            else:
                lin_bias = module.lin.bias.data.clone()
            lin_weight = module.lin.weight.data.clone()
            if edge_dim is not None:
                if module.lin_edge.bias is None:
                    lin_edge_bias = None
                else:
                    lin_edge_bias = module.lin_edge.bias.data.clone()
                lin_edge_weight = module.lin_edge.weight.data.clone()
                att_edge = module.att_edge.data.clone()
            else:
                lin_edge_bias = None
                lin_edge_weight = None
                att_edge = None
            if module.bias is None:
                bias = None
            else:
                bias = module.bias.data.clone()
            ctx.save_for_backward(x, edge_index, att_edge,
                                  lin_weight, lin_bias,
                                  lin_edge_weight, lin_edge_bias,
                                  bias, eps_tensor, *values)

        ctx.saved_rels = module.saved_rels
        ctx.ref_name = module.ref_name
        ctx.is_GATv2 = is_GATv2
        ctx.skip_layer_2 = skip_layer_2
        ctx.layer_1_no_self_loops = layer_1_no_self_loops
        ctx.layer_2_no_self_loops = layer_2_no_self_loops
        ctx.concat = concat
        ctx.edge_dim = edge_dim
        # print('GCNConv ctx.needs_input_grad', ctx.needs_input_grad)
        # print(x.device, edge_index.device)
        module.to(module.lin.weight.device)
        return module.forward(x, edge_index)

    @staticmethod
    def backward(ctx, grad_output):
        is_GATv2 = ctx.is_GATv2
        if is_GATv2:
            input_, edge_index, att, \
            lin_l_weight, lin_l_bias, \
            lin_r_weight, lin_r_bias, \
            lin_edge_weight, lin_edge_bias, \
            bias, eps_tensor, *values = ctx.saved_tensors
        else:
            input_, edge_index, att_edge, \
            lin_weight, lin_bias, \
            lin_edge_weight, lin_edge_bias, \
            bias, eps_tensor, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        ref_name = ctx.ref_name
        skip_layer_2 = ctx.skip_layer_2
        layer_1_no_self_loops = ctx.layer_1_no_self_loops
        layer_2_no_self_loops = ctx.layer_2_no_self_loops
        concat = ctx.concat
        edge_dim = ctx.edge_dim
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = ['in_channels', 'out_channels', 'heads', 'negative_slope', 'dropout']
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
        params_dict['concat'] = concat
        params_dict['edge_dim'] = edge_dim

        if is_GATv2:
            if bias is None:
                if layer_1_no_self_loops or layer_2_no_self_loops:
                    add_self_loops = False
                else:
                    add_self_loops = True
                module = geom_nn.GATv2Conv(**params_dict, bias=False, add_self_loops=add_self_loops)
                if lin_l_bias is not None:
                    module.lin_l.bias = nn.Parameter(lin_l_bias)
                if lin_r_bias is not None:
                    module.lin_r.bias = nn.Parameter(lin_r_bias)
            else:
                if layer_1_no_self_loops or layer_2_no_self_loops:
                    add_self_loops = False
                else:
                    add_self_loops = True
                module = geom_nn.GATv2Conv(**params_dict, bias=True, add_self_loops=add_self_loops)
                if lin_l_bias is not None:
                    module.lin_l.bias = nn.Parameter(lin_l_bias)
                if lin_r_bias is not None:
                    module.lin_r.bias = nn.Parameter(lin_r_bias)
            module.lin_l.weight = nn.Parameter(lin_l_weight)
            module.lin_r.weight = nn.Parameter(lin_r_weight)
            module.att = nn.Parameter(att)
        else:
            if bias is None:
                if layer_1_no_self_loops or layer_2_no_self_loops:
                    add_self_loops = False
                else:
                    add_self_loops = True
                module = geom_nn.conv.gat_conv.GATConv(**params_dict, bias=False, add_self_loops=add_self_loops)
                if lin_bias is not None:
                    module.lin.bias = nn.Parameter(lin_bias)
            else:
                if layer_1_no_self_loops or layer_2_no_self_loops:
                    add_self_loops = False
                else:
                    add_self_loops = True
                module = geom_nn.conv.gat_conv.GATConv(**params_dict, bias=True, add_self_loops=add_self_loops)
                if lin_bias is not None:
                    module.lin.bias = nn.Parameter(lin_bias)
            module.lin.weight = nn.Parameter(lin_weight)
            module.att_edge = nn.Parameter(att_edge)

        if bias is not None:
            module.bias = nn.Parameter(bias)
        if edge_dim is not None:
            module.lin_edge.weight = nn.Parameter(lin_edge_weight)

        # print('GCNConv custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        if is_GATv2:
            lin_l = LRPLinear(module.lin_l, EB_Linear_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                              ref_name=ref_name + '_lin_l')
            lin_r = LRPLinear(module.lin_r, EB_Linear_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                              ref_name=ref_name + '_lin_r')
            module.lin_l = lin_l
            module.lin_r = lin_r
        else:
            lin = LRPLinear(module.lin, EB_Linear_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                            ref_name=ref_name + '_lin')
            module.lin = lin
        if edge_dim is not None:
            lin_edge = LRPLinear(module.lin_edge, EB_Linear_Autograd_Fn, params={'linear_eps': eps}, saved_rels=saved_rels,
                                 ref_name=ref_name + '_lin_edge')
            module.lin_edge = lin_edge
        # print(lin.module.weight.device)
        X = input_.clone().detach().requires_grad_(True)
        if skip_layer_2:
            edge_index = torch.as_tensor([[], []], dtype=edge_index.dtype, device=edge_index.device)
        R = eb_gatconv(input_=X,
                        edge_index=edge_index,
                        layer=module,
                        relevance_output=grad_output)
        # print('GCNConv custom R', R.shape)
        return R, None, None, None, None


def lrp_gatconv(input_, edge_index, layer, relevance_output, eps0, eps):
    if input_.grad is not None:
        input_.grad.zero()
    relevance_output_data = relevance_output.clone().detach()
    layer.to(input_.device)
    with torch.enable_grad():
        Z = layer(input_, edge_index)
    S = safe_divide(relevance_output_data, Z.clone().detach(), eps0, eps)
    Z.backward(S)
    relevance_input = input_.data * input_.grad
    return relevance_input


def cam_gatconv(input_, edge_index, layer, relevance_output):
    if input_.grad is not None:
        input_.grad.zero()
    relevance_output_data = relevance_output.clone().detach()
    layer.to(input_.device)
    with torch.enable_grad():
        Z = layer(input_, edge_index)
    Z.backward(relevance_output_data)
    relevance_input = F.relu(input_.data * input_.grad)
    layer.saved_relevance = relevance_output
    return relevance_input


def eb_gatconv(input_, edge_index, layer, relevance_output):
    if input_.grad is not None:
        input_.grad.zero()
    layer.to(input_.device)
    with torch.enable_grad():
        Z = layer(input_, edge_index)  # X = W^{+}^T * A_{n}
    relevance_output_data = relevance_output.clone().detach()  # P_{n-1}
    # print(Z.clone().detach().shape)
    X = Z.clone().detach()
    Y = relevance_output_data / X  # Y = P_{n-1} (/) X
    # print('gcnconv: ', X.shape, Y.shape)
    Z.backward(Y)  # Use backward pass to compute Z = W^{+} * Y
    relevance_input = input_.data * input_.grad  # P_{n} = A_{n} (*) Z
    layer.saved_relevance = relevance_output
    return relevance_input
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from lrp_pytorch.modules.base import safe_divide
from lrp_pytorch.modules.gcn_conv import LRPGCNConv, GCNConv_Autograd_Fn, CAM_GCNConv_Autograd_Fn, \
    EB_GCNConv_Autograd_Fn
from lrp_pytorch.modules.linear import LRPLinear, Linear_Epsilon_Autograd_Fn, CAM_Linear_Autograd_Fn, \
    EB_Linear_Autograd_Fn, C_EB_Linear_Autograd_Fn
from model.Twitter.BiGCN_Twitter import TDrumorGCN, BUrumorGCN
from torch_geometric.data import Data


class LRPBiGCNRumourGCN(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(LRPBiGCNRumourGCN, self).__init__()
        self.module = module
        self.autograd_fn = autograd_fn
        self.module.is_BU = kwargs.get('is_BU', False)
        self.params = kwargs['params']
        self.module.x_in = []

    def forward(self, data, skip_layer_2=False, layer_1_no_self_loops=False, layer_2_no_self_loops=False):
        x = data.x
        try:
            x.requires_grad = True
        except:
            pass
        edge_index = data.edge_index
        BU_edge_index = data.BU_edge_index
        rootindex = data.rootindex
        root = data.root
        self.params['skip_layer_2'] = skip_layer_2
        self.params['layer_1_no_self_loops'] = layer_1_no_self_loops
        self.params['layer_2_no_self_loops'] = layer_2_no_self_loops
        return self.autograd_fn.apply(x, edge_index, BU_edge_index, rootindex, root, self.module, self.params)


class BiGCNRumourGCN_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, edge_index, BU_edge_index, rootindex, root, module, params):
        eps = params.get('bigcn', 1e-9)
        skip_layer_2 = params.get('skip_layer_2', False)
        layer_1_no_self_loops = params.get('layer_1_no_self_loops', False)
        layer_2_no_self_loops = params.get('layer_2_no_self_loops', False)

        def config_values_to_tensors(module):
            if isinstance(module, TDrumorGCN) or isinstance(module, BUrumorGCN):
                property_names0 = ['device']
            else:
                print('Error: module not BiGCN TDrumorGCN layer')
                raise Exception

            if isinstance(module.conv1, geom_nn.GCNConv):
                property_names1 = ['in_channels', 'out_channels']
            else:
                print('Error: module not GCNConv layer')
                raise Exception

            if isinstance(module.conv2, geom_nn.GCNConv):
                property_names2 = ['out_channels']
            else:
                print('Error: module not GCNConv layer')
                raise Exception

            property_names = property_names0 + property_names1 + property_names2

            values = []
            for attr in property_names0:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.conv1.lin.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.conv1.lin.weight.device)
                else:
                    # print('error: property value is neither int nor tuple', attr, value)
                    value = torch.zeros(1, dtype=torch.int32, device=module.conv1.lin.weight.device)
                    # exit()
                values.append(value)
            for attr in property_names1:
                value = getattr(module.conv1, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.conv1.lin.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.conv1.lin.weight.device)
                else:
                    print('error: property value is neither int nor tuple', attr, value)
                    # exit()
                values.append(value)
            for attr in property_names2:
                value = getattr(module.conv2, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.conv2.lin.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.conv2.lin.weight.device)
                else:
                    print('error: property value is neither int nor tuple', attr, value)
                    # exit()
                values.append(value)
            return property_names, values

        property_names, values = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)
        # print(eps_tensor)

        if module.conv1.lin.bias is None:
            conv1_lin_bias = None
        else:
            conv1_lin_bias = module.conv1.lin.bias.data.clone()

        if module.conv2.lin.bias is None:
            conv2_lin_bias = None
        else:
            conv2_lin_bias = module.conv2.lin.bias.data.clone()

        ctx.save_for_backward(x, edge_index, BU_edge_index, rootindex, root,
                              module.conv1.lin.weight.data.clone(), conv1_lin_bias,
                              module.conv2.lin.weight.data.clone(), conv2_lin_bias,
                              eps_tensor, *values)
        ctx.saved_rels = module.saved_rels
        ctx.is_BU = module.is_BU
        ctx.skip_layer_2 = skip_layer_2
        ctx.layer_1_no_self_loops = layer_1_no_self_loops
        ctx.layer_2_no_self_loops = layer_2_no_self_loops

        # print('BiGCNRumourGCN ctx.needs_input_grad', ctx.needs_input_grad)

        data = Data(x=x, edge_index=edge_index, BU_edge_index=BU_edge_index, rootindex=rootindex, root=root,
                    batch=torch.zeros(x.shape[0]).long().to(x.device))
        data.requires_grad = True
        return module.forward(data)

    @staticmethod
    def backward(ctx, grad_output):
        input_, edge_index, BU_edge_index, rootindex, root, \
        conv1_lin_weight, conv1_lin_bias, conv2_lin_weight, conv2_lin_bias, \
        eps_tensor, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        is_BU = ctx.is_BU
        skip_layer_2 = ctx.skip_layer_2
        layer_1_no_self_loops = ctx.layer_1_no_self_loops
        layer_2_no_self_loops = ctx.layer_2_no_self_loops
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names0 = ['device']
            property_names1 = ['in_feats', 'hid_feats']
            property_names2 = ['out_feats']
            # params_dict0 = {}
            # params_dict1 = {}
            # params_dict2 = {}
            params_dict = {}

            property_names = property_names0 + property_names1 + property_names2

            for i, property_name in enumerate(property_names):
                value = values[i]
                # if i == 0:
                #     params_dict = params_dict0
                # elif 1 <= i <= 2:
                #     params_dict = params_dict1
                # else:
                #     params_dict = params_dict2
                if property_name == 'device':
                    params_dict[property_name] = value.device
                elif value.numel == 1:
                    params_dict[property_name] = value.item()
                else:
                    value_list = value.tolist()
                    if len(value_list) == 1:
                        params_dict[property_name] = value_list[0]
                    else:
                        params_dict[property_name] = tuple(value_list)
            return params_dict

        params_dict = tensors_to_dict(values)

        if is_BU:
            module = BUrumorGCN(**params_dict)
        else:
            module = TDrumorGCN(**params_dict)

        if conv1_lin_bias is None:
            pass
        else:
            module.conv1.lin.bias = nn.Parameter(conv1_lin_bias)
        module.conv1.lin.weight = nn.Parameter(conv1_lin_weight)

        if conv2_lin_bias is None:
            pass
        else:
            module.conv2.lin.bias = nn.Parameter(conv2_lin_bias)
        module.conv2.lin.weight = nn.Parameter(conv2_lin_weight)

        # print('BiGCNRumourGCN custom input_.shape', input_.shape)
        eps = eps_tensor.item()
        # print(module.conv1.lin.weight, module.conv2.lin.weight)

        if is_BU:
            module.conv1.ref_name = 'bu_conv1'
            module.conv2.ref_name = 'bu_conv2'
        else:
            module.conv1.ref_name = 'td_conv1'
            module.conv2.ref_name = 'td_conv2'
        skip_layer_2 = ctx.skip_layer_2
        layer_1_no_self_loops = ctx.layer_1_no_self_loops
        layer_2_no_self_loops = ctx.layer_2_no_self_loops
        conv1 = LRPGCNConv(module.conv1, GCNConv_Autograd_Fn, params={'gcn_conv': eps,
                                                                      'layer_1_no_self_loops': layer_1_no_self_loops},
                           saved_rels=saved_rels)
        conv2 = LRPGCNConv(module.conv2, GCNConv_Autograd_Fn, params={'gcn_conv': eps,
                                                                      'skip_layer_2': skip_layer_2,
                                                                      'layer_2_no_self_loops': layer_2_no_self_loops},
                           saved_rels=saved_rels)
        module.conv1 = conv1
        module.conv2 = conv2

        # print(conv1.module.lin.weight.device, conv2.module.lin.weight.device)

        X = input_.clone().detach().requires_grad_(True)
        # print(X.device, edge_index.device, BU_edge_index.device, rootindex.device, input_.device)
        data = Data(x=X, edge_index=edge_index, BU_edge_index=BU_edge_index, rootindex=rootindex, root=root,
                    batch=torch.zeros(input_.shape[0]).long().to(input_.device))
        # print(data.batch.device)
        # R = lrp_rumourgcn(input_=data,
        #                   layer=module,
        #                   relevance_output=grad_output,
        #                   eps0=eps,
        #                   eps=eps)
        R = lrp_rumourgcn(input_=data,
                          layer=module,
                          relevance_output=grad_output,
                          eps0=eps,
                          eps=0)
        # print('BiGCNRumourGCN custom R', R.shape)
        return R, None, None, None, None, None, None


class CAM_BiGCNRumourGCN_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, edge_index, BU_edge_index, rootindex, root, module, params):
        eps = params.get('bigcn', 1e-6)

        def config_values_to_tensors(module):
            if isinstance(module, TDrumorGCN) or isinstance(module, BUrumorGCN):
                property_names0 = ['device']
            else:
                print('Error: module not BiGCN TDrumorGCN layer')
                raise Exception

            if isinstance(module.conv1, geom_nn.GCNConv):
                property_names1 = ['in_channels', 'out_channels']
            else:
                print('Error: module not GCNConv layer')
                raise Exception

            if isinstance(module.conv2, geom_nn.GCNConv):
                property_names2 = ['out_channels']
            else:
                print('Error: module not GCNConv layer')
                raise Exception

            property_names = property_names0 + property_names1 + property_names2

            values = []
            for attr in property_names0:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.conv1.lin.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.conv1.lin.weight.device)
                else:
                    # print('error: property value is neither int nor tuple', attr, value)
                    value = torch.zeros(1, dtype=torch.int32, device=module.conv1.lin.weight.device)
                    # exit()
                values.append(value)
            for attr in property_names1:
                value = getattr(module.conv1, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.conv1.lin.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.conv1.lin.weight.device)
                else:
                    print('error: property value is neither int nor tuple', attr, value)
                    # exit()
                values.append(value)
            for attr in property_names2:
                value = getattr(module.conv2, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.conv2.lin.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.conv2.lin.weight.device)
                else:
                    print('error: property value is neither int nor tuple', attr, value)
                    # exit()
                values.append(value)
            return property_names, values

        property_names, values = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)
        # print(eps_tensor)

        if module.conv1.lin.bias is None:
            conv1_lin_bias = None
        else:
            conv1_lin_bias = module.conv1.lin.bias.data.clone()

        if module.conv2.lin.bias is None:
            conv2_lin_bias = None
        else:
            conv2_lin_bias = module.conv2.lin.bias.data.clone()

        ctx.save_for_backward(x, edge_index, BU_edge_index, rootindex, root,
                              module.conv1.lin.weight.data.clone(), conv1_lin_bias,
                              module.conv2.lin.weight.data.clone(), conv2_lin_bias,
                              eps_tensor, *values)
        ctx.saved_rels = module.saved_rels
        ctx.is_BU = module.is_BU

        # print('BiGCNRumourGCN ctx.needs_input_grad', ctx.needs_input_grad)

        data = Data(x=x, edge_index=edge_index, BU_edge_index=BU_edge_index, rootindex=rootindex, root=root,
                    batch=torch.zeros(x.shape[0]).long().to(x.device))
        data.requires_grad = True
        return module.forward(data)

    @staticmethod
    def backward(ctx, grad_output):
        input_, edge_index, BU_edge_index, rootindex, root, \
        conv1_lin_weight, conv1_lin_bias, conv2_lin_weight, conv2_lin_bias, \
        eps_tensor, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        is_BU = ctx.is_BU
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names0 = ['device']
            property_names1 = ['in_feats', 'hid_feats']
            property_names2 = ['out_feats']
            # params_dict0 = {}
            # params_dict1 = {}
            # params_dict2 = {}
            params_dict = {}

            property_names = property_names0 + property_names1 + property_names2

            for i, property_name in enumerate(property_names):
                value = values[i]
                # if i == 0:
                #     params_dict = params_dict0
                # elif 1 <= i <= 2:
                #     params_dict = params_dict1
                # else:
                #     params_dict = params_dict2
                if property_name == 'device':
                    params_dict[property_name] = value.device
                elif value.numel == 1:
                    params_dict[property_name] = value.item()
                else:
                    value_list = value.tolist()
                    if len(value_list) == 1:
                        params_dict[property_name] = value_list[0]
                    else:
                        params_dict[property_name] = tuple(value_list)
            return params_dict

        params_dict = tensors_to_dict(values)

        if is_BU:
            module = BUrumorGCN(**params_dict)
        else:
            module = TDrumorGCN(**params_dict)

        if conv1_lin_bias is None:
            pass
        else:
            module.conv1.lin.bias = nn.Parameter(conv1_lin_bias)
        module.conv1.lin.weight = nn.Parameter(conv1_lin_weight)

        if conv2_lin_bias is None:
            pass
        else:
            module.conv2.lin.bias = nn.Parameter(conv2_lin_bias)
        module.conv2.lin.weight = nn.Parameter(conv2_lin_weight)

        # print('BiGCNRumourGCN custom input_.shape', input_.shape)
        eps = eps_tensor.item()
        # print(module.conv1.lin.weight, module.conv2.lin.weight)

        if is_BU:
            module.conv1.ref_name = 'bu_conv1'
            module.conv2.ref_name = 'bu_conv2'
        else:
            module.conv1.ref_name = 'td_conv1'
            module.conv2.ref_name = 'td_conv2'
        conv1 = LRPGCNConv(module.conv1, CAM_GCNConv_Autograd_Fn, params={'gcn_conv': eps}, saved_rels=saved_rels)
        conv2 = LRPGCNConv(module.conv2, CAM_GCNConv_Autograd_Fn, params={'gcn_conv': eps}, saved_rels=saved_rels)
        module.conv1 = conv1
        module.conv2 = conv2

        # print(conv1.module.lin.weight.device, conv2.module.lin.weight.device)

        X = input_.clone().detach().requires_grad_(True)
        # print(X.device, edge_index.device, BU_edge_index.device, rootindex.device, input_.device)
        data = Data(x=X, edge_index=edge_index, BU_edge_index=BU_edge_index, rootindex=rootindex, root=root,
                    batch=torch.zeros(input_.shape[0]).long().to(input_.device))
        # print(data.batch.device)
        R = cam_rumourgcn(input_=data,
                          layer=module,
                          relevance_output=grad_output)
        # print('BiGCNRumourGCN custom R', R.shape)
        return R, None, None, None, None, None, None


class EB_BiGCNRumourGCN_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, edge_index, BU_edge_index, rootindex, root, module, params):
        eps = params.get('bigcn', 1e-6)

        def config_values_to_tensors(module):
            if isinstance(module, TDrumorGCN) or isinstance(module, BUrumorGCN):
                property_names0 = ['device']
            else:
                print('Error: module not BiGCN TDrumorGCN layer')
                raise Exception

            if isinstance(module.conv1, geom_nn.GCNConv):
                property_names1 = ['in_channels', 'out_channels']
            else:
                print('Error: module not GCNConv layer')
                raise Exception

            if isinstance(module.conv2, geom_nn.GCNConv):
                property_names2 = ['out_channels']
            else:
                print('Error: module not GCNConv layer')
                raise Exception

            property_names = property_names0 + property_names1 + property_names2

            values = []
            for attr in property_names0:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.conv1.lin.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.conv1.lin.weight.device)
                else:
                    # print('error: property value is neither int nor tuple', attr, value)
                    value = torch.zeros(1, dtype=torch.int32, device=module.conv1.lin.weight.device)
                    # exit()
                values.append(value)
            for attr in property_names1:
                value = getattr(module.conv1, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.conv1.lin.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.conv1.lin.weight.device)
                else:
                    print('error: property value is neither int nor tuple', attr, value)
                    # exit()
                values.append(value)
            for attr in property_names2:
                value = getattr(module.conv2, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.conv2.lin.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.conv2.lin.weight.device)
                else:
                    print('error: property value is neither int nor tuple', attr, value)
                    # exit()
                values.append(value)
            return property_names, values

        property_names, values = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)
        # print(eps_tensor)

        if module.conv1.lin.bias is None:
            conv1_lin_bias = None
        else:
            conv1_lin_bias = module.conv1.lin.bias.data.clone()

        if module.conv2.lin.bias is None:
            conv2_lin_bias = None
        else:
            conv2_lin_bias = module.conv2.lin.bias.data.clone()

        ctx.save_for_backward(x, edge_index, BU_edge_index, rootindex, root,
                              module.conv1.lin.weight.data.clone(), conv1_lin_bias,
                              module.conv2.lin.weight.data.clone(), conv2_lin_bias,
                              eps_tensor, *values)
        ctx.saved_rels = module.saved_rels
        ctx.is_BU = module.is_BU

        # print('BiGCNRumourGCN ctx.needs_input_grad', ctx.needs_input_grad)

        data = Data(x=x, edge_index=edge_index, BU_edge_index=BU_edge_index, rootindex=rootindex, root=root,
                    batch=torch.zeros(x.shape[0]).long().to(x.device))
        data.requires_grad = True
        return module.forward(data)

    @staticmethod
    def backward(ctx, grad_output):
        input_, edge_index, BU_edge_index, rootindex, root, \
        conv1_lin_weight, conv1_lin_bias, conv2_lin_weight, conv2_lin_bias, \
        eps_tensor, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        is_BU = ctx.is_BU
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names0 = ['device']
            property_names1 = ['in_feats', 'hid_feats']
            property_names2 = ['out_feats']
            # params_dict0 = {}
            # params_dict1 = {}
            # params_dict2 = {}
            params_dict = {}

            property_names = property_names0 + property_names1 + property_names2

            for i, property_name in enumerate(property_names):
                value = values[i]
                # if i == 0:
                #     params_dict = params_dict0
                # elif 1 <= i <= 2:
                #     params_dict = params_dict1
                # else:
                #     params_dict = params_dict2
                if property_name == 'device':
                    params_dict[property_name] = value.device
                elif value.numel == 1:
                    params_dict[property_name] = value.item()
                else:
                    value_list = value.tolist()
                    if len(value_list) == 1:
                        params_dict[property_name] = value_list[0]
                    else:
                        params_dict[property_name] = tuple(value_list)
            return params_dict

        params_dict = tensors_to_dict(values)

        if is_BU:
            module = BUrumorGCN(**params_dict)
        else:
            module = TDrumorGCN(**params_dict)

        if conv1_lin_bias is None:
            pass
        else:
            module.conv1.lin.bias = nn.Parameter(conv1_lin_bias)
        module.conv1.lin.weight = nn.Parameter(conv1_lin_weight)

        if conv2_lin_bias is None:
            pass
        else:
            module.conv2.lin.bias = nn.Parameter(conv2_lin_bias)
        module.conv2.lin.weight = nn.Parameter(conv2_lin_weight)

        # print('BiGCNRumourGCN custom input_.shape', input_.shape)
        eps = eps_tensor.item()
        # print(module.conv1.lin.weight, module.conv2.lin.weight)

        if is_BU:
            module.conv1.ref_name = 'bu_conv1'
            module.conv2.ref_name = 'bu_conv2'
        else:
            module.conv1.ref_name = 'td_conv1'
            module.conv2.ref_name = 'td_conv2'
        conv1 = LRPGCNConv(module.conv1, EB_GCNConv_Autograd_Fn, params={'gcn_conv': eps}, saved_rels=saved_rels)
        conv2 = LRPGCNConv(module.conv2, EB_GCNConv_Autograd_Fn, params={'gcn_conv': eps}, saved_rels=saved_rels)
        module.conv1 = conv1
        module.conv2 = conv2

        # print(conv1.module.lin.weight.device, conv2.module.lin.weight.device)

        X = input_.clone().detach().requires_grad_(True)
        # print(X.device, edge_index.device, BU_edge_index.device, rootindex.device, input_.device)
        data = Data(x=X, edge_index=edge_index, BU_edge_index=BU_edge_index, rootindex=rootindex, root=root,
                    batch=torch.zeros(input_.shape[0]).long().to(input_.device))
        # print(data.batch.device)
        R = eb_rumourgcn(input_=data,
                         layer=module,
                         relevance_output=grad_output)
        # print('BiGCNRumourGCN custom R', R.shape, data.x.grad.shape)
        # print(saved_rels.keys())
        return R, None, None, None, None, None, None


def lrp_rumourgcn(input_, layer, relevance_output, eps0, eps):
    if input_.x.grad is not None:
        input_.x.grad.zero()
    relevance_output_data = relevance_output.clone().detach()
    # print(input_)
    with torch.enable_grad():
        Z = layer(input_)
    # print(Z)
    S = safe_divide(relevance_output_data, Z.clone().detach(), eps0, eps)
    Z.backward(S)
    relevance_input = input_.x.data * input_.x.grad
    return relevance_input


def cam_rumourgcn(input_, layer, relevance_output):
    if input_.x.grad is not None:
        input_.x.grad.zero()
    relevance_output_data = relevance_output.clone().detach()
    with torch.enable_grad():
        Z = layer(input_)
    Z.backward(relevance_output_data)
    relevance_input = F.relu(input_.x.data * input_.x.grad)
    # layer.saved_relevance = relevance_output
    return relevance_input


def eb_rumourgcn(input_, layer, relevance_output):
    if input_.x.grad is not None:
        input_.x.grad.zero()
    with torch.enable_grad():
        Z = layer(input_)  # X = W^{+}^T * A_{n}
    relevance_output_data = relevance_output.clone().detach()  # P_{n-1}
    X = Z.clone().detach()
    Y = relevance_output_data / X  # Y = P_{n-1} (/) X
    # print('gcn: ', X.shape, Y.shape)
    Z.backward(Y)  # Use backward pass to compute Z = W^{+} * Y
    relevance_input = input_.x.data * input_.x.grad  # P_{n} = A_{n} (*) Z
    # layer.saved_relevance = relevance_input
    # print(relevance_input.shape, input_.x.grad.shape)
    return relevance_input


class LRPBiGCN(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(LRPBiGCN, self).__init__()
        # self.module = module
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']
        self.saved_rels = {}
        module.TDrumorGCN.saved_rels = self.saved_rels
        module.BUrumorGCN.saved_rels = self.saved_rels

        self.TDrumorGCN = LRPBiGCNRumourGCN(module.TDrumorGCN, BiGCNRumourGCN_Autograd_Fn, params=self.params,
                                            is_BU=False)
        self.BUrumorGCN = LRPBiGCNRumourGCN(module.BUrumorGCN, BiGCNRumourGCN_Autograd_Fn, params=self.params,
                                            is_BU=True)
        self.fc = LRPLinear(module.fc, Linear_Epsilon_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                            ref_name='fc')

    def forward(self, data, skip_layer_2=False, layer_1_no_self_loops=False, layer_2_no_self_loops=False):
        TD_x = self.TDrumorGCN(data, skip_layer_2=skip_layer_2, layer_1_no_self_loops=layer_1_no_self_loops,
                               layer_2_no_self_loops=layer_2_no_self_loops)
        BU_x = self.BUrumorGCN(data, skip_layer_2=skip_layer_2, layer_1_no_self_loops=layer_1_no_self_loops,
                               layer_2_no_self_loops=layer_2_no_self_loops)
        self.x = torch.cat((BU_x, TD_x), 1)
        self.out = self.fc(self.x)
        x = F.log_softmax(self.out, dim=1)
        return x


class CAM_BiGCN(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(CAM_BiGCN, self).__init__()
        # self.module = module
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']
        self.saved_rels = {}
        module.TDrumorGCN.saved_rels = self.saved_rels
        module.BUrumorGCN.saved_rels = self.saved_rels

        self.TDrumorGCN = LRPBiGCNRumourGCN(module.TDrumorGCN, CAM_BiGCNRumourGCN_Autograd_Fn, params=self.params,
                                            is_BU=False)
        self.BUrumorGCN = LRPBiGCNRumourGCN(module.BUrumorGCN, CAM_BiGCNRumourGCN_Autograd_Fn, params=self.params,
                                            is_BU=True)
        self.fc = LRPLinear(module.fc, CAM_Linear_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                            ref_name='fc')

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        self.x = torch.cat((BU_x, TD_x), 1)
        self.out = self.fc(self.x)
        x = F.log_softmax(self.out, dim=1)
        return x


class EB_BiGCN(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(EB_BiGCN, self).__init__()
        # self.module = module
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']
        self.is_contrastive = kwargs.get('is_contrastive', False)
        self.saved_rels = dict()
        module.TDrumorGCN.saved_rels = self.saved_rels
        module.BUrumorGCN.saved_rels = self.saved_rels

        self.TDrumorGCN = LRPBiGCNRumourGCN(module.TDrumorGCN, EB_BiGCNRumourGCN_Autograd_Fn, params=self.params,
                                            is_BU=False)
        self.BUrumorGCN = LRPBiGCNRumourGCN(module.BUrumorGCN, EB_BiGCNRumourGCN_Autograd_Fn, params=self.params,
                                            is_BU=True)
        if self.is_contrastive:
            with torch.no_grad():  # contrastive, flip weights
                module.fc.weight.copy_(-module.fc.weight.float())
            self.fc = LRPLinear(module.fc, EB_Linear_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                                ref_name='fc')
        else:
            self.fc = LRPLinear(module.fc, C_EB_Linear_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                                ref_name='fc')

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        self.x = torch.cat((BU_x, TD_x), 1)
        self.out = self.fc(self.x)
        x = F.log_softmax(self.out, dim=1)
        return x
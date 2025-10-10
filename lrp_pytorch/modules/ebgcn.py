import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_geometric.nn as geom_nn
from lrp_pytorch.modules.base import safe_divide
from lrp_pytorch.modules.gcn_conv import LRPGCNConv, GCNConv_Autograd_Fn, CAM_GCNConv_Autograd_Fn, \
    EB_GCNConv_Autograd_Fn
from lrp_pytorch.modules.linear import LRPLinear, Linear_Epsilon_Autograd_Fn, CAM_Linear_Autograd_Fn, \
    EB_Linear_Autograd_Fn, C_EB_Linear_Autograd_Fn
from lrp_pytorch.modules.conv import LRPConv1d, Conv1d_Autograd_Fn, CAM_Conv1d_Autograd_Fn, EB_Conv1d_Autograd_Fn
from lrp_pytorch.modules.batchnorm import LRPBatchNorm1d, BatchNorm1d_Autograd_Fn, CAM_BatchNorm1d_Autograd_Fn, \
    EB_BatchNorm1d_Autograd_Fn
from model.Twitter.EBGCN import TDrumorGCN, BUrumorGCN
from torch_geometric.data import Data
from argparse import ArgumentParser


class LRPEBGCNRumourGCN(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(LRPEBGCNRumourGCN, self).__init__()
        self.module = module
        self.autograd_fn = autograd_fn
        self.module.is_BU = kwargs.get('is_BU', False)
        self.params = kwargs['params']

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


class EBGCNRumourGCN_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, edge_index, BU_edge_index, rootindex, root, module, params):
        eps = params.get('ebgcn', 1e-6)
        skip_layer_2 = params.get('skip_layer_2', False)
        layer_1_no_self_loops = params.get('layer_1_no_self_loops', False)
        layer_2_no_self_loops = params.get('layer_2_no_self_loops', False)

        def config_values_to_tensors(module):
            if isinstance(module, TDrumorGCN) or isinstance(module, BUrumorGCN):
                property_names = ['input_features', 'hidden_features', 'output_features', 'edge_num',
                                  'dropout', 'edge_infer_td', 'edge_infer_bu', 'device', 'training']
            else:
                print('Error: module not EBGCN TDrumorGCN or BUrumorGCN layer')
                raise Exception

            values = []
            for attr in property_names:
                value = getattr(module.args, attr)
                if attr == 'device':
                    value = torch.zeros(1, dtype=torch.int32, device=module.fc1.weight.device)
                elif isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.fc1.weight.device)
                elif isinstance(value, float):
                    value = torch.tensor([value], dtype=torch.float, device=module.fc1.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.fc1.weight.device)
                elif isinstance(value, bool):
                    value = torch.tensor(value, dtype=torch.bool, device=module.fc1.weight.device)
                else:
                    print('error: property value is neither int nor tuple', attr, value)
                    # exit()
                values.append(value)
            return property_names, values

        property_names, values = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)
        # print(eps_tensor)

        # GCNConv 1
        if module.conv1.lin.bias is None:
            conv1_lin_bias = None
        else:
            conv1_lin_bias = module.conv1.lin.bias.data.clone()
        conv1_lin_weight = module.conv1.lin.weight.data.clone()

        # GCNConv 2
        if module.conv2.lin.bias is None:
            conv2_lin_bias = None
        else:
            conv2_lin_bias = module.conv2.lin.bias.data.clone()
        conv2_lin_weight = module.conv2.lin.weight.data.clone()

        # sim_network
        # print(module.sim_network)
        if module.sim_network.sim_valconv0.bias is None:
            sim_network_sim_valconv0_bias = None
        else:
            sim_network_sim_valconv0_bias = module.sim_network.sim_valconv0.bias.data.clone()
        sim_network_sim_valconv0_weight = module.sim_network.sim_valconv0.weight.data.clone()
        if module.sim_network.sim_valnorm0.bias is None:
            sim_network_sim_valnorm0_bias = None
        else:
            sim_network_sim_valnorm0_bias = module.sim_network.sim_valnorm0.bias.data.clone()
        sim_network_sim_valnorm0_weight = module.sim_network.sim_valnorm0.weight.data.clone()
        sim_network_sim_valnorm0_running_mean = module.sim_network.sim_valnorm0.running_mean.data.clone()
        sim_network_sim_valnorm0_running_var = module.sim_network.sim_valnorm0.running_var.data.clone()
        if module.sim_network.sim_valconv_out.bias is None:
            sim_network_sim_valconv_out_bias = None
        else:
            sim_network_sim_valconv_out_bias = module.sim_network.sim_valconv_out.bias.data.clone()
        sim_network_sim_valconv_out_weight = module.sim_network.sim_valconv_out.weight.data.clone()

        # W_mean
        if module.W_mean.W_meanconv0.bias is None:
            W_mean_W_meanconv0_bias = None
        else:
            W_mean_W_meanconv0_bias = module.W_mean.W_meanconv0.bias.data.clone()
        W_mean_W_meanconv0_weight = module.W_mean.W_meanconv0.weight.data.clone()
        if module.W_mean.W_meannorm0.bias is None:
            W_mean_W_meannorm0_bias = None
        else:
            W_mean_W_meannorm0_bias = module.W_mean.W_meannorm0.bias.data.clone()
        W_mean_W_meannorm0_weight = module.W_mean.W_meannorm0.weight.data.clone()
        W_mean_W_meannorm0_running_mean = module.W_mean.W_meannorm0.running_mean.data.clone()
        W_mean_W_meannorm0_running_var = module.W_mean.W_meannorm0.running_var.data.clone()
        if module.W_mean.W_meanconv_out.bias is None:
            W_mean_W_meanconv_out_bias = None
        else:
            W_mean_W_meanconv_out_bias = module.W_mean.W_meanconv_out.bias.data.clone()
        W_mean_W_meanconv_out_weight = module.W_mean.W_meanconv_out.weight.data.clone()

        # W_bias
        if module.W_bias.W_biasconv0.bias is None:
            W_bias_W_biasconv0_bias = None
        else:
            W_bias_W_biasconv0_bias = module.W_bias.W_biasconv0.bias.data.clone()
        W_bias_W_biasconv0_weight = module.W_bias.W_biasconv0.weight.data.clone()
        if module.W_bias.W_biasnorm0.bias is None:
            W_bias_W_biasnorm0_bias = None
        else:
            W_bias_W_biasnorm0_bias = module.W_bias.W_biasnorm0.bias.data.clone()
        W_bias_W_biasnorm0_weight = module.W_bias.W_biasnorm0.weight.data.clone()
        W_bias_W_biasnorm0_running_mean = module.W_bias.W_biasnorm0.running_mean.data.clone()
        W_bias_W_biasnorm0_running_var = module.W_bias.W_biasnorm0.running_var.data.clone()
        if module.W_bias.W_biasconv_out.bias is None:
            W_bias_W_biasconv_out_bias = None
        else:
            W_bias_W_biasconv_out_bias = module.W_bias.W_biasconv_out.bias.data.clone()
        W_bias_W_biasconv_out_weight = module.W_bias.W_biasconv_out.weight.data.clone()

        # B_mean
        if module.B_mean.B_meanconv0.bias is None:
            B_mean_B_meanconv0_bias = None
        else:
            B_mean_B_meanconv0_bias = module.B_mean.B_meanconv0.bias.data.clone()
        B_mean_B_meanconv0_weight = module.B_mean.B_meanconv0.weight.data.clone()
        if module.B_mean.B_meannorm0.bias is None:
            B_mean_B_meannorm0_bias = None
        else:
            B_mean_B_meannorm0_bias = module.B_mean.B_meannorm0.bias.data.clone()
        B_mean_B_meannorm0_weight = module.B_mean.B_meannorm0.weight.data.clone()
        B_mean_B_meannorm0_running_mean = module.B_mean.B_meannorm0.running_mean.data.clone()
        B_mean_B_meannorm0_running_var = module.B_mean.B_meannorm0.running_var.data.clone()
        if module.B_mean.B_meanconv_out.bias is None:
            B_mean_B_meanconv_out_bias = None
        else:
            B_mean_B_meanconv_out_bias = module.B_mean.B_meanconv_out.bias.data.clone()
        B_mean_B_meanconv_out_weight = module.B_mean.B_meanconv_out.weight.data.clone()

        # B_bias
        if module.B_bias.B_biasconv0.bias is None:
            B_bias_B_biasconv0_bias = None
        else:
            B_bias_B_biasconv0_bias = module.B_bias.B_biasconv0.bias.data.clone()
        B_bias_B_biasconv0_weight = module.B_bias.B_biasconv0.weight.data.clone()
        if module.B_bias.B_biasnorm0.bias is None:
            B_bias_B_biasnorm0_bias = None
        else:
            B_bias_B_biasnorm0_bias = module.B_bias.B_biasnorm0.bias.data.clone()
        B_bias_B_biasnorm0_weight = module.B_bias.B_biasnorm0.weight.data.clone()
        B_bias_B_biasnorm0_running_mean = module.B_bias.B_biasnorm0.running_mean.data.clone()
        B_bias_B_biasnorm0_running_var = module.B_bias.B_biasnorm0.running_var.data.clone()
        if module.B_bias.B_biasconv_out.bias is None:
            B_bias_B_biasconv_out_bias = None
        else:
            B_bias_B_biasconv_out_bias = module.B_bias.B_biasconv_out.bias.data.clone()
        B_bias_B_biasconv_out_weight = module.B_bias.B_biasconv_out.weight.data.clone()

        # fc1
        if module.fc1.bias is None:
            fc1_bias = None
        else:
            fc1_bias = module.fc1.bias.data.clone()
        fc1_weight = module.fc1.weight.data.clone()

        # fc2
        if module.fc2.bias is None:
            fc2_bias = None
        else:
            fc2_bias = module.fc2.bias.data.clone()
        fc2_weight = module.fc2.weight.data.clone()

        # bn1
        if module.bn1.bias is None:
            bn1_bias = None
        else:
            bn1_bias = module.bn1.bias.data.clone()
        bn1_weight = module.bn1.weight.data.clone()
        bn1_running_mean = module.bn1.running_mean.data.clone()
        bn1_running_var = module.bn1.running_var.data.clone()

        ctx.save_for_backward(x, edge_index, BU_edge_index, rootindex, root,
                              conv1_lin_weight, conv1_lin_bias,
                              conv2_lin_weight, conv2_lin_bias,
                              sim_network_sim_valconv0_weight, sim_network_sim_valconv0_bias,
                              sim_network_sim_valnorm0_weight, sim_network_sim_valnorm0_bias,
                              sim_network_sim_valnorm0_running_mean, sim_network_sim_valnorm0_running_var,
                              sim_network_sim_valconv_out_weight, sim_network_sim_valconv_out_bias,
                              W_mean_W_meanconv0_weight, W_mean_W_meanconv0_bias,
                              W_mean_W_meannorm0_weight, W_mean_W_meannorm0_bias,
                              W_mean_W_meannorm0_running_mean, W_mean_W_meannorm0_running_var,
                              W_mean_W_meanconv_out_weight, W_mean_W_meanconv_out_bias,
                              W_bias_W_biasconv0_weight, W_bias_W_biasconv0_bias,
                              W_bias_W_biasnorm0_weight, W_bias_W_biasnorm0_bias,
                              W_bias_W_biasnorm0_running_mean, W_bias_W_biasnorm0_running_var,
                              W_bias_W_biasconv_out_weight, W_bias_W_biasconv_out_bias,
                              B_mean_B_meanconv0_weight, B_mean_B_meanconv0_bias,
                              B_mean_B_meannorm0_weight, B_mean_B_meannorm0_bias,
                              B_mean_B_meannorm0_running_mean, B_mean_B_meannorm0_running_var,
                              B_mean_B_meanconv_out_weight, B_mean_B_meanconv_out_bias,
                              B_bias_B_biasconv0_weight, B_bias_B_biasconv0_bias,
                              B_bias_B_biasnorm0_weight, B_bias_B_biasnorm0_bias,
                              B_bias_B_biasnorm0_running_mean, B_bias_B_biasnorm0_running_var,
                              B_bias_B_biasconv_out_weight, B_bias_B_biasconv_out_bias,
                              fc1_weight, fc1_bias,
                              fc2_weight, fc2_bias,
                              bn1_weight, bn1_bias,
                              bn1_running_mean, bn1_running_var,
                              eps_tensor, *values)
        ctx.saved_rels = module.saved_rels
        ctx.is_BU = module.is_BU
        ctx.skip_layer_2 = skip_layer_2
        ctx.layer_1_no_self_loops = layer_1_no_self_loops
        ctx.layer_2_no_self_loops = layer_2_no_self_loops

        # print('EBGCNRumourGCN ctx.needs_input_grad', ctx.needs_input_grad)

        data = Data(x=x, edge_index=edge_index, BU_edge_index=BU_edge_index, rootindex=rootindex, root=root,
                    batch=torch.zeros(x.shape[0]).long().to(x.device))
        # data.requires_grad = True
        return module.forward(data)

    @staticmethod
    def backward(ctx, grad_output, edge_loss_output):
        input_, edge_index, BU_edge_index, rootindex, root, \
        conv1_lin_weight, conv1_lin_bias, \
        conv2_lin_weight, conv2_lin_bias, \
        sim_network_sim_valconv0_weight, sim_network_sim_valconv0_bias, \
        sim_network_sim_valnorm0_weight, sim_network_sim_valnorm0_bias, \
        sim_network_sim_valnorm0_running_mean, sim_network_sim_valnorm0_running_var, \
        sim_network_sim_valconv_out_weight, sim_network_sim_valconv_out_bias, \
        W_mean_W_meanconv0_weight, W_mean_W_meanconv0_bias, \
        W_mean_W_meannorm0_weight, W_mean_W_meannorm0_bias, \
        W_mean_W_meannorm0_running_mean, W_mean_W_meannorm0_running_var, \
        W_mean_W_meanconv_out_weight, W_mean_W_meanconv_out_bias, \
        W_bias_W_biasconv0_weight, W_bias_W_biasconv0_bias, \
        W_bias_W_biasnorm0_weight, W_bias_W_biasnorm0_bias, \
        W_bias_W_biasnorm0_running_mean, W_bias_W_biasnorm0_running_var, \
        W_bias_W_biasconv_out_weight, W_bias_W_biasconv_out_bias, \
        B_mean_B_meanconv0_weight, B_mean_B_meanconv0_bias, \
        B_mean_B_meannorm0_weight, B_mean_B_meannorm0_bias, \
        B_mean_B_meannorm0_running_mean, B_mean_B_meannorm0_running_var, \
        B_mean_B_meanconv_out_weight, B_mean_B_meanconv_out_bias, \
        B_bias_B_biasconv0_weight, B_bias_B_biasconv0_bias, \
        B_bias_B_biasnorm0_weight, B_bias_B_biasnorm0_bias, \
        B_bias_B_biasnorm0_running_mean, B_bias_B_biasnorm0_running_var, \
        B_bias_B_biasconv_out_weight, B_bias_B_biasconv_out_bias, \
        fc1_weight, fc1_bias, \
        fc2_weight, fc2_bias, \
        bn1_weight, bn1_bias, \
        bn1_running_mean, bn1_running_var, \
        eps_tensor, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        is_BU = ctx.is_BU
        skip_layer_2 = ctx.skip_layer_2
        layer_1_no_self_loops = ctx.layer_1_no_self_loops
        layer_2_no_self_loops = ctx.layer_2_no_self_loops
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = ['input_features', 'hidden_features', 'output_features', 'edge_num',
                              'dropout', 'edge_infer_td', 'edge_infer_bu', 'device', 'training']
            params_dict = {}

            for i, property_name in enumerate(property_names):
                value = values[i]
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
            else:
                new_values = values[i+1:]
            return params_dict, new_values

        params_dict, values = tensors_to_dict(values)

        parser = ArgumentParser()
        args = parser.parse_args()
        for k, v in params_dict.items():
            args.__setattr__(k, v)

        if is_BU:
            module = BUrumorGCN(args).to(args.device)
        else:
            module = TDrumorGCN(args).to(args.device)

        # GCNConv 1
        if conv1_lin_bias is not None:
            module.conv1.lin.bias = nn.Parameter(conv1_lin_bias)
        module.conv1.lin.weight = nn.Parameter(conv1_lin_weight)

        # GCNConv 2
        if conv2_lin_bias is not None:
            module.conv2.lin.bias = nn.Parameter(conv2_lin_bias)
        module.conv2.lin.weight = nn.Parameter(conv2_lin_weight)

        # sim_network
        if sim_network_sim_valconv0_bias is not None:
            module.sim_network.sim_valconv0.bias = nn.Parameter(sim_network_sim_valconv0_bias)
        module.sim_network.sim_valconv0.weight = nn.Parameter(sim_network_sim_valconv0_weight)
        if sim_network_sim_valnorm0_bias is not None:
            module.sim_network.sim_valnorm0.bias = nn.Parameter(sim_network_sim_valnorm0_bias)
        module.sim_network.sim_valnorm0.weight = nn.Parameter(sim_network_sim_valnorm0_weight)
        module.sim_network.sim_valnorm0.running_mean = nn.Parameter(sim_network_sim_valnorm0_running_mean)
        module.sim_network.sim_valnorm0.running_var = nn.Parameter(sim_network_sim_valnorm0_running_var)
        if sim_network_sim_valconv_out_bias is not None:
            module.sim_network.sim_valconv_out.bias = nn.Parameter(sim_network_sim_valconv_out_bias)
        module.sim_network.sim_valconv_out.weight = nn.Parameter(sim_network_sim_valconv_out_weight)

        # W_mean
        if W_mean_W_meanconv0_bias is not None:
            module.W_mean.W_meanconv0.bias = nn.Parameter(W_mean_W_meanconv0_bias)
        module.W_mean.W_meanconv0.weight = nn.Parameter(W_mean_W_meanconv0_weight)
        if W_mean_W_meannorm0_bias is not None:
            module.W_mean.W_meannorm0.bias = nn.Parameter(W_mean_W_meannorm0_bias)
        module.W_mean.W_meannorm0.weight = nn.Parameter(W_mean_W_meannorm0_weight)
        module.W_mean.W_meannorm0.running_mean = nn.Parameter(W_mean_W_meannorm0_running_mean)
        module.W_mean.W_meannorm0.running_var = nn.Parameter(W_mean_W_meannorm0_running_var)
        if W_mean_W_meanconv_out_bias is not None:
            module.W_mean.W_meanconv_out.bias = nn.Parameter(W_mean_W_meanconv_out_bias)
        module.W_mean.W_meanconv_out.weight = nn.Parameter(W_mean_W_meanconv_out_weight)

        # W_bias
        if W_mean_W_meanconv0_bias is not None:
            module.W_mean.W_meanconv0.bias = nn.Parameter(W_mean_W_meanconv0_bias)
        module.W_mean.W_meanconv0.weight = nn.Parameter(W_mean_W_meanconv0_weight)
        if W_mean_W_meannorm0_bias is not None:
            module.W_mean.W_meannorm0.bias = nn.Parameter(W_mean_W_meannorm0_bias)
        module.W_mean.W_meannorm0.weight = nn.Parameter(W_mean_W_meannorm0_weight)
        module.W_mean.W_meannorm0.running_mean = nn.Parameter(W_mean_W_meannorm0_running_mean)
        module.W_mean.W_meannorm0.running_var = nn.Parameter(W_mean_W_meannorm0_running_var)
        if W_mean_W_meanconv_out_bias is not None:
            module.W_mean.W_meanconv_out.bias = nn.Parameter(W_mean_W_meanconv_out_bias)
        module.W_mean.W_meanconv_out.weight = nn.Parameter(W_mean_W_meanconv_out_weight)

        # B_mean
        if B_mean_B_meanconv0_bias is not None:
            module.B_mean.B_meanconv0.bias = nn.Parameter(B_mean_B_meanconv0_bias)
        module.B_mean.B_meanconv0.weight = nn.Parameter(B_mean_B_meanconv0_weight)
        if B_mean_B_meannorm0_bias is not None:
            module.B_mean.B_meannorm0.bias = nn.Parameter(B_mean_B_meannorm0_bias)
        module.B_mean.B_meannorm0.weight = nn.Parameter(B_mean_B_meannorm0_weight)
        module.B_mean.B_meannorm0.running_mean = nn.Parameter(B_mean_B_meannorm0_running_mean)
        module.B_mean.B_meannorm0.running_var = nn.Parameter(B_mean_B_meannorm0_running_var)
        if B_mean_B_meanconv_out_bias is not None:
            module.B_mean.B_meanconv_out.bias = nn.Parameter(B_mean_B_meanconv_out_bias)
        module.B_mean.B_meanconv_out.weight = nn.Parameter(B_mean_B_meanconv_out_weight)

        # B_bias
        if B_bias_B_biasconv0_bias is not None:
            module.B_bias.B_biasconv0.bias = nn.Parameter(B_bias_B_biasconv0_bias)
        module.B_bias.B_biasconv0.weight = nn.Parameter(B_bias_B_biasconv0_weight)
        if B_bias_B_biasnorm0_bias is not None:
            module.B_bias.B_biasnorm0.bias = nn.Parameter(B_bias_B_biasnorm0_bias)
        module.B_bias.B_biasnorm0.weight = nn.Parameter(B_bias_B_biasnorm0_weight)
        module.B_bias.B_biasnorm0.running_mean = nn.Parameter(B_bias_B_biasnorm0_running_mean)
        module.B_bias.B_biasnorm0.running_var = nn.Parameter(B_bias_B_biasnorm0_running_var)
        if B_bias_B_biasconv_out_bias is not None:
            module.B_bias.B_biasconv_out.bias = nn.Parameter(B_bias_B_biasconv_out_bias)
        module.B_bias.B_biasconv_out.weight = nn.Parameter(B_bias_B_biasconv_out_weight)

        # fc1
        if fc1_bias is not None:
            module.fc1.bias = nn.Parameter(fc1_bias)
        module.fc1.weight = nn.Parameter(fc1_weight)

        # fc2
        if fc2_bias is not None:
            module.fc2.bias = nn.Parameter(fc2_bias)
        module.fc2.weight = nn.Parameter(fc2_weight)

        # bn1
        if bn1_bias is not None:
            module.bn1.bias = nn.Parameter(bn1_bias)
        module.bn1.weight = nn.Parameter(bn1_weight)
        module.bn1.running_mean = nn.Parameter(bn1_running_mean)
        module.bn1.running_var = nn.Parameter(bn1_running_var)

        # print('EBGCNRumourGCN custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        if is_BU:
            module.conv1.ref_name = 'bu_conv1'
            module.conv2.ref_name = 'bu_conv2'
        else:
            module.conv1.ref_name = 'td_conv1'
            module.conv2.ref_name = 'td_conv2'
        module.conv1 = LRPGCNConv(module.conv1, GCNConv_Autograd_Fn, params={'gcn_conv': eps,
                                                                      'layer_1_no_self_loops': layer_1_no_self_loops},
                                  saved_rels=saved_rels)
        module.conv2 = LRPGCNConv(module.conv2, GCNConv_Autograd_Fn, params={'gcn_conv': eps,
                                                                      'skip_layer_2': skip_layer_2,
                                                                      'layer_2_no_self_loops': layer_2_no_self_loops},
                                  saved_rels=saved_rels)

        module.sim_network.sim_valconv0 = LRPConv1d(module.sim_network.sim_valconv0, Conv1d_Autograd_Fn,
                                                    params={'conv1d_eps': eps})
        module.sim_network.sim_valnorm0 = LRPBatchNorm1d(module.sim_network.sim_valnorm0, BatchNorm1d_Autograd_Fn,
                                                         params={'bn1d_eps': eps})
        module.sim_network.sim_valconv_out = LRPConv1d(module.sim_network.sim_valconv_out, Conv1d_Autograd_Fn,
                                                       params={'conv1d_eps': eps})

        module.W_mean.W_meanconv0 = LRPConv1d(module.W_mean.W_meanconv0, Conv1d_Autograd_Fn,
                                                    params={'conv1d_eps': eps})
        module.W_mean.W_meannorm0 = LRPBatchNorm1d(module.W_mean.W_meannorm0, BatchNorm1d_Autograd_Fn,
                                                         params={'bn1d_eps': eps})
        module.W_mean.W_meanconv_out = LRPConv1d(module.W_mean.W_meanconv_out, Conv1d_Autograd_Fn,
                                                       params={'conv1d_eps': eps})
        module.W_bias.W_biasconv0 = LRPConv1d(module.W_bias.W_biasconv0, Conv1d_Autograd_Fn,
                                                    params={'conv1d_eps': eps})
        module.W_bias.W_biasnorm0 = LRPBatchNorm1d(module.W_bias.W_biasnorm0, BatchNorm1d_Autograd_Fn,
                                                         params={'bn1d_eps': eps})
        module.W_bias.W_biasconv_out = LRPConv1d(module.W_bias.W_biasconv_out, Conv1d_Autograd_Fn,
                                                       params={'conv1d_eps': eps})

        module.B_mean.B_meanconv0 = LRPConv1d(module.B_mean.B_meanconv0, Conv1d_Autograd_Fn,
                                              params={'conv1d_eps': eps})
        module.B_mean.B_meannorm0 = LRPBatchNorm1d(module.B_mean.B_meannorm0, BatchNorm1d_Autograd_Fn,
                                                   params={'bn1d_eps': eps})
        module.B_mean.B_meanconv_out = LRPConv1d(module.B_mean.B_meanconv_out, Conv1d_Autograd_Fn,
                                                 params={'conv1d_eps': eps})
        module.B_bias.B_biasconv0 = LRPConv1d(module.B_bias.B_biasconv0, Conv1d_Autograd_Fn,
                                              params={'conv1d_eps': eps})
        module.B_bias.B_biasnorm0 = LRPBatchNorm1d(module.B_bias.B_biasnorm0, BatchNorm1d_Autograd_Fn,
                                                   params={'bn1d_eps': eps})
        module.B_bias.B_biasconv_out = LRPConv1d(module.B_bias.B_biasconv_out, Conv1d_Autograd_Fn,
                                                 params={'conv1d_eps': eps})

        module.fc1 = LRPLinear(module.fc1, Linear_Epsilon_Autograd_Fn, params={'linear_eps': eps},
                               ref_name='ebgcn_fc1', saved_rels=saved_rels)
        module.fc2 = LRPLinear(module.fc2, Linear_Epsilon_Autograd_Fn, params={'linear_eps': eps},
                               ref_name='ebgcn_fc2', saved_rels=saved_rels)
        module.bn1 = LRPBatchNorm1d(module.bn1, BatchNorm1d_Autograd_Fn,
                                    params={'bn1d_eps': eps})

        X = input_.clone().detach().requires_grad_(True)

        data = Data(x=X, edge_index=edge_index, BU_edge_index=BU_edge_index, rootindex=rootindex, root=root,
                    batch=torch.zeros(input_.shape[0]).long().to(input_.device))
        R = lrp_rumourgcn(input_=data,
                          layer=module,
                          relevance_output=grad_output,
                          eps0=eps,
                          eps=eps)
        # print('EBGCNRumourGCN custom R', R.shape)
        return R, None, None, None, None, None, None


class CAM_EBGCNRumourGCN_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, edge_index, BU_edge_index, rootindex, root, module, params):
        eps = params.get('ebgcn', 1e-6)

        def config_values_to_tensors(module):
            if isinstance(module, TDrumorGCN) or isinstance(module, BUrumorGCN):
                property_names = ['input_features', 'hidden_features', 'output_features', 'edge_num',
                                  'dropout', 'edge_infer_td', 'edge_infer_bu', 'device', 'training']
            else:
                print('Error: module not EBGCN TDrumorGCN or BUrumorGCN layer')
                raise Exception

            values = []
            for attr in property_names:
                value = getattr(module.args, attr)
                if attr == 'device':
                    value = torch.zeros(1, dtype=torch.int32, device=module.fc1.weight.device)
                elif isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.fc1.weight.device)
                elif isinstance(value, float):
                    value = torch.tensor([value], dtype=torch.float, device=module.fc1.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.fc1.weight.device)
                elif isinstance(value, bool):
                    value = torch.tensor(value, dtype=torch.bool, device=module.fc1.weight.device)
                else:
                    print('error: property value is neither int nor tuple', attr, value)
                    # exit()
                values.append(value)
            return property_names, values

        property_names, values = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)
        # print(eps_tensor)

        # GCNConv 1
        if module.conv1.lin.bias is None:
            conv1_lin_bias = None
        else:
            conv1_lin_bias = module.conv1.lin.bias.data.clone()
        conv1_lin_weight = module.conv1.lin.weight.data.clone()

        # GCNConv 2
        if module.conv2.lin.bias is None:
            conv2_lin_bias = None
        else:
            conv2_lin_bias = module.conv2.lin.bias.data.clone()
        conv2_lin_weight = module.conv2.lin.weight.data.clone()

        # sim_network
        # print(module.sim_network)
        if module.sim_network.sim_valconv0.bias is None:
            sim_network_sim_valconv0_bias = None
        else:
            sim_network_sim_valconv0_bias = module.sim_network.sim_valconv0.bias.data.clone()
        sim_network_sim_valconv0_weight = module.sim_network.sim_valconv0.weight.data.clone()
        if module.sim_network.sim_valnorm0.bias is None:
            sim_network_sim_valnorm0_bias = None
        else:
            sim_network_sim_valnorm0_bias = module.sim_network.sim_valnorm0.bias.data.clone()
        sim_network_sim_valnorm0_weight = module.sim_network.sim_valnorm0.weight.data.clone()
        sim_network_sim_valnorm0_running_mean = module.sim_network.sim_valnorm0.running_mean.data.clone()
        sim_network_sim_valnorm0_running_var = module.sim_network.sim_valnorm0.running_var.data.clone()
        if module.sim_network.sim_valconv_out.bias is None:
            sim_network_sim_valconv_out_bias = None
        else:
            sim_network_sim_valconv_out_bias = module.sim_network.sim_valconv_out.bias.data.clone()
        sim_network_sim_valconv_out_weight = module.sim_network.sim_valconv_out.weight.data.clone()

        # W_mean
        if module.W_mean.W_meanconv0.bias is None:
            W_mean_W_meanconv0_bias = None
        else:
            W_mean_W_meanconv0_bias = module.W_mean.W_meanconv0.bias.data.clone()
        W_mean_W_meanconv0_weight = module.W_mean.W_meanconv0.weight.data.clone()
        if module.W_mean.W_meannorm0.bias is None:
            W_mean_W_meannorm0_bias = None
        else:
            W_mean_W_meannorm0_bias = module.W_mean.W_meannorm0.bias.data.clone()
        W_mean_W_meannorm0_weight = module.W_mean.W_meannorm0.weight.data.clone()
        W_mean_W_meannorm0_running_mean = module.W_mean.W_meannorm0.running_mean.data.clone()
        W_mean_W_meannorm0_running_var = module.W_mean.W_meannorm0.running_var.data.clone()
        if module.W_mean.W_meanconv_out.bias is None:
            W_mean_W_meanconv_out_bias = None
        else:
            W_mean_W_meanconv_out_bias = module.W_mean.W_meanconv_out.bias.data.clone()
        W_mean_W_meanconv_out_weight = module.W_mean.W_meanconv_out.weight.data.clone()

        # W_bias
        if module.W_bias.W_biasconv0.bias is None:
            W_bias_W_biasconv0_bias = None
        else:
            W_bias_W_biasconv0_bias = module.W_bias.W_biasconv0.bias.data.clone()
        W_bias_W_biasconv0_weight = module.W_bias.W_biasconv0.weight.data.clone()
        if module.W_bias.W_biasnorm0.bias is None:
            W_bias_W_biasnorm0_bias = None
        else:
            W_bias_W_biasnorm0_bias = module.W_bias.W_biasnorm0.bias.data.clone()
        W_bias_W_biasnorm0_weight = module.W_bias.W_biasnorm0.weight.data.clone()
        W_bias_W_biasnorm0_running_mean = module.W_bias.W_biasnorm0.running_mean.data.clone()
        W_bias_W_biasnorm0_running_var = module.W_bias.W_biasnorm0.running_var.data.clone()
        if module.W_bias.W_biasconv_out.bias is None:
            W_bias_W_biasconv_out_bias = None
        else:
            W_bias_W_biasconv_out_bias = module.W_bias.W_biasconv_out.bias.data.clone()
        W_bias_W_biasconv_out_weight = module.W_bias.W_biasconv_out.weight.data.clone()

        # B_mean
        if module.B_mean.B_meanconv0.bias is None:
            B_mean_B_meanconv0_bias = None
        else:
            B_mean_B_meanconv0_bias = module.B_mean.B_meanconv0.bias.data.clone()
        B_mean_B_meanconv0_weight = module.B_mean.B_meanconv0.weight.data.clone()
        if module.B_mean.B_meannorm0.bias is None:
            B_mean_B_meannorm0_bias = None
        else:
            B_mean_B_meannorm0_bias = module.B_mean.B_meannorm0.bias.data.clone()
        B_mean_B_meannorm0_weight = module.B_mean.B_meannorm0.weight.data.clone()
        B_mean_B_meannorm0_running_mean = module.B_mean.B_meannorm0.running_mean.data.clone()
        B_mean_B_meannorm0_running_var = module.B_mean.B_meannorm0.running_var.data.clone()
        if module.B_mean.B_meanconv_out.bias is None:
            B_mean_B_meanconv_out_bias = None
        else:
            B_mean_B_meanconv_out_bias = module.B_mean.B_meanconv_out.bias.data.clone()
        B_mean_B_meanconv_out_weight = module.B_mean.B_meanconv_out.weight.data.clone()

        # B_bias
        if module.B_bias.B_biasconv0.bias is None:
            B_bias_B_biasconv0_bias = None
        else:
            B_bias_B_biasconv0_bias = module.B_bias.B_biasconv0.bias.data.clone()
        B_bias_B_biasconv0_weight = module.B_bias.B_biasconv0.weight.data.clone()
        if module.B_bias.B_biasnorm0.bias is None:
            B_bias_B_biasnorm0_bias = None
        else:
            B_bias_B_biasnorm0_bias = module.B_bias.B_biasnorm0.bias.data.clone()
        B_bias_B_biasnorm0_weight = module.B_bias.B_biasnorm0.weight.data.clone()
        B_bias_B_biasnorm0_running_mean = module.B_bias.B_biasnorm0.running_mean.data.clone()
        B_bias_B_biasnorm0_running_var = module.B_bias.B_biasnorm0.running_var.data.clone()
        if module.B_bias.B_biasconv_out.bias is None:
            B_bias_B_biasconv_out_bias = None
        else:
            B_bias_B_biasconv_out_bias = module.B_bias.B_biasconv_out.bias.data.clone()
        B_bias_B_biasconv_out_weight = module.B_bias.B_biasconv_out.weight.data.clone()

        # fc1
        if module.fc1.bias is None:
            fc1_bias = None
        else:
            fc1_bias = module.fc1.bias.data.clone()
        fc1_weight = module.fc1.weight.data.clone()

        # fc2
        if module.fc2.bias is None:
            fc2_bias = None
        else:
            fc2_bias = module.fc2.bias.data.clone()
        fc2_weight = module.fc2.weight.data.clone()

        # bn1
        if module.bn1.bias is None:
            bn1_bias = None
        else:
            bn1_bias = module.bn1.bias.data.clone()
        bn1_weight = module.bn1.weight.data.clone()
        bn1_running_mean = module.bn1.running_mean.data.clone()
        bn1_running_var = module.bn1.running_var.data.clone()

        ctx.save_for_backward(x, edge_index, BU_edge_index, rootindex, root,
                              conv1_lin_weight, conv1_lin_bias,
                              conv2_lin_weight, conv2_lin_bias,
                              sim_network_sim_valconv0_weight, sim_network_sim_valconv0_bias,
                              sim_network_sim_valnorm0_weight, sim_network_sim_valnorm0_bias,
                              sim_network_sim_valnorm0_running_mean, sim_network_sim_valnorm0_running_var,
                              sim_network_sim_valconv_out_weight, sim_network_sim_valconv_out_bias,
                              W_mean_W_meanconv0_weight, W_mean_W_meanconv0_bias,
                              W_mean_W_meannorm0_weight, W_mean_W_meannorm0_bias,
                              W_mean_W_meannorm0_running_mean, W_mean_W_meannorm0_running_var,
                              W_mean_W_meanconv_out_weight, W_mean_W_meanconv_out_bias,
                              W_bias_W_biasconv0_weight, W_bias_W_biasconv0_bias,
                              W_bias_W_biasnorm0_weight, W_bias_W_biasnorm0_bias,
                              W_bias_W_biasnorm0_running_mean, W_bias_W_biasnorm0_running_var,
                              W_bias_W_biasconv_out_weight, W_bias_W_biasconv_out_bias,
                              B_mean_B_meanconv0_weight, B_mean_B_meanconv0_bias,
                              B_mean_B_meannorm0_weight, B_mean_B_meannorm0_bias,
                              B_mean_B_meannorm0_running_mean, B_mean_B_meannorm0_running_var,
                              B_mean_B_meanconv_out_weight, B_mean_B_meanconv_out_bias,
                              B_bias_B_biasconv0_weight, B_bias_B_biasconv0_bias,
                              B_bias_B_biasnorm0_weight, B_bias_B_biasnorm0_bias,
                              B_bias_B_biasnorm0_running_mean, B_bias_B_biasnorm0_running_var,
                              B_bias_B_biasconv_out_weight, B_bias_B_biasconv_out_bias,
                              fc1_weight, fc1_bias,
                              fc2_weight, fc2_bias,
                              bn1_weight, bn1_bias,
                              bn1_running_mean, bn1_running_var,
                              eps_tensor, *values)
        ctx.saved_rels = module.saved_rels
        ctx.is_BU = module.is_BU

        # print('EBGCNRumourGCN ctx.needs_input_grad', ctx.needs_input_grad)

        data = Data(x=x, edge_index=edge_index, BU_edge_index=BU_edge_index, rootindex=rootindex, root=root,
                    batch=torch.zeros(x.shape[0]).long().to(x.device))
        # data.requires_grad = True
        return module.forward(data)

    @staticmethod
    def backward(ctx, grad_output, edge_loss_output):
        input_, edge_index, BU_edge_index, rootindex, root, \
        conv1_lin_weight, conv1_lin_bias, \
        conv2_lin_weight, conv2_lin_bias, \
        sim_network_sim_valconv0_weight, sim_network_sim_valconv0_bias, \
        sim_network_sim_valnorm0_weight, sim_network_sim_valnorm0_bias, \
        sim_network_sim_valnorm0_running_mean, sim_network_sim_valnorm0_running_var, \
        sim_network_sim_valconv_out_weight, sim_network_sim_valconv_out_bias, \
        W_mean_W_meanconv0_weight, W_mean_W_meanconv0_bias, \
        W_mean_W_meannorm0_weight, W_mean_W_meannorm0_bias, \
        W_mean_W_meannorm0_running_mean, W_mean_W_meannorm0_running_var, \
        W_mean_W_meanconv_out_weight, W_mean_W_meanconv_out_bias, \
        W_bias_W_biasconv0_weight, W_bias_W_biasconv0_bias, \
        W_bias_W_biasnorm0_weight, W_bias_W_biasnorm0_bias, \
        W_bias_W_biasnorm0_running_mean, W_bias_W_biasnorm0_running_var, \
        W_bias_W_biasconv_out_weight, W_bias_W_biasconv_out_bias, \
        B_mean_B_meanconv0_weight, B_mean_B_meanconv0_bias, \
        B_mean_B_meannorm0_weight, B_mean_B_meannorm0_bias, \
        B_mean_B_meannorm0_running_mean, B_mean_B_meannorm0_running_var, \
        B_mean_B_meanconv_out_weight, B_mean_B_meanconv_out_bias, \
        B_bias_B_biasconv0_weight, B_bias_B_biasconv0_bias, \
        B_bias_B_biasnorm0_weight, B_bias_B_biasnorm0_bias, \
        B_bias_B_biasnorm0_running_mean, B_bias_B_biasnorm0_running_var, \
        B_bias_B_biasconv_out_weight, B_bias_B_biasconv_out_bias, \
        fc1_weight, fc1_bias, \
        fc2_weight, fc2_bias, \
        bn1_weight, bn1_bias, \
        bn1_running_mean, bn1_running_var, \
        eps_tensor, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        is_BU = ctx.is_BU
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = ['input_features', 'hidden_features', 'output_features', 'edge_num',
                              'dropout', 'edge_infer_td', 'edge_infer_bu', 'device', 'training']
            params_dict = {}

            for i, property_name in enumerate(property_names):
                value = values[i]
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
            else:
                new_values = values[i+1:]
            return params_dict, new_values

        params_dict, values = tensors_to_dict(values)

        parser = ArgumentParser()
        args = parser.parse_args()
        for k, v in params_dict.items():
            args.__setattr__(k, v)

        if is_BU:
            module = BUrumorGCN(args).to(args.device)
        else:
            module = TDrumorGCN(args).to(args.device)

        # GCNConv 1
        if conv1_lin_bias is not None:
            module.conv1.lin.bias = nn.Parameter(conv1_lin_bias)
        module.conv1.lin.weight = nn.Parameter(conv1_lin_weight)

        # GCNConv 2
        if conv2_lin_bias is not None:
            module.conv2.lin.bias = nn.Parameter(conv2_lin_bias)
        module.conv2.lin.weight = nn.Parameter(conv2_lin_weight)

        # sim_network
        if sim_network_sim_valconv0_bias is not None:
            module.sim_network.sim_valconv0.bias = nn.Parameter(sim_network_sim_valconv0_bias)
        module.sim_network.sim_valconv0.weight = nn.Parameter(sim_network_sim_valconv0_weight)
        if sim_network_sim_valnorm0_bias is not None:
            module.sim_network.sim_valnorm0.bias = nn.Parameter(sim_network_sim_valnorm0_bias)
        module.sim_network.sim_valnorm0.weight = nn.Parameter(sim_network_sim_valnorm0_weight)
        module.sim_network.sim_valnorm0.running_mean = nn.Parameter(sim_network_sim_valnorm0_running_mean)
        module.sim_network.sim_valnorm0.running_var = nn.Parameter(sim_network_sim_valnorm0_running_var)
        if sim_network_sim_valconv_out_bias is not None:
            module.sim_network.sim_valconv_out.bias = nn.Parameter(sim_network_sim_valconv_out_bias)
        module.sim_network.sim_valconv_out.weight = nn.Parameter(sim_network_sim_valconv_out_weight)

        # W_mean
        if W_mean_W_meanconv0_bias is not None:
            module.W_mean.W_meanconv0.bias = nn.Parameter(W_mean_W_meanconv0_bias)
        module.W_mean.W_meanconv0.weight = nn.Parameter(W_mean_W_meanconv0_weight)
        if W_mean_W_meannorm0_bias is not None:
            module.W_mean.W_meannorm0.bias = nn.Parameter(W_mean_W_meannorm0_bias)
        module.W_mean.W_meannorm0.weight = nn.Parameter(W_mean_W_meannorm0_weight)
        module.W_mean.W_meannorm0.running_mean = nn.Parameter(W_mean_W_meannorm0_running_mean)
        module.W_mean.W_meannorm0.running_var = nn.Parameter(W_mean_W_meannorm0_running_var)
        if W_mean_W_meanconv_out_bias is not None:
            module.W_mean.W_meanconv_out.bias = nn.Parameter(W_mean_W_meanconv_out_bias)
        module.W_mean.W_meanconv_out.weight = nn.Parameter(W_mean_W_meanconv_out_weight)

        # W_bias
        if W_mean_W_meanconv0_bias is not None:
            module.W_mean.W_meanconv0.bias = nn.Parameter(W_mean_W_meanconv0_bias)
        module.W_mean.W_meanconv0.weight = nn.Parameter(W_mean_W_meanconv0_weight)
        if W_mean_W_meannorm0_bias is not None:
            module.W_mean.W_meannorm0.bias = nn.Parameter(W_mean_W_meannorm0_bias)
        module.W_mean.W_meannorm0.weight = nn.Parameter(W_mean_W_meannorm0_weight)
        module.W_mean.W_meannorm0.running_mean = nn.Parameter(W_mean_W_meannorm0_running_mean)
        module.W_mean.W_meannorm0.running_var = nn.Parameter(W_mean_W_meannorm0_running_var)
        if W_mean_W_meanconv_out_bias is not None:
            module.W_mean.W_meanconv_out.bias = nn.Parameter(W_mean_W_meanconv_out_bias)
        module.W_mean.W_meanconv_out.weight = nn.Parameter(W_mean_W_meanconv_out_weight)

        # B_mean
        if B_mean_B_meanconv0_bias is not None:
            module.B_mean.B_meanconv0.bias = nn.Parameter(B_mean_B_meanconv0_bias)
        module.B_mean.B_meanconv0.weight = nn.Parameter(B_mean_B_meanconv0_weight)
        if B_mean_B_meannorm0_bias is not None:
            module.B_mean.B_meannorm0.bias = nn.Parameter(B_mean_B_meannorm0_bias)
        module.B_mean.B_meannorm0.weight = nn.Parameter(B_mean_B_meannorm0_weight)
        module.B_mean.B_meannorm0.running_mean = nn.Parameter(B_mean_B_meannorm0_running_mean)
        module.B_mean.B_meannorm0.running_var = nn.Parameter(B_mean_B_meannorm0_running_var)
        if B_mean_B_meanconv_out_bias is not None:
            module.B_mean.B_meanconv_out.bias = nn.Parameter(B_mean_B_meanconv_out_bias)
        module.B_mean.B_meanconv_out.weight = nn.Parameter(B_mean_B_meanconv_out_weight)

        # B_bias
        if B_bias_B_biasconv0_bias is not None:
            module.B_bias.B_biasconv0.bias = nn.Parameter(B_bias_B_biasconv0_bias)
        module.B_bias.B_biasconv0.weight = nn.Parameter(B_bias_B_biasconv0_weight)
        if B_bias_B_biasnorm0_bias is not None:
            module.B_bias.B_biasnorm0.bias = nn.Parameter(B_bias_B_biasnorm0_bias)
        module.B_bias.B_biasnorm0.weight = nn.Parameter(B_bias_B_biasnorm0_weight)
        module.B_bias.B_biasnorm0.running_mean = nn.Parameter(B_bias_B_biasnorm0_running_mean)
        module.B_bias.B_biasnorm0.running_var = nn.Parameter(B_bias_B_biasnorm0_running_var)
        if B_bias_B_biasconv_out_bias is not None:
            module.B_bias.B_biasconv_out.bias = nn.Parameter(B_bias_B_biasconv_out_bias)
        module.B_bias.B_biasconv_out.weight = nn.Parameter(B_bias_B_biasconv_out_weight)

        # fc1
        if fc1_bias is not None:
            module.fc1.bias = nn.Parameter(fc1_bias)
        module.fc1.weight = nn.Parameter(fc1_weight)

        # fc2
        if fc2_bias is not None:
            module.fc2.bias = nn.Parameter(fc2_bias)
        module.fc2.weight = nn.Parameter(fc2_weight)

        # bn1
        if bn1_bias is not None:
            module.bn1.bias = nn.Parameter(bn1_bias)
        module.bn1.weight = nn.Parameter(bn1_weight)
        module.bn1.running_mean = nn.Parameter(bn1_running_mean)
        module.bn1.running_var = nn.Parameter(bn1_running_var)

        # print('EBGCNRumourGCN custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        if is_BU:
            module.conv1.ref_name = 'bu_conv1'
            module.conv2.ref_name = 'bu_conv2'
        else:
            module.conv1.ref_name = 'td_conv1'
            module.conv2.ref_name = 'td_conv2'
        module.conv1 = LRPGCNConv(module.conv1, CAM_GCNConv_Autograd_Fn, params={'gcn_conv': eps}, saved_rels=saved_rels)
        module.conv2 = LRPGCNConv(module.conv2, CAM_GCNConv_Autograd_Fn, params={'gcn_conv': eps}, saved_rels=saved_rels)

        module.sim_network.sim_valconv0 = LRPConv1d(module.sim_network.sim_valconv0, CAM_Conv1d_Autograd_Fn,
                                                    params={'conv1d_eps': eps})
        module.sim_network.sim_valnorm0 = LRPBatchNorm1d(module.sim_network.sim_valnorm0, CAM_BatchNorm1d_Autograd_Fn,
                                                         params={'bn1d_eps': eps})
        module.sim_network.sim_valconv_out = LRPConv1d(module.sim_network.sim_valconv_out, CAM_Conv1d_Autograd_Fn,
                                                       params={'conv1d_eps': eps})

        module.W_mean.W_meanconv0 = LRPConv1d(module.W_mean.W_meanconv0, CAM_Conv1d_Autograd_Fn,
                                                    params={'conv1d_eps': eps})
        module.W_mean.W_meannorm0 = LRPBatchNorm1d(module.W_mean.W_meannorm0, CAM_BatchNorm1d_Autograd_Fn,
                                                         params={'bn1d_eps': eps})
        module.W_mean.W_meanconv_out = LRPConv1d(module.W_mean.W_meanconv_out, CAM_Conv1d_Autograd_Fn,
                                                       params={'conv1d_eps': eps})
        module.W_bias.W_biasconv0 = LRPConv1d(module.W_bias.W_biasconv0, CAM_Conv1d_Autograd_Fn,
                                                    params={'conv1d_eps': eps})
        module.W_bias.W_biasnorm0 = LRPBatchNorm1d(module.W_bias.W_biasnorm0, CAM_BatchNorm1d_Autograd_Fn,
                                                         params={'bn1d_eps': eps})
        module.W_bias.W_biasconv_out = LRPConv1d(module.W_bias.W_biasconv_out, CAM_Conv1d_Autograd_Fn,
                                                       params={'conv1d_eps': eps})

        module.B_mean.B_meanconv0 = LRPConv1d(module.B_mean.B_meanconv0, CAM_Conv1d_Autograd_Fn,
                                              params={'conv1d_eps': eps})
        module.B_mean.B_meannorm0 = LRPBatchNorm1d(module.B_mean.B_meannorm0, CAM_BatchNorm1d_Autograd_Fn,
                                                   params={'bn1d_eps': eps})
        module.B_mean.B_meanconv_out = LRPConv1d(module.B_mean.B_meanconv_out, CAM_Conv1d_Autograd_Fn,
                                                 params={'conv1d_eps': eps})
        module.B_bias.B_biasconv0 = LRPConv1d(module.B_bias.B_biasconv0, CAM_Conv1d_Autograd_Fn,
                                              params={'conv1d_eps': eps})
        module.B_bias.B_biasnorm0 = LRPBatchNorm1d(module.B_bias.B_biasnorm0, CAM_BatchNorm1d_Autograd_Fn,
                                                   params={'bn1d_eps': eps})
        module.B_bias.B_biasconv_out = LRPConv1d(module.B_bias.B_biasconv_out, CAM_Conv1d_Autograd_Fn,
                                                 params={'conv1d_eps': eps})

        module.fc1 = LRPLinear(module.fc1, CAM_Linear_Autograd_Fn, params={'linear_eps': eps},
                               ref_name='ebgcn_fc1', saved_rels=saved_rels)
        module.fc2 = LRPLinear(module.fc2, CAM_Linear_Autograd_Fn, params={'linear_eps': eps},
                               ref_name='ebgcn_fc2', saved_rels=saved_rels)
        module.bn1 = LRPBatchNorm1d(module.bn1, CAM_BatchNorm1d_Autograd_Fn,
                                    params={'bn1d_eps': eps})

        X = input_.clone().detach().requires_grad_(True)
        data = Data(x=X, edge_index=edge_index, BU_edge_index=BU_edge_index, rootindex=rootindex, root=root,
                    batch=torch.zeros(input_.shape[0]).long().to(input_.device))
        R = cam_rumourgcn(input_=data,
                          layer=module,
                          relevance_output=grad_output)
        # print('EBGCNRumourGCN custom R', R.shape)
        return R, None, None, None, None, None, None


class EB_EBGCNRumourGCN_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, edge_index, BU_edge_index, rootindex, root, module, params):
        eps = params.get('ebgcn', 1e-6)

        def config_values_to_tensors(module):
            if isinstance(module, TDrumorGCN) or isinstance(module, BUrumorGCN):
                property_names = ['input_features', 'hidden_features', 'output_features', 'edge_num',
                                  'dropout', 'edge_infer_td', 'edge_infer_bu', 'device', 'training']
            else:
                print('Error: module not EBGCN TDrumorGCN or BUrumorGCN layer')
                raise Exception

            values = []
            for attr in property_names:
                value = getattr(module.args, attr)
                if attr == 'device':
                    value = torch.zeros(1, dtype=torch.int32, device=module.fc1.weight.device)
                elif isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.fc1.weight.device)
                elif isinstance(value, float):
                    value = torch.tensor([value], dtype=torch.float, device=module.fc1.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.fc1.weight.device)
                elif isinstance(value, bool):
                    value = torch.tensor(value, dtype=torch.bool, device=module.fc1.weight.device)
                else:
                    print('error: property value is neither int nor tuple', attr, value)
                    # exit()
                values.append(value)
            return property_names, values

        property_names, values = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)
        # print(eps_tensor)

        # GCNConv 1
        if module.conv1.lin.bias is None:
            conv1_lin_bias = None
        else:
            conv1_lin_bias = module.conv1.lin.bias.data.clone()
        conv1_lin_weight = module.conv1.lin.weight.data.clone()

        # GCNConv 2
        if module.conv2.lin.bias is None:
            conv2_lin_bias = None
        else:
            conv2_lin_bias = module.conv2.lin.bias.data.clone()
        conv2_lin_weight = module.conv2.lin.weight.data.clone()

        # sim_network
        # print(module.sim_network)
        if module.sim_network.sim_valconv0.bias is None:
            sim_network_sim_valconv0_bias = None
        else:
            sim_network_sim_valconv0_bias = module.sim_network.sim_valconv0.bias.data.clone()
        sim_network_sim_valconv0_weight = module.sim_network.sim_valconv0.weight.data.clone()
        if module.sim_network.sim_valnorm0.bias is None:
            sim_network_sim_valnorm0_bias = None
        else:
            sim_network_sim_valnorm0_bias = module.sim_network.sim_valnorm0.bias.data.clone()
        sim_network_sim_valnorm0_weight = module.sim_network.sim_valnorm0.weight.data.clone()
        sim_network_sim_valnorm0_running_mean = module.sim_network.sim_valnorm0.running_mean.data.clone()
        sim_network_sim_valnorm0_running_var = module.sim_network.sim_valnorm0.running_var.data.clone()
        if module.sim_network.sim_valconv_out.bias is None:
            sim_network_sim_valconv_out_bias = None
        else:
            sim_network_sim_valconv_out_bias = module.sim_network.sim_valconv_out.bias.data.clone()
        sim_network_sim_valconv_out_weight = module.sim_network.sim_valconv_out.weight.data.clone()

        # W_mean
        if module.W_mean.W_meanconv0.bias is None:
            W_mean_W_meanconv0_bias = None
        else:
            W_mean_W_meanconv0_bias = module.W_mean.W_meanconv0.bias.data.clone()
        W_mean_W_meanconv0_weight = module.W_mean.W_meanconv0.weight.data.clone()
        if module.W_mean.W_meannorm0.bias is None:
            W_mean_W_meannorm0_bias = None
        else:
            W_mean_W_meannorm0_bias = module.W_mean.W_meannorm0.bias.data.clone()
        W_mean_W_meannorm0_weight = module.W_mean.W_meannorm0.weight.data.clone()
        W_mean_W_meannorm0_running_mean = module.W_mean.W_meannorm0.running_mean.data.clone()
        W_mean_W_meannorm0_running_var = module.W_mean.W_meannorm0.running_var.data.clone()
        if module.W_mean.W_meanconv_out.bias is None:
            W_mean_W_meanconv_out_bias = None
        else:
            W_mean_W_meanconv_out_bias = module.W_mean.W_meanconv_out.bias.data.clone()
        W_mean_W_meanconv_out_weight = module.W_mean.W_meanconv_out.weight.data.clone()

        # W_bias
        if module.W_bias.W_biasconv0.bias is None:
            W_bias_W_biasconv0_bias = None
        else:
            W_bias_W_biasconv0_bias = module.W_bias.W_biasconv0.bias.data.clone()
        W_bias_W_biasconv0_weight = module.W_bias.W_biasconv0.weight.data.clone()
        if module.W_bias.W_biasnorm0.bias is None:
            W_bias_W_biasnorm0_bias = None
        else:
            W_bias_W_biasnorm0_bias = module.W_bias.W_biasnorm0.bias.data.clone()
        W_bias_W_biasnorm0_weight = module.W_bias.W_biasnorm0.weight.data.clone()
        W_bias_W_biasnorm0_running_mean = module.W_bias.W_biasnorm0.running_mean.data.clone()
        W_bias_W_biasnorm0_running_var = module.W_bias.W_biasnorm0.running_var.data.clone()
        if module.W_bias.W_biasconv_out.bias is None:
            W_bias_W_biasconv_out_bias = None
        else:
            W_bias_W_biasconv_out_bias = module.W_bias.W_biasconv_out.bias.data.clone()
        W_bias_W_biasconv_out_weight = module.W_bias.W_biasconv_out.weight.data.clone()

        # B_mean
        if module.B_mean.B_meanconv0.bias is None:
            B_mean_B_meanconv0_bias = None
        else:
            B_mean_B_meanconv0_bias = module.B_mean.B_meanconv0.bias.data.clone()
        B_mean_B_meanconv0_weight = module.B_mean.B_meanconv0.weight.data.clone()
        if module.B_mean.B_meannorm0.bias is None:
            B_mean_B_meannorm0_bias = None
        else:
            B_mean_B_meannorm0_bias = module.B_mean.B_meannorm0.bias.data.clone()
        B_mean_B_meannorm0_weight = module.B_mean.B_meannorm0.weight.data.clone()
        B_mean_B_meannorm0_running_mean = module.B_mean.B_meannorm0.running_mean.data.clone()
        B_mean_B_meannorm0_running_var = module.B_mean.B_meannorm0.running_var.data.clone()
        if module.B_mean.B_meanconv_out.bias is None:
            B_mean_B_meanconv_out_bias = None
        else:
            B_mean_B_meanconv_out_bias = module.B_mean.B_meanconv_out.bias.data.clone()
        B_mean_B_meanconv_out_weight = module.B_mean.B_meanconv_out.weight.data.clone()

        # B_bias
        if module.B_bias.B_biasconv0.bias is None:
            B_bias_B_biasconv0_bias = None
        else:
            B_bias_B_biasconv0_bias = module.B_bias.B_biasconv0.bias.data.clone()
        B_bias_B_biasconv0_weight = module.B_bias.B_biasconv0.weight.data.clone()
        if module.B_bias.B_biasnorm0.bias is None:
            B_bias_B_biasnorm0_bias = None
        else:
            B_bias_B_biasnorm0_bias = module.B_bias.B_biasnorm0.bias.data.clone()
        B_bias_B_biasnorm0_weight = module.B_bias.B_biasnorm0.weight.data.clone()
        B_bias_B_biasnorm0_running_mean = module.B_bias.B_biasnorm0.running_mean.data.clone()
        B_bias_B_biasnorm0_running_var = module.B_bias.B_biasnorm0.running_var.data.clone()
        if module.B_bias.B_biasconv_out.bias is None:
            B_bias_B_biasconv_out_bias = None
        else:
            B_bias_B_biasconv_out_bias = module.B_bias.B_biasconv_out.bias.data.clone()
        B_bias_B_biasconv_out_weight = module.B_bias.B_biasconv_out.weight.data.clone()

        # fc1
        if module.fc1.bias is None:
            fc1_bias = None
        else:
            fc1_bias = module.fc1.bias.data.clone()
        fc1_weight = module.fc1.weight.data.clone()

        # fc2
        if module.fc2.bias is None:
            fc2_bias = None
        else:
            fc2_bias = module.fc2.bias.data.clone()
        fc2_weight = module.fc2.weight.data.clone()

        # bn1
        if module.bn1.bias is None:
            bn1_bias = None
        else:
            bn1_bias = module.bn1.bias.data.clone()
        bn1_weight = module.bn1.weight.data.clone()
        bn1_running_mean = module.bn1.running_mean.data.clone()
        bn1_running_var = module.bn1.running_var.data.clone()

        ctx.save_for_backward(x, edge_index, BU_edge_index, rootindex, root,
                              conv1_lin_weight, conv1_lin_bias,
                              conv2_lin_weight, conv2_lin_bias,
                              sim_network_sim_valconv0_weight, sim_network_sim_valconv0_bias,
                              sim_network_sim_valnorm0_weight, sim_network_sim_valnorm0_bias,
                              sim_network_sim_valnorm0_running_mean, sim_network_sim_valnorm0_running_var,
                              sim_network_sim_valconv_out_weight, sim_network_sim_valconv_out_bias,
                              W_mean_W_meanconv0_weight, W_mean_W_meanconv0_bias,
                              W_mean_W_meannorm0_weight, W_mean_W_meannorm0_bias,
                              W_mean_W_meannorm0_running_mean, W_mean_W_meannorm0_running_var,
                              W_mean_W_meanconv_out_weight, W_mean_W_meanconv_out_bias,
                              W_bias_W_biasconv0_weight, W_bias_W_biasconv0_bias,
                              W_bias_W_biasnorm0_weight, W_bias_W_biasnorm0_bias,
                              W_bias_W_biasnorm0_running_mean, W_bias_W_biasnorm0_running_var,
                              W_bias_W_biasconv_out_weight, W_bias_W_biasconv_out_bias,
                              B_mean_B_meanconv0_weight, B_mean_B_meanconv0_bias,
                              B_mean_B_meannorm0_weight, B_mean_B_meannorm0_bias,
                              B_mean_B_meannorm0_running_mean, B_mean_B_meannorm0_running_var,
                              B_mean_B_meanconv_out_weight, B_mean_B_meanconv_out_bias,
                              B_bias_B_biasconv0_weight, B_bias_B_biasconv0_bias,
                              B_bias_B_biasnorm0_weight, B_bias_B_biasnorm0_bias,
                              B_bias_B_biasnorm0_running_mean, B_bias_B_biasnorm0_running_var,
                              B_bias_B_biasconv_out_weight, B_bias_B_biasconv_out_bias,
                              fc1_weight, fc1_bias,
                              fc2_weight, fc2_bias,
                              bn1_weight, bn1_bias,
                              bn1_running_mean, bn1_running_var,
                              eps_tensor, *values)
        ctx.saved_rels = module.saved_rels
        ctx.is_BU = module.is_BU

        # print('EBGCNRumourGCN ctx.needs_input_grad', ctx.needs_input_grad)

        data = Data(x=x, edge_index=edge_index, BU_edge_index=BU_edge_index, rootindex=rootindex, root=root,
                    batch=torch.zeros(x.shape[0]).long().to(x.device))
        # data.requires_grad = True
        return module.forward(data)

    @staticmethod
    def backward(ctx, grad_output, edge_loss_output):
        input_, edge_index, BU_edge_index, rootindex, root, \
        conv1_lin_weight, conv1_lin_bias, \
        conv2_lin_weight, conv2_lin_bias, \
        sim_network_sim_valconv0_weight, sim_network_sim_valconv0_bias, \
        sim_network_sim_valnorm0_weight, sim_network_sim_valnorm0_bias, \
        sim_network_sim_valnorm0_running_mean, sim_network_sim_valnorm0_running_var, \
        sim_network_sim_valconv_out_weight, sim_network_sim_valconv_out_bias, \
        W_mean_W_meanconv0_weight, W_mean_W_meanconv0_bias, \
        W_mean_W_meannorm0_weight, W_mean_W_meannorm0_bias, \
        W_mean_W_meannorm0_running_mean, W_mean_W_meannorm0_running_var, \
        W_mean_W_meanconv_out_weight, W_mean_W_meanconv_out_bias, \
        W_bias_W_biasconv0_weight, W_bias_W_biasconv0_bias, \
        W_bias_W_biasnorm0_weight, W_bias_W_biasnorm0_bias, \
        W_bias_W_biasnorm0_running_mean, W_bias_W_biasnorm0_running_var, \
        W_bias_W_biasconv_out_weight, W_bias_W_biasconv_out_bias, \
        B_mean_B_meanconv0_weight, B_mean_B_meanconv0_bias, \
        B_mean_B_meannorm0_weight, B_mean_B_meannorm0_bias, \
        B_mean_B_meannorm0_running_mean, B_mean_B_meannorm0_running_var, \
        B_mean_B_meanconv_out_weight, B_mean_B_meanconv_out_bias, \
        B_bias_B_biasconv0_weight, B_bias_B_biasconv0_bias, \
        B_bias_B_biasnorm0_weight, B_bias_B_biasnorm0_bias, \
        B_bias_B_biasnorm0_running_mean, B_bias_B_biasnorm0_running_var, \
        B_bias_B_biasconv_out_weight, B_bias_B_biasconv_out_bias, \
        fc1_weight, fc1_bias, \
        fc2_weight, fc2_bias, \
        bn1_weight, bn1_bias, \
        bn1_running_mean, bn1_running_var, \
        eps_tensor, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        is_BU = ctx.is_BU
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = ['input_features', 'hidden_features', 'output_features', 'edge_num',
                              'dropout', 'edge_infer_td', 'edge_infer_bu', 'device', 'training']
            params_dict = {}

            for i, property_name in enumerate(property_names):
                value = values[i]
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
            else:
                new_values = values[i+1:]
            return params_dict, new_values

        params_dict, values = tensors_to_dict(values)

        parser = ArgumentParser()
        args = parser.parse_args()
        for k, v in params_dict.items():
            args.__setattr__(k, v)

        if is_BU:
            module = BUrumorGCN(args).to(args.device)
        else:
            module = TDrumorGCN(args).to(args.device)

        # GCNConv 1
        if conv1_lin_bias is not None:
            module.conv1.lin.bias = nn.Parameter(conv1_lin_bias)
        module.conv1.lin.weight = nn.Parameter(conv1_lin_weight)

        # GCNConv 2
        if conv2_lin_bias is not None:
            module.conv2.lin.bias = nn.Parameter(conv2_lin_bias)
        module.conv2.lin.weight = nn.Parameter(conv2_lin_weight)

        # sim_network
        if sim_network_sim_valconv0_bias is not None:
            module.sim_network.sim_valconv0.bias = nn.Parameter(sim_network_sim_valconv0_bias)
        module.sim_network.sim_valconv0.weight = nn.Parameter(sim_network_sim_valconv0_weight)
        if sim_network_sim_valnorm0_bias is not None:
            module.sim_network.sim_valnorm0.bias = nn.Parameter(sim_network_sim_valnorm0_bias)
        module.sim_network.sim_valnorm0.weight = nn.Parameter(sim_network_sim_valnorm0_weight)
        module.sim_network.sim_valnorm0.running_mean = nn.Parameter(sim_network_sim_valnorm0_running_mean)
        module.sim_network.sim_valnorm0.running_var = nn.Parameter(sim_network_sim_valnorm0_running_var)
        if sim_network_sim_valconv_out_bias is not None:
            module.sim_network.sim_valconv_out.bias = nn.Parameter(sim_network_sim_valconv_out_bias)
        module.sim_network.sim_valconv_out.weight = nn.Parameter(sim_network_sim_valconv_out_weight)

        # W_mean
        if W_mean_W_meanconv0_bias is not None:
            module.W_mean.W_meanconv0.bias = nn.Parameter(W_mean_W_meanconv0_bias)
        module.W_mean.W_meanconv0.weight = nn.Parameter(W_mean_W_meanconv0_weight)
        if W_mean_W_meannorm0_bias is not None:
            module.W_mean.W_meannorm0.bias = nn.Parameter(W_mean_W_meannorm0_bias)
        module.W_mean.W_meannorm0.weight = nn.Parameter(W_mean_W_meannorm0_weight)
        module.W_mean.W_meannorm0.running_mean = nn.Parameter(W_mean_W_meannorm0_running_mean)
        module.W_mean.W_meannorm0.running_var = nn.Parameter(W_mean_W_meannorm0_running_var)
        if W_mean_W_meanconv_out_bias is not None:
            module.W_mean.W_meanconv_out.bias = nn.Parameter(W_mean_W_meanconv_out_bias)
        module.W_mean.W_meanconv_out.weight = nn.Parameter(W_mean_W_meanconv_out_weight)

        # W_bias
        if W_mean_W_meanconv0_bias is not None:
            module.W_mean.W_meanconv0.bias = nn.Parameter(W_mean_W_meanconv0_bias)
        module.W_mean.W_meanconv0.weight = nn.Parameter(W_mean_W_meanconv0_weight)
        if W_mean_W_meannorm0_bias is not None:
            module.W_mean.W_meannorm0.bias = nn.Parameter(W_mean_W_meannorm0_bias)
        module.W_mean.W_meannorm0.weight = nn.Parameter(W_mean_W_meannorm0_weight)
        module.W_mean.W_meannorm0.running_mean = nn.Parameter(W_mean_W_meannorm0_running_mean)
        module.W_mean.W_meannorm0.running_var = nn.Parameter(W_mean_W_meannorm0_running_var)
        if W_mean_W_meanconv_out_bias is not None:
            module.W_mean.W_meanconv_out.bias = nn.Parameter(W_mean_W_meanconv_out_bias)
        module.W_mean.W_meanconv_out.weight = nn.Parameter(W_mean_W_meanconv_out_weight)

        # B_mean
        if B_mean_B_meanconv0_bias is not None:
            module.B_mean.B_meanconv0.bias = nn.Parameter(B_mean_B_meanconv0_bias)
        module.B_mean.B_meanconv0.weight = nn.Parameter(B_mean_B_meanconv0_weight)
        if B_mean_B_meannorm0_bias is not None:
            module.B_mean.B_meannorm0.bias = nn.Parameter(B_mean_B_meannorm0_bias)
        module.B_mean.B_meannorm0.weight = nn.Parameter(B_mean_B_meannorm0_weight)
        module.B_mean.B_meannorm0.running_mean = nn.Parameter(B_mean_B_meannorm0_running_mean)
        module.B_mean.B_meannorm0.running_var = nn.Parameter(B_mean_B_meannorm0_running_var)
        if B_mean_B_meanconv_out_bias is not None:
            module.B_mean.B_meanconv_out.bias = nn.Parameter(B_mean_B_meanconv_out_bias)
        module.B_mean.B_meanconv_out.weight = nn.Parameter(B_mean_B_meanconv_out_weight)

        # B_bias
        if B_bias_B_biasconv0_bias is not None:
            module.B_bias.B_biasconv0.bias = nn.Parameter(B_bias_B_biasconv0_bias)
        module.B_bias.B_biasconv0.weight = nn.Parameter(B_bias_B_biasconv0_weight)
        if B_bias_B_biasnorm0_bias is not None:
            module.B_bias.B_biasnorm0.bias = nn.Parameter(B_bias_B_biasnorm0_bias)
        module.B_bias.B_biasnorm0.weight = nn.Parameter(B_bias_B_biasnorm0_weight)
        module.B_bias.B_biasnorm0.running_mean = nn.Parameter(B_bias_B_biasnorm0_running_mean)
        module.B_bias.B_biasnorm0.running_var = nn.Parameter(B_bias_B_biasnorm0_running_var)
        if B_bias_B_biasconv_out_bias is not None:
            module.B_bias.B_biasconv_out.bias = nn.Parameter(B_bias_B_biasconv_out_bias)
        module.B_bias.B_biasconv_out.weight = nn.Parameter(B_bias_B_biasconv_out_weight)

        # fc1
        if fc1_bias is not None:
            module.fc1.bias = nn.Parameter(fc1_bias)
        module.fc1.weight = nn.Parameter(fc1_weight)

        # fc2
        if fc2_bias is not None:
            module.fc2.bias = nn.Parameter(fc2_bias)
        module.fc2.weight = nn.Parameter(fc2_weight)

        # bn1
        if bn1_bias is not None:
            module.bn1.bias = nn.Parameter(bn1_bias)
        module.bn1.weight = nn.Parameter(bn1_weight)
        module.bn1.running_mean = nn.Parameter(bn1_running_mean)
        module.bn1.running_var = nn.Parameter(bn1_running_var)

        # print('EBGCNRumourGCN custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        if is_BU:
            module.conv1.ref_name = 'bu_conv1'
            module.conv2.ref_name = 'bu_conv2'
        else:
            module.conv1.ref_name = 'td_conv1'
            module.conv2.ref_name = 'td_conv2'
        module.conv1 = LRPGCNConv(module.conv1, EB_GCNConv_Autograd_Fn, params={'gcn_conv': eps}, saved_rels=saved_rels)
        module.conv2 = LRPGCNConv(module.conv2, EB_GCNConv_Autograd_Fn, params={'gcn_conv': eps}, saved_rels=saved_rels)

        module.sim_network.sim_valconv0 = LRPConv1d(module.sim_network.sim_valconv0, EB_Conv1d_Autograd_Fn,
                                                    params={'conv1d_eps': eps})
        module.sim_network.sim_valnorm0 = LRPBatchNorm1d(module.sim_network.sim_valnorm0, EB_BatchNorm1d_Autograd_Fn,
                                                         params={'bn1d_eps': eps})
        module.sim_network.sim_valconv_out = LRPConv1d(module.sim_network.sim_valconv_out, EB_Conv1d_Autograd_Fn,
                                                       params={'conv1d_eps': eps})

        module.W_mean.W_meanconv0 = LRPConv1d(module.W_mean.W_meanconv0, EB_Conv1d_Autograd_Fn,
                                                    params={'conv1d_eps': eps})
        module.W_mean.W_meannorm0 = LRPBatchNorm1d(module.W_mean.W_meannorm0, EB_BatchNorm1d_Autograd_Fn,
                                                         params={'bn1d_eps': eps})
        module.W_mean.W_meanconv_out = LRPConv1d(module.W_mean.W_meanconv_out, EB_Conv1d_Autograd_Fn,
                                                       params={'conv1d_eps': eps})
        module.W_bias.W_biasconv0 = LRPConv1d(module.W_bias.W_biasconv0, EB_Conv1d_Autograd_Fn,
                                                    params={'conv1d_eps': eps})
        module.W_bias.W_biasnorm0 = LRPBatchNorm1d(module.W_bias.W_biasnorm0, EB_BatchNorm1d_Autograd_Fn,
                                                         params={'bn1d_eps': eps})
        module.W_bias.W_biasconv_out = LRPConv1d(module.W_bias.W_biasconv_out, EB_Conv1d_Autograd_Fn,
                                                       params={'conv1d_eps': eps})

        module.B_mean.B_meanconv0 = LRPConv1d(module.B_mean.B_meanconv0, EB_Conv1d_Autograd_Fn,
                                              params={'conv1d_eps': eps})
        module.B_mean.B_meannorm0 = LRPBatchNorm1d(module.B_mean.B_meannorm0, EB_BatchNorm1d_Autograd_Fn,
                                                   params={'bn1d_eps': eps})
        module.B_mean.B_meanconv_out = LRPConv1d(module.B_mean.B_meanconv_out, EB_Conv1d_Autograd_Fn,
                                                 params={'conv1d_eps': eps})
        module.B_bias.B_biasconv0 = LRPConv1d(module.B_bias.B_biasconv0, EB_Conv1d_Autograd_Fn,
                                              params={'conv1d_eps': eps})
        module.B_bias.B_biasnorm0 = LRPBatchNorm1d(module.B_bias.B_biasnorm0, EB_BatchNorm1d_Autograd_Fn,
                                                   params={'bn1d_eps': eps})
        module.B_bias.B_biasconv_out = LRPConv1d(module.B_bias.B_biasconv_out, EB_Conv1d_Autograd_Fn,
                                                 params={'conv1d_eps': eps})

        module.fc1 = LRPLinear(module.fc1, EB_Linear_Autograd_Fn, params={'linear_eps': eps},
                               ref_name='ebgcn_fc1', saved_rels=saved_rels)
        module.fc2 = LRPLinear(module.fc2, EB_Linear_Autograd_Fn, params={'linear_eps': eps},
                               ref_name='ebgcn_fc2', saved_rels=saved_rels)
        module.bn1 = LRPBatchNorm1d(module.bn1, EB_BatchNorm1d_Autograd_Fn,
                                    params={'bn1d_eps': eps})

        X = input_.clone().detach().requires_grad_(True)
        data = Data(x=X, edge_index=edge_index, BU_edge_index=BU_edge_index, rootindex=rootindex, root=root,
                    batch=torch.zeros(input_.shape[0]).long().to(input_.device))
        R = eb_rumourgcn(input_=data,
                         layer=module,
                         relevance_output=grad_output)
        # print('EBGCNRumourGCN custom R', R.shape)
        return R, None, None, None, None, None, None


def lrp_rumourgcn(input_, layer, relevance_output, eps0, eps):
    if input_.x.grad is not None:
        input_.x.grad.zero()
    relevance_output_data = relevance_output.clone().detach()
    with torch.enable_grad():
        Z, _ = layer(input_)
    # print(Z)
    S = safe_divide(relevance_output_data, Z.clone().detach(), eps0, eps)
    # Z.backward(S, retain_graph=True)
    Z.backward(S)
    relevance_input = input_.x.data * input_.x.grad
    return relevance_input


def cam_rumourgcn(input_, layer, relevance_output):
    if input_.x.grad is not None:
        input_.x.grad.zero()
    relevance_output_data = relevance_output.clone().detach()
    with torch.enable_grad():
        Z, _ = layer(input_)
    Z.backward(relevance_output_data)
    relevance_input = F.relu(input_.x.data * input_.x.grad)
    # layer.saved_relevance = relevance_output
    return relevance_input


def eb_rumourgcn(input_, layer, relevance_output):
    if input_.x.grad is not None:
        input_.x.grad.zero()
    with torch.enable_grad():
        Z, _ = layer(input_)  # X = W^{+}^T * A_{n}
    relevance_output_data = relevance_output.clone().detach()  # P_{n-1}
    X = Z.clone().detach().sum()
    Y = relevance_output_data / X  # Y = P_{n-1} (/) X
    Z.backward(Y)  # Use backward pass to compute Z = W^{+} * Y
    relevance_input = input_.x.data * input_.x.grad  # P_{n} = A_{n} (*) Z
    # layer.saved_relevance = relevance_input
    return relevance_input


class LRPEBGCN(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(LRPEBGCN, self).__init__()
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']
        self.saved_rels = {}
        self.base = module
        module.TDrumorGCN.saved_rels = self.saved_rels
        module.BUrumorGCN.saved_rels = self.saved_rels

        self.TDrumorGCN = LRPEBGCNRumourGCN(module.TDrumorGCN, EBGCNRumourGCN_Autograd_Fn, params=self.params,
                                            is_BU=False)
        self.BUrumorGCN = LRPEBGCNRumourGCN(module.BUrumorGCN, EBGCNRumourGCN_Autograd_Fn, params=self.params,
                                            is_BU=True)
        self.fc = LRPLinear(module.fc, Linear_Epsilon_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                            ref_name='fc')

    def forward(self, data, skip_layer_2=False, layer_1_no_self_loops=False, layer_2_no_self_loops=False):
        TD_x, TD_edge_loss = self.TDrumorGCN(data, skip_layer_2=skip_layer_2, layer_1_no_self_loops=layer_1_no_self_loops,
                                             layer_2_no_self_loops=layer_2_no_self_loops)
        BU_x, BU_edge_loss = self.BUrumorGCN(data, skip_layer_2=skip_layer_2, layer_1_no_self_loops=layer_1_no_self_loops,
                                             layer_2_no_self_loops=layer_2_no_self_loops)
        self.x = torch.cat((BU_x, TD_x), 1)
        self.out = self.fc(self.x)
        x = F.log_softmax(self.out, dim=1)
        return x, TD_edge_loss, BU_edge_loss


class CAM_EBGCN(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(CAM_EBGCN, self).__init__()
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']
        self.saved_rels = {}
        self.base = module
        module.TDrumorGCN.saved_rels = self.saved_rels
        module.BUrumorGCN.saved_rels = self.saved_rels

        self.TDrumorGCN = LRPEBGCNRumourGCN(module.TDrumorGCN, CAM_EBGCNRumourGCN_Autograd_Fn, params=self.params,
                                            is_BU=False)
        self.BUrumorGCN = LRPEBGCNRumourGCN(module.BUrumorGCN, CAM_EBGCNRumourGCN_Autograd_Fn, params=self.params,
                                            is_BU=True)
        self.fc = LRPLinear(module.fc, CAM_Linear_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                            ref_name='fc')

    def forward(self, data):
        TD_x, TD_edge_loss = self.TDrumorGCN(data)
        BU_x, BU_edge_loss = self.BUrumorGCN(data)
        self.x = torch.cat((BU_x, TD_x), 1)
        self.out = self.fc(self.x)
        x = F.log_softmax(self.out, dim=1)
        return x, TD_edge_loss, BU_edge_loss


class EB_EBGCN(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(EB_EBGCN, self).__init__()
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']
        self.is_contrastive = kwargs.get('is_contrastive', False)
        self.saved_rels = {}
        self.base = module
        module.TDrumorGCN.saved_rels = self.saved_rels
        module.BUrumorGCN.saved_rels = self.saved_rels

        self.TDrumorGCN = LRPEBGCNRumourGCN(module.TDrumorGCN, EB_EBGCNRumourGCN_Autograd_Fn, params=self.params,
                                            is_BU=False)
        self.BUrumorGCN = LRPEBGCNRumourGCN(module.BUrumorGCN, EB_EBGCNRumourGCN_Autograd_Fn, params=self.params,
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
        TD_x, TD_edge_loss = self.TDrumorGCN(data)
        BU_x, BU_edge_loss = self.BUrumorGCN(data)
        self.x = torch.cat((BU_x, TD_x), 1)
        self.out = self.fc(self.x)
        x = F.log_softmax(self.out, dim=1)
        return x, TD_edge_loss, BU_edge_loss

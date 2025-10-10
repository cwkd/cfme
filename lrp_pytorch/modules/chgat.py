import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import copy
# import torch_geometric.nn as geom_nn
from lrp_pytorch.modules.base import safe_divide
from lrp_pytorch.modules.gat2_conv import LRPGATv2Conv, GATv2Conv_Autograd_Fn, CAM_GATv2Conv_Autograd_Fn, \
    EB_GATv2Conv_Autograd_Fn
from lrp_pytorch.modules.linear import LRPLinear, Linear_Epsilon_Autograd_Fn, CAM_Linear_Autograd_Fn, \
    EB_Linear_Autograd_Fn, C_EB_Linear_Autograd_Fn
from model.Twitter.BiGAT_Twitter import PostLevelAttention, PostLevelAttention2
from argparse import ArgumentParser


class LRPPostLevelAttention(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(LRPPostLevelAttention, self).__init__()
        self.module = module
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']
        self.module.saved_rels = kwargs.get('saved_rels', None)

    def forward(self, x):
        return self.autograd_fn.apply(x, self.module, self.params)


class LRPPostLevelAttention2(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(LRPPostLevelAttention2, self).__init__()
        self.module = module
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']
        self.module.saved_rels = kwargs.get('saved_rels', None)

    def forward(self, x):
        return self.autograd_fn.apply(x, self.module, self.params)


class LRPPostLevelAttention_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        eps = params.get('ebgcn', 1e-6)
        if isinstance(module, PostLevelAttention):
            is_post_attn1 = True
        else:
            is_post_attn1 = False

        def config_values_to_tensors(module):
            if isinstance(module, PostLevelAttention):
                property_names = ['in_feats', 'out_feats']
            elif isinstance(module, PostLevelAttention2):
                property_names = ['hidden_feats', 'claim_feats', 'out_feats']
            else:
                print('Error: module not CHGAT PostLevelAttention block')
                raise Exception

            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.claim_gate.weight.device)
                else:
                    print('error: property value is neither int nor tuple', attr, value)
                    # exit()
                values.append(value)
            return property_names, values

        property_names, values = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)
        # print(eps_tensor)

        # PostLevelAttention
        claim_gate_weight = module.claim_gate.weight.data.clone()
        post_gate_weight = module.post_gate.weight.data.clone()

        ctx.save_for_backward(x, claim_gate_weight, post_gate_weight,
                              eps_tensor, *values)
        ctx.saved_rels = module.saved_rels
        ctx.is_post_attn1 = is_post_attn1

        # print('EBGCNRumourGCN ctx.needs_input_grad', ctx.needs_input_grad)

        # data = Data(x=x,
        #             batch=torch.zeros(x.shape[0]).long().to(x.device))
        # data.requires_grad = True
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, claim_gate_weight, post_gate_weight, \
        eps_tensor, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        is_post_attn1 = ctx.is_post_attn1
        # print('retrieved', len(values))

        def tensors_to_dict(values, is_post_attn1):
            if is_post_attn1:
                property_names = ['in_feats', 'out_feats']
            else:
                property_names = ['hidden_feats', 'claim_feats', 'out_feats']
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
            else:
                new_values = values[i+1:]
            return params_dict, new_values

        params_dict, values = tensors_to_dict(values, is_post_attn1)

        parser = ArgumentParser()
        args = parser.parse_args()
        for k, v in params_dict.items():
            args.__setattr__(k, v)

        # print('EBGCNRumourGCN custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        if is_post_attn1:
            module = PostLevelAttention(**params_dict).to(input_.device)
            module.ref_name = 'post_attn1'
        else:
            module = PostLevelAttention2(**params_dict).to(input_.device)
            module.ref_name = 'post_attn2'
        module.post_gate.weight = nn.Parameter(post_gate_weight)
        module.claim_gate.weight = nn.Parameter(claim_gate_weight)

        module.post_gate = LRPLinear(module.post_gate, Linear_Epsilon_Autograd_Fn, params={'linear_eps': eps},
                               ref_name='post_gate', saved_rels=saved_rels)
        module.claim_gate = LRPLinear(module.claim_gate, Linear_Epsilon_Autograd_Fn, params={'linear_eps': eps},
                               ref_name='claim_gate', saved_rels=saved_rels)

        X = input_.clone().detach().requires_grad_(True)

        # data = Data(x=X,
        #             batch=torch.zeros(input_.shape[0]).long().to(input_.device))
        R = lrp_postattn(input_=X,
                         layer=module,
                         relevance_output=grad_output,
                         eps0=eps,
                         eps=eps)
        # print('EBGCNRumourGCN custom R', R.shape)
        return R, None, None, None, None, None


class CAM_PostLevelAttention_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        eps = params.get('ebgcn', 1e-6)
        if isinstance(module, PostLevelAttention):
            is_post_attn1 = True
        else:
            is_post_attn1 = False

        def config_values_to_tensors(module):
            if isinstance(module, PostLevelAttention):
                property_names = ['in_feats', 'out_feats']
            elif isinstance(module, PostLevelAttention2):
                property_names = ['hidden_feats', 'claim_feats', 'out_feats']
            else:
                print('Error: module not CHGAT PostLevelAttention block')
                raise Exception

            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.claim_gate.weight.device)
                else:
                    print('error: property value is neither int nor tuple', attr, value)
                    # exit()
                values.append(value)
            return property_names, values

        property_names, values = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)
        # print(eps_tensor)

        # PostLevelAttention
        claim_gate_weight = module.claim_gate.weight.data.clone()
        post_gate_weight = module.post_gate.weight.data.clone()

        ctx.save_for_backward(x, claim_gate_weight, post_gate_weight,
                              eps_tensor, *values)
        ctx.saved_rels = module.saved_rels
        ctx.is_post_attn1 = is_post_attn1

        # print('EBGCNRumourGCN ctx.needs_input_grad', ctx.needs_input_grad)

        # data = Data(x=x,
        #             batch=torch.zeros(x.shape[0]).long().to(x.device))
        # data.requires_grad = True
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, claim_gate_weight, post_gate_weight, \
        eps_tensor, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        is_post_attn1 = ctx.is_post_attn1
        # print('retrieved', len(values))

        def tensors_to_dict(values, is_post_attn1):
            if is_post_attn1:
                property_names = ['in_feats', 'out_feats']
            else:
                property_names = ['hidden_feats', 'claim_feats', 'out_feats']
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
            else:
                new_values = values[i+1:]
            return params_dict, new_values

        params_dict, values = tensors_to_dict(values, is_post_attn1)

        parser = ArgumentParser()
        args = parser.parse_args()
        for k, v in params_dict.items():
            args.__setattr__(k, v)

        # print('EBGCNRumourGCN custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        if is_post_attn1:
            module = PostLevelAttention(**params_dict).to(input_.device)
            module.ref_name = 'post_attn1'
        else:
            module = PostLevelAttention2(**params_dict).to(input_.device)
            module.ref_name = 'post_attn2'
        module.post_gate.weight = nn.Parameter(post_gate_weight)
        module.claim_gate.weight = nn.Parameter(claim_gate_weight)

        module.post_gate = LRPLinear(module.post_gate, CAM_Linear_Autograd_Fn, params={'linear_eps': eps},
                               ref_name='post_gate', saved_rels=saved_rels)
        module.claim_gate = LRPLinear(module.claim_gate, CAM_Linear_Autograd_Fn, params={'linear_eps': eps},
                               ref_name='claim_gate', saved_rels=saved_rels)

        X = input_.clone().detach().requires_grad_(True)

        # data = Data(x=X)
        R = cam_postattn(input_=X,
                         layer=module,
                         relevance_output=grad_output)
        # print('EBGCNRumourGCN custom R', R.shape)
        return R, None, None, None, None, None


class EB_PostLevelAttention_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        eps = params.get('ebgcn', 1e-6)
        if isinstance(module, PostLevelAttention):
            is_post_attn1 = True
        else:
            is_post_attn1 = False

        def config_values_to_tensors(module):
            if isinstance(module, PostLevelAttention):
                property_names = ['in_feats', 'out_feats']
            elif isinstance(module, PostLevelAttention2):
                property_names = ['hidden_feats', 'claim_feats', 'out_feats']
            else:
                print('Error: module not CHGAT PostLevelAttention block')
                raise Exception

            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.claim_gate.weight.device)
                else:
                    print('error: property value is neither int nor tuple', attr, value)
                    # exit()
                values.append(value)
            return property_names, values

        property_names, values = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)
        # print(eps_tensor)

        # PostLevelAttention
        claim_gate_weight = module.claim_gate.weight.data.clone()
        post_gate_weight = module.post_gate.weight.data.clone()

        ctx.save_for_backward(x, claim_gate_weight, post_gate_weight,
                              eps_tensor, *values)
        ctx.saved_rels = module.saved_rels
        ctx.is_post_attn1 = is_post_attn1

        # print('EBGCNRumourGCN ctx.needs_input_grad', ctx.needs_input_grad)

        # data = Data(x=x,
        #             batch=torch.zeros(x.shape[0]).long().to(x.device))
        # data.requires_grad = True
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, claim_gate_weight, post_gate_weight, \
        eps_tensor, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        is_post_attn1 = ctx.is_post_attn1
        # print('retrieved', len(values))

        def tensors_to_dict(values, is_post_attn1):
            if is_post_attn1:
                property_names = ['in_feats', 'out_feats']
            else:
                property_names = ['hidden_feats', 'claim_feats', 'out_feats']
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
            else:
                new_values = values[i+1:]
            return params_dict, new_values

        params_dict, values = tensors_to_dict(values, is_post_attn1)

        parser = ArgumentParser()
        args = parser.parse_args()
        for k, v in params_dict.items():
            args.__setattr__(k, v)

        # print('EBGCNRumourGCN custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        if is_post_attn1:
            module = PostLevelAttention(**params_dict).to(input_.device)
            module.ref_name = 'post_attn1'
        else:
            module = PostLevelAttention2(**params_dict).to(input_.device)
            module.ref_name = 'post_attn2'
        module.post_gate.weight = nn.Parameter(post_gate_weight)
        module.claim_gate.weight = nn.Parameter(claim_gate_weight)

        module.post_gate = LRPLinear(module.post_gate, EB_Linear_Autograd_Fn, params={'linear_eps': eps},
                               ref_name='post_gate', saved_rels=saved_rels)
        module.claim_gate = LRPLinear(module.claim_gate, EB_Linear_Autograd_Fn, params={'linear_eps': eps},
                               ref_name='claim_gate', saved_rels=saved_rels)

        X = input_.clone().detach().requires_grad_(True)

        # data = Data(x=X)
        R = eb_postattn(input_=X,
                        layer=module,
                        relevance_output=grad_output)
        # print('EBGCNRumourGCN custom R', R.shape)
        return R, None, None, None, None, None


def lrp_postattn(input_, layer, relevance_output, eps0, eps):
    if input_.grad is not None:
        input_.grad.zero()
    relevance_output_data = relevance_output.clone().detach()
    with torch.enable_grad():
        Z = layer(input_)
    # print(Z)
    S = safe_divide(relevance_output_data, Z.clone().detach(), eps0, eps)
    # Z.backward(S, retain_graph=True)
    Z.backward(S)
    relevance_input = input_.data * input_.grad
    return relevance_input


def cam_postattn(input_, layer, relevance_output):
    if input_.grad is not None:
        input_.grad.zero()
    relevance_output_data = relevance_output.clone().detach()
    with torch.enable_grad():
        Z = layer(input_)
    Z.backward(relevance_output_data)
    relevance_input = F.relu(input_.data * input_.grad)
    # layer.saved_relevance = relevance_output
    return relevance_input


def eb_postattn(input_, layer, relevance_output):
    if input_.grad is not None:
        input_.grad.zero()
    with torch.enable_grad():
        Z = layer(input_)  # X = W^{+}^T * A_{n}
    relevance_output_data = relevance_output.clone().detach()  # P_{n-1}
    X = Z.clone().detach()#.sum()
    Y = relevance_output_data / X  # Y = P_{n-1} (/) X
    Z.backward(Y)  # Use backward pass to compute Z = W^{+} * Y
    relevance_input = input_.data * input_.grad  # P_{n} = A_{n} (*) Z
    # layer.saved_relevance = relevance_input
    return relevance_input


class LRPCHGAT(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(LRPCHGAT, self).__init__()
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']
        self.saved_rels = {}
        self.module = module
        module.conv1.saved_rels = self.saved_rels
        module.conv2.saved_rels = self.saved_rels
        module.post_attn1.saved_rels = self.saved_rels
        module.post_attn1.saved_rels = self.saved_rels
        module.fc.saved_rels = self.saved_rels
        module.clf.saved_rels = self.saved_rels

        self.conv1 = LRPGATv2Conv(module.conv1, GATv2Conv_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                                  ref_name='conv1')
        self.conv2 = LRPGATv2Conv(module.conv2, GATv2Conv_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                                  ref_name='conv2')
        self.post_attn1 = LRPPostLevelAttention(module.post_attn1, LRPPostLevelAttention_Autograd_Fn, params=self.params,
                                                saved_rels=self.saved_rels)
        self.post_attn2 = LRPPostLevelAttention2(module.post_attn2, LRPPostLevelAttention_Autograd_Fn, params=self.params,
                                                 saved_rels=self.saved_rels)
        self.fc = LRPLinear(module.fc, Linear_Epsilon_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                            ref_name='fc')
        self.clf = LRPLinear(module.clf, Linear_Epsilon_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                            ref_name='clf')

    def forward(self, data, skip_layer_2=False, layer_1_no_self_loops=False, layer_2_no_self_loops=False):
        # out = self.module.forward(data, skip_layer_2, layer_1_no_self_loops, layer_2_no_self_loops)
        # data.x.requires_grad = True
        x, edge_index, BU_edge_index = data.x, data.edge_index, data.BU_edge_index
        root = data.root
        merged_edge_index = torch.cat((edge_index, BU_edge_index), dim=-1).to(x.device)
        x1 = copy.copy(x.float())
        # x_copy = copy.copy(x.float())

        rootindex = data.rootindex
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(x.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            # root_extend[index] = x1[rootindex[num_batch]]
            root_extend[index] = root[num_batch]
        x = torch.cat((x, root_extend), 1)  # 768 + 768

        # x, (_, self.attention_weights1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.post_attn1(x)  # 128
        # print(x)
        x = self.conv1(x, merged_edge_index, skip_layer_2=False, layer_1_no_self_loops=layer_1_no_self_loops,
                       layer_2_no_self_loops=layer_2_no_self_loops)  # 128 * 4
        self.emb1 = x
        # self.emb1.requires_grad = True
        # self.emb1.retain_grad()
        x2 = copy.copy(x)
        x = F.relu(x)  # 128 * 4

        slice_len = x.shape[-1] // self.module.K
        h_list = []
        for k in range(self.module.K):
            if k != self.module.K - 1:
                x_slice = x[:, k * slice_len:(k + 1) * slice_len]
            else:
                x_slice = x[:, k * slice_len:]
            # print(slice_len, x_slice.shape, k * slice_len, (k+1) * slice_len)
            root_extend = torch.zeros(len(data.batch), x1.size(1)).to(x.device)
            # print(root_extend.shape)
            # print(rootindex.shape, x2.shape)
            for num_batch in range(batch_size):
                index = (torch.eq(data.batch, num_batch))
                # print(rootindex[num_batch])
                # root_extend[index] = x1[rootindex[num_batch]]
                root_extend[index] = root[num_batch]
            x_slice = torch.cat((x_slice, root_extend), 1)  # 128 + 768
            h_list.append(self.post_attn2(x_slice))  # 128
        x = torch.cat(h_list, -1)  # 128 * 4

        # x = F.dropout(x, training=self.training)

        # x, (_, self.attention_weights2) = self.conv2(x, edge_index, return_attention_weights=True)

        x = self.conv2(x, merged_edge_index, skip_layer_2=skip_layer_2, layer_1_no_self_loops=layer_1_no_self_loops,
                       layer_2_no_self_loops=layer_2_no_self_loops)  # 64
        self.emb2 = x
        x = F.relu(x)

        # self.emb2.requires_grad = True
        # self.emb2.retain_grad()

        root_embed = torch.zeros(len(data.ptr) - 1, x.size(1)).to(x.device)
        for index in range(len(data.ptr) - 1):
            # index = (torch.eq(data.batch, num_batch))
            root_embed[index] = x[data.ptr[index]]
        h_c = torch.zeros(len(data.batch), x.size(1)).to(x.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            h_c[index] = root_embed[num_batch]
        h_prod = x * h_c
        h_diff = torch.abs(h_c - x)
        h_joint = torch.cat((h_c, x, h_prod, h_diff), 1)
        # print(h_joint.shape)
        h_joint = F.tanh(self.fc(h_joint))
        beta = F.softmax(h_joint, dim=-1)
        s_hat = beta * x
        # print(s_hat.shape)
        s_hat = scatter_mean(s_hat, data.batch, dim=0)
        s = scatter_mean(x, data.batch, dim=0)
        s = torch.cat((s_hat, s), 1)
        self.x = s
        # self.x.requires_grad = True
        # self.x.retain_grad()
        self.out = self.clf(s)
        out = F.log_softmax(self.out, dim=1)
        return out


class CAM_CHGAT(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(CAM_CHGAT, self).__init__()
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']
        self.saved_rels = {}
        self.module = module
        module.conv1.saved_rels = self.saved_rels
        module.conv2.saved_rels = self.saved_rels
        module.post_attn1.saved_rels = self.saved_rels
        module.post_attn1.saved_rels = self.saved_rels
        module.fc.saved_rels = self.saved_rels
        module.clf.saved_rels = self.saved_rels

        self.conv1 = LRPGATv2Conv(module.conv1, CAM_GATv2Conv_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                                  ref_name='conv1')
        self.conv2 = LRPGATv2Conv(module.conv2, CAM_GATv2Conv_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                                  ref_name='conv2')
        self.post_attn1 = LRPPostLevelAttention(module.post_attn1, CAM_PostLevelAttention_Autograd_Fn, params=self.params,
                                                saved_rels=self.saved_rels)
        self.post_attn2 = LRPPostLevelAttention2(module.post_attn2, CAM_PostLevelAttention_Autograd_Fn, params=self.params,
                                                 saved_rels=self.saved_rels)
        self.fc = LRPLinear(module.fc, CAM_Linear_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                            ref_name='fc')
        self.clf = LRPLinear(module.clf, CAM_Linear_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                            ref_name='clf')

    def forward(self, data, skip_layer_2=False, layer_1_no_self_loops=False, layer_2_no_self_loops=False):
        # data.x.requires_grad = True
        x, edge_index, BU_edge_index = data.x, data.edge_index, data.BU_edge_index
        root = data.root
        merged_edge_index = torch.cat((edge_index, BU_edge_index), dim=-1).to(x.device)
        x1 = copy.copy(x.float())
        x_copy = copy.copy(x.float())

        rootindex = data.rootindex
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(x.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            # root_extend[index] = x1[rootindex[num_batch]]
            root_extend[index] = root[num_batch]
        x = torch.cat((x_copy, root_extend), 1)  # 768 + 768

        # x, (_, self.attention_weights1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.post_attn1(x)  # 128
        # print(x)
        x = self.conv1(x, merged_edge_index, skip_layer_2=False, layer_1_no_self_loops=layer_1_no_self_loops,
                       layer_2_no_self_loops=layer_2_no_self_loops)  # 128 * 4
        self.emb1 = x
        # self.emb1.requires_grad = True
        # self.emb1.retain_grad()
        x2 = copy.copy(x)
        x = F.relu(x)  # 128 * 4

        slice_len = x.shape[-1] // self.module.K
        h_list = []
        for k in range(self.module.K):
            if k != self.module.K - 1:
                x_slice = x[:, k * slice_len:(k + 1) * slice_len]
            else:
                x_slice = x[:, k * slice_len:]
            # print(slice_len, x_slice.shape, k * slice_len, (k+1) * slice_len)
            root_extend = torch.zeros(len(data.batch), x1.size(1)).to(x.device)
            # print(root_extend.shape)
            # print(rootindex.shape, x2.shape)
            for num_batch in range(batch_size):
                index = (torch.eq(data.batch, num_batch))
                # print(rootindex[num_batch])
                # root_extend[index] = x1[rootindex[num_batch]]
                root_extend[index] = root[num_batch]
            x_slice = torch.cat((x_slice, root_extend), 1)  # 128 + 768
            h_list.append(self.post_attn2(x_slice))  # 128
        x = torch.cat(h_list, -1)  # 128 * 4

        # x = F.dropout(x, training=self.training)

        # x, (_, self.attention_weights2) = self.conv2(x, edge_index, return_attention_weights=True)

        x = self.conv2(x, merged_edge_index, skip_layer_2=skip_layer_2, layer_1_no_self_loops=layer_1_no_self_loops,
                       layer_2_no_self_loops=layer_2_no_self_loops)  # 64
        self.emb2 = x
        x = F.relu(x)

        # self.emb2.requires_grad = True
        # self.emb2.retain_grad()

        root_embed = torch.zeros(len(data.ptr) - 1, x.size(1)).to(x.device)
        for index in range(len(data.ptr) - 1):
            # index = (torch.eq(data.batch, num_batch))
            root_embed[index] = x[data.ptr[index]]
        h_c = torch.zeros(len(data.batch), x.size(1)).to(x.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            h_c[index] = root_embed[num_batch]
        h_prod = x * h_c
        h_diff = torch.abs(h_c - x)
        h_joint = torch.cat((h_c, x, h_prod, h_diff), 1)
        # print(h_joint.shape)
        h_joint = F.tanh(self.fc(h_joint))
        beta = F.softmax(h_joint, dim=-1)
        s_hat = beta * x
        # print(s_hat.shape)
        s_hat = scatter_mean(s_hat, data.batch, dim=0)
        s = scatter_mean(x, data.batch, dim=0)
        s = torch.cat((s_hat, s), 1)
        self.x = s
        # self.x.requires_grad = True
        # self.x.retain_grad()
        self.out = self.clf(s)
        out = F.log_softmax(self.out, dim=1)
        return out


class EB_CHGAT(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(EB_CHGAT, self).__init__()
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']
        self.is_contrastive = kwargs.get('is_contrastive', False)
        self.saved_rels = {}
        self.module = module
        module.conv1.saved_rels = self.saved_rels
        module.conv2.saved_rels = self.saved_rels
        module.post_attn1.saved_rels = self.saved_rels
        module.post_attn1.saved_rels = self.saved_rels
        module.fc.saved_rels = self.saved_rels
        module.clf.saved_rels = self.saved_rels

        self.conv1 = LRPGATv2Conv(module.conv1, EB_GATv2Conv_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                                  ref_name='conv1')
        self.conv2 = LRPGATv2Conv(module.conv2, EB_GATv2Conv_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                                  ref_name='conv2')
        self.post_attn1 = LRPPostLevelAttention(module.post_attn1, EB_PostLevelAttention_Autograd_Fn, params=self.params,
                                                saved_rels=self.saved_rels)
        self.post_attn2 = LRPPostLevelAttention2(module.post_attn2, EB_PostLevelAttention_Autograd_Fn, params=self.params,
                                                 saved_rels=self.saved_rels)
        self.fc = LRPLinear(module.fc, EB_Linear_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                            ref_name='fc')
        if self.is_contrastive:
            with torch.no_grad():  # contrastive, flip weights
                module.clf.weight.copy_(-module.clf.weight.float())
            self.clf = LRPLinear(module.clf, EB_Linear_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                                 ref_name='clf')
        else:
            self.clf = LRPLinear(module.clf, C_EB_Linear_Autograd_Fn, params=self.params, saved_rels=self.saved_rels,
                                 ref_name='clf')

    def forward(self, data, skip_layer_2=False, layer_1_no_self_loops=False, layer_2_no_self_loops=False):
        # data.x.requires_grad = True
        x, edge_index, BU_edge_index = data.x, data.edge_index, data.BU_edge_index
        root = data.root
        merged_edge_index = torch.cat((edge_index, BU_edge_index), dim=-1).to(x.device)
        x1 = copy.copy(x.float())
        # x_copy = copy.copy(x.float())

        rootindex = data.rootindex
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(x.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            # root_extend[index] = x1[rootindex[num_batch]]
            root_extend[index] = root[num_batch]
        x = torch.cat((x, root_extend), 1)  # 768 + 768

        # x, (_, self.attention_weights1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.post_attn1(x)  # 128
        # print(x)
        x = self.conv1(x, merged_edge_index, skip_layer_2=False, layer_1_no_self_loops=layer_1_no_self_loops,
                       layer_2_no_self_loops=layer_2_no_self_loops)  # 128 * 4
        self.emb1 = x
        # self.emb1.requires_grad = True
        # self.emb1.retain_grad()
        x2 = copy.copy(x)
        x = F.relu(x)  # 128 * 4

        slice_len = x.shape[-1] // self.module.K
        h_list = []
        for k in range(self.module.K):
            if k != self.module.K - 1:
                x_slice = x[:, k * slice_len:(k + 1) * slice_len]
            else:
                x_slice = x[:, k * slice_len:]
            # print(slice_len, x_slice.shape, k * slice_len, (k+1) * slice_len)
            root_extend = torch.zeros(len(data.batch), x1.size(1)).to(x.device)
            # print(root_extend.shape)
            # print(rootindex.shape, x2.shape)
            for num_batch in range(batch_size):
                index = (torch.eq(data.batch, num_batch))
                # print(rootindex[num_batch])
                # root_extend[index] = x1[rootindex[num_batch]]
                root_extend[index] = root[num_batch]
            x_slice = torch.cat((x_slice, root_extend), 1)  # 128 + 768
            h_list.append(self.post_attn2(x_slice))  # 128
        x = torch.cat(h_list, -1)  # 128 * 4

        # x = F.dropout(x, training=self.training)

        # x, (_, self.attention_weights2) = self.conv2(x, edge_index, return_attention_weights=True)

        x = self.conv2(x, merged_edge_index, skip_layer_2=skip_layer_2, layer_1_no_self_loops=layer_1_no_self_loops,
                       layer_2_no_self_loops=layer_2_no_self_loops)  # 64
        self.emb2 = x
        x = F.relu(x)

        # self.emb2.requires_grad = True
        # self.emb2.retain_grad()

        root_embed = torch.zeros(len(data.ptr) - 1, x.size(1)).to(x.device)
        for index in range(len(data.ptr) - 1):
            # index = (torch.eq(data.batch, num_batch))
            root_embed[index] = x[data.ptr[index]]
        h_c = torch.zeros(len(data.batch), x.size(1)).to(x.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            h_c[index] = root_embed[num_batch]
        h_prod = x * h_c
        h_diff = torch.abs(h_c - x)
        h_joint = torch.cat((h_c, x, h_prod, h_diff), 1)
        # print(h_joint.shape)
        h_joint = F.tanh(self.fc(h_joint))
        beta = F.softmax(h_joint, dim=-1)
        s_hat = beta * x
        # print(s_hat.shape)
        s_hat = scatter_mean(s_hat, data.batch, dim=0)
        s = scatter_mean(x, data.batch, dim=0)
        s = torch.cat((s_hat, s), 1)
        self.x = s
        # self.x.requires_grad = True
        # self.x.retain_grad()
        self.out = self.clf(s)
        out = F.log_softmax(self.out, dim=1)
        return out
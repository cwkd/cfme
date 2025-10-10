import sys, os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
sys.path.append(os.getcwd())
from Process.process import *
import torch
from torch_scatter import scatter_mean
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from Process.pheme9fold import *
from tools.evaluate import *
from torch_geometric.nn import GATv2Conv, GATConv
import copy


class TDrumorGAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, device):
        super(TDrumorGAT, self).__init__()
        self.conv1 = GATv2Conv(in_feats, hid_feats, heads=4, concat=False)
        self.conv2 = GATv2Conv(hid_feats + in_feats, out_feats, heads=4, concat=False)
        self.device = device

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        # x, (_, self.attention_weights1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.conv1(x, edge_index)
        # print(x.shape)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # print(x.shape)
        # x, (_, self.attention_weights2) = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # print(x.shape)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        self.x = x
        x = torch.cat((x, root_extend), 1)
        # print(x.shape)
        x = scatter_mean(x, data.batch, dim=0)
        # print(x.shape)
        return x


class BUrumorGAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, device):
        super(BUrumorGAT, self).__init__()
        self.conv1 = GATv2Conv(in_feats, hid_feats, heads=4, concat=False)
        self.conv2 = GATv2Conv(hid_feats + in_feats, out_feats, heads=4, concat=False)
        self.device = device

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())

        # x, (_, self.attention_weights1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        rootindex = data.rootindex
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # x, (_, self.attention_weights2) = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        self.x = x
        x = torch.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        return x


class BiGAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_class=4, device=None):
        super(BiGAT, self).__init__()
        self.TDrumorGAT = TDrumorGAT(in_feats, hid_feats, out_feats, device)
        self.BUrumorGAT = BUrumorGAT(in_feats, hid_feats, out_feats, device)
        self.fc = torch.nn.Linear((out_feats + hid_feats) * 2, num_class)
        self.device = device

    def forward(self, data):
        TD_x = self.TDrumorGAT(data)
        BU_x = self.BUrumorGAT(data)
        self.x = torch.cat((BU_x, TD_x), 1)
        out = self.fc(self.x)
        out = F.log_softmax(out, dim=1)
        return out

    def get_node_embeds(self, data, layer=1):
        self.forward(data)
        if layer == 1:
            td_x = self.TDrumorGAT.emb1
            bu_x = self.BUrumorGAT.emb1
        elif layer == 2:
            td_x = self.TDrumorGAT.emb2
            bu_x = self.BUrumorGAT.emb2
        return torch.cat((td_x, bu_x), 1)


class PostLevelAttention(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(PostLevelAttention, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.claim_gate = nn.Linear(in_feats, out_feats, bias=False)
        self.post_gate = nn.Linear(in_feats, out_feats, bias=False)

    def forward(self, x):
        half = x.shape[-1] // 2  # 768
        # print(x[:, :half])
        # print(x[:, half:])
        # print(self.post_gate.weight)
        h_x = self.post_gate(x[:, :half])   # 64, 64
        h_c = self.claim_gate(x[:, half:])
        # print(h_x.shape, h_c.shape)
        g = F.sigmoid(h_x + h_c)
        # print(h_x.shape, h_c.shape, g.shape)
        h = g * h_x + (1 - g) * h_c

        return torch.cat((h, h_x), -1)  # 128


class PostLevelAttention2(nn.Module):
    def __init__(self, hidden_feats, claim_feats, out_feats):
        super(PostLevelAttention2, self).__init__()
        self.hidden_feats = hidden_feats
        self.claim_feats = claim_feats
        self.out_feats = out_feats
        self.split = out_feats  # 64
        self.claim_gate = nn.Linear(claim_feats, out_feats, bias=False)
        self.post_gate = nn.Linear(hidden_feats, out_feats, bias=False)

    def forward(self, x):
        # print(self.split, x[:, :self.split].shape, x[:, self.split:].shape)
        # print(self.post_gate.weight.shape, self.claim_gate.weight.shape)
        h_x = self.post_gate(x[:, :self.split])  # 64, 64
        h_c = self.claim_gate(x[:, self.split:])
        g = F.sigmoid(h_x + h_c)
        # print(h_x.shape, h_c.shape, g.shape)
        h = g * h_x + (1 - g) * h_c

        return torch.cat((h, h_x), -1)  # 128


class CHGAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_class=4, device=None):
        super(CHGAT, self).__init__()
        self.K = 4
        self.conv1 = GATConv(hid_feats * 2, hid_feats, heads=self.K, concat=True)
        self.conv2 = GATConv(hid_feats * 8, out_feats, heads=self.K, concat=False)
        self.post_attn1 = PostLevelAttention(in_feats, hid_feats)
        self.post_attn2 = PostLevelAttention2(hid_feats, in_feats, hid_feats)
        self.fc = nn.Linear(out_feats * 4, out_feats)
        self.clf = nn.Linear(out_feats * 2, num_class)
        self.device = device

    def forward(self, data, skip_layer_2=False, layer_1_no_self_loops=False, layer_2_no_self_loops=False):
        # self.x0 = data.x
        # self.x0.retain_grad()
        x, edge_index, BU_edge_index = data.x, data.edge_index, data.BU_edge_index
        root = data.root
        merged_edge_index = torch.cat((edge_index, BU_edge_index), dim=-1).to(self.device)
        x1 = copy.copy(x.float())
        # x_copy = copy.copy(x.float())

        rootindex = data.rootindex
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            # root_extend[index] = x1[rootindex[num_batch]]
            root_extend[index] = root[num_batch]
        x = torch.cat((x, root_extend), 1)  # 768 + 768
        # self.x_in = x
        # self.x_in.retain_grad()
        # x, (_, self.attention_weights1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.post_attn1(x)  # 128
        # print(x)
        if layer_1_no_self_loops:
            self.conv1.add_self_loops = False
        x = self.conv1(x, merged_edge_index)  # 128 * 4
        self.emb1 = x
        # self.emb1.requires_grad = True
        # self.emb1.retain_grad()
        x2 = copy.copy(x)
        x = F.relu(x)  # 128 * 4

        slice_len = x.shape[-1] // self.K
        h_list = []
        for k in range(self.K):
            if k != self.K - 1:
                x_slice = x[:, k * slice_len:(k+1) * slice_len]
            else:
                x_slice = x[:, k * slice_len:]
            # print(slice_len, x_slice.shape, k * slice_len, (k+1) * slice_len)
            root_extend = torch.zeros(len(data.batch), x1.size(1)).to(self.device)
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
        if layer_2_no_self_loops:
            self.conv2.add_self_loops = False
        if skip_layer_2:
            temp_edge_index = torch.as_tensor([[], []], dtype=merged_edge_index.dtype, device=merged_edge_index.device)
            x = self.conv2(x, temp_edge_index)  # 64
        else:
            x = self.conv2(x, merged_edge_index)  # 64
        self.emb2 = x
        x = F.relu(x)

        # self.emb2.requires_grad = True
        # self.emb2.retain_grad()

        root_embed = torch.zeros(len(data.ptr) - 1, x.size(1)).to(self.device)
        for index in range(len(data.ptr) - 1):
            # index = (torch.eq(data.batch, num_batch))
            root_embed[index] = x[data.ptr[index]]
        h_c = torch.zeros(len(data.batch), x.size(1)).to(self.device)
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

    def get_node_embeds(self, data, layer=1):
        self.forward(data)
        return self.get_node_embeds_after_forward(layer=layer)

    def get_node_embeds_after_forward(self, layer=1):
        if layer == 1:
            x = self.emb1
        elif layer == 2:
            x = self.emb2
        elif layer == 3:
            x = self.x
        return x


class CHGATv2(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_class=4, device=None):
        super(CHGATv2, self).__init__()
        self.K = 4
        self.conv1 = GATConv(hid_feats * 2, hid_feats, heads=self.K, concat=True)
        self.conv2 = GATConv(hid_feats * 8, out_feats, heads=self.K, concat=False)
        self.post_attn1 = PostLevelAttention(in_feats, hid_feats)
        self.post_attn2 = PostLevelAttention2(hid_feats, in_feats, hid_feats)
        self.fc = nn.Linear(out_feats * 4, out_feats)
        self.clf = nn.Linear(out_feats * 2, num_class)
        self.device = device

    def forward(self, x, edge_index, batch_data, skip_layer_2=False, layer_1_no_self_loops=False, layer_2_no_self_loops=False):
        data = batch_data
        # self.x0 = data.x
        # self.x0.retain_grad()
        x, edge_index, BU_edge_index = data.x, data.edge_index, data.BU_edge_index
        root = data.root
        merged_edge_index = torch.cat((edge_index, BU_edge_index), dim=-1).to(self.device)
        x1 = copy.copy(x.float())
        # x_copy = copy.copy(x.float())

        rootindex = data.rootindex
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            # root_extend[index] = x1[rootindex[num_batch]]
            root_extend[index] = root[num_batch]
        x = torch.cat((x, root_extend), 1)  # 768 + 768
        # self.x_in = x
        # self.x_in.retain_grad()
        # x, (_, self.attention_weights1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.post_attn1(x)  # 128
        # print(x)
        if layer_1_no_self_loops:
            self.conv1.add_self_loops = False
        x = self.conv1(x, merged_edge_index)  # 128 * 4
        self.emb1 = x
        # self.emb1.requires_grad = True
        # self.emb1.retain_grad()
        x2 = copy.copy(x)
        x = F.relu(x)  # 128 * 4

        slice_len = x.shape[-1] // self.K
        h_list = []
        for k in range(self.K):
            if k != self.K - 1:
                x_slice = x[:, k * slice_len:(k+1) * slice_len]
            else:
                x_slice = x[:, k * slice_len:]
            # print(slice_len, x_slice.shape, k * slice_len, (k+1) * slice_len)
            root_extend = torch.zeros(len(data.batch), x1.size(1)).to(self.device)
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
        if layer_2_no_self_loops:
            self.conv2.add_self_loops = False
        if skip_layer_2:
            temp_edge_index = torch.as_tensor([[], []], dtype=merged_edge_index.dtype, device=merged_edge_index.device)
            x = self.conv2(x, temp_edge_index)  # 64
        else:
            x = self.conv2(x, merged_edge_index)  # 64
        self.emb2 = x
        x = F.relu(x)

        # self.emb2.requires_grad = True
        # self.emb2.retain_grad()

        root_embed = torch.zeros(len(data.ptr) - 1, x.size(1)).to(self.device)
        for index in range(len(data.ptr) - 1):
            # index = (torch.eq(data.batch, num_batch))
            root_embed[index] = x[data.ptr[index]]
        h_c = torch.zeros(len(data.batch), x.size(1)).to(self.device)
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

    def get_node_embeds(self, data, layer=1):
        self.forward(data)
        return self.get_node_embeds_after_forward(layer=layer)

    def get_node_embeds_after_forward(self, layer=1):
        if layer == 1:
            x = self.emb1
        elif layer == 2:
            x = self.emb2
        elif layer == 3:
            x = self.x
        return x
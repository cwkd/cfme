import sys, os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
sys.path.append(os.getcwd())
from Process.process import *
import torch
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from Process.pheme9fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
import copy


class TDrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, device):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.device = device
        self.conv1.ref_name = 'td_conv1'
        self.conv2.ref_name = 'td_conv2'
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def forward(self, data):
        # print(data)
        # print(data.batch)
        x, edge_index = data.x, data.edge_index
        root = data.root
        try:
            edge_weights = data.edge_weights
        except:
            edge_weights = None
        x1 = copy.copy(x.float())
        # print('x1', x1.shape)
        # if x.shape[0] > 1:
        #     print("TD")
        #     print("x shape:", x.shape)
        #     print("Edge_index shape:", edge_index.shape)
        #     print("Edge_index:", edge_index)
        #     try:
        #         print("Maxes within edge:", th.max(edge_index, dim=1)[0])
        #         for item in th.max(edge_index, dim=1)[0]:
        #             if item >= x.shape[0]:
        #                 print("VIOLATION:", item, "    VS     ", x.shape)
        #     except IndexError:
        #         print("no links.")
        # print(x.shape)
        # print(x.device, edge_index.device, self.device)
        if edge_weights is None:
            try:
                x = self.conv1(x, edge_index)
            except:
                print(x.shape, edge_index.shape)
                print(x.device, edge_index.device)
                raise Exception
        else:
            x = self.conv1(x, edge_index, edge_weights)
        self.emb1 = x
        # self.emb1.requires_grad = True
        # self.emb1.retain_grad()
        # print('x', x.shape)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        # root_extend = th.zeros(x1.size(0), x1.size(1)).to(self.device)
        # batch_size = rootindex.size(0)
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            # print(num_batch, rootindex[num_batch], 'root_extend', root_extend.shape, x1.shape)
            # print(x1[rootindex[num_batch]].shape)
            # root_extend[index] = x1[rootindex[num_batch]]
            root_extend[index] = root[num_batch]
            # break
        # print(x.shape, root_extend.shape)
        x = torch.cat((x, root_extend), 1)

        # x = F.relu(x)
        x = self.relu1(x)
        x = F.dropout(x, training=self.training)
        # print(x.shape)
        if edge_weights is None:
            x = self.conv2(x, edge_index)
        else:
            x = self.conv2(x, edge_index, edge_weights)
        # x = F.relu(x)
        x = self.relu2(x)
        # print(x.shape)
        # root_extend = th.zeros(x1.size(0), x2.size(1)).to(self.device)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        self.emb2 = x
        # self.emb2.requires_grad = True
        # self.emb2.retain_grad()
        x = torch.cat((x, root_extend), 1)
        # print(x.shape)
        x = scatter_mean(x, data.batch, dim=0)
        # print(x.shape)
        # print('stop')
        # raise Exception
        return x


class BUrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, device):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.device = device
        self.conv1.ref_name = 'bu_conv1'
        self.conv2.ref_name = 'bu_conv2'
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        root = data.root
        try:
            edge_weights = data.edge_weights
        except:
            edge_weights = None
        x1 = copy.copy(x.float())
        # if x.shape[0] > 1:
        #     print("BU")
        #     print("x shape:",x.shape)
        #     print("Edge_index shape:",edge_index.shape)
        #     print("Edge_index:",edge_index)
        #     try:
        #         print("Maxes within edge:",th.max(edge_index,dim=1)[0])
        #         for item in th.max(edge_index,dim=1)[0]:
        #             if item>=x.shape[0]:
        #                 print("VIOLATION:",item, "    VS     ",x.shape)
        #     except IndexError:
        #         print("no links.")
        if edge_weights is None:
            x = self.conv1(x, edge_index)
        else:
            x = self.conv1(x, edge_index, edge_weights)
        self.emb1 = x
        # self.emb1.requires_grad = True
        # self.emb1.retain_grad()
        x2 = copy.copy(x)

        rootindex = data.rootindex
        # root_extend = th.zeros(x1.size(0), x1.size(1)).to(self.device)
        # batch_size = rootindex.size(0)
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            # root_extend[index] = x1[rootindex[num_batch]]
            root_extend[index] = root[num_batch]
        x = torch.cat((x, root_extend), 1)

        # x = F.relu(x)
        x = self.relu1(x)
        x = F.dropout(x, training=self.training)
        if edge_weights is None:
            x = self.conv2(x, edge_index)
        else:
            x = self.conv2(x, edge_index, edge_weights)
        # x = F.relu(x)
        x = self.relu2(x)
        # root_extend = th.zeros(x1.size(0), x2.size(1)).to(self.device)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        self.emb2 = x
        # self.emb2.requires_grad = True
        # self.emb2.retain_grad()
        x = torch.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        return x


class BiGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_class=4, device=None):
        super(BiGCN, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats, device)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats, device)
        self.fc = torch.nn.Linear((out_feats + hid_feats) * 2, num_class)
        self.device = device

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        self.x = torch.cat((BU_x, TD_x), 1)
        self.out = self.fc(self.x)
        out = F.log_softmax(self.out, dim=1)
        return out

    def get_node_embeds(self, data, layer=1):
        self.forward(data)
        return self.get_node_embeds_after_forward(layer=layer)

    def get_node_embeds_after_forward(self, layer=1):
        if layer == 1:
            td_x = self.TDrumorGCN.emb1
            bu_x = self.BUrumorGCN.emb1
        elif layer == 2:
            td_x = self.TDrumorGCN.emb2
            bu_x = self.BUrumorGCN.emb2
        return (td_x, bu_x)  # torch.cat((td_x, bu_x), 1)


class TDrumorGCNv2(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, device):
        super(TDrumorGCNv2, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.device = device
        self.conv1.ref_name = 'td_conv1'
        self.conv2.ref_name = 'td_conv2'
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # print(data)
        # print(data.batch)
        # x, edge_index = data.x, data.edge_index
        root = x[0].detach().clone()  # data.root
        # try:
        #     edge_weights = data.edge_weights
        # except:
        #     edge_weights = None
        edge_weights = None
        x1 = copy.copy(x.float())
        # print('x1', x1.shape)
        # if x.shape[0] > 1:
        #     print("TD")
        #     print("x shape:", x.shape)
        #     print("Edge_index shape:", edge_index.shape)
        #     print("Edge_index:", edge_index)
        #     try:
        #         print("Maxes within edge:", th.max(edge_index, dim=1)[0])
        #         for item in th.max(edge_index, dim=1)[0]:
        #             if item >= x.shape[0]:
        #                 print("VIOLATION:", item, "    VS     ", x.shape)
        #     except IndexError:
        #         print("no links.")
        # print(x.shape)
        # print(x.device, edge_index.device, self.device)
        if edge_weights is None:
            try:
                x = self.conv1(x, edge_index)
            except:
                print(x.shape, edge_index.shape)
                print(x.device, edge_index.device)
                raise Exception
        else:
            x = self.conv1(x, edge_index, edge_weights)
        self.emb1 = x
        # self.emb1.requires_grad = True
        # self.emb1.retain_grad()
        # print('x', x.shape)
        x2 = copy.copy(x)
        rootindex = 0  # data.rootindex
        # root_extend = th.zeros(x1.size(0), x1.size(1)).to(self.device)
        # batch_size = rootindex.size(0)
        root_extend = torch.zeros(x1.size(0), x1.size(1)).to(self.device)
        batch_size = 1  # max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(torch.zeros(x1.size(0), 1), num_batch))
            # index = (torch.eq(data.batch, num_batch))
            # print(num_batch, rootindex[num_batch], 'root_extend', root_extend.shape, x1.shape)
            # print(x1[rootindex[num_batch]].shape)
            # root_extend[index] = x1[rootindex[num_batch]]
            root_extend[index] = root[num_batch]
            # break
        # print(x.shape, root_extend.shape)
        x = torch.cat((x, root_extend), 1)

        # x = F.relu(x)
        x = self.relu1(x)
        x = F.dropout(x, training=self.training)
        # print(x.shape)
        if edge_weights is None:
            x = self.conv2(x, edge_index)
        else:
            x = self.conv2(x, edge_index, edge_weights)
        # x = F.relu(x)
        x = self.relu2(x)
        # print(x.shape)
        # root_extend = th.zeros(x1.size(0), x2.size(1)).to(self.device)
        # root_extend = torch.zeros(len(data.batch), x2.size(1)).to(self.device)
        root_extend = torch.zeros(x1.size(0), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (torch.eq(torch.zeros(x1.size(0), 1), num_batch))
            # index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        self.emb2 = x
        # self.emb2.requires_grad = True
        # self.emb2.retain_grad()
        x = torch.cat((x, root_extend), 1)
        # print(x.shape)
        x = scatter_mean(x, torch.zeros(x1.size(0), 1), dim=0)
        # print(x.shape)
        # print('stop')
        # raise Exception
        return x


class BUrumorGCNv2(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, device):
        super(BUrumorGCNv2, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.device = device
        self.conv1.ref_name = 'bu_conv1'
        self.conv2.ref_name = 'bu_conv2'
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.BU_edge_index
        # root = data.root
        root = x[0].detach().clone()
        edge_weights = None
        # try:
        #     edge_weights = data.edge_weights
        # except:
        #     edge_weights = None
        x1 = copy.copy(x.float())
        # if x.shape[0] > 1:
        #     print("BU")
        #     print("x shape:",x.shape)
        #     print("Edge_index shape:",edge_index.shape)
        #     print("Edge_index:",edge_index)
        #     try:
        #         print("Maxes within edge:",th.max(edge_index,dim=1)[0])
        #         for item in th.max(edge_index,dim=1)[0]:
        #             if item>=x.shape[0]:
        #                 print("VIOLATION:",item, "    VS     ",x.shape)
        #     except IndexError:
        #         print("no links.")
        if edge_weights is None:
            x = self.conv1(x, edge_index)
        else:
            x = self.conv1(x, edge_index, edge_weights)
        self.emb1 = x
        # self.emb1.requires_grad = True
        # self.emb1.retain_grad()
        x2 = copy.copy(x)

        rootindex = 0  # data.rootindex
        # root_extend = th.zeros(x1.size(0), x1.size(1)).to(self.device)
        # batch_size = rootindex.size(0)
        # root_extend = torch.zeros(len(data.batch), x1.size(1)).to(self.device)
        root_extend = torch.zeros(x1.size(0), x1.size(1)).to(self.device)
        batch_size = 1  # max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(torch.zeros(x1.size(0), 1), num_batch))
            # index = (torch.eq(data.batch, num_batch))
            # root_extend[index] = x1[rootindex[num_batch]]
            root_extend[index] = root[num_batch]
        x = torch.cat((x, root_extend), 1)

        # x = F.relu(x)
        x = self.relu1(x)
        x = F.dropout(x, training=self.training)
        if edge_weights is None:
            x = self.conv2(x, edge_index)
        else:
            x = self.conv2(x, edge_index, edge_weights)
        # x = F.relu(x)
        x = self.relu2(x)
        # root_extend = th.zeros(x1.size(0), x2.size(1)).to(self.device)
        # root_extend = torch.zeros(len(data.batch), x2.size(1)).to(self.device)
        root_extend = torch.zeros(x1.size(0), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            # index = (torch.eq(data.batch, num_batch))
            index = (torch.eq(torch.zeros(x1.size(0), 1), num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        self.emb2 = x
        # self.emb2.requires_grad = True
        # self.emb2.retain_grad()
        x = torch.cat((x, root_extend), 1)

        x = scatter_mean(x, torch.zeros(x1.size(0), 1), dim=0)
        return x


class BiGCNv2(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_class=4, device=None):
        super(BiGCNv2, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats, device)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats, device)
        self.fc = torch.nn.Linear((out_feats + hid_feats) * 2, num_class)
        self.device = device

    def forward(self, x, edge_index, batch_data):
        TD_x = self.TDrumorGCN(batch_data)
        BU_x = self.BUrumorGCN(batch_data)
        # TD_x = self.TDrumorGCN(x, edge_index)
        # BU_edge_index = torch.ones_like(edge_index)
        # BU_edge_index[0] = edge_index[1].detach().clone()
        # BU_edge_index[1] = edge_index[0].detach().clone()
        # BU_x = self.BUrumorGCN(x, edge_index)
        self.x = torch.cat((BU_x, TD_x), 1)
        self.out = self.fc(self.x)
        out = F.log_softmax(self.out, dim=1)
        return out

    def get_node_embeds(self, data, layer=1):
        self.forward(data)
        return self.get_node_embeds_after_forward(layer=layer)

    def get_node_embeds_after_forward(self, layer=1):
        if layer == 1:
            td_x = self.TDrumorGCN.emb1
            bu_x = self.BUrumorGCN.emb1
        elif layer == 2:
            td_x = self.TDrumorGCN.emb2
            bu_x = self.BUrumorGCN.emb2
        return (td_x, bu_x)  # torch.cat((td_x, bu_x), 1)


def train_GCN(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs,
              batchsize, datasetname, iter_num, fold, device, **kwargs):
    version = kwargs.get('version', 2)
    log_file_path = kwargs['log_file_path']
    class_weight = kwargs.get('class_weight', None)
    split_type = kwargs.get('split_type', None)
    if class_weight is not None:
        class_weight = class_weight.to(device)
    if datasetname == "PHEME":
        if version == 2.2:
            model = BiGCN(768, 128, 128, device).to(device)
        elif 2 <= version <= 3:
            model = BiGCN(768, 64, 64, device).to(device)
        else:
            model = BiGCN(256 * 768, 64, 64, device).to(device)
    else:  # Twitter
        model = BiGCN(5000, 64, 64, device).to(device)

    BU_params = list(map(id, model.BUrumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.BUrumorGCN.conv1.parameters(), 'lr': lr/5},
        {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr/5}
    ], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    if version != 0:
        modelname = f'BiGCNv{version}'
    else:
        modelname = 'BiGCN'
    if split_type is not None:
        modelname = f'{split_type}-{modelname}'
    current_loss = None
    try:
        for epoch in range(n_epochs):
            model.train()
            traindata_list, testdata_list = loadBiData(datasetname,
                                                       treeDic,
                                                       x_train,
                                                       x_test,
                                                       TDdroprate,
                                                       BUdroprate)
            train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
            test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
            avg_loss = []
            avg_acc = []
            batch_idx = 0
            tqdm_train_loader = tqdm(train_loader)
            for Batch_data, tweetid in tqdm_train_loader:
                # print(Batch_data, tweetid)
                Batch_data.to(device)
                if 2 <= version <= 3:  # Version 2
                    # new_x = Batch_data.x
                    # new_x = new_x.reshape(new_x.shape[0], -1, 768)
                    # new_x = new_x[:, 0]
                    # Batch_data.x = new_x
                    try:
                        Batch_data.x = Batch_data.cls
                    except:
                        pass
                out_labels = model(Batch_data)
                finalloss = F.nll_loss(out_labels, Batch_data.y, class_weight)
                loss = finalloss
                optimizer.zero_grad()
                loss.backward()
                avg_loss.append(loss.item())
                optimizer.step()
                _, pred = out_labels.max(dim=-1)
                correct = pred.eq(Batch_data.y).sum().item()
                train_acc = correct / len(Batch_data.y)
                avg_acc.append(train_acc)
                print("Fold {} | Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(
                    fold, iter_num, epoch, batch_idx, loss.item(), train_acc))
                batch_idx = batch_idx + 1
            train_losses.append(np.mean(avg_loss))
            train_accs.append(np.mean(avg_acc))

            temp_val_losses = []
            temp_val_accs = []
            temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
            temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
            temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
            temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            model.eval()
            tqdm_test_loader = tqdm(test_loader)
            for Batch_data, tweetid in tqdm_test_loader:
                Batch_data.to(device)
                if 2 <= version <= 3:  # Version 2
                    # new_x = Batch_data.x
                    # new_x = new_x.reshape(new_x.shape[0], -1, 768)
                    # new_x = new_x[:, 0]
                    # Batch_data.x = new_x
                    try:
                        Batch_data.x = Batch_data.cls
                    except:
                        pass
                val_out = model(Batch_data)
                # val_loss = F.nll_loss(val_out, Batch_data.y, class_weight)
                val_loss = F.nll_loss(val_out, Batch_data.y)
                temp_val_losses.append(val_loss.item())
                _, val_pred = val_out.max(dim=1)
                correct = val_pred.eq(Batch_data.y).sum().item()
                val_acc = correct / len(Batch_data.y)
                Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                    val_pred, Batch_data.y)
                temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                    Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
                temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                    Recll2), temp_val_F2.append(F2), \
                temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                    Recll3), temp_val_F3.append(F3), \
                temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                    Recll4), temp_val_F4.append(F4)
                temp_val_accs.append(val_acc)
            val_losses.append(np.mean(temp_val_losses))
            val_accs.append(np.mean(temp_val_accs))
            current_loss = np.mean(temp_val_losses)
            accs = np.mean(temp_val_accs)
            scheduler.step(current_loss)
            print("Fold {} | Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(
                fold, epoch, current_loss, accs))

            res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
                   'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                           np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
                   'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                           np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
                   'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                           np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
                   'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                           np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
            print('results:', res)
            with open(log_file_path, 'a') as f:
                f.write(f'Fold: {fold}| Iter: {iter_num:03d} | Epoch {epoch:05d} | '
                        f'Val_loss {current_loss:.4f} | Val_acc: {accs:.4f}'
                        f'Results: {res}\n')
            checkpoint = {
                'fold': fold,
                'iter_num': iter_num,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_losses[-1],
                'val_loss': current_loss,
                'acc': train_accs[-1],
                'val_acc': accs,
                'res': res
            }
            F1 = np.mean(temp_val_F1)
            F2 = np.mean(temp_val_F2)
            F3 = np.mean(temp_val_F3)
            F4 = np.mean(temp_val_F4)
            early_stopping(train_losses[-1], current_loss, accs, F1, F2, F3, F4, model, modelname, datasetname,
                           checkpoint=checkpoint)

            if early_stopping.early_stop:
                print("Early stopping")
                accs = early_stopping.accs
                F1 = early_stopping.F1
                F2 = early_stopping.F2
                F3 = early_stopping.F3
                F4 = early_stopping.F4
                # Added model snapshot saving
                # checkpoint = {
                #     'iter_num': iter_num,
                #     'epoch': epoch,
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'loss': train_losses[-1],
                #     'res': res
                # }
                # root_dir = os.path.dirname(os.path.abspath(__file__))
                # save_dir = os.path.join(root_dir, 'checkpoints')
                # if not os.path.exists(save_dir):
                #     os.makedirs(save_dir)
                # save_path = os.path.join(save_dir, f'bigcn_f{fold}_i{iter_num}_e{epoch:05d}_l{loss:.5f}.pt')
                # th.save(checkpoint, save_path)
                return train_losses, val_losses, train_accs, val_accs, accs, F1, F2, F3, F4
            # checkpoint = {
            #     'iter_num': iter_num,
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': train_losses[-1],
            #     'res': res
            # }
            # root_dir = os.path.dirname(os.path.abspath(__file__))
            # save_dir = os.path.join(root_dir, 'checkpoints')
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # save_path = os.path.join(save_dir, f'bigcn_f{fold}_i{iter_num}_e{epoch:05d}_l{loss:.5f}.pt')
            # th.save(checkpoint, save_path)
        else:
            checkpoint = {
                'fold': fold,
                'iter_num': iter_num,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_losses[-1],
                'val_loss': current_loss,
                'acc': train_accs[-1],
                'val_acc': accs,
                'res': res
            }
            root_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(root_dir, 'checkpoints')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(
                save_dir,
                f'final_{modelname}_{datasetname}_f{fold}_i{iter_num}_e{epoch:05d}_l{current_loss:.5f}.pt')
            torch.save(checkpoint, save_path)
    except KeyboardInterrupt:
        # Added model snapshot saving
        checkpoint = {
            'fold': fold,
            'iter_num': iter_num,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        root_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(root_dir, 'checkpoints')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(
            save_dir,
            f'interrupt_{modelname}_{datasetname}_f{fold}_i{iter_num}_e{epoch:05d}_l{current_loss:.5f}.pt')
        torch.save(checkpoint, save_path)
    # except:
    #     t = th.cuda.get_device_properties(0).total_memory
    #     r = th.cuda.memory_reserved(0)
    #     a = th.cuda.memory_allocated(0)
    #     f = r - a  # free inside reserved
    #     print(f'{e}\n')
    #     print(Batch_data)
    #     print(f'GPU Memory:\nTotal: {t}\tReserved: {r}\tAllocated: {a}\tFree: {f}\n')
    return train_losses, val_losses, train_accs, val_accs, accs, F1, F2, F3, F4


if __name__ == '__main__':
    lr = 0.0005
    weight_decay = 1e-4
    patience = 10
    n_epochs = 200
    batchsize = 128
    TDdroprate = 0.2
    BUdroprate = 0.2
    # datasetname=sys.argv[1] #"Twitter15"、"Twitter16", 'PHEME'
    datasetname = 'Twitter'  # 'Twitter', 'PHEME'
    # iterations=int(sys.argv[2])
    if datasetname == 'PHEME':
        batchsize = 32  # 24
    # else:
    #     batchsize = 128
    iterations = 1
    model = "GCN"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_accs = []
    NR_F1 = []
    FR_F1 = []
    TR_F1 = []
    UR_F1 = []
    version = 2  # 2.122
    split_type = '5fold'  # '5fold', '9fold'
    if version != 0:
        log_file_path = f'BiGCNv{version}_log.txt'
    else:
        log_file_path = f'BiGCN_log.txt'
    if split_type is not None:
        log_file_path = f'{datasetname}-{split_type}-{log_file_path}'
    else:
        log_file_path = f'{datasetname}-{log_file_path}'
    summary = f'BiGCN:\t' \
              f'Version: {version}\t' \
              f'Dataset: {datasetname}\t' \
              f'LR: {lr}\t' \
              f'Weight Decay: {weight_decay}\t' \
              f'Batchsize: {batchsize}\t' \
              f'Patience: {patience}\t' \
              f'TDdroprate: {TDdroprate}\t' \
              f'BUdroprate: {BUdroprate}'
    with open(log_file_path, 'a') as f:
        f.write(f'{summary}\n')
    for iter_num in range(iterations):
        if datasetname != 'PHEME':
            fold0_x_test, fold0_x_train, \
            fold1_x_test, fold1_x_train, \
            fold2_x_test, fold2_x_train, \
            fold3_x_test, fold3_x_train, \
            fold4_x_test, fold4_x_train = load5foldData(datasetname)
            treeDic = loadTree(datasetname)
            train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(treeDic,
                                                                                                       fold0_x_test,
                                                                                                       fold0_x_train,
                                                                                                       TDdroprate, BUdroprate,
                                                                                                       lr, weight_decay,
                                                                                                       patience,
                                                                                                       n_epochs,
                                                                                                       batchsize,
                                                                                                       datasetname,
                                                                                                       iter_num,
                                                                                                       fold=0,
                                                                                                       device=device,
                                                                                                       log_file_path=log_file_path,
                                                                                                       version=version,
                                                                                                       split_type=split_type)
            train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(treeDic,
                                                                                                       fold1_x_test,
                                                                                                       fold1_x_train,
                                                                                                       TDdroprate, BUdroprate, lr,
                                                                                                       weight_decay,
                                                                                                       patience,
                                                                                                       n_epochs,
                                                                                                       batchsize,
                                                                                                       datasetname,
                                                                                                       iter_num,
                                                                                                       fold=1,
                                                                                                       device=device,
                                                                                                       log_file_path=log_file_path,
                                                                                                       version=version,
                                                                                                       split_type=split_type)
            train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(treeDic,
                                                                                                       fold2_x_test,
                                                                                                       fold2_x_train,
                                                                                                       TDdroprate, BUdroprate, lr,
                                                                                                       weight_decay,
                                                                                                       patience,
                                                                                                       n_epochs,
                                                                                                       batchsize,
                                                                                                       datasetname,
                                                                                                       iter_num,
                                                                                                       fold=2,
                                                                                                       device=device,
                                                                                                       log_file_path=log_file_path,
                                                                                                       version=version,
                                                                                                       split_type=split_type)
            train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(treeDic,
                                                                                                       fold3_x_test,
                                                                                                       fold3_x_train,
                                                                                                       TDdroprate, BUdroprate, lr,
                                                                                                       weight_decay,
                                                                                                       patience,
                                                                                                       n_epochs,
                                                                                                       batchsize,
                                                                                                       datasetname,
                                                                                                       iter_num,
                                                                                                       fold=3,
                                                                                                       device=device,
                                                                                                       log_file_path=log_file_path,
                                                                                                       version=version,
                                                                                                       split_type=split_type)
            train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(treeDic,
                                                                                                       fold4_x_test,
                                                                                                       fold4_x_train,
                                                                                                       TDdroprate, BUdroprate, lr,
                                                                                                       weight_decay,
                                                                                                       patience,
                                                                                                       n_epochs,
                                                                                                       batchsize,
                                                                                                       datasetname,
                                                                                                       iter_num,
                                                                                                       fold=4,
                                                                                                       device=device,
                                                                                                       log_file_path=log_file_path,
                                                                                                       version=version,
                                                                                                       split_type=split_type)
            test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
            NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
            FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
            TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
            UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
            print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
                sum(test_accs) / iterations, sum(NR_F1) /iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))
        else:
            treeDic = None
            train_losses_dict, val_losses_dict, train_accs_dict, val_accs_dict = {}, {}, {}, {}
            test_accs_dict, F1_dict, F2_dict, F3_dict, F4_dict = {}, {}, {}, {}, {}
            if split_type == '5fold':
                for fold_num, fold in enumerate(load5foldDataStratified(datasetname)):
                    c0_train, c0_test, c1_train, c1_test, c2_train, c2_test, c3_train, c3_test = fold
                    c0_labels = np.zeros_like(c0_train, dtype=int)
                    c1_labels = np.ones_like(c1_train, dtype=int)
                    c2_labels = np.ones_like(c1_train, dtype=int) * 2
                    c3_labels = np.ones_like(c1_train, dtype=int) * 3
                    labels = np.concatenate((c0_labels, c1_labels, c2_labels, c3_labels))
                    classes = np.asarray([0, 1, 2, 3])
                    class_weight = compute_class_weight('balanced', classes=classes, y=labels)
                    class_weight = torch.FloatTensor(class_weight)
                    fold_train = np.concatenate((c0_train, c1_train, c2_train, c3_train)).tolist()
                    fold_test = np.concatenate((c0_test, c1_test, c2_test, c3_test)).tolist()
                    output = train_GCN(treeDic,
                                       fold_test,
                                       fold_train,
                                       TDdroprate,
                                       BUdroprate,
                                       lr,
                                       weight_decay,
                                       patience,
                                       n_epochs,
                                       batchsize,
                                       datasetname,
                                       iter_num,
                                       fold=fold_num,
                                       device=device, log_file_path=log_file_path,
                                       version=version,
                                       class_weight=class_weight,
                                       split_type=split_type)
                    train_losses, val_losses, train_accs, val_accs, accs, F1, F2, F3, F4 = output
                    train_losses_dict[f'train_losses_{fold_num}'] = train_losses
                    val_losses_dict[f'val_losses_{fold_num}'] = val_losses
                    train_accs_dict[f'train_accs_{fold_num}'] = train_accs
                    val_accs_dict[f'val_accs_{fold_num}'] = val_accs
                    test_accs_dict[f'test_accs_{fold_num}'] = accs
                    F1_dict[f'F1_{fold_num}'] = F1
                    F2_dict[f'F2_{fold_num}'] = F2
                    F3_dict[f'F3_{fold_num}'] = F3
                    F4_dict[f'F4_{fold_num}'] = F4
                test_accs.append(sum([v for k, v in test_accs_dict.items()]) / 5)
                NR_F1.append(sum([v for k, v in F1_dict.items()]) / 5)
                FR_F1.append(sum([v for k, v in F2_dict.items()]) / 5)
                TR_F1.append(sum([v for k, v in F3_dict.items()]) / 5)
                UR_F1.append(sum([v for k, v in F4_dict.items()]) / 5)
            elif split_type == '9fold':
                for fold_num, (fold_train, fold_test) in enumerate(load9foldData(datasetname)):
                    fold_train, fold_train_labels = fold_train
                    fold_test, fold_test_labels = fold_test
                    fold_train_labels = np.asarray(fold_train_labels)
                    # fold_test_labels = np.asarray(fold_test_labels)
                    classes = np.asarray([0, 1, 2, 3])
                    class_weight = compute_class_weight('balanced', classes=classes, y=fold_train_labels)
                    class_weight = torch.FloatTensor(class_weight)
                    output = train_GCN(treeDic,
                                       fold_test,
                                       fold_train,
                                       TDdroprate,
                                       BUdroprate,
                                       lr,
                                       weight_decay,
                                       patience,
                                       n_epochs,
                                       batchsize,
                                       datasetname,
                                       iter_num,
                                       fold=fold_num,
                                       device=device, log_file_path=log_file_path,
                                       version=version,
                                       class_weight=class_weight,
                                       split_type=split_type)
                    train_losses, val_losses, train_accs, val_accs, accs, F1, F2, F3, F4 = output
                    train_losses_dict[f'train_losses_{fold_num}'] = train_losses
                    val_losses_dict[f'val_losses_{fold_num}'] = val_losses
                    train_accs_dict[f'train_accs_{fold_num}'] = train_accs
                    val_accs_dict[f'val_accs_{fold_num}'] = val_accs
                    test_accs_dict[f'test_accs_{fold_num}'] = accs
                    F1_dict[f'F1_{fold_num}'] = F1
                    F2_dict[f'F2_{fold_num}'] = F2
                    F3_dict[f'F3_{fold_num}'] = F3
                    F4_dict[f'F4_{fold_num}'] = F4
                test_accs.append(sum([v for k, v in test_accs_dict.items()]) / 9)
                NR_F1.append(sum([v for k, v in F1_dict.items()]) / 9)
                FR_F1.append(sum([v for k, v in F2_dict.items()]) / 9)
                TR_F1.append(sum([v for k, v in F3_dict.items()]) / 9)
                UR_F1.append(sum([v for k, v in F4_dict.items()]) / 9)
            summary = "Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
                sum(test_accs) / iterations, sum(NR_F1) / iterations, sum(FR_F1) / iterations, sum(TR_F1) / iterations,
                sum(UR_F1) / iterations)
            print(summary)
            with open(log_file_path, 'a') as f:
                f.write(f'{summary}\n')

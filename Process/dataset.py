import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
# from torch_sparse import SparseTensor
# from transformers import BertTokenizer, BertModel
# import json

cwd = os.getcwd()
log_path = os.path.join(cwd, '..', 'data', 'droplist.txt')


class GraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))


def collate_fn(data):
    return data


class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, tddroprate=0, budroprate=0,
                 data_path=os.path.join('..', '..', 'data', 'Weibograph'), implicit=False):
        if data_path.find('PHEME') != -1 or data_path.find('MaWeibo') != -1 or data_path.find('NewTwitter') != -1 or \
            data_path.find('NewWeibo') != -1 or data_path.find('Cls') != -1:
            # pheme_dir_path = os.path.join('..', '..', 'data', 'PHEME')
            self.fold_x = fold_x
            # self.data_path = data_path
            # self.tddroprate = tddroprate
            # self.budroprate = budroprate
        else:
            self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
            self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate
        self.implicit = implicit

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        max_size = 100
        id = self.fold_x[index]
        # print(id, index)
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        try:
            x = data['cls']
        except:
            x = data['x']
        original_size = x.shape[0]
        root = x[0, :]

        if original_size > max_size:
            x = x[:max_size]
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            new_row, new_col = [], []
            for src, dst in zip(row, col):
                if src < max_size or dst < max_size:
                    new_row.append(src)
                    new_col.append(dst)
            edgeindex = [np.array(new_row), np.array(new_col)]
        # print(data)


        # if self.implicit:
        #     try:
        #         implicit_edge_index = data['implicit_edge_index']
        #     except:
        #         # print(edgeindex)
        #         direct_links = {f'{u}-{v}': 1 for u, v in zip(*edgeindex)}
        #         try:
        #             nodes = list(range(data['x'].shape[0]))
        #             size = data['x'].shape[0]
        #         except:
        #             nodes = list(range(data['cls'].shape[0]))
        #             size = data['cls'].shape[0]
        #             # print(data.files)
        #
        #         implicit_src, implicit_dst = [], []
        #         for src in nodes:
        #             for dst in range(src, size):
        #                 if direct_links.get(f'{src}-{dst}', None) is None:
        #                     implicit_src.append(src)
        #                     implicit_dst.append(dst)
        #         implicit_edge_index = [implicit_src, implicit_dst]
        #     implicit_BU_edge_index = [implicit_edge_index[1], implicit_edge_index[0]]
        #     print(implicit_edge_index, implicit_BU_edge_index)
        # raise Exception
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow, bucol]
        try:
            if data['tweetids'].shape[0] > max_size:
                tweetids = np.zeros(data['tweetids'][:max_size].shape)
            else:
                tweetids = np.zeros(data['tweetids'].shape)
            for i, value in enumerate(data['tweetids']):
                if i == max_size:
                    break
                tweetids[i] = int(value)
        except:
            pass
        if self.data_path.find('PHEME') != -1:  # If PHEME
            # try:
            #     return Data(x=torch.tensor(data['x'], dtype=torch.float32),
            #                 cls=torch.tensor(data['cls'], dtype=torch.float32),
            #                 edge_index=torch.LongTensor(new_edgeindex), BU_edge_index=torch.LongTensor(bunew_edgeindex),
            #                 y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
            #                 rootindex=torch.LongTensor([int(data['rootindex'])]),
            #                 tweetids=torch.LongTensor(tweetids)), data['tweetids'][int(data['rootindex'])]
            # except:
            try:  # Latest version with only CLS, user id, timestamp
                return Data(x=torch.tensor(x, dtype=torch.float32),
                            edge_index=torch.LongTensor(new_edgeindex),
                            BU_edge_index=torch.LongTensor(bunew_edgeindex),
                            y=torch.LongTensor([int(data['y'])]),
                            rootindex=torch.LongTensor([int(data['rootindex'])]),
                            root=torch.tensor(root, dtype=torch.float32).unsqueeze(0),
                            user_ids=torch.LongTensor(data['user_ids']),
                            timestamps=torch.tensor(data['timestamps'], dtype=torch.float32),
                            event_num=torch.LongTensor([int(data['event_num'])]),
                            tweetids=torch.LongTensor(tweetids)), \
                       (data['tweetids'][int(data['rootindex'])])
            except:  # Original data organisation
                print(data.files)
                return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                            edge_index=torch.LongTensor(new_edgeindex), BU_edge_index=torch.LongTensor(bunew_edgeindex),
                            y=torch.LongTensor([int(data['y'])]), event_num=torch.LongTensor([int(data['event_num'])]),
                            root=torch.LongTensor(data['root']),
                            rootindex=torch.LongTensor([int(data['rootindex'])]),
                            tweetids=torch.LongTensor(tweetids)), data['tweetids'][int(data['rootindex'])]
        elif self.data_path.find('MaWeibo') != -1:
            return Data(x=torch.tensor(data['cls'], dtype=torch.float32),
                        edge_index=torch.LongTensor(new_edgeindex),
                        BU_edge_index=torch.LongTensor(bunew_edgeindex),
                        y=torch.LongTensor([int(data['y'])]),
                        rootindex=torch.LongTensor([int(data['rootindex'])]),
                        tweetids=torch.LongTensor(tweetids)), data['tweetids'][int(data['rootindex'])]
        elif self.data_path.find('NewWeibo') != -1 or self.data_path.find('ClsWeibo') != -1:
            return Data(x=torch.tensor(data['cls'], dtype=torch.float32),
                        edge_index=torch.LongTensor(new_edgeindex),
                        BU_edge_index=torch.LongTensor(bunew_edgeindex),
                        y=torch.LongTensor([int(data['y'])]),
                        rootindex=torch.LongTensor([int(data['rootindex'])]),
                        root=torch.tensor(root, dtype=torch.float32).unsqueeze(0),
                        user_ids=torch.LongTensor(data['user_ids']),
                        timestamps=torch.tensor(data['timestamps'], dtype=torch.float32),
                        tweetids=torch.LongTensor(tweetids)), data['tweetids'][int(data['rootindex'])]
        elif self.data_path.find('NewTwitter') != -1 or self.data_path.find('ClsTwitter') != -1:
            return Data(x=torch.tensor(data['cls'], dtype=torch.float32),
                        edge_index=torch.LongTensor(new_edgeindex),
                        BU_edge_index=torch.LongTensor(bunew_edgeindex),
                        y=torch.LongTensor([int(data['y'])]),
                        event_num=torch.LongTensor([int(data['event_num'])]),
                        rootindex=torch.LongTensor([int(data['rootindex'])]),
                        root=torch.tensor(root, dtype=torch.float32).unsqueeze(0)), id
        else:
            return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                        edge_index=torch.LongTensor(new_edgeindex), BU_edge_index=torch.LongTensor(bunew_edgeindex),
                        y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                        rootindex=torch.LongTensor([int(data['rootindex'])])), [id]
        # return Data(x=torch.tensor(data['x'], dtype=torch.float32),
        #             edge_index=SparseTensor.from_edge_index(torch.LongTensor(new_edgeindex),
        #                                                     sparse_sizes=torch.Tensor([data['x'].shape[0]])),
        #             BU_edge_index=SparseTensor.from_edge_index(torch.LongTensor(bunew_edgeindex), is_sorted=True),
        #             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
        #             rootindex=torch.LongTensor([int(data['rootindex'])]),
        #             tweetids=torch.LongTensor(tweetids)), data['tweetids'][int(data['rootindex'])]


class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..','data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]

        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))


class BERTDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, tddroprate=0, budroprate=0,
                 data_path=os.path.join('..', '..', 'data', 'Weibograph')):
        if data_path.find('PHEME') != -1:
            # pheme_dir_path = os.path.join('..', '..', 'data', 'PHEME')
            self.fold_x = fold_x
            self.data_path = data_path
            self.tddroprate = tddroprate
            self.budroprate = budroprate
        else:
            self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
            self.treeDic = treeDic
            self.data_path = data_path
            self.tddroprate = tddroprate
            self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow, bucol]
        tweetids = np.zeros(data['tweetids'].shape)
        for i, value in enumerate(data['tweetids']):
            tweetids[i] = int(value)
        return torch.tensor(data['x'], dtype=torch.float32), \
               torch.LongTensor([int(data['y'])]), int(data['tweetids'][int(data['rootindex'])])
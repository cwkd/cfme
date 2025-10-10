import os, sys
import random
import string
import time, datetime
import itertools
import traceback
from functools import reduce

import joblib
import joblib as jl

import scipy.stats
from scipy.optimize import linprog
import sklearn.metrics
from sklearn.linear_model import LassoLars
from sklearn.manifold import TSNE
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.data import DataLoader, Data
import numpy as np

import matplotlib.pyplot as plt

from model.Twitter.BiGCN_Twitter import BiGCN, BiGCNv2
from model.Twitter.BiGAT_Twitter import CHGAT, CHGATv2
from model.Twitter.EBGCN import EBGCN, EBGCNv2
from Process.process import loadBiData, loadTree
from Process.pheme9fold import load9foldData
from Process.rand5fold import load5foldData
from Process.getPHEMEgraph import getRawData

from tools.earlystopping import EarlyStopping
from tools.evaluate import *

# from torch_geometric.utils import add_remaining_self_loops
# from torch_geometric.utils.num_nodes import maybe_num_nodes

import lrp_pytorch.modules.utils as lrp_utils
from lrp_pytorch.modules.base import safe_divide
from tqdm import tqdm
import copy
import argparse
import json

import nltk
from nltk.corpus import stopwords

from transformers import BertTokenizer, BertModel
# from interpret_bert.interpret_nlp.visualization.heatmap import html_heatmap
# from IPython.core.display import display, HTML

# nltk.download('stopwords')
STOPWORDS = {}
for word in stopwords.words('english'):
    STOPWORDS[word] = 0
FOLD_2_EVENTNAME = {0: 'charliehebdo',
                    1: 'ebola',
                    2: 'ferguson',
                    3: 'germanwings',
                    4: 'gurlitt',
                    5: 'ottawashooting',
                    6: 'prince',
                    7: 'putinmissing',
                    8: 'sydneysiege'}
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXPLAIN_DIR = os.path.join(DATA_DIR, 'explain')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'model', 'Twitter', 'checkpoints')
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = False

LAYERS = ['input', 'td_conv1', 'td_conv2', 'bu_conv2', 'bu_conv2']

LRP_PARAMS = {
    'linear_eps': 1e-6,
    'gcn_conv': 1e-6,
    'bigcn': 1e-6,
    'ebgcn': 1e-6,
    'mode': 'lrp'
}


class GraphLIME:

    def compute_kernel(self, x, is_L):
        n, d = x.shape
        dist = x.reshape(1, n, d) - x.reshape(n, 1, d)  # (n, n, d)
        dist = dist ** 2
        # print(f'dist: {dist.shape}', flush=True)
        if is_L:
            # dist = dist.sum(-1, keepdims=True)
            dist = np.sum(dist, axis=-1, keepdims=True)  # (n, n, 1)
        std = np.sqrt(d)
        K = np.exp(-dist / (2 * std ** 2 * 0.1 + 1e-10))  # (n, n, 1) or (n, n, d)
        # print(f'K: {K.shape}', flush=True)
        return K

    def compute_gram_matrix(self, x):
        G = x - np.mean(x, axis=0, keepdims=True)
        G = G - np.mean(G, axis=1, keepdims=True)
        G = G / (np.linalg.norm(G, ord='fro', axis=(0, 1), keepdims=True) + 1e-10)
        # print(f'G: {G.shape}', flush=True)
        return G

    def compute_betas(self, x, y, rho):
        n, d = x.shape
        # print('x shape', n, d, flush=True)
        K = self.compute_kernel(x, is_L=False)  # (n, n, d)
        L = self.compute_kernel(y, is_L=True)  # (n, n, 1)
        K_bar = self.compute_gram_matrix(K)  # (n, n, d)
        L_bar = self.compute_gram_matrix(L)  # (n, n, 1)
        # L_bar = L.mean(-1)
        K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
        L_bar = L_bar.reshape(n ** 2, )  # (n ** 2,)
        solver = LassoLars(rho, fit_intercept=False, positive=True)
        # print('fitting', flush=True)
        solver.fit(K_bar * n, L_bar * n)
        # print('fitted', flush=True)
        return solver.coef_


def show_memory_usage(device):
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    print(f'gpu: {device}\t\ttotal memory: {t}\t\treserved memory: {r}\t\tallocated memory: {a}\t\t'
          f'unallocated memory: {r-a}\t\tfree memory: {t-r}')


def display_mem_usage(device, verbose=False):
    free_mem, total_mem = torch.cuda.mem_get_info(device)
    used_mem = total_mem - free_mem
    mem_used_MB = (total_mem - free_mem) / 1024 ** 2
    mem_proportion_used = used_mem / total_mem
    if verbose:
        print(f'free_mem: {free_mem}\n'
              f'total_mem: {total_mem}\nused_mem: {used_mem}\tmem_used_MB: {mem_used_MB}\n'
              f'mem_proportion_used: {mem_proportion_used}', flush=True)
    return mem_proportion_used


def get_raw_texts_PHEME(root_ids):
    batch_raw_texts = []
    for root_id in root_ids:
        raw_texts = None
        for fold_num in range(9):  # Load raw texts
            try:
                raw_texts, _ = getRawData(FOLD_2_EVENTNAME[fold_num], root_id)
                break
            except:
                pass
        batch_raw_texts.append(raw_texts)
    return batch_raw_texts


def get_model_copy(original_model_type, input_size, hidden_size, output_size, num_class, device, ebgcn_args):
    if original_model_type == 'BiGCN':
        base_model = BiGCN(input_size, hidden_size, output_size, num_class, device).to(device)
    elif original_model_type == 'BiGCNv2':
        base_model = BiGCNv2(input_size, hidden_size, output_size, num_class, device).to(device)
    elif original_model_type == 'EBGCN':
        ebgcn_args.input_features = input_size
        ebgcn_args.num_class = num_class
        base_model = EBGCN(ebgcn_args).to(device)
    elif original_model_type == 'EBGCNv2':
        ebgcn_args.input_features = input_size
        ebgcn_args.num_class = num_class
        base_model = EBGCNv2(ebgcn_args).to(device)
    elif original_model_type == 'CHGAT':
        base_model = CHGAT(input_size, hidden_size, output_size, num_class, device).to(device)
    elif original_model_type == 'CHGATv2':
        base_model = CHGATv2(input_size, hidden_size, output_size, num_class, device).to(device)
    return base_model


def test_GCN(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs,
              batchsize, datasetname, iter_num, fold, device, **kwargs):
    # version = kwargs.get('version', 2)
    log_file_path = kwargs['log_file_path']
    split_type = kwargs.get('split_type', None)
    model_type = kwargs.get('model_type', None)
    hidden_size = kwargs.get('hidden_size', None)
    output_size = kwargs.get('output_size', None)
    ebgcn_args = kwargs.get('ebgcn_args', None)
    exp_method = kwargs.get('exp_method', None)
    tokeniser = kwargs.get('tokeniser', None)
    text_encoder = kwargs.get('text_encoder', None)
    load_checkpoint = kwargs.get('load_checkpoint', True)
    if datasetname.find('PHEME') != -1 or datasetname.find('Twitter') != -1:
        input_size = 768
        num_class = 4
    elif datasetname.find('Weibo') != -1:
        input_size = 768
        num_class = 2
    model = get_model_copy(model_type, input_size, hidden_size, output_size, num_class, device,
                                ebgcn_args)
    # Load pretrained model according to model type from path, match iteration and fold
    if load_checkpoint:
        baseline_checkpoints_path = os.path.join(ROOT_DIR, 'testing', f'{datasetname}')
        filenames = os.listdir(baseline_checkpoints_path)
        for criterion in [model_type, f'i{iter_num}', f'f{fold}']:
            filenames = list(filter(lambda x: x.find(f'{criterion}') != -1, filenames))
        savepoint_filename = filenames[0]
        savepoint_name = savepoint_filename[:-3]
        checkpoint_path = os.path.join(baseline_checkpoints_path, savepoint_filename)
        checkpoint_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_dict['model_state_dict'])
        print(f"model checkpoint loaded from {checkpoint_path}")
        print(f"Epoch: {checkpoint_dict['epoch']}\tTrain Loss: {checkpoint_dict['loss']}\tTrain Acc: {checkpoint_dict['acc']}\n"
              f"Val Loss: {checkpoint_dict['val_loss']}\tVal Acc: {checkpoint_dict['val_acc']}\n"
              f"{checkpoint_dict['res']}\n")
        del checkpoint_dict
    version = f'[{hidden_size},{output_size}]'
    if model_type == 'EBGCN':
        model0 = f'{model_type}-ie'
    else:
        model0 = model_type
    if split_type is not None:
        modelname = f'{model0}-{version}-lr{lr}-wd{weight_decay}-bs{batchsize}-p{patience}'
    expl_save_dir = os.path.join(DATA_DIR, 'explain', f'{datasetname}', f'{model_type}')
    if not os.path.exists(expl_save_dir):
        os.makedirs(expl_save_dir)
    model_ref = copy.deepcopy(model)
    LRP_PARAMS['mode'] = 'lrp'
    try:  # For EBGCN
        model_ref.args.training = False
    except:
        pass
    model_copy = lrp_utils.get_lrpwrappermodule(model_ref, LRP_PARAMS).to(device)
    if model_copy is None:
        assert False
    model_copy.eval()
    model.eval()
    traindata_list, testdata_list = loadBiData(datasetname,
                                               treeDic,
                                               x_train,
                                               x_test,
                                               TDdroprate,
                                               BUdroprate)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    data_loader = DataLoader(traindata_list, batch_size=1, shuffle=False, num_workers=5)
    batch_idx = 0
    tqdm_data_loader = tqdm(data_loader)

    # def enable_grad_logging(obj):
    #     """
    #     function to enable gradient retrieval after backprop from specified Tensor
    #     @param obj: usually a Tensor
    #     @return:
    #     """
    #     try:
    #         obj.requires_grad = True
    #     except:
    #         pass
    #     obj.retain_grad()

    def get_embeds_from_token_ids(input_ids, text_encoder, device):
        if type(input_ids) is list:
            input_ids = torch.LongTensor(input_ids).to(device)
        with torch.no_grad():
            # encoded_texts: BatchEncoding = tokeniser(text, padding='longest', max_length=256, truncation=True,
            #                                          return_tensors='pt')
            # if pooling_mode != 'pooler':
            #     embeddings = model.embeddings.word_embeddings(
            #         encoded_texts['input_ids'].to(device)).cpu().detach().numpy()
            #     if pooling_mode == 'mean':
            #         cls = embeddings.mean(-2)
            #         # root_feat = cls[0]
            #     if pooling_mode == 'max':
            #         cls = embeddings.max(-2)
            #         # root_feat = cls[0]
            # elif pooling_mode == 'pooler':
            #     cls = model(encoded_texts['input_ids'].to(device)).pooler_output.cpu().detach().numpy()
                # root_feat = cls[0]
            embeds = text_encoder.embeddings.word_embeddings(input_ids)
        return embeds

    # def get_cls_from_token_ids(input_ids, model, device):
    #     if type(input_ids) is list:
    #         input_ids = torch.LongTensor(input_ids).to(device).unsqueeze(0)
    #     model: BertModel
    #     bert_output = model(input_ids, output_hidden_states=True)
    #     embeds = bert_output.hidden_states[0].squeeze(0)
    #     cls = bert_output.pooler_output.squeeze(0)
    #     return cls, embeds

    def load_and_get_token_embeds(root_ids, model_type, datasetname, text_encoder, device):
        new_x_list, token_embeds_list = [], []
        if datasetname.find('PHEME') != -1:
            token_ids_dir = os.path.join(ROOT_DIR, 'data', 'PHEME', 'raw_text')
        elif datasetname == 'NewTwitter':
            token_ids_dir = os.path.join(ROOT_DIR, 'data', 'NewTwitter')
        elif datasetname == 'NewWeibo':
            token_ids_dir = os.path.join(ROOT_DIR, 'data', 'NewWeibo')
        # elif datatsetname.find('Weibo') != -1:
        #
        # print(Batch_data.x.shape)
        for root_id in root_ids:
            with open(os.path.join(token_ids_dir, f'{root_id}.json'), 'r', encoding='utf-8') as f:
                json_obj = json.load(f)
            token_ids_list = json_obj['token_ids']
            if datasetname.find('PHEME') != -1:
                tokenised_text = json_obj['texts']
            elif datasetname == 'NewTwitter':
                tokenised_text = json_obj['tokenised_texts']
            elif datasetname == 'NewWeibo':
                tokenised_text = json_obj['texts']
            if model_type == 'GACL':
                source_claim = None
            for token_ids in token_ids_list:
                if model_type == 'GACL':
                    prob = torch.ones_like(token_ids) * 0.2
                    prob[0], prob[-1] = 1, 1
                    mask = torch.bernoulli(prob) > 0
                    token_ids = token_ids[mask]
                embeds = get_embeds_from_token_ids(token_ids, text_encoder, device)
                # cls, embeds = get_cls_from_token_ids(token_ids, text_encoder, device)
                embeds.requires_grad = True
                embeds.retain_grad()
                token_embeds_list.append(embeds)
                new_x = embeds.mean(-2)
                # new_x = cls
                if model_type == 'GACL':
                    if source_claim is None:
                        source_claim = new_x
                    root_extend = torch.ones_like(source_claim, dtype=new_x.dtype, device=new_x.device)
                    root_extend *= source_claim
                    new_x = torch.cat((new_x, root_extend), 0)
                new_x_list.append(new_x)
        new_x = torch.stack(new_x_list)
        return new_x, token_embeds_list, (token_ids_list, tokenised_text)

    sample_num = 0
    # flipped = [0, 0, 0, 0, 0]
    # sample_count = 0
    # valid = [0, 0, 0, 0, 0]
    # original_eq_all_masked = [0, 0, 0, 0, 0]
    unconstrained_sparsities = []
    flush_count = 0

    topks = ['top10', 'top5', 'top1']
    test_sparsities = [0.0, 0.2, 0.4, 0.6, 0.8]
    topk_test = {int(key[3:]): value for value, key in enumerate(topks)}

    flipped = [0] * len(test_sparsities)  # [0, 0, 0, 0, 0]
    sample_count = 0
    valid = [0] * len(test_sparsities)  # [0, 0, 0, 0, 0]
    original_eq_all_masked = [0] * len(test_sparsities)  # [0, 0, 0, 0, 0]
    flipped_with_graph_bias = [0] * len(test_sparsities)  # [0, 0, 0, 0, 0]
    flipped_without_graph_bias = [0] * len(test_sparsities)  # [0, 0, 0, 0, 0]
    valid_with_graph_bias = [0] * len(test_sparsities)  # [0, 0, 0, 0, 0]
    valid_without_graph_bias = [0] * len(test_sparsities)  # [0, 0, 0, 0, 0]
    solution_with_graph_bias = [0] * len(test_sparsities)  # [0, 0, 0, 0, 0]
    solution_without_graph_bias = [0] * len(test_sparsities)  # [0, 0, 0, 0, 0]
    no_solution_with_graph_bias = [0] * len(test_sparsities)  # [0, 0, 0, 0, 0]
    no_solution_without_graph_bias = [0] * len(test_sparsities)  # [0, 0, 0, 0, 0]
    no_solution_with_allclass_non_neg_infl_gap = [0] * len(test_sparsities)  # [0, 0, 0, 0, 0]
    sensitivity_log_probs = [None] * len(test_sparsities)
    topk_sensitivity_log_probs = [None] * len(topk_test)

    classwise_graph_expl_token_id_counts_dict = [None] * num_class
    classwise_claim_expl_token_id_counts_dict = [None] * num_class
    ablated_classwise_graph_expl_token_id_counts_dict = [None] * num_class
    ablated_classwise_claim_expl_token_id_counts_dict = [None] * num_class
    topk_classwise_graph_expl_token_id_counts_dict = [None] * len(topk_test)
    topk_classwise_claim_expl_token_id_counts_dict = [None] * len(topk_test)
    topk_ablated_classwise_graph_expl_token_id_counts_dict = [None] * len(topk_test)
    topk_ablated_classwise_claim_expl_token_id_counts_dict = [None] * len(topk_test)
    classwise_pred_counts = [0] * num_class
    jaccard_similarity_list = []
    ss_similarity_list = []
    topk_jaccard_similarity_list = [None] * len(topk_test)
    topk_ss_similarity_list = [None] * len(topk_test)

    for Batch_data, root_ids in tqdm_data_loader:
        try:
            Batch_data.x = Batch_data.cls
        except:
            pass
        Batch_data.to(device)
        flat_idx_2_nested_idx = {}
        nested_idx_2_flat_idx = {}
        test_summary = ''
        graph_expl_dict = {}
        root_id = root_ids[0]
        # print(root_ids, model_type, datasetname, text_encoder)
        if datasetname.find('PHEME') != -1:
            for check_fold in range(9):  # iterate through all source files
                event_name = FOLD_2_EVENTNAME[check_fold]
                with open(os.path.join(DATA_DIR, 'PHEME', f'{event_name}.json'), 'r', encoding='utf-8') as f:
                    json_obj = json.load(f)
                try:
                    root_text = json_obj[f'{root_id}'][f'{root_id}']['text']  # check if entry exists, will throw error if not found
                    break  # forcefully terminate search if found
                except:  # will throw error if no entry, catch and let for loop continue
                    pass
            else:  # if not forcefully termintated, this will execute, skip this iteration in outer loop
                print(f'{root_id} not found')
                continue
        elif datasetname == 'NewTwitter':
            with open(os.path.join(DATA_DIR, 'NewTwitter', f'{root_id}.json'), 'r', encoding='utf-8') as f:
                json_obj = json.load(f)
            root_text = json_obj['tweets'][0]
        elif datasetname == 'NewWeibo':
            with open(os.path.join(DATA_DIR, 'MaWeibo', 'Threads', f'{root_id}.json'), 'r', encoding='utf-8') as f:
                json_obj = json.load(f)
            root_text = json_obj[0]['original_text']
        graph_expl_dict = {'root_id': root_id,
                           'root_text': root_text}
        test_summary += f'{root_id}\t'
        new_x, token_embeds_list, (token_ids_list, tokenised_texts) = load_and_get_token_embeds(
            root_ids, model_type, datasetname, text_encoder=text_encoder, device=device)
        original_claim = Batch_data.root.clone().detach()
        Batch_data.x = new_x.clone().detach()
        Batch_data.root = original_claim.clone().detach()
        num_nodes = new_x.shape[0]
        test_summary += f'{num_nodes}\t'
        # check size of graph
        if num_nodes < 20:  # skip threads with nodes smaller than 20 posts
            sample_num += 1
            continue
        include_graph_bias = True
        include_root_extend = True
        relax_pos_constraint = False
        relax_graph_bias = False
        if exp_method == 'ct-lrp':
            model.eval()
            new_x.retain_grad()
            if include_root_extend:
                original_claim = Batch_data.root
                new_root = Batch_data.root.clone().detach()
                new_root.requires_grad_()
                new_root.retain_grad()
                Batch_data.root = new_root
            # get the flattened index for token and their nested index equivalent
            component_count = 0
            root_components_range = [0, 0]
            for node_num, token_embeds in enumerate(token_embeds_list):
                for token_num in range(token_embeds.shape[0]):
                    flat_idx_2_nested_idx[component_count] = (node_num, token_num)
                    nested_idx_2_flat_idx[(node_num, token_num)] = component_count
                    component_count += 1
            else:  # get root tokens
                if include_root_extend:
                    root_components_range[0] = component_count
                    for token_num in range(token_embeds_list[0].shape[0]):
                        flat_idx_2_nested_idx[component_count] = (-1, token_num)
                        nested_idx_2_flat_idx[(-1, token_num)] = component_count
                        component_count += 1
                    else:
                        root_components_range[1] = component_count
            temp_x = torch.zeros_like(new_x, device=new_x.device)
            Batch_data.x = temp_x
            if include_root_extend:
                temp_claim = torch.zeros_like(original_claim, device=original_claim.device)
                Batch_data.root = temp_claim
            all_masked_probs = model(Batch_data)
            if type(all_masked_probs) is tuple:
                all_masked_probs = all_masked_probs[0]
            _, all_masked_pred = all_masked_probs.max(-1)
            all_masked_logits = model.out.clone().detach()
            Batch_data.x = new_x
            if include_root_extend:
                Batch_data.root = new_root
            original_probs = model(Batch_data)
            if type(original_probs) is tuple:
                original_probs = original_probs[0]
            _, original_pred = original_probs.max(-1)
            original_pred_idx = original_pred.item()
            out_logits = model.out.clone().detach()

            component_scores = torch.zeros((component_count, num_class), device=new_x.device)
            for class_num in range(num_class):
                if Batch_data.x.grad is not None:
                    Batch_data.x.grad.zero_()
                    if include_root_extend:
                        Batch_data.root.grad.zero_()
                with torch.enable_grad():
                    output = model_copy(Batch_data)  # forward
                    if type(output) is tuple:
                        output = output[0]
                output[0, class_num].backward()
                class_x_R = Batch_data.x.grad.detach().clone()
                if include_root_extend:
                    class_claim_R = Batch_data.root.grad.detach().clone()
                token_R_list = []
                for node_num, token_embeds in enumerate(token_embeds_list):
                    temp_list = []
                    node_R = class_x_R[node_num]
                    token_embeds.requires_grad_()
                    token_embeds.retain_grad()
                    if token_embeds.grad is None:
                        pass
                    else:
                        token_embeds.grad.zero_()
                    token_Z = token_embeds.mean(0)
                    token_S = safe_divide(node_R, token_Z, 1e-6, 1e-6)
                    token_Z.backward(token_S)
                    token_R = token_embeds.data * token_embeds.grad
                    token_R_list.append(token_R.clone().detach())
                else:  # compute score for all inputs masked
                    if include_root_extend:
                        token_embeds = token_embeds_list[0]
                        claim_R = class_claim_R
                        token_embeds.requires_grad_()
                        token_embeds.retain_grad()
                        if token_embeds.grad is None:
                            pass
                        else:
                            token_embeds.grad.zero_()
                        token_Z = token_embeds.mean(0, keepdim=True)
                        token_S = safe_divide(claim_R, token_Z, 1e-6, 1e-6)
                        token_Z.backward(token_S)
                        token_R = token_embeds.data * token_embeds.grad
                        claim_R_list = token_R.clone().detach()
                class_scores = torch.cat(token_R_list, dim=0)
                if include_root_extend:
                    class_scores = torch.cat((class_scores, claim_R_list), dim=0)
                component_scores[:, class_num] = class_scores.sum(-1)

            # do the minimsation using ILP
            A_ub, b_ub = [], []  # np.ones((component_count, 3))  # constraints
            A_eq, b_eq = [], []
            # this is the constraint for ILP
            threshold = 0.00
            is_pos = (component_scores > threshold).sum(-1)
            is_shared = is_pos >= 2
            pred_class_scores = component_scores[:, original_pred_idx]
            pos_components = torch.where(
                pred_class_scores > threshold,
                torch.zeros_like(pred_class_scores),
                torch.ones_like(pred_class_scores))
            A_eq.append(pos_components)
            b_eq.append(0)
            is_shared = (pred_class_scores > threshold) & is_shared
            shared_components = torch.where(
                is_shared,
                torch.ones_like(pred_class_scores),
                torch.zeros_like(pred_class_scores))
            # disambiguate shared tokens
            for idx, is_shared_component in enumerate(is_shared):
                if is_shared_component:
                    node_num, token_num = flat_idx_2_nested_idx[idx]
                    if node_num != -1:  # graph nodes
                        temp_x = new_x.clone().detach()
                        temp_token_embeds = token_embeds_list[node_num].clone().detach()
                        temp_token_embeds[token_num] = torch.zeros_like(temp_token_embeds[token_num])
                        temp_x[node_num] = temp_token_embeds.mean(0)
                        Batch_data.x = temp_x
                        Batch_data.root = original_claim
                    else:  # root
                        temp_claim = token_embeds_list[0].clone().detach()
                        temp_claim[token_num] = torch.zeros_like(temp_claim[token_num])
                        temp_claim = temp_claim.mean(0)
                        Batch_data.x = new_x
                        Batch_data.root = temp_claim
                    _ = model(Batch_data)
                    logits = model.out.clone().detach()
                    logits_diff = out_logits - logits
                    if original_pred_idx == logits_diff.argmax():
                        shared_components[idx] = 0
            A_eq.append(shared_components)
            b_eq.append(0)
            A_eq_in = torch.stack(A_eq).cpu().numpy()
            b_eq_in = np.array(b_eq)
            for sparsity_num, sparsity in enumerate([0.0, 0.2, 0.4, 0.6, 0.8]):
                fixed_sparsity = 1 - sparsity
                if sparsity_num == 0:
                    A_ub.append(torch.ones_like(pred_class_scores))
                    max_components = int(component_count * fixed_sparsity)
                    b_ub.append(max_components)
                else:
                    max_components = int(component_count * fixed_sparsity)
                    b_ub[-1] = max_components
                # convert to matrix and vector form
                A_ub_in = torch.stack(A_ub).cpu().numpy()
                b_ub_in = np.array(b_ub)
                c = -pred_class_scores.detach().cpu().numpy()
                new_mask = np.ones_like(c)
                try:
                    solution = linprog(c, A_ub_in, b_ub_in, A_eq_in, b_eq_in, bounds=(0, 1),
                                       method='highs', integrality=np.ones_like(c),
                                       options={'presolve': False})
                    if solution.success:
                        new_mask = solution.x
                except Exception as exce:
                    print(traceback.print_exc(), flush=True)
                    pass
                if sparsity_num == 0:
                    unconstrained_sparsities.append(1 - ((new_mask.sum()) / component_count))
                # generate new masked sample and conduct fidelity test
                masked_x = new_x.clone().detach()
                masked_claim = None
                masked_token_embeds_list = []
                temp_claim_token_embeds = token_embeds_list[0].clone().detach()
                for token_embeds in token_embeds_list:
                    masked_token_embeds_list.append(token_embeds.clone().detach())
                # print(new_mask.shape)
                num_selected_per_node = {}
                for component_num, i in enumerate(new_mask):
                    if i == 1 and component_num != component_count:
                        node_num, token_num = flat_idx_2_nested_idx[component_num]
                        if num_selected_per_node.get(node_num, None) is None:
                            num_selected_per_node[node_num] = 1
                        else:
                            num_selected_per_node[node_num] += 1
                        if node_num != -1:  # mask graph tokens
                            temp_token_embeds = masked_token_embeds_list[node_num]
                            temp_token_embeds[token_num] = torch.zeros_like(
                                temp_token_embeds[token_num],
                                device=temp_token_embeds.device)
                        else:  # mask claim tokens
                            temp_claim_token_embeds[token_num] = torch.zeros_like(
                                temp_claim_token_embeds[token_num],
                                device=temp_claim_token_embeds.device)
                else:
                    if num_selected_per_node.get(-1, None) is not None:
                        num_tokens_removed = num_selected_per_node[-1]
                    else:
                        num_tokens_removed = 0
                    num_tokens = temp_claim_token_embeds.shape[0]
                    masked_claim = temp_claim_token_embeds.sum(0) / (num_tokens - num_tokens_removed)
                for node_num, token_embeds in enumerate(masked_token_embeds_list):
                    if num_selected_per_node.get(node_num, None) is not None:
                        num_tokens_removed = num_selected_per_node[node_num]
                    else:
                        num_tokens_removed = 0
                    num_tokens = token_embeds.shape[0]
                    new_node = token_embeds.sum(0) / (num_tokens - num_tokens_removed)
                    masked_x[node_num] = new_node
                Batch_data.x = masked_x
                if include_root_extend:
                    Batch_data.root = masked_claim
                else:
                    Batch_data.root = masked_x[0]
                fidelity_probs = model(Batch_data)
                if type(fidelity_probs) is tuple:
                    fidelity_probs = fidelity_probs[0]
                _, fidelity_pred = fidelity_probs.max(-1)
                # generate explanation and conduct validity test
                expl_x = torch.zeros_like(new_x, device=new_x.device)
                expl_claim = None
                expl_token_embeds_list = []
                temp_claim_token_embeds = token_embeds_list[0].clone().detach()
                for token_embeds in token_embeds_list:
                    expl_token_embeds_list.append(token_embeds.clone().detach())
                num_selected_per_node = {}
                for component_num, i in enumerate(new_mask):
                    if i != 1 and component_num != component_count:
                        node_num, token_num = flat_idx_2_nested_idx[component_num]
                        if num_selected_per_node.get(node_num, None) is None:
                            num_selected_per_node[node_num] = 1
                        else:
                            num_selected_per_node[node_num] += 1
                        if node_num != -1:  # select explanation graph tokens
                            temp_token_embeds = expl_token_embeds_list[node_num]
                            temp_token_embeds[token_num] = torch.zeros_like(
                                temp_token_embeds[token_num],
                                device=temp_token_embeds.device)
                        else:  # select explanation claim tokens
                            temp_claim_token_embeds[token_num] = torch.zeros_like(
                                temp_claim_token_embeds[token_num],
                                device=temp_claim_token_embeds.device)
                else:
                    if num_selected_per_node.get(-1, None) is not None:
                        num_tokens_selected = num_selected_per_node[-1]
                        expl_claim = temp_claim_token_embeds.sum(0) / num_tokens_selected
                    else:
                        expl_claim = temp_claim_token_embeds.mean(0)
                for node_num, token_embeds in enumerate(expl_token_embeds_list):
                    if num_selected_per_node.get(node_num, None) is not None:
                        num_tokens_selected = num_selected_per_node[node_num]
                        new_node = token_embeds.sum(0) / num_tokens_selected
                        expl_x[node_num] = new_node
                Batch_data.x = expl_x
                if include_root_extend:
                    Batch_data.root = expl_claim
                else:
                    Batch_data.root = expl_x[0]
                validity_probs = model(Batch_data)
                if type(validity_probs) is tuple:
                    validity_probs = validity_probs[0]
                _, validity_pred = validity_probs.max(-1)
                flush = False
                if torch.eq(validity_pred, original_pred):
                    valid[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    # print(
                    #     f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\tnot valid\n'
                    #     f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #     f'validity pred:{validity_pred.item()}', flush=flush)
                if not torch.eq(fidelity_pred, original_pred):
                    flipped[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    # print(f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\t'
                    #       f'flip failed', solution.success, solution.status, solution.nit, solution.fun)
                    # print(
                    #     f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #     f'fidelity pred: {fidelity_pred.item()}', flush=flush)
                    if original_pred.item() == all_masked_pred.item():
                        original_eq_all_masked[sparsity_num] += 1
            sample_count += 1
            sample_num += 1
            continue
        elif exp_method == 'lrp':
            model.eval()
            new_x.retain_grad()
            if include_root_extend:
                original_claim = Batch_data.root
                new_root = Batch_data.root.clone().detach()
                new_root.requires_grad_()
                new_root.retain_grad()
                Batch_data.root = new_root
            # get the flattened index for token and their nested index equivalent
            component_count = 0
            root_components_range = [0, 0]
            for node_num, token_embeds in enumerate(token_embeds_list):
                for token_num in range(token_embeds.shape[0]):
                    flat_idx_2_nested_idx[component_count] = (node_num, token_num)
                    nested_idx_2_flat_idx[(node_num, token_num)] = component_count
                    component_count += 1
            else:  # get root tokens
                if include_root_extend:
                    root_components_range[0] = component_count
                    for token_num in range(token_embeds_list[0].shape[0]):
                        flat_idx_2_nested_idx[component_count] = (-1, token_num)
                        nested_idx_2_flat_idx[(-1, token_num)] = component_count
                        component_count += 1
                    else:
                        root_components_range[1] = component_count
            temp_x = torch.zeros_like(new_x, device=new_x.device)
            Batch_data.x = temp_x
            if include_root_extend:
                temp_claim = torch.zeros_like(original_claim, device=original_claim.device)
                Batch_data.root = temp_claim
            all_masked_probs = model(Batch_data)
            if type(all_masked_probs) is tuple:
                all_masked_probs = all_masked_probs[0]
            _, all_masked_pred = all_masked_probs.max(-1)
            all_masked_logits = model.out.clone().detach()
            Batch_data.root = original_claim
            claim_only_probs = model(Batch_data)
            if type(claim_only_probs) is tuple:
                claim_only_probs = claim_only_probs[0]
            _, claim_only_pred = claim_only_probs.max(-1)
            claim_only_logits = model.out.clone().detach()
            Batch_data.x = new_x
            Batch_data.root = torch.zeros_like(original_claim, device=original_claim.device)
            graph_only_probs = model(Batch_data)
            if type(graph_only_probs) is tuple:
                graph_only_probs = graph_only_probs[0]
            _, graph_only_pred = graph_only_probs.max(-1)
            graph_only_logits = model.out.clone().detach()
            Batch_data.x = new_x
            if include_root_extend:
                Batch_data.root = new_root
            original_probs = model(Batch_data)
            if type(original_probs) is tuple:
                original_probs = original_probs[0]
            _, original_pred = original_probs.max(-1)
            original_pred_idx = original_pred.item()
            out_logits = model.out.clone().detach()
            logits_range = out_logits - all_masked_logits
            graph_logits_range = claim_only_logits - all_masked_logits
            claim_logits_range = graph_only_logits - all_masked_logits
            graph_plus_claim_logits_range = graph_logits_range + claim_logits_range

            if Batch_data.x.grad is not None:
                Batch_data.x.grad.zero_()
                if include_root_extend:
                    Batch_data.root.grad.zero_()
            with torch.enable_grad():
                output = model_copy(Batch_data)  # forward
                if type(output) is tuple:
                    output = output[0]
            output[0, original_pred_idx].backward()
            class_x_R = Batch_data.x.grad.detach().clone()
            if include_root_extend:
                class_claim_R = Batch_data.root.grad.detach().clone()
            token_R_list = []
            for node_num, token_embeds in enumerate(token_embeds_list):
                temp_list = []
                node_R = class_x_R[node_num]
                token_embeds.requires_grad_()
                token_embeds.retain_grad()
                if token_embeds.grad is None:
                    pass
                else:
                    token_embeds.grad.zero_()
                token_Z = token_embeds.mean(0)
                token_S = safe_divide(node_R, token_Z, 1e-6, 1e-6)
                token_Z.backward(token_S)
                token_R = token_embeds.data * token_embeds.grad
                token_R_list.append(token_R.clone().detach())
            else:  # compute score for all inputs masked
                if include_root_extend:
                    token_embeds = token_embeds_list[0]
                    claim_R = class_claim_R
                    token_embeds.requires_grad_()
                    token_embeds.retain_grad()
                    if token_embeds.grad is None:
                        pass
                    else:
                        token_embeds.grad.zero_()
                    token_Z = token_embeds.mean(0, keepdim=True)
                    token_S = safe_divide(claim_R, token_Z, 1e-6, 1e-6)
                    token_Z.backward(token_S)
                    token_R = token_embeds.data * token_embeds.grad
                    claim_R_list = token_R.clone().detach()
            component_scores = torch.cat(token_R_list, dim=0)
            if include_root_extend:
                component_scores = torch.cat((component_scores, claim_R_list), dim=0)
            component_scores = component_scores.sum(-1)
            # do the minimsation using ILP
            A_ub, b_ub = [], []  # np.ones((component_count, 3))  # constraints
            A_eq, b_eq = [], []
            # this is the constraint for ILP
            threshold = 0.00
            pos_components = torch.where(
                component_scores > threshold,
                torch.zeros_like(component_scores),
                torch.ones_like(component_scores))
            A_eq.append(pos_components)
            b_eq.append(0)
            A_eq_in = torch.stack(A_eq).cpu().numpy()
            b_eq_in = np.array(b_eq)
            for sparsity_num, sparsity in enumerate([0.0, 0.2, 0.4, 0.6, 0.8]):
                fixed_sparsity = 1 - sparsity
                if sparsity_num == 0:
                    A_ub.append(torch.ones_like(component_scores))
                    max_components = int(component_count * fixed_sparsity)
                    b_ub.append(max_components)
                else:
                    max_components = int(component_count * fixed_sparsity)
                    b_ub[-1] = max_components
                # convert to matrix and vector form
                A_ub_in = torch.stack(A_ub).cpu().numpy()
                b_ub_in = np.array(b_ub)
                solution_intersect = None
                c = -component_scores.detach().cpu().numpy()
                new_mask = np.ones_like(c)
                try:
                    solution = linprog(c, A_ub_in, b_ub_in, A_eq_in, b_eq_in, bounds=(0, 1),
                                       method='highs', integrality=np.ones_like(c),
                                       options={'presolve': False})
                    if solution.success:
                        new_mask = solution.x
                except Exception as exce:
                    print(traceback.print_exc(), flush=True)
                    pass
                if sparsity_num == 0:
                    unconstrained_sparsities.append(1 - ((new_mask.sum()) / component_count))
                # generate new masked sample and conduct fidelity test
                masked_x = new_x.clone().detach()
                masked_claim = None
                masked_token_embeds_list = []
                temp_claim_token_embeds = token_embeds_list[0].clone().detach()
                for token_embeds in token_embeds_list:
                    masked_token_embeds_list.append(token_embeds.clone().detach())
                num_selected_per_node = {}
                for component_num, i in enumerate(new_mask):
                    if i == 1 and component_num != component_count:
                        node_num, token_num = flat_idx_2_nested_idx[component_num]
                        if num_selected_per_node.get(node_num, None) is None:
                            num_selected_per_node[node_num] = 1
                        else:
                            num_selected_per_node[node_num] += 1
                        if node_num != -1:  # mask graph tokens
                            temp_token_embeds = masked_token_embeds_list[node_num]
                            temp_token_embeds[token_num] = torch.zeros_like(
                                temp_token_embeds[token_num],
                                device=temp_token_embeds.device)
                        else:  # mask claim tokens
                            temp_claim_token_embeds[token_num] = torch.zeros_like(
                                temp_claim_token_embeds[token_num],
                                device=temp_token_embeds.device)
                else:
                    if num_selected_per_node.get(-1, None) is not None:
                        num_tokens_removed = num_selected_per_node[-1]
                    else:
                        num_tokens_removed = 0
                    num_tokens = temp_claim_token_embeds.shape[0]
                    masked_claim = temp_claim_token_embeds.sum(0) / (num_tokens - num_tokens_removed)
                for node_num, token_embeds in enumerate(masked_token_embeds_list):
                    if num_selected_per_node.get(node_num, None) is not None:
                        num_tokens_removed = num_selected_per_node[node_num]
                    else:
                        num_tokens_removed = 0
                    num_tokens = token_embeds.shape[0]
                    new_node = token_embeds.sum(0) / (num_tokens - num_tokens_removed)
                    masked_x[node_num] = new_node
                Batch_data.x = masked_x
                if include_root_extend:
                    Batch_data.root = masked_claim
                else:
                    Batch_data.root = masked_x[0]
                fidelity_probs = model(Batch_data)
                if type(fidelity_probs) is tuple:
                    fidelity_probs = fidelity_probs[0]
                _, fidelity_pred = fidelity_probs.max(-1)
                # generate explanation and conduct validity test
                # expl_x = new_x.clone().detach()
                expl_x = torch.zeros_like(new_x, device=new_x.device)
                expl_claim = None
                expl_token_embeds_list = []
                temp_claim_token_embeds = token_embeds_list[0].clone().detach()
                for token_embeds in token_embeds_list:
                    expl_token_embeds_list.append(token_embeds.clone().detach())
                num_selected_per_node = {}
                for component_num, i in enumerate(new_mask):
                    if i != 1 and component_num != component_count:
                        node_num, token_num = flat_idx_2_nested_idx[component_num]
                        if num_selected_per_node.get(node_num, None) is None:
                            num_selected_per_node[node_num] = 1
                        else:
                            num_selected_per_node[node_num] += 1
                        if node_num != -1:  # select explanation graph tokens
                            temp_token_embeds = expl_token_embeds_list[node_num]
                            temp_token_embeds[token_num] = torch.zeros_like(
                                temp_token_embeds[token_num],
                                device=temp_token_embeds.device)
                        else:  # select explanation claim tokens
                            temp_claim_token_embeds[token_num] = torch.zeros_like(
                                temp_claim_token_embeds[token_num],
                                device=temp_token_embeds.device)
                else:
                    if num_selected_per_node.get(-1, None) is not None:
                        num_tokens_selected = num_selected_per_node[-1]
                        expl_claim = temp_claim_token_embeds.sum(0) / num_tokens_selected
                    else:
                        expl_claim = temp_claim_token_embeds.mean(0)
                for node_num, token_embeds in enumerate(expl_token_embeds_list):
                    if num_selected_per_node.get(node_num, None) is not None:
                        num_tokens_selected = num_selected_per_node[node_num]
                        new_node = token_embeds.sum(0) / num_tokens_selected
                        expl_x[node_num] = new_node
                Batch_data.x = expl_x
                if include_root_extend:
                    Batch_data.root = expl_claim
                else:
                    Batch_data.root = expl_x[0]
                validity_probs = model(Batch_data)
                if type(validity_probs) is tuple:
                    validity_probs = validity_probs[0]
                _, validity_pred = validity_probs.max(-1)
                flush = False
                if torch.eq(validity_pred, original_pred):
                    valid[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    # print(
                    #     f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\tnot valid\n'
                    #     f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #     f'validity pred:{validity_pred.item()}', flush=flush)
                if not torch.eq(fidelity_pred, original_pred):
                    flipped[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    # print(f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\t'
                    #       f'flip failed', solution.success, solution.status, solution.nit, solution.fun)
                    # print(
                    #     f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #     f'fidelity pred: {fidelity_pred.item()}', flush=flush)
                    if original_pred.item() == all_masked_pred.item():
                        original_eq_all_masked[sparsity_num] += 1
            sample_count += 1
            sample_num += 1
            continue
        elif exp_method == 'GraphLIME':
            with torch.no_grad():
                original_probs = model(Batch_data)
                if type(original_probs) is tuple:
                    original_probs = original_probs[0]
                _, original_pred = original_probs.max(-1)
                original_pred_idx = original_pred.item()
                original_output = model.out.clone().detach()
                Batch_data.x = torch.zeros_like(new_x, device=new_x.device)
                if include_root_extend:
                    Batch_data.root = torch.zeros_like(original_claim, device=original_claim.device)
                else:
                    Batch_data.root = original_claim
                all_masked_probs = model(Batch_data)
                if type(all_masked_probs) is tuple:
                    all_masked_probs = all_masked_probs[0]
                _, all_masked_pred = all_masked_probs.max(-1)
                # print(Batch_data.y, original_pred, baseline)
                component_count = 0
                root_components_range = [0, 0]  # [start, end)
                # get the flattened index for token and their nested index equivalent
                for node_num, token_embeds in enumerate(token_embeds_list):
                    for token_num in range(token_embeds.shape[0]):
                        # if token_num == 0 or token_num == token_embeds.shape[0] - 1:
                        #     continue
                        flat_idx_2_nested_idx[component_count] = (node_num, token_num)
                        nested_idx_2_flat_idx[(node_num, token_num)] = component_count
                        component_count += 1
                else:  # get root tokens
                    if include_root_extend:
                        root_components_range[0] = component_count
                        for token_num in range(token_embeds_list[0].shape[0]):
                            flat_idx_2_nested_idx[component_count] = (-1, token_num)
                            nested_idx_2_flat_idx[(-1, token_num)] = component_count
                            component_count += 1
                        else:
                            root_components_range[1] = component_count

                if include_root_extend:
                    node_scores = torch.zeros((new_x.shape[0] + 1, num_class), device=new_x.device)
                else:
                    node_scores = torch.zeros((new_x.shape[0], num_class), device=new_x.device)
                # get node scores by sampling
                for node_num in range(new_x.shape[0]):
                    temp_x = torch.zeros_like(new_x, device=new_x.device)
                    temp_x[node_num] = new_x[node_num].clone().detach()
                    Batch_data.x = temp_x
                    if include_root_extend:
                        Batch_data.root = torch.zeros_like(original_claim, device=original_claim.device)
                    else:
                        Batch_data.root = original_claim
                    probs = model(Batch_data)
                    node_scores[node_num] = torch.exp(probs).clone().detach()
                else:
                    if include_root_extend:
                        Batch_data.root = original_claim
                        Batch_data.x = torch.zeros_like(new_x, device=new_x.device)
                        probs = model(Batch_data)
                        node_scores[-1] = torch.exp(probs).clone().detach()
                flattened_x = torch.cat((new_x, original_claim), dim=0).detach().cpu().numpy()
                flattened_y = node_scores.detach().cpu().numpy()

                betas = GraphLIME().compute_betas(flattened_x, flattened_y, rho=0.0001)
                betas = torch.as_tensor(betas, device=new_x.device)
                betas_pos = torch.clamp(betas, min=0)
                flattened_components = torch.cat(token_embeds_list, dim=0)
                if include_root_extend:
                    flattened_components = torch.cat((flattened_components, token_embeds_list[0]), dim=0)
                # multiply token embeds by feature coefficients to get scores for individual tokens
                flattened_components = flattened_components * betas_pos.unsqueeze(0)
                flattened_component_scores = flattened_components.sum(-1)
                A_ub, b_ub = [], []  # np.ones((component_count, 3))  # constraints
                # use LP to find explanation with fixed sparsities
                for sparsity_num, sparsity in enumerate([0.0, 0.2, 0.4, 0.6, 0.8]):
                    fixed_sparsity = 1 - sparsity
                    if sparsity_num == 0:
                        A_ub.append(torch.ones_like(flattened_component_scores))
                        max_components = int(component_count * fixed_sparsity)
                        b_ub.append(max_components)
                    else:
                        max_components = int(component_count * fixed_sparsity)
                        b_ub[-1] = max_components
                    # convert to matrix and vector form
                    A_ub_in = torch.stack(A_ub).cpu().numpy()
                    b_ub_in = np.array(b_ub)
                    c = torch.nan_to_num(flattened_component_scores).detach().cpu().numpy()
                    solution = linprog(c, A_ub_in, b_ub_in, bounds=(0, 1),
                                       method='highs', integrality=np.ones_like(c),
                                       options={'presolve': True})
                    new_mask = solution.x
                    if sparsity_num == 0:
                        unconstrained_sparsities.append(1 - (new_mask.sum() / component_count))
                    # generate new masked sample and conduct fidelity test
                    masked_x = new_x.clone().detach()
                    masked_claim = None
                    masked_token_embeds_list = []
                    temp_claim_token_embeds = token_embeds_list[0].clone().detach()
                    for token_embeds in token_embeds_list:
                        masked_token_embeds_list.append(token_embeds.clone().detach())
                    # print(new_mask.shape)
                    for component_num, i in enumerate(new_mask):
                        if i == 1 and component_num != component_count:
                            node_num, token_num = flat_idx_2_nested_idx[component_num]
                            if node_num != -1:  # mask graph tokens
                                temp_token_embeds = masked_token_embeds_list[node_num]
                                temp_token_embeds[token_num] = torch.zeros_like(
                                    temp_token_embeds[token_num],
                                    device=temp_token_embeds.device)
                            else:  # mask claim tokens
                                temp_claim_token_embeds[token_num] = torch.zeros_like(
                                    temp_claim_token_embeds[token_num],
                                    device=temp_token_embeds.device)
                    else:
                        masked_claim = temp_claim_token_embeds.mean(0)
                    for node_num, token_embeds in enumerate(masked_token_embeds_list):
                        new_node = token_embeds.mean(0)
                        masked_x[node_num] = new_node
                    Batch_data.x = masked_x
                    if include_root_extend:
                        Batch_data.root = masked_claim
                    else:
                        Batch_data.root = masked_x[0]
                    fidelity_probs = model(Batch_data)
                    if type(fidelity_probs) is tuple:
                        fidelity_probs = fidelity_probs[0]
                    _, fidelity_pred = fidelity_probs.max(-1)
                    # generate explanation and conduct validity test
                    expl_x = new_x.clone().detach()
                    expl_claim = None
                    expl_token_embeds_list = []
                    temp_claim_token_embeds = token_embeds_list[0].clone().detach()
                    for token_embeds in token_embeds_list:
                        expl_token_embeds_list.append(token_embeds.clone().detach())
                    for component_num, i in enumerate(new_mask):
                        if i == 0 and component_num != component_count:
                            node_num, token_num = flat_idx_2_nested_idx[component_num]
                            if node_num != -1:  # select explanation graph tokens
                                temp_token_embeds = expl_token_embeds_list[node_num]
                                temp_token_embeds[token_num] = torch.zeros_like(
                                    temp_token_embeds[token_num],
                                    device=temp_token_embeds.device)
                            else:  # select explanation claim tokens
                                temp_claim_token_embeds[token_num] = torch.zeros_like(
                                    temp_claim_token_embeds[token_num],
                                    device=temp_token_embeds.device)
                    else:
                        expl_claim = temp_claim_token_embeds.mean(0)
                    for node_num, token_embeds in enumerate(expl_token_embeds_list):
                        new_node = token_embeds.mean(0)
                        expl_x[node_num] = new_node
                    Batch_data.x = expl_x
                    if include_root_extend:
                        Batch_data.root = expl_claim
                    else:
                        Batch_data.root = expl_x[0]
                    validity_probs = model(Batch_data)
                    if type(validity_probs) is tuple:
                        validity_probs = validity_probs[0]
                    _, validity_pred = validity_probs.max(-1)
                    flush = False
                    if torch.eq(validity_pred, original_pred):
                        valid[sparsity_num] += 1
                    else:
                        flush_count += 1
                        if flush_count >= 50:
                            flush_count = 0
                            flush = True
                        print(
                            f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\tnot valid\n'
                            f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                            f'validity pred:{validity_pred.item()}', flush=flush)
                    if not torch.eq(fidelity_pred, original_pred):
                        flipped[sparsity_num] += 1
                    else:
                        flush_count += 1
                        if flush_count >= 50:
                            flush_count = 0
                            flush = True
                        print(f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\t'
                              f'flip failed', solution.success, solution.status, solution.nit, solution.fun)
                        print(
                            f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                            f'fidelity pred: {fidelity_pred.item()}', flush=flush)
                        if original_pred.item() == all_masked_pred.item():
                            original_eq_all_masked[sparsity_num] += 1
                sample_count += 1
                sample_num += 1
                continue
        elif exp_method == 'sglrp':
            model.eval()
            new_x.retain_grad()
            if include_root_extend:
                original_claim = Batch_data.root
                new_root = Batch_data.root.clone().detach()
                new_root.requires_grad_()
                new_root.retain_grad()
                Batch_data.root = new_root
            # get the flattened index for token and their nested index equivalent
            component_count = 0
            root_components_range = [0, 0]
            for node_num, token_embeds in enumerate(token_embeds_list):
                for token_num in range(token_embeds.shape[0]):
                    flat_idx_2_nested_idx[component_count] = (node_num, token_num)
                    nested_idx_2_flat_idx[(node_num, token_num)] = component_count
                    component_count += 1
            else:  # get root tokens
                if include_root_extend:
                    root_components_range[0] = component_count
                    for token_num in range(token_embeds_list[0].shape[0]):
                        flat_idx_2_nested_idx[component_count] = (-1, token_num)
                        nested_idx_2_flat_idx[(-1, token_num)] = component_count
                        component_count += 1
                    else:
                        root_components_range[1] = component_count
            temp_x = torch.zeros_like(new_x, device=new_x.device)
            Batch_data.x = temp_x
            if include_root_extend:
                temp_claim = torch.zeros_like(original_claim, device=original_claim.device)
                Batch_data.root = temp_claim
            all_masked_probs = model(Batch_data)
            if type(all_masked_probs) is tuple:
                all_masked_probs = all_masked_probs[0]
            _, all_masked_pred = all_masked_probs.max(-1)
            all_masked_logits = model.out.clone().detach()
            Batch_data.root = original_claim
            claim_only_probs = model(Batch_data)
            if type(claim_only_probs) is tuple:
                claim_only_probs = claim_only_probs[0]
            _, claim_only_pred = claim_only_probs.max(-1)
            claim_only_logits = model.out.clone().detach()
            Batch_data.x = new_x
            Batch_data.root = torch.zeros_like(original_claim, device=original_claim.device)
            graph_only_probs = model(Batch_data)
            if type(graph_only_probs) is tuple:
                graph_only_probs = graph_only_probs[0]
            _, graph_only_pred = graph_only_probs.max(-1)
            graph_only_logits = model.out.clone().detach()
            # print(f'graph_only_logits {graph_only_logits}')
            Batch_data.x = new_x
            if include_root_extend:
                Batch_data.root = new_root
            original_probs = model(Batch_data)
            if type(original_probs) is tuple:
                original_probs = original_probs[0]
            _, original_pred = original_probs.max(-1)
            out_logits = model.out.clone().detach()
            logits_range = out_logits - all_masked_logits
            graph_logits_range = claim_only_logits - all_masked_logits
            claim_logits_range = graph_only_logits - all_masked_logits
            graph_plus_claim_logits_range = graph_logits_range + claim_logits_range

            if Batch_data.x.grad is not None:
                Batch_data.x.grad.zero_()
                if include_root_extend:
                    Batch_data.root.grad.zero_()
            with torch.enable_grad():
                output = model_copy(Batch_data)  # forward
                if type(output) is tuple:
                    output = output[0]
            base_R = torch.exp(output.clone().detach())
            coeff_R = torch.ones_like(base_R, device=base_R.device)
            for class_num in range(num_class):
                if class_num == original_pred:
                    coeff_R[0, class_num] = 1 - base_R[0, class_num]
                else:
                    coeff_R[0, class_num] = -base_R[0, class_num]
            base_R *= coeff_R
            output.backward(base_R)
            class_x_R = Batch_data.x.grad.detach().clone()
            if include_root_extend:
                class_claim_R = Batch_data.root.grad.detach().clone()
            token_R_list = []
            for node_num, token_embeds in enumerate(token_embeds_list):
                temp_list = []
                node_R = class_x_R[node_num]
                token_embeds.requires_grad_()
                token_embeds.retain_grad()
                if token_embeds.grad is None:
                    pass
                else:
                    token_embeds.grad.zero_()
                token_Z = token_embeds.mean(0)
                token_S = safe_divide(node_R, token_Z, 1e-6, 1e-6)
                token_Z.backward(token_S)
                token_R = token_embeds.data * token_embeds.grad
                token_R_list.append(token_R.clone().detach())
            else:  # compute score for all inputs masked
                if include_root_extend:
                    token_embeds = token_embeds_list[0]
                    claim_R = class_claim_R
                    token_embeds.requires_grad_()
                    token_embeds.retain_grad()
                    if token_embeds.grad is None:
                        pass
                    else:
                        token_embeds.grad.zero_()
                    token_Z = token_embeds.mean(0, keepdim=True)
                    token_S = safe_divide(claim_R, token_Z, 1e-6, 1e-6)
                    token_Z.backward(token_S)
                    token_R = token_embeds.data * token_embeds.grad
                    claim_R_list = token_R.clone().detach()
            component_scores = torch.cat(token_R_list, dim=0)
            if include_root_extend:
                component_scores = torch.cat((component_scores, claim_R_list), dim=0)
            pos_scores = component_scores.clamp(min=0)
            neg_scores = component_scores.clamp(max=0)
            component_scores = component_scores.sum(-1)
            # do the minimsation using ILP
            A_ub, b_ub = [], []  # np.ones((component_count, 3))  # constraints
            A_eq, b_eq = [], []
            # get influence diff, this is the constraint for ILP
            threshold = 0.00
            pos_components = torch.where(
                component_scores > threshold,
                torch.zeros_like(component_scores),
                torch.ones_like(component_scores))
            A_eq.append(pos_components)
            b_eq.append(0)
            A_eq_in = torch.stack(A_eq).cpu().numpy()
            b_eq_in = np.array(b_eq)
            for sparsity_num, sparsity in enumerate([0.0, 0.2, 0.4, 0.6, 0.8]):
                fixed_sparsity = 1 - sparsity
                if sparsity_num == 0:
                    A_ub.append(torch.ones_like(component_scores))
                    max_components = int(component_count * fixed_sparsity)
                    b_ub.append(max_components)
                else:
                    max_components = int(component_count * fixed_sparsity)
                    b_ub[-1] = max_components
                # convert to matrix and vector form
                A_ub_in = torch.stack(A_ub).cpu().numpy()
                b_ub_in = np.array(b_ub)
                c = -component_scores.detach().cpu().numpy()
                new_mask = np.ones_like(c)
                try:
                    solution = linprog(c, A_ub_in, b_ub_in, A_eq_in, b_eq_in, bounds=(0, 1),
                                       method='highs', integrality=np.ones_like(c),
                                       options={'presolve': False})
                    if solution.success:
                        new_mask = solution.x
                except Exception as exce:
                    print(traceback.print_exc(), flush=True)
                    continue
                if sparsity_num == 0:
                    unconstrained_sparsities.append(1 - ((new_mask.sum()) / component_count))
                # generate new masked sample and conduct fidelity test
                masked_x = new_x.clone().detach()
                masked_claim = None
                masked_token_embeds_list = []
                temp_claim_token_embeds = token_embeds_list[0].clone().detach()
                for token_embeds in token_embeds_list:
                    masked_token_embeds_list.append(token_embeds.clone().detach())
                num_selected_per_node = {}
                for component_num, i in enumerate(new_mask):
                    if i == 1 and component_num != component_count:
                        node_num, token_num = flat_idx_2_nested_idx[component_num]
                        if num_selected_per_node.get(node_num, None) is None:
                            num_selected_per_node[node_num] = 1
                        else:
                            num_selected_per_node[node_num] += 1
                        if node_num != -1:  # mask graph tokens
                            temp_token_embeds = masked_token_embeds_list[node_num]
                            temp_token_embeds[token_num] = torch.zeros_like(
                                temp_token_embeds[token_num],
                                device=temp_token_embeds.device)
                        else:  # mask claim tokens
                            temp_claim_token_embeds[token_num] = torch.zeros_like(
                                temp_claim_token_embeds[token_num],
                                device=temp_token_embeds.device)
                else:
                    if num_selected_per_node.get(-1, None) is not None:
                        num_tokens_removed = num_selected_per_node[-1]
                    else:
                        num_tokens_removed = 0
                    num_tokens = temp_claim_token_embeds.shape[0]
                    masked_claim = temp_claim_token_embeds.sum(0) / (num_tokens - num_tokens_removed)
                for node_num, token_embeds in enumerate(masked_token_embeds_list):
                    if num_selected_per_node.get(node_num, None) is not None:
                        num_tokens_removed = num_selected_per_node[node_num]
                    else:
                        num_tokens_removed = 0
                    num_tokens = token_embeds.shape[0]
                    new_node = token_embeds.sum(0) / (num_tokens - num_tokens_removed)
                    masked_x[node_num] = new_node
                Batch_data.x = masked_x
                if include_root_extend:
                    Batch_data.root = masked_claim
                else:
                    Batch_data.root = masked_x[0]
                fidelity_probs = model(Batch_data)
                if type(fidelity_probs) is tuple:
                    fidelity_probs = fidelity_probs[0]
                _, fidelity_pred = fidelity_probs.max(-1)
                # generate explanation and conduct validity test
                expl_x = torch.zeros_like(new_x, device=new_x.device)
                expl_claim = None
                expl_token_embeds_list = []
                temp_claim_token_embeds = token_embeds_list[0].clone().detach()
                for token_embeds in token_embeds_list:
                    expl_token_embeds_list.append(token_embeds.clone().detach())
                num_selected_per_node = {}
                for component_num, i in enumerate(new_mask):
                    if i != 1 and component_num != component_count:
                        node_num, token_num = flat_idx_2_nested_idx[component_num]
                        if num_selected_per_node.get(node_num, None) is None:
                            num_selected_per_node[node_num] = 1
                        else:
                            num_selected_per_node[node_num] += 1
                        if node_num != -1:  # select explanation graph tokens
                            temp_token_embeds = expl_token_embeds_list[node_num]
                            temp_token_embeds[token_num] = torch.zeros_like(
                                temp_token_embeds[token_num],
                                device=temp_token_embeds.device)
                        else:  # select explanation claim tokens
                            temp_claim_token_embeds[token_num] = torch.zeros_like(
                                temp_claim_token_embeds[token_num],
                                device=temp_token_embeds.device)
                else:
                    if num_selected_per_node.get(-1, None) is not None:
                        num_tokens_selected = num_selected_per_node[-1]
                        expl_claim = temp_claim_token_embeds.sum(0) / num_tokens_selected
                    else:
                        expl_claim = temp_claim_token_embeds.mean(0)
                for node_num, token_embeds in enumerate(expl_token_embeds_list):
                    if num_selected_per_node.get(node_num, None) is not None:
                        num_tokens_selected = num_selected_per_node[node_num]
                        new_node = token_embeds.sum(0) / num_tokens_selected
                        expl_x[node_num] = new_node
                Batch_data.x = expl_x
                if include_root_extend:
                    Batch_data.root = expl_claim
                else:
                    Batch_data.root = expl_x[0]
                validity_probs = model(Batch_data)
                if type(validity_probs) is tuple:
                    validity_probs = validity_probs[0]
                _, validity_pred = validity_probs.max(-1)
                flush = False
                if torch.eq(validity_pred, original_pred):
                    valid[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    print(
                        f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\tnot valid\n'
                        f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                        f'validity pred:{validity_pred.item()}', flush=flush)
                if not torch.eq(fidelity_pred, original_pred):
                    flipped[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    print(f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\t'
                          f'flip failed', solution.success, solution.status, solution.nit, solution.fun)
                    print(
                        f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                        f'fidelity pred: {fidelity_pred.item()}', flush=flush)
                    if original_pred.item() == all_masked_pred.item():
                        original_eq_all_masked[sparsity_num] += 1
            sample_count += 1
            sample_num += 1
            continue
        elif exp_method == 'CFME':
            with torch.no_grad():
                original_probs = model(Batch_data)
                if type(original_probs) is tuple:
                    original_probs = original_probs[0]
                _, original_pred = original_probs.max(-1)
                classwise_pred_counts[original_pred.item()] += 1
                baseline = model.out.clone().detach()
                component_count = 0
                root_components_range = [0, 0]  # [start, end)
                graph_component_count = 0
                # get the flattened index for token and their nested index equivalent
                for node_num, token_embeds in enumerate(token_embeds_list):
                    for token_num in range(token_embeds.shape[0]):
                        flat_idx_2_nested_idx[component_count] = (node_num, token_num)
                        nested_idx_2_flat_idx[(node_num, token_num)] = component_count
                        component_count += 1
                else:  # get root tokens
                    graph_component_count = component_count
                    if include_root_extend:
                        root_components_range[0] = component_count
                        for token_num in range(token_embeds_list[0].shape[0]):
                            flat_idx_2_nested_idx[component_count] = (-1, token_num)
                            nested_idx_2_flat_idx[(-1, token_num)] = component_count
                            component_count += 1
                        else:
                            root_components_range[1] = component_count
                extension = 0
                if include_graph_bias:
                    extension += 1
                component_scores = torch.zeros((component_count + extension, num_class),
                                               device=baseline.device)
                # compute the score for the input with the component masked
                for component_num in range(component_count):
                    node_num, token_num = flat_idx_2_nested_idx[component_num]
                    if node_num != -1:  # compute score for masked graph tokens
                        temp_x = new_x.clone().detach()
                        temp_token_embeds = token_embeds_list[node_num].clone().detach()
                        temp_token_embeds[token_num] = torch.zeros_like(temp_token_embeds[token_num],
                                                                        device=temp_token_embeds.device)
                        new_node = temp_token_embeds.sum(0)
                        new_node /= (temp_token_embeds.shape[0] - 1)
                        temp_x[node_num] = new_node
                        Batch_data.x = temp_x
                        probs = model(Batch_data)
                        if type(probs) is tuple:
                            probs = probs[0]
                        _, pred = probs.max(-1)
                        component_scores[component_num] = model.out.clone().detach()
                    else:  # compute score for masked claim tokens
                        Batch_data.x = new_x
                        temp_token_embeds = token_embeds_list[0].clone().detach()
                        temp_token_embeds[token_num] = torch.zeros_like(temp_token_embeds[token_num],
                                                                        device=temp_token_embeds.device)
                        temp_claim = temp_token_embeds.sum(0)
                        temp_claim /= (temp_token_embeds.shape[0] - 1)
                        Batch_data.root = temp_claim
                        probs = model(Batch_data)
                        if type(probs) is tuple:
                            probs = probs[0]
                        _, pred = probs.max(-1)
                        component_scores[component_num] = model.out.clone().detach()
                else:  # compute score for all graph tokens and claim tokens masked
                    temp_x = torch.zeros_like(new_x, device=new_x.device)
                    Batch_data.x = temp_x
                    temp_claim = torch.zeros_like(original_claim, device=original_claim.device)
                    Batch_data.root = temp_claim
                    all_masked_probs = model(Batch_data)
                    if type(all_masked_probs) is tuple:
                        all_masked_probs = all_masked_probs[0]
                    _, all_masked_pred = all_masked_probs.max(-1)
                    if include_graph_bias:
                        component_scores[component_count] = model.out.clone().detach()

                influence_scores = baseline - component_scores
                influence_scores[-1] = component_scores[-1]
                A_ub, b_ub = [], []  # np.ones((component_count, 3))  # constraints
                A_eq, b_eq = [], []
                A_ub2, b_ub2 = [], []  # ablated
                A_eq2, b_eq2 = [], []
                # get influence diff, this is the constraint for LP
                obj_scores = torch.zeros_like(influence_scores[:, original_pred.item()])
                ablated_scores = -influence_scores[:, original_pred.item()]
                all_pos = None
                for class_num in range(num_class):
                    if class_num == original_pred:
                        continue
                    else:
                        influence_scores_diff = influence_scores[:, class_num] - influence_scores[:,
                                                                                 original_pred.item()]
                        obj_scores += influence_scores_diff
                        pos_components = torch.where(
                            influence_scores_diff > 0,
                            torch.ones_like(influence_scores[:, class_num]),
                            torch.zeros_like(influence_scores[:, class_num])
                        )
                        pos_components[-1] = 0
                        if pos_components.sum() == pos_components.shape[0] - 1:
                            print(f'sample: {sample_count} class {class_num} '
                                  f'no pos influence gaps with class {original_pred.item()}')
                            continue
                        if all_pos is None:
                            all_pos = pos_components
                        else:
                            all_pos += pos_components
                        if not include_root_extend and not include_graph_bias:
                            A_eq.append(pos_components[:root_components_range[0]])
                        else:
                            A_eq.append(pos_components)
                        b_eq.append(0)
                else:
                    obj_scores /= (num_class - 1)
                    A_ub.append(obj_scores)
                    b_ub.append(0)
                    # ablated
                    ablated_scores[-1] = obj_scores[-1]
                    if not include_root_extend and not include_graph_bias:
                        A_ub2.append(influence_scores_diff[:root_components_range[0]])
                    else:
                        A_ub2.append(ablated_scores)
                    b_ub2.append(0)
                    if include_graph_bias:
                        graph_bias = torch.zeros_like(influence_scores[:, original_pred.item()])
                        graph_bias[-1] = 1
                        A_eq.append(graph_bias)
                        b_eq.append(1)
                        # ablated
                        graph_bias2 = torch.zeros_like(ablated_scores)
                        graph_bias2[-1] = 1
                        A_eq2.append(graph_bias2)
                        b_eq2.append(1)
                # sparisty enforcement
                A_eq_in = torch.stack(A_eq).cpu().numpy()
                b_eq_in = np.array(b_eq)
                # ablated
                A_eq_in2 = torch.stack(A_eq2).cpu().numpy()
                b_eq_in2 = np.array(b_eq2)

                for sparsity_num, sparsity in enumerate(test_sparsities):
                    has_solution_with_graph_bias = False
                    has_solution_without_graph_bias = False
                    if type(sparsity) is not str:
                        fixed_sparsity = 1 - sparsity
                    curr_topk = None
                    curr_topk_idx = None
                    if sparsity_num == 0:
                        A_ub.append(torch.ones_like(influence_scores[:, original_pred.item()]))
                        max_components = int(component_count * fixed_sparsity) + 1
                        b_ub.append(max_components)
                        # ablated
                        A_ub2.append(torch.ones_like(influence_scores[:, original_pred.item()]))
                        max_components = int(component_count * fixed_sparsity) + 1
                        b_ub2.append(max_components)
                    elif type(sparsity) is str:
                        max_components = int(sparsity[3:]) + 1
                        curr_topk = int(sparsity[3:])
                        curr_topk_idx = topk_test[curr_topk]
                        b_ub[-1] = max_components
                        b_ub2[-1] = max_components
                    else:
                        max_components = int(component_count * fixed_sparsity) + 1
                        b_ub[-1] = max_components
                    # convert to matrix and vector form
                    A_ub_in = torch.stack(A_ub).cpu().numpy()
                    b_ub_in = np.array(b_ub)
                    # ablated
                    A_ub_in2 = torch.stack(A_ub2).cpu().numpy()
                    b_ub_in2 = np.array(b_ub2)
                    c = A_ub_in[0]
                    solution = linprog(c, A_ub_in, b_ub_in, A_eq_in, b_eq_in, bounds=(0, 1),
                                       method='highs', integrality=np.ones_like(c),
                                       options={'presolve': False})
                    if sparsity_num == 0 or type(sparsity) is str:
                        c2 = A_ub_in2[0]
                        solution2 = linprog(c2, A_ub_in2, b_ub_in2, A_eq_in2, b_eq_in2, bounds=(0, 1),
                                            method='highs', integrality=np.ones_like(c2),
                                            options={'presolve': False})
                        if solution2.success:
                            new_mask2 = solution2.x
                        else:
                            new_mask2 = np.zeros_like(c)
                            new_mask2[-1] = 1
                    if solution.success:
                        new_mask = solution.x
                        has_solution_with_graph_bias = True
                        solution_with_graph_bias[sparsity_num] += 1
                        if sparsity_num == 0:
                            assert 1 - ((new_mask.sum() - 1) / component_count) <= 1
                            unconstrained_sparsities.append(
                                1 - ((new_mask.sum() - 1) / component_count))
                        # assert False
                    elif not relax_graph_bias:
                        # print('no solution with graph bias', flush=True)
                        new_mask = np.zeros_like(c)
                        new_mask[-1] = 1
                        # assert False
                    else:  # since solution is not found, try again without graph bias
                        # print('no solution with graph bias, trying without graph bias')
                        no_solution_with_graph_bias[sparsity_num] += 1
                        A_ub_no_gb = A_ub_in[:, :-1]
                        b_ub_no_gb = copy.deepcopy(b_ub_in)
                        b_ub_no_gb[-1] -= 1
                        A_eq_no_gb = A_eq_in[:-1, :-1]
                        b_eq_no_gb = b_eq_in[:-1]
                        c = A_ub_no_gb[0]
                        solution = linprog(c, A_ub_no_gb, b_ub_no_gb, A_eq_no_gb, b_eq_no_gb, bounds=(0, 1),
                                           method='highs', integrality=np.ones_like(c),
                                           options={'presolve': False})
                        if solution.success:
                            new_mask = solution.x
                            solution_without_graph_bias[sparsity_num] += 1
                            has_solution_without_graph_bias = True
                        else:
                            new_mask = np.zeros_like(c)
                            new_mask[-1] = 1
                            no_solution_without_graph_bias[sparsity_num] += 1
                            # print('no solution')
                            # continue
                        if sparsity_num == 0:
                            assert 1 - (new_mask.sum() / component_count) <= 1
                            if new_mask.sum() != 1:
                                unconstrained_sparsities.append(1 - (new_mask.sum() / component_count))

                    if sparsity_num == 0 or type(sparsity) is str:
                        smaller = min((new_mask.sum(), new_mask2.sum()))
                        if smaller == 1:  # one of the explanations is an empty set
                            pass
                        else:
                            intersection = np.bitwise_and(new_mask.astype(int), new_mask2.astype(int))
                            union = np.bitwise_or(new_mask.astype(int), new_mask2.astype(int))
                            jac = (intersection.sum() - 1) / (union.sum() - 1)
                            ss = (intersection.sum() - 1) / (smaller - 1)
                            if sparsity_num != 0:
                                if topk_jaccard_similarity_list[curr_topk_idx] is None:
                                    topk_jaccard_similarity_list[curr_topk_idx] = [jac]
                                    topk_ss_similarity_list[curr_topk_idx] = [ss]
                                else:
                                    topk_jaccard_similarity_list[curr_topk_idx].append(jac)
                                    topk_ss_similarity_list[curr_topk_idx].append(ss)
                            else:
                                jaccard_similarity_list.append(jac)
                                ss_similarity_list.append(ss)
                    # generate new masked sample and conduct fidelity test
                    masked_x = new_x.clone().detach()
                    masked_claim = None
                    masked_token_embeds_list = []
                    temp_claim_token_embeds = token_embeds_list[0].clone().detach()
                    # ablated
                    if type(sparsity) is str:
                        masked_x2 = new_x.clone().detach()
                        masked_claim2 = None
                        masked_token_embeds_list2 = []
                        temp_claim_token_embeds2 = token_embeds_list[0].clone().detach()
                    for token_embeds in token_embeds_list:
                        masked_token_embeds_list.append(token_embeds.clone().detach())
                        if type(sparsity) is str:
                            masked_token_embeds_list2.append(token_embeds.clone().detach())
                    for component_num, i in enumerate(new_mask):
                        if i == 1 and component_num != component_count:
                            node_num, token_num = flat_idx_2_nested_idx[component_num]
                            if node_num != -1:  # mask graph tokens
                                temp_token_embeds = masked_token_embeds_list[node_num]
                                temp_token_embeds[token_num] = torch.zeros_like(
                                    temp_token_embeds[token_num],
                                    device=temp_token_embeds.device)
                                if sparsity_num == 0:
                                    token_id = token_ids_list[node_num][token_num]
                                    if classwise_graph_expl_token_id_counts_dict[original_pred.item()] is None:
                                        new_dict = {token_id: 1}
                                        classwise_graph_expl_token_id_counts_dict[original_pred.item()] = new_dict
                                    elif classwise_graph_expl_token_id_counts_dict[original_pred.item()].get(
                                            token_id, None) is None:
                                        classwise_graph_expl_token_id_counts_dict[original_pred.item()][
                                            token_id] = 1
                                    else:
                                        classwise_graph_expl_token_id_counts_dict[original_pred.item()][
                                            token_id] += 1
                                if type(sparsity) is str:
                                    token_id = token_ids_list[node_num][token_num]
                                    if topk_classwise_graph_expl_token_id_counts_dict[curr_topk_idx] is None:
                                        topk_classwise_graph_expl_token_id_counts_dict[curr_topk_idx] = [
                                                                                                            None] * num_class
                                    if topk_classwise_graph_expl_token_id_counts_dict[curr_topk_idx][
                                        original_pred.item()] is None:
                                        new_dict = {token_id: 1}
                                        topk_classwise_graph_expl_token_id_counts_dict[curr_topk_idx][
                                            original_pred.item()] = new_dict
                                    elif topk_classwise_graph_expl_token_id_counts_dict[curr_topk_idx][
                                        original_pred.item()].get(token_id, None) is None:
                                        topk_classwise_graph_expl_token_id_counts_dict[curr_topk_idx][
                                            original_pred.item()][token_id] = 1
                                    else:
                                        topk_classwise_graph_expl_token_id_counts_dict[curr_topk_idx][
                                            original_pred.item()][token_id] += 1
                            else:  # mask claim tokens
                                temp_claim_token_embeds[token_num] = torch.zeros_like(
                                    temp_claim_token_embeds[token_num],
                                    device=temp_claim_token_embeds.device)
                                if sparsity_num == 0:
                                    token_id = token_ids_list[0][token_num]
                                    if classwise_claim_expl_token_id_counts_dict[original_pred.item()] is None:
                                        new_dict = {token_id: 1}
                                        classwise_claim_expl_token_id_counts_dict[original_pred.item()] = new_dict
                                    elif classwise_claim_expl_token_id_counts_dict[original_pred.item()].get(
                                            token_id, None) is None:
                                        classwise_claim_expl_token_id_counts_dict[original_pred.item()][
                                            token_id] = 1
                                    else:
                                        classwise_claim_expl_token_id_counts_dict[original_pred.item()][
                                            token_id] += 1
                                if type(sparsity) is str:
                                    token_id = token_ids_list[0][token_num]
                                    if topk_classwise_claim_expl_token_id_counts_dict[curr_topk_idx] is None:
                                        topk_classwise_claim_expl_token_id_counts_dict[curr_topk_idx] = [
                                                                                                            None] * num_class
                                    if topk_classwise_claim_expl_token_id_counts_dict[curr_topk_idx][
                                        original_pred.item()] is None:
                                        new_dict = {token_id: 1}
                                        topk_classwise_claim_expl_token_id_counts_dict[curr_topk_idx][
                                            original_pred.item()] = new_dict
                                    elif topk_classwise_claim_expl_token_id_counts_dict[curr_topk_idx][
                                        original_pred.item()].get(token_id, None) is None:
                                        topk_classwise_claim_expl_token_id_counts_dict[curr_topk_idx][
                                            original_pred.item()][token_id] = 1
                                    else:
                                        topk_classwise_claim_expl_token_id_counts_dict[curr_topk_idx][
                                            original_pred.item()][token_id] += 1
                            # node_tokens_left[node_num] -= 1
                        if sparsity_num == 0:
                            if new_mask2[component_num] == 1 and component_num != component_count:
                                node_num2, token_num2 = flat_idx_2_nested_idx[component_num]
                                if node_num2 != -1:  # mask graph tokens
                                    token_id2 = token_ids_list[node_num2][token_num2]
                                    if ablated_classwise_graph_expl_token_id_counts_dict[
                                        original_pred.item()] is None:
                                        new_dict2 = {token_id2: 1}
                                        ablated_classwise_graph_expl_token_id_counts_dict[
                                            original_pred.item()] = new_dict2
                                    elif ablated_classwise_graph_expl_token_id_counts_dict[
                                        original_pred.item()].get(token_id2, None) is None:
                                        ablated_classwise_graph_expl_token_id_counts_dict[
                                            original_pred.item()][token_id2] = 1
                                    else:
                                        ablated_classwise_graph_expl_token_id_counts_dict[
                                            original_pred.item()][token_id2] += 1
                                else:  # mask claim tokens
                                    token_id2 = token_ids_list[0][token_num2]
                                    if ablated_classwise_claim_expl_token_id_counts_dict[
                                        original_pred.item()] is None:
                                        new_dict2 = {token_id2: 1}
                                        ablated_classwise_claim_expl_token_id_counts_dict[
                                            original_pred.item()] = new_dict2
                                    elif ablated_classwise_claim_expl_token_id_counts_dict[
                                        original_pred.item()].get(token_id2, None) is None:
                                        ablated_classwise_claim_expl_token_id_counts_dict[
                                            original_pred.item()][token_id2] = 1
                                    else:
                                        ablated_classwise_claim_expl_token_id_counts_dict[
                                            original_pred.item()][token_id2] += 1
                        if type(sparsity) is str:
                            if new_mask2[component_num] == 1 and component_num != component_count:
                                node_num2, token_num2 = flat_idx_2_nested_idx[component_num]
                                if node_num2 != -1:  # mask graph tokens
                                    temp_token_embeds2 = masked_token_embeds_list2[node_num2]
                                    temp_token_embeds2[token_num2] = torch.zeros_like(
                                        temp_token_embeds2[token_num2],
                                        device=temp_token_embeds2.device)
                                    token_id2 = token_ids_list[node_num2][token_num2]
                                    if topk_ablated_classwise_graph_expl_token_id_counts_dict[
                                        curr_topk_idx] is None:
                                        topk_ablated_classwise_graph_expl_token_id_counts_dict[curr_topk_idx] = [
                                                                                                                    None] * num_class
                                    if topk_ablated_classwise_graph_expl_token_id_counts_dict[curr_topk_idx][
                                        original_pred.item()] is None:
                                        new_dict2 = {token_id2: 1}
                                        topk_ablated_classwise_graph_expl_token_id_counts_dict[curr_topk_idx][
                                            original_pred.item()] = new_dict2
                                    elif topk_ablated_classwise_graph_expl_token_id_counts_dict[curr_topk_idx][
                                        original_pred.item()].get(token_id2, None) is None:
                                        topk_ablated_classwise_graph_expl_token_id_counts_dict[curr_topk_idx][
                                            original_pred.item()][token_id2] = 1
                                    else:
                                        topk_ablated_classwise_graph_expl_token_id_counts_dict[curr_topk_idx][
                                            original_pred.item()][token_id2] += 1
                                else:  # mask claim tokens
                                    temp_claim_token_embeds2[token_num2] = torch.zeros_like(
                                        temp_claim_token_embeds2[token_num2],
                                        device=temp_claim_token_embeds2.device)
                                    token_id2 = token_ids_list[0][token_num2]
                                    if topk_ablated_classwise_claim_expl_token_id_counts_dict[
                                        curr_topk_idx] is None:
                                        topk_ablated_classwise_claim_expl_token_id_counts_dict[curr_topk_idx] = [
                                                                                                                    None] * num_class
                                    if topk_ablated_classwise_claim_expl_token_id_counts_dict[curr_topk_idx][
                                        original_pred.item()] is None:
                                        new_dict2 = {token_id2: 1}
                                        topk_ablated_classwise_claim_expl_token_id_counts_dict[curr_topk_idx][
                                            original_pred.item()] = new_dict2
                                    elif topk_ablated_classwise_claim_expl_token_id_counts_dict[curr_topk_idx][
                                        original_pred.item()].get(token_id2, None) is None:
                                        topk_ablated_classwise_claim_expl_token_id_counts_dict[curr_topk_idx][
                                            original_pred.item()][token_id2] = 1
                                    else:
                                        topk_ablated_classwise_claim_expl_token_id_counts_dict[curr_topk_idx][
                                            original_pred.item()][token_id2] += 1
                    else:
                        masked_claim = temp_claim_token_embeds.mean(0)
                        if type(sparsity) is str:
                            masked_claim2 = temp_claim_token_embeds2.mean(0)
                    for node_num, token_embeds in enumerate(masked_token_embeds_list):
                        new_node = token_embeds.mean(0)
                        masked_x[node_num] = new_node
                    if type(sparsity) is str:
                        for node_num2, token_embeds2 in enumerate(masked_token_embeds_list2):
                            new_node2 = token_embeds2.mean(0)
                            masked_x2[node_num2] = new_node2
                    Batch_data.x = masked_x
                    if include_root_extend:
                        Batch_data.root = masked_claim
                    else:
                        Batch_data.root = masked_x[0]
                    fidelity_probs = model(Batch_data)
                    if type(fidelity_probs) is tuple:
                        fidelity_probs = fidelity_probs[0]
                    _, fidelity_pred = fidelity_probs.max(-1)
                    log_probs_change = original_probs[0, original_pred.item()] - \
                                       fidelity_probs[0, original_pred.item()]
                    try:
                        sensitivity_log_probs[sparsity_num].append(log_probs_change)
                    except:
                        sensitivity_log_probs[sparsity_num] = [log_probs_change]
                    if type(sparsity) is str:
                        Batch_data.x = masked_x2
                        if include_root_extend:
                            Batch_data.root = masked_claim2
                        else:
                            Batch_data.root = masked_x2[0]
                        fidelity_probs2 = model(Batch_data)
                        if type(fidelity_probs2) is tuple:
                            fidelity_probs2 = fidelity_probs2[0]
                        _, fidelity_pred2 = fidelity_probs2.max(-1)
                        log_probs_change2 = original_probs[0, original_pred.item()] - \
                                            fidelity_probs2[0, original_pred.item()]
                        try:
                            topk_sensitivity_log_probs[curr_topk_idx].append(log_probs_change2)
                        except:
                            topk_sensitivity_log_probs[curr_topk_idx] = [log_probs_change2]
                    # generate explanation and conduct validity test
                    expl_x = new_x.clone().detach()
                    expl_claim = None
                    expl_token_embeds_list = []
                    temp_claim_token_embeds = token_embeds_list[0].clone().detach()
                    for token_embeds in token_embeds_list:
                        expl_token_embeds_list.append(token_embeds.clone().detach())
                    for component_num, i in enumerate(new_mask):
                        if i == 0 and component_num != component_count:
                            node_num, token_num = flat_idx_2_nested_idx[component_num]
                            if node_num != -1:  # select explanation graph tokens
                                temp_token_embeds = expl_token_embeds_list[node_num]
                                temp_token_embeds[token_num] = torch.zeros_like(
                                    temp_token_embeds[token_num],
                                    device=temp_token_embeds.device)
                            else:  # select explanation claim tokens
                                temp_claim_token_embeds[token_num] = torch.zeros_like(
                                    temp_claim_token_embeds[token_num],
                                    device=temp_token_embeds.device)
                    else:
                        expl_claim = temp_claim_token_embeds.mean(0)
                    for node_num, token_embeds in enumerate(expl_token_embeds_list):
                        new_node = token_embeds.mean(0)
                        expl_x[node_num] = new_node
                    Batch_data.x = expl_x
                    if include_root_extend:
                        Batch_data.root = expl_claim
                    else:
                        Batch_data.root = expl_x[0]
                    validity_probs = model(Batch_data)
                    if type(validity_probs) is tuple:
                        validity_probs = validity_probs[0]
                    _, validity_pred = validity_probs.max(-1)
                    flush = False
                    if torch.eq(validity_pred, original_pred):
                        valid[sparsity_num] += 1
                        if has_solution_with_graph_bias:
                            valid_with_graph_bias[sparsity_num] += 1
                        if has_solution_without_graph_bias:
                            valid_without_graph_bias[sparsity_num] += 1
                    # else:
                    #     flush_count += 1
                    #     if flush_count >= 50:
                    #         flush_count = 0
                    #         flush = True
                    #     print(f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\tnot valid\n'
                    #           f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #           f'validity pred:{validity_pred.item()}', flush=flush)
                    if not torch.eq(fidelity_pred, original_pred):
                        flipped[sparsity_num] += 1
                        if has_solution_with_graph_bias:
                            flipped_with_graph_bias[sparsity_num] += 1
                        if has_solution_without_graph_bias:
                            flipped_without_graph_bias[sparsity_num] += 1
                    # else:
                    #     flush_count += 1
                    #     if flush_count >= 50:
                    #         flush_count = 0
                    #         flush = True
                    #     print(f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\t'
                    #           f'flip failed', solution.success, solution.status, solution.nit, solution.fun)
                    #     print(f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #           f'fidelity pred: {fidelity_pred.item()}', flush=flush)
                    #     if original_pred.item() == all_masked_pred.item():
                    #         original_eq_all_masked[sparsity_num] += 1
                sample_count += 1
                sample_num += 1

                continue
    fold_summary = ''
    print(f'fold {fold}\tavg sparsity (unconstrained): {np.array(unconstrained_sparsities).mean():.4f}')
    fold_summary += f'fold {fold}\tavg sparsity (unconstrained): {np.array(unconstrained_sparsities).mean():.4f}\n'
    for sparsity_num, sparsity in enumerate([0.0, 0.2, 0.4, 0.6, 0.8]):
        print(f'sparsity: {sparsity}')
        print(f'validity: {valid[sparsity_num]/sample_count:.4f} [{valid[sparsity_num]}/{sample_count}]')
        print(f'fidelity: {flipped[sparsity_num]/sample_count:.4f} [{flipped[sparsity_num]}/{sample_count}]')
        no_flip_count = sample_count - flipped[sparsity_num]
        if no_flip_count != 0:
            print(f'original == all masked: {original_eq_all_masked[sparsity_num]/no_flip_count:.4f} '
                  f'[{original_eq_all_masked[sparsity_num]}/{no_flip_count}]')
        fold_summary += f'sparsity: {sparsity}\n' \
                        f'validity: {valid[sparsity_num]/sample_count:.4f} [{valid[sparsity_num]}/{sample_count}]\n' \
                        f'fidelity: {flipped[sparsity_num]/sample_count:.4f} [{flipped[sparsity_num]}/{sample_count}]\n'
        if no_flip_count != 0:
            fold_summary += f'original == all masked: {original_eq_all_masked[sparsity_num]/no_flip_count:.4f} ' \
                            f'[{original_eq_all_masked[sparsity_num]}/{no_flip_count}]\n'
    with open(log_file_path, 'a',) as f:
        f.write(fold_summary)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasetname', type=str, default="Twitter", metavar='dataname',
                        help='dataset name, option: Twitter/PHEME/Weibo', choices=['Twitter', 'PHEME', 'Weibo'])
    parser.add_argument('-m', '--modelname', type=str, default="BiGCN", metavar='modeltype',
                        help='model type, option: BiGCN/EBGCN/CHGAT', choices=['BiGCN', 'EBGCN', 'CHGAT'])
    parser.add_argument('--input_features', type=int, default=768, metavar='inputF',
                        help='dimension of input features (BERT)')
    parser.add_argument('--hidden_features', type=int, default=64, metavar='graph_hidden',
                        help='dimension of graph hidden state')
    parser.add_argument('--output_features', type=int, default=64, metavar='output_features',
                        help='dimension of output features')
    parser.add_argument('--num_class', type=int, default=4, metavar='numclass',
                        help='number of classes')
    parser.add_argument('--num_workers', type=int, default=0, metavar='num_workers',
                        help='number of workers for training')

    # Parameters for training the model
    parser.add_argument('--seed', type=int, default=2020, help='random state seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='does not use GPU')
    parser.add_argument('--num_cuda', type=int, default=0,
                        help='index of GPU 0/1')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr_scale_bu', type=int, default=5, metavar='LRSB',
                        help='learning rate scale for bottom-up direction')
    parser.add_argument('--lr_scale_td', type=int, default=1, metavar='LRST',
                        help='learning rate scale for top-down direction')
    parser.add_argument('--l2', type=float, default=1e-4, metavar='L2',
                        help='L2 regularization weight')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--patience', type=int, default=10, metavar='patience',
                        help='patience for early stop')
    parser.add_argument('--batchsize', type=int, default=128, metavar='BS',
                        help='batch size')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='E',
                        help='number of max epochs')
    parser.add_argument('--iterations', type=int, default=1, metavar='F',
                        help='number of iterations for 5-fold cross-validation')

    # Parameters for the proposed model
    parser.add_argument('--TDdroprate', type=float, default=0, metavar='TDdroprate',
                        help='drop rate for edges in the top-down propagation graph')
    parser.add_argument('--BUdroprate', type=float, default=0, metavar='BUdroprate',
                        help='drop rate for edges in the bottom-up dispersion graph')
    parser.add_argument('--edge_infer_td', action='store_true', default=True,  # default=False,
                        help='edge inference in the top-down graph')
    parser.add_argument('--edge_infer_bu', action='store_true', default=True,  # default=True,
                        help='edge inference in the bottom-up graph')
    parser.add_argument('--edge_loss_td', type=float, default=0.2, metavar='edge_loss_td',
                        help='a hyperparameter gamma to weight the unsupervised relation learning loss in the top-down propagation graph')
    parser.add_argument('--edge_loss_bu', type=float, default=0.2, metavar='edge_loss_bu',
                        help='a hyperparameter gamma to weight the unsupervised relation learning loss in the bottom-up dispersion graph')
    parser.add_argument('--edge_num', type=int, default=2, metavar='edgenum',
                        help='latent relation types T in the edge inference')

    parser.add_argument('--exp_method', type=str, default='lrp', metavar='exp_method',
                        help='explanation method, option: ct-lrp/lrp-token/lrp/grad-cam/c-eb',
                        choices=['ct-lrp', 'lrp-token', 'lrp', 'grad-cam', 'c-eb'])

    args = parser.parse_args()

    # some admin stuff
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.num_cuda}' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    datasetname = f'New{args.datasetname}'  # 'NewTwitter', 'NewPHEME', 'NewWeibo'
    # iterations = int(sys.argv[2])
    args.datasetname = datasetname
    args.input_features = 768
    args.device = device
    args.training = True

    model = args.modelname
    model = 'EBGCN'  # 'BiGCN', 'EBGCN', 'CHGAT'
    treeDic = None  # Not required for PHEME
    min_graph_size = 20  # Exclude graphs with less that this number of nodes

    lr = args.lr  # 5e-4
    weight_decay = args.l2  # 1e-4
    patience = args.patience  # 10
    n_epochs = args.n_epochs  # 200
    batchsize = args.batchsize
    iterations = args.iterations
    hidden_size = args.hidden_features
    output_size = args.output_features
    TDdroprate = args.TDdroprate
    BUdroprate = args.BUdroprate
    # edge_dropout = 0.2  # 0.2
    # exp_method = args.exp_method
    exp_method = 'GraphLIME'  # ['CFME','GraphLIME', 'sglrp', 'ct-lrp', 'lrp']
    if exp_method == 'GraphLIME':
        model += 'v2'

    SAVE_DIR_PATH = os.path.join(EXPLAIN_DIR, datasetname, 'temp', exp_method)
    if not os.path.exists(SAVE_DIR_PATH):
        os.makedirs(SAVE_DIR_PATH)

    print(device)
    if datasetname in ['NewTwitter', 'NewWeibo', 'NewPHEME']:
        # bert_tokeniser = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        # bert_tokeniser = BertTokenizer(torch.load("tokenizer_config.json"))
        # bert_tokeniser.load_state_dict(torch.load("tokenizer.json"))
        # bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased').to(device)
        bert_tokeniser = BertTokenizer.from_pretrained('./bert-dir')
        bert_model = BertModel.from_pretrained('./bert-dir').to(device)
        # torch.save(bert_model.config, "temp-bert-config.pt")
        # torch.save(bert_model.state_dict(), "temp-bert.pt")
        bert_model.load_state_dict(torch.load("temp-bert.pt", map_location='cpu'))
        bert_model.eval()
        print(next(bert_model.parameters()).device)
    else:
        bert_tokeniser, bert_model = None, None

    for datasetname in ['NewTwitter', 'NewWeibo', 'NewPHEME']:
        args.datasetname = datasetname
        version = f'[{hidden_size},{output_size}]'
        split_type = '5fold' if datasetname.find('PHEME') != -1 else '9fold'  # '5fold', '9fold'
        log_file_path = f'{datasetname}_log.txt'
        if model == 'EBGCN':
            model0 = f'{model}-ie'
        else:
            model0 = model
        log_file_path = f'{model0}-{version}-lr{lr}-wd{weight_decay}-bs{batchsize}-p{patience}-{exp_method}_{log_file_path}'
        summary = f'{log_file_path}\n' \
                  f'{model0}:\t' \
                  f'Version: {version}\t' \
                  f'Dataset: {datasetname}\t' \
                  f'LR: {lr}\t' \
                  f'Weight Decay: {weight_decay}\n' \
                  f'Batchsize: {batchsize}\t' \
                  f'Patience: {patience}\t' \
                  f'TDdroprate: {TDdroprate}\t' \
                  f'BUdroprate: {BUdroprate}\t' \
                  f'Explanation Method: {exp_method}\n'
        start_datetime = datetime.datetime.now()
        print(start_datetime)
        print(summary)
        with open(log_file_path, 'a') as f:
            f.write(f'{start_datetime}\n')
            f.write(f'{summary}\n')
        for iter_num in range(iterations):
            torch.manual_seed(iter_num)
            np.random.seed(iter_num)
            random.seed(iter_num)
            if datasetname in ['NewTwitter', 'NewWeibo']:
                dataset_tuple = load5foldData(datasetname)
                treeDic = None  # loadTree(datasetname)
                for fold_num in range(5):
                    seed = int(f'{iter_num}{fold_num}')
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    output = test_GCN(treeDic, dataset_tuple[fold_num * 2], dataset_tuple[fold_num * 2 + 1],
                                      TDdroprate, BUdroprate, lr,
                                      weight_decay, patience, n_epochs, batchsize, datasetname, iter_num,
                                      fold=fold_num, device=device, log_file_path=log_file_path, model_type=model,
                                      split_type=split_type, hidden_size=hidden_size, output_size=output_size,
                                      ebgcn_args=args, exp_method=exp_method,
                                      tokeniser=bert_tokeniser, text_encoder=bert_model)
            elif datasetname in ['NewPHEME']:
                treeDic = None
                for fold_num, (fold_train, fold_test) in enumerate(load9foldData(datasetname, upsample=False)):
                    fold_train, fold_train_labels = fold_train
                    fold_test, fold_test_labels = fold_test
                    fold_train_labels = np.asarray(fold_train_labels)
                    # fold_test_labels = np.asarray(fold_test_labels)
                    classes = np.asarray([0, 1, 2, 3])
                    class_weight = compute_class_weight('balanced', classes=classes, y=fold_train_labels)
                    class_weight = torch.FloatTensor(class_weight)
                    seed = int(f'{iter_num}{fold_num}')
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    output = test_GCN(treeDic, fold_test, fold_train, TDdroprate, BUdroprate, lr, weight_decay,
                                      patience, n_epochs, batchsize, datasetname, iter_num, fold=fold_num,
                                      device=device, log_file_path=log_file_path, class_weight=class_weight,
                                      model_type=model, split_type=split_type, hidden_size=hidden_size,
                                      output_size=output_size, ebgcn_args=args, exp_method=exp_method,
                                      tokeniser=bert_tokeniser, text_encoder=bert_model)
        print('End of programme')
        end_datetime = datetime.datetime.now()
        print(end_datetime)
        with open(log_file_path, 'a') as f:
            f.write('End of programme')
            f.write(f'{end_datetime}\n')
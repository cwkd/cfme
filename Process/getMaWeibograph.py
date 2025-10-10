# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import transformers
from joblib import Parallel, delayed
from tqdm import tqdm

from transformers import BertTokenizer, BertModel
import torch
import json
import gc

cwd = os.getcwd()


# class Node_tweet(object):
#     def __init__(self, idx=None):
#         self.children = []
#         self.idx = idx
#         self.word = []
#         self.index = []
#         self.parent = None

def get_cls_from_text(text, tokeniser, model, pooling_mode, device):
    if type(text) is list:
        text = text[0]
    with torch.no_grad():
        encoded_texts: transformers.BatchEncoding = tokeniser(text,
                                                              padding='longest',
                                                              max_length=256,
                                                              truncation=True,
                                                              return_tensors='pt')
        tokenised_text = tokeniser.tokenize(text)
        token_ids = encoded_texts['input_ids'][0].cpu().detach().tolist()
        if pooling_mode != 'pooler':
            embeddings = model.embeddings.word_embeddings(
                encoded_texts['input_ids'].to(device)).cpu().detach().numpy()
            if pooling_mode == 'mean':
                cls = embeddings.mean(-2)
                # root_feat = cls[0]
            if pooling_mode == 'max':
                cls = embeddings.max(-2)
                # root_feat = cls[0]
        elif pooling_mode == 'pooler':
            cls = model(encoded_texts['input_ids'].to(device)).pooler_output.cpu().detach().numpy()
            # root_feat = cls[0]
    return cls[0], tokenised_text, token_ids


def constructDataMatrix(thread, tokeniser, model, device, label, max_tree_size=100, verbose=False, **kwargs):
    pooling_mode = kwargs.get('pooling_mode', 'mean')  # ['pooler', 'mean', 'max']
    # msg_ids: [str] = list(filter(lambda x: x.isnumeric(), thread_json.keys()))
    # if len(thread) > max_tree_size:
    #     thread = thread[:max_tree_size]
    msg_ids: [str] = list(map(lambda x: x['mid'], thread))
    root_msg_id: str = thread[0]['mid']
    # print(root_msg_id, len(thread))
    # print(msg_ids)
    row, col = [], []  # sparse matrix representation of adjacency matrix
    idx_counter = 1  # Counter to track current node index to be assigned
    id2index: {str: int} = {f'{root_msg_id}': 0}  # Dictionary for fast node index lookup from message ID
    root_index: int = 0  # Root tweet node index; set to 0 by default
    # texts: [str] = [thread_json[f'{root_msg_id}']['original_text']]  # First row of texts
    texts: [str] = [thread[0]['original_text']]  # First row of texts
    # texts_dict: {str: str} = {}
    # label: int = thread_json['label']
    if type(label) is str:
        label = int(label)
    no_parent_tweetids, missing_parent_tweetids = set(), set()
    # temp_graph: {str: [str]} = {k: [] for k in msg_ids}
    if len(thread) == 1:  # Skip threads with no replies
        return None
    root_feat, tokenised_text, token_ids = get_cls_from_text(texts, tokeniser, model, pooling_mode, device)  # for reference
    cls_list = [root_feat]
    tokenised_text_list = [tokenised_text]
    token_ids_list = [token_ids]
    t_list = []
    uid_list = []
    msg_id_list = []
    # Assign children to parents, iterate through list of tweetids, node index increasing by temporal order\
    for i, msg_id in enumerate(msg_ids):
        if idx_counter == max_tree_size:  # terminate loop, max graph size reached
            break
        msg_id: str
        if i == 0:  # skip root
            uid_list.append(thread[i]['uid'])
            t_list.append(thread[i]['t'])
            msg_id_list.append(msg_id)
            continue
        msg_text = thread[i]['original_text']
        cls, tokenised_text, token_ids = get_cls_from_text(msg_text, tokeniser, model, pooling_mode, device)  # get text embeds
        # Skip if all content is the same, i.e retweet
        if torch.all(torch.eq(torch.as_tensor(cls), torch.as_tensor(root_feat))):
            continue
        else:  # start adding edges
            parent_msg_id: str = thread[i]['parent']
            parent_id = id2index.get(f'{parent_msg_id}', None)
            if parent_id is None:  # do nothing, start of disjoint sub graph
                pass
            else:  # add edge
                row.append(parent_id)
                col.append(idx_counter)
            cls_list.append(cls)  # add cls to list
            tokenised_text_list.append(tokenised_text)
            token_ids_list.append(token_ids)
            uid_list.append(thread[i]['uid'])  # add user id
            t_list.append(thread[i]['t'])  # add timestamp
            msg_id_list.append(msg_id)
            idx_counter += 1
    if idx_counter == 1:
        return None
    # print(len(msg_ids), msg_ids)
    # # Assign children to parents
    # for i, msg_id in enumerate(msg_ids):
    #     msg_id: str
    #     parent_msg_id: str = thread[i]['parent']
    #     texts_dict[msg_id] = thread[i]['original_text']
    #     if parent_msg_id is not None:
    #         # check = msg_id_check.get(f'{parent_msg_id}', False)
    #         # print(parent_msg_id, msg_id, check)
    #         # if check:
    #         try:
    #             temp_graph[f'{parent_msg_id}'].append(msg_id)
    #         except:
    #             temp_graph[f'{parent_msg_id}'] = [msg_id]
    # # Add first level of reactions
    # msg_id_check: {str: bool} = {child_msg_id: True for child_msg_id in temp_graph[f'{root_msg_id}']}
    # for child_msg_id in msg_id_check.keys():
    #     texts.append(texts_dict[msg_id])
    #     row.append(root_index)
    #     col.append(idx_counter)
    #     id2index[child_msg_id] = idx_counter
    #     idx_counter += 1
    # msg_id_check[f'{root_msg_id}'] = True
    # # Progressively construct thread_json
    # for i, msg_id in enumerate(msg_ids):
    #     parent_msg_id: int = thread[i]['parent']
    #     if parent_msg_id is None:  # Skip tweets without parent
    #         if msg_id != f'{root_msg_id}':
    #             no_parent_tweetids.add(msg_id)
    #         continue
    #     if msg_id != f'{root_msg_id}':  # Check that tweet ID is not root tweet ID
    #         if msg_id_check.get(f'{parent_msg_id}', False):  # Check that tweet parent is in current thread_json
    #             for child_msg_id in temp_graph[msg_id]:
    #                 assert type(child_msg_id) is str
    #                 try:
    #                     id2index[child_msg_id]
    #                 except:
    #                     texts.append(thread[i]['original_text'])
    #                     row.append(id2index[msg_id])
    #                     col.append(idx_counter)
    #                     msg_id_check[child_msg_id] = True  # Add child tweets to current thread_json
    #                     id2index[child_msg_id] = idx_counter
    #                     idx_counter += 1
    #         else:
    #             missing_parent_tweetids.add(msg_id)
    #             # print(f'Node Error: {parent_msg_id} not in current thread_json {root_msg_id}')
    # Log for sanity checking
    if verbose:
        if len(row) != 0:
            check = False
            if max(row) < len(texts) and max(col) < len(texts):
                check = True
            print(f'Sanity check: Root ID: {root_msg_id}\tNum Tweet IDs: {len(msg_ids)}\tNum Texts: {len(texts)}\t'
                  f'Max Origin Index: {max(row)}\tMax Dest Index: {max(col)}\tMax Index < Num Texts: {check}')
            print('Parents not in thread_json: ', missing_parent_tweetids)
            print('No parent IDs: ', no_parent_tweetids)
        else:
            print(f'Sanity check: Root ID: {root_msg_id}\tNum Tweet IDs: {len(msg_ids)}\tNum Texts: {len(texts)}\t'
                  f'No Reactions')
    try:
        assert idx_counter == len(cls_list)
        assert idx_counter <= len(msg_id_list)
    except:
        pass
    processing_metadata = {'num_tweetids': len(msg_id_list),
                           'num_embeddings': len(cls_list),
                           'origin_index_max': max(row) if len(row) != 0 else None,
                           'dest_index_max': max(col) if len(col) != 0 else None,
                           'num_missing_parents': len(missing_parent_tweetids),
                           'num_no_parents': len(no_parent_tweetids),
                           'label': label}
    return cls_list, [row, col], root_index, label, uid_list, t_list, msg_ids, tokenised_text_list, token_ids_list, \
           processing_metadata
    # Batch encode texts with BERT
    # if len(thread) > 100:
    #     minibatch_size = 100
    # else:
    #     minibatch_size = None
    # try:
    #     if minibatch_size is not None:
    #         embeddings, cls = None, None
    #         if len(thread) >= max_tree_size:
    #             num_minibatches = max_tree_size // minibatch_size   # 10000 / 100 = 100
    #             print(num_minibatches)
    #         else:
    #             num_minibatches = (len(thread) // minibatch_size) + 1  # 201 / 100 = 2 => 3
    #         # quotient = len(thread) % minibatch_size  # 201 % 100 = 1
    #         for minibatch_num in range(num_minibatches):
    #             if minibatch_num != num_minibatches - 1:
    #                 temp_texts = texts[minibatch_num * minibatch_size: (minibatch_num + 1) * minibatch_size]
    #             else:
    #                 temp_texts = texts[minibatch_num * minibatch_size:]
    #             if len(temp_texts) != 0:
    #                 with torch.no_grad():
    #                     encoded_texts: transformers.BatchEncoding = tokeniser(temp_texts, padding='longest',
    #                                                                           max_length=256, truncation=True,
    #                                                                           return_tensors='pt')
    #                     # tokens = []
    #                     # for text in temp_texts:
    #                     #     tokens.append(tokeniser.tokenize(text))
    #                     # for text in encoded_texts['input_ids']
    #                     # for text in texts:
    #                     #     tokens.append(tokeniser(text,
    #                     #                             padding='max_length',
    #                     #                             max_length=256,
    #                     #                             truncation=True,
    #                     #                             return_tensors='pt'))
    #                     if pooling_mode != 'pooler':
    #                         temp_embeddings = model.embeddings.word_embeddings(
    #                             encoded_texts['input_ids'].to(device)).cpu().detach().numpy()
    #                         if pooling_mode == 'mean':
    #                             temp_cls = temp_embeddings.mean(-2)
    #                         if pooling_mode == 'max':
    #                             temp_cls = temp_embeddings.max(-2)
    #                         try:
    #                             cls = np.concatenate((cls, temp_cls), axis=0)
    #                         except:
    #                             cls = temp_cls
    #                         del encoded_texts
    #                         gc.collect()
    #                         torch.cuda.empty_cache()
    #                     # temp_embeddings = model.embeddings(encoded_texts['input_ids'].to(device)).cpu().detach().numpy()
    #                     # attention_mask = encoded_texts['attention_mask']
    #                     elif pooling_mode == 'pooler':
    #                         temp_cls = model(encoded_texts['input_ids'].to(device)).pooler_output.cpu().detach().numpy()
    #                         # root_feat = embeddings[root_index].reshape(-1, 256 * 768).cpu().detach().numpy()
    #                         # x_word = torch.cat([embeddings[:root_index], embeddings[root_index+1:]],
    #                         #                    dim=0).reshape(-1, 256*768).cpu().detach().numpy()
    #                         del encoded_texts
    #                         gc.collect()
    #                         torch.cuda.empty_cache()
    #                         if embeddings is not None:
    #                             # print(embeddings.shape, temp_embeddings.shape, cls.shape, temp_cls.shape)
    #                             # embeddings = np.concatenate((embeddings, temp_embeddings), axis=0)
    #                             cls = np.concatenate((cls, temp_cls), axis=0)
    #                         else:
    #                             # embeddings = temp_embeddings
    #                             cls = temp_cls
    #         # print(embeddings.shape, cls.shape)
    #     else:
    #         with torch.no_grad():
    #             encoded_texts: transformers.BatchEncoding = tokeniser(texts, padding='longest', max_length=256,
    #                                                                   truncation=True, return_tensors='pt')
    #             # tokens = []
    #             # for text in texts:
    #             #     tokens.append(tokeniser.tokenize(text))
    #             # for text in encoded_texts['input_ids']
    #             # for text in texts:
    #             #     tokens.append(tokeniser(text,
    #             #                             padding='max_length',
    #             #                             max_length=256,
    #             #                             truncation=True,
    #             #                             return_tensors='pt'))
    #             if pooling_mode != 'pooler':
    #                 embeddings = model.embeddings.word_embeddings(encoded_texts['input_ids'].to(device)).cpu().detach().numpy()
    #                 if pooling_mode == 'mean':
    #                     cls = embeddings.mean(-2)
    #                 if pooling_mode == 'max':
    #                     cls = embeddings.max(-2)
    #             # attention_mask = encoded_texts['attention_mask']
    #             elif pooling_mode == 'pooler':
    #                 cls = model(encoded_texts['input_ids'].to(device)).pooler_output.cpu().detach().numpy()
    #             # root_feat = embeddings[root_index].reshape(-1, 256 * 768).cpu().detach().numpy()
    #             # x_word = torch.cat([embeddings[:root_index], embeddings[root_index+1:]],
    #             #                    dim=0).reshape(-1, 256*768).cpu().detach().numpy()
    #             del encoded_texts
    #             gc.collect()
    #             torch.cuda.empty_cache()
    #     # x_word = None
    #     # x_word = embeddings.reshape(-1, 256 * 768)
    #     # root_feat = x_word[0]
    #     # root_feat = cls[0]
    #     return cls, [row, col], root_index, label, msg_ids, processing_metadata
    # except:
    #     # print(root_msg_id, msg_ids, texts)
    #     raise Exception


def saveTree(thread_json, threads_dir_path, tokeniser, model, device, processing_metadata_dict, labels_dict,
             max_tree_size):
    eid = os.path.basename(thread_json).split('.')[0]
    label = int(labels_dict[eid])
    thread_json_path = os.path.join(threads_dir_path, thread_json)
    with open(thread_json_path, 'r', encoding='utf-8', errors='replace') as thread_json_file:
        thread = json.load(thread_json_file)
    data_matrix = constructDataMatrix(thread, tokeniser, model, device, label, max_tree_size)
    if data_matrix is None:
        return None
    cls, edgeindex, root_index, label, user_ids, timestamps, tweetids, tokenised_texts, token_ids, \
        processing_metadata = data_matrix
    root_tweetid = f'{eid}'
    if label is None:
        print(f'{root_tweetid}: Label is None')
        return None
    processing_metadata_dict[root_tweetid] = processing_metadata

    edgeindex = np.array(edgeindex)
    root_index = np.array(root_index)
    label = np.array(label)
    tweetids = np.array(tweetids)
    if not os.path.exists(os.path.join(cwd, 'data', 'NewWeibograph')):
        os.makedirs(os.path.join(cwd, 'data', 'NewWeibograph'))
        print(f"Created graph directory: {os.path.join(cwd, 'data', 'NewWeibograph')}")
    # np_obj = np.load(os.path.join(cwd, 'data', 'NewWeibograph', f'{root_tweetid}.npz'))
    # saved_cls = np_obj['cls']
    # print(torch.as_tensor(saved_cls), torch.as_tensor(cls))
    # raise Exception
    try:
        if not os.path.exists(os.path.join(cwd, 'data', 'NewWeibo')):
            os.makedirs(os.path.join(cwd, 'data', 'NewWeibo'))
        with open(os.path.join(cwd, 'data', 'NewWeibo', f'{root_tweetid}.json'), 'w', encoding='utf-8') as f:
            json.dump({'texts': tokenised_texts,
                       'token_ids': token_ids}, f, ensure_ascii=False, indent=4)
        np.savez(os.path.join(cwd, 'data', 'NewWeibograph', f'{root_tweetid}.npz'),
                 cls=cls,
                 edgeindex=edgeindex,
                 rootindex=root_index,
                 y=label,
                 user_ids=user_ids,
                 timestamps=timestamps,
                 tweetids=tweetids)
        del cls, edgeindex, root_index, label, user_ids, timestamps, tweetids, processing_metadata, data_matrix
        gc.collect()
        torch.cuda.empty_cache()
    except:
        print(f'Thread {eid}: failed to process')
        # try:
        #     os.makedirs(os.path.join(cwd, 'data', 'MaWeibograph'))
        #     print(f"Created graph directory: {os.path.join(cwd, 'data', 'MaWeibograph')}")
        # except:
        #     pass
    # raise Exception
    # else:
    #     print(f'Root: {root_tweetid}\t\tTweetid: {tweetids.shape}'
    #           f'\t\tEmbeds: {x_word.shape}\t\tCLS: {cls.shape}')
    # return processing_metadata


def main():
    threads_dir_path = os.path.join(cwd, 'data', 'MaWeibo', 'Threads')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased').to(device)
    tokeniser = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased').to(device)
    # tokeniser = None
    # model = None
    print('loading labels')
    labels_path = os.path.join(cwd, 'data', 'MaWeibo', 'Weibo.txt')
    # labels_output_path = os.path.join(cwd, 'data', 'MaWeibo', 'MaWeibolabels.txt')
    labels = []
    labels_dict = {}
    with open(labels_path, 'r') as labels_file:
        lines = labels_file.readlines()
    for line in lines:
        line = line.split(' ')[0]
        line = line.split('\t')
        eid = line[0].split(':')[1]
        label = line[1].split(':')[1]
        labels.append((eid, label))
        labels_dict[eid] = label
    # print(len(labels))
    print('loading trees')
    processing_metadata_dict = {}
    max_tree_size = 100
    for thread_json in tqdm(filter(lambda x: x.find('.json') != -1,
                                   os.listdir(threads_dir_path))):
        saveTree(thread_json, threads_dir_path, tokeniser, model, device, processing_metadata_dict, labels_dict,
                 max_tree_size)
    # Parallel(n_jobs=5, backend='threading')(delayed(saveTree)(
    #     thread_json, threads_dir_path, tokeniser, model, device,
    #     processing_metadata_dict, labels_dict) for thread_json in tqdm(filter(lambda x: x.find('.json') != -1,
    #                                                                           os.listdir(threads_dir_path))))

    summary = ''
    for thread, event_tweet_list in processing_metadata_dict.items():
        event_num_trees = 0
        event_num_tweetids = 0
        event_num_embeddings = 0
        event_num_missing_parents = 0
        event_num_no_parents = 0
        labels = [0, 0]
        for _, tree_processing_metadata in event_tweet_list.items():
            event_num_trees += 1
            event_num_tweetids += tree_processing_metadata['num_tweetids']
            event_num_embeddings += tree_processing_metadata['num_embeddings']
            event_num_missing_parents += tree_processing_metadata['num_missing_parents']
            event_num_no_parents += tree_processing_metadata['num_no_parents']
            labels[tree_processing_metadata['label']] += 1
        summary += f'Event Name: {thread}\n' \
                   f'Num Trees: {event_num_trees}|\tNum Tweets: {event_num_tweetids}|\t' \
                   f'Num Embeddings: {event_num_embeddings}|\n' \
                   f'Num Tweets with Parents not in Tree: {event_num_missing_parents}|\t' \
                   f'Num Tweets which are not Roots with no Parents: {event_num_no_parents}\n' \
                   f'Label Counts: {labels}\n'
    print(summary)
    return processing_metadata_dict


def filter_ids():
    if not os.path.exists(os.path.join(cwd, 'data', 'NewWeibograph')):  # make directory if non-existent
        os.makedirs(os.path.join(cwd, 'data', 'NewWeibograph'))
    data_files = os.listdir(os.path.join(cwd, 'data', 'NewWeibograph'))
    # print(data_files)
    with open(os.path.join(cwd, 'data', 'MaWeibo', 'weibo_id_label.txt'), 'r') as f:
        lines = f.readlines()
    new_file_path = os.path.join(cwd, 'data', 'MaWeibo', 'new_weibo_id_label.txt')
    for line in lines:
        id, label = line.split(' ')
        # print(id, label)
        has_id = list(filter(lambda x: x.find(id) != -1, data_files))
        # print(has_id)
        if len(has_id) == 1:
            with open(new_file_path, 'a') as f:
                f.write(line)
        else:
            print(f'{id} not found in data files')


if __name__ == '__main__':
    main()
    # filter_ids()
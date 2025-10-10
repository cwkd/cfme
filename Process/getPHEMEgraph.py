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
import datetime

cwd = os.getcwd()


def getRawData(eventname, tree_id, verbose=False):
    event_dir_path = os.path.join(cwd, 'data', 'PHEME')
    event_json_path = os.path.join(event_dir_path, f'{eventname}.json')
    with open(event_json_path, 'r') as event_json_file:
        event = json.load(event_json_file)
    # event_tweets = list(event.keys())
    # print(tree_id)
    tree = event[tree_id]
    # for event_json in filter(lambda x: x.find('.json') != -1, os.listdir(event_dir_path)):
    #     event_json_path = os.path.join(event_dir_path, event_json)
    #     with open(event_json_path, 'r') as event_json_file:
    #         event = json.load(event_json_file)
    #     # for root_tweetid in event.keys():
    #     #     tree = event[root_tweetid]
    #     print('loading dataset')
    #     eventname = event_json.split('.')[0]
    #     event_tweets = list(event.keys())
    #     print(event_tweets)
    #     raise Exception
    tweetids: [str] = list(filter(lambda x: x.isnumeric(), tree.keys()))
    root_tweetid: int = tree['root_tweetid']
    row, col = [], []  # sparse matrix representation of adjacency matrix
    idx_counter = 1  # Counter to track current node index to be assigned
    id2index: {str: int} = {f'{root_tweetid}': 0}  # Dictionary to for fast node index lookup from tweet ID
    root_index: int = 0  # Root tweet node index; set to 0 by default
    texts: [str] = [tree[f'{root_tweetid}']['text']]  # First row of texts
    label: int = tree['label']
    no_parent_tweetids, missing_parent_tweetids = set(), set()
    temp_graph: {str: [str]} = {k: [] for k in tweetids}
    # Assign children to parents
    for tweetid in tweetids:
        tweetid: str
        parent_tweetid: int = tree[tweetid]['parent_tweetid']
        if parent_tweetid is not None:
            # check = tweetid_check.get(f'{parent_tweetid}', False)
            # print(parent_tweetid, tweetid, check)
            # if check:
            try:
                temp_graph[f'{parent_tweetid}'].append(tweetid)
            except:
                temp_graph[f'{parent_tweetid}'] = [tweetid]
    # Add first level of reactions
    tweetid_check: {str: bool} = {child_tweetid: True for child_tweetid in temp_graph[f'{root_tweetid}']}
    for child_tweetid in tweetid_check.keys():
        texts.append(tree[tweetid]['text'])
        row.append(root_index)
        col.append(idx_counter)
        id2index[child_tweetid] = idx_counter
        idx_counter += 1
    tweetid_check[f'{root_tweetid}'] = True
    # Progressively construct tree
    for tweetid in tweetids:
        parent_tweetid: int = tree[tweetid]['parent_tweetid']
        if parent_tweetid is None:  # Skip tweets without parent
            if tweetid != f'{root_tweetid}':
                no_parent_tweetids.add(tweetid)
            continue
        if tweetid != f'{root_tweetid}':  # Check that tweet ID is not root tweet ID
            if tweetid_check.get(f'{parent_tweetid}', False):  # Check that tweet parent is in current tree
                for child_tweetid in temp_graph[tweetid]:
                    assert type(child_tweetid) is str
                    try:
                        id2index[child_tweetid]
                    except:
                        texts.append(tree[child_tweetid]['text'])
                        row.append(id2index[tweetid])
                        col.append(idx_counter)
                        tweetid_check[child_tweetid] = True  # Add child tweets to current tree
                        id2index[child_tweetid] = idx_counter
                        idx_counter += 1
            else:
                missing_parent_tweetids.add(tweetid)
                # print(f'Node Error: {parent_tweetid} not in current tree {root_tweetid}')

    # Log for sanity checking
    if verbose:
        if len(row) != 0:
            check = False
            if max(row) < len(texts) and max(col) < len(texts):
                check = True
            print(f'Sanity check: Root ID: {root_tweetid}\tNum Tweet IDs: {len(tweetids)}\tNum Texts: {len(texts)}\t'
                  f'Max Origin Index: {max(row)}\tMax Dest Index: {max(col)}\tMax Index < Num Texts: {check}')
            print('Parents not in tree: ', missing_parent_tweetids)
            print('No parent IDs: ', no_parent_tweetids)
        else:
            print(f'Sanity check: Root ID: {root_tweetid}\tNum Tweet IDs: {len(tweetids)}\tNum Texts: {len(texts)}\t'
                  f'No Reactions')
    try:
        assert idx_counter == len(texts)
        assert idx_counter <= len(tweetids)
    except:
        pass
    processing_metadata = {'num_tweetids': len(tweetids),
                           'num_embeddings': len(texts),
                           'origin_index_max': max(row) if len(row) != 0 else None,
                           'dest_index_max': max(col) if len(col) != 0 else None,
                           'num_missing_parents': len(missing_parent_tweetids),
                           'num_no_parents': len(no_parent_tweetids)}
    return texts, processing_metadata


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


def constructDataMatrix(tree, pooling_mode, max_tree_size, tokeniser, model, device, verbose=False):
    tweetids: [str] = list(filter(lambda x: x.isnumeric(), tree.keys()))
    root_tweetid: int = tree['root_tweetid']
    row, col = [], []  # sparse matrix representation of adjacency matrix
    idx_counter = 1  # Counter to track current node index to be assigned
    id2index: {str: int} = {f'{root_tweetid}': 0}  # Dictionary to for fast node index lookup from tweet ID
    root_index: int = 0  # Root tweet node index; set to 0 by default
    texts: [str] = [tree[f'{root_tweetid}']['text']]  # First row of texts
    label: int = tree['label']
    no_parent_tweetids, missing_parent_tweetids = set(), set()
    # temp_graph: {str: [str]} = {k: [] for k in tweetids}
    # print(texts)
    temp, tokenised_text, token_ids = get_cls_from_text(texts, tokeniser, model, pooling_mode, device)
    root_feat = temp
    tokenised_text_list = [tokenised_text]
    token_ids_list = [token_ids]
    cls_list = [root_feat]
    user_id_list = []
    timestamp_list = []
    tweetid_list = []
    if len(tweetids) == 1:  # Skip threads with no replies
        return None
    # Assign children to parents, iterate through list of tweetids, node index increasing by temporal order\
    for tweet_num, tweetid in enumerate(tweetids):
        if idx_counter == max_tree_size:  # terminate loop, max graph size reached
            break
        tweetid: str
        if tweet_num == 0:  # First tweet in list is root node
            user_id_list.append(tree[tweetid]['userid'])  # add user id to list
            tweet_time = tree[tweetid]['tweet_time']
            time_obj = datetime.datetime.strptime(tweet_time, "%a %b %d %H:%M:%S %z %Y")
            timestamp = time_obj.timestamp()
            timestamp_list.append(timestamp)
            tweetid_list.append(tweetid)
            continue
        tweet_text = tree[tweetid]['text']
        cls, tokenised_text, token_ids = get_cls_from_text(tweet_text, tokeniser, model, pooling_mode, device)  # get text embeds
        # Skip if all content is the same, i.e retweet
        if torch.all(torch.eq(torch.as_tensor(cls), torch.as_tensor(root_feat))):
            continue
        else:
            parent_tweetid: int = tree[tweetid]['parent_tweetid']
            parent_id = id2index.get(f'{parent_tweetid}', None)
            if parent_id is None:  # do nothing, start of disjoint sub graph
                pass
            else:  # add edge
                row.append(parent_id)
                col.append(idx_counter)
            cls_list.append(cls)  # add cls to list
            tokenised_text_list.append(tokenised_text)
            token_ids_list.append(token_ids)
            user_id_list.append(tree[tweetid]['userid'])
            tweet_time = tree[tweetid]['tweet_time']
            time_obj = datetime.datetime.strptime(tweet_time, "%a %b %d %H:%M:%S %z %Y")
            timestamp = time_obj.timestamp()
            timestamp_list.append(timestamp)
            tweetid_list.append(tweetid)
            idx_counter += 1
    if idx_counter == 1:
        return None
    # for tweetid in tweetids:
    #     tweetid: str
    #     parent_tweetid: int = tree[tweetid]['parent_tweetid']
    #
    #     # print(torch.all(torch.isclose(torch.as_tensor(cls), torch.as_tensor(root_feat))),
    #     #       torch.all(torch.eq(torch.as_tensor(cls), torch.as_tensor(root_feat))))
    #     if parent_tweetid is not None:  # Node is start of sub tree
    #         # check = tweetid_check.get(f'{parent_tweetid}', False)
    #         # print(parent_tweetid, tweetid, check)
    #         # if check:
    #         try:
    #             temp_graph[f'{parent_tweetid}'].append(tweetid)
    #         except:
    #             temp_graph[f'{parent_tweetid}'] = [tweetid]
    # # Add first level of reactions
    # tweetid_check: {str: bool} = {child_tweetid: True for child_tweetid in temp_graph[f'{root_tweetid}']}
    # # check first level of tweets directly connected to root node
    # for num, child_tweetid in enumerate(tweetid_check.keys()):
    #     # check if parent is retweet
    #     tweet_text = tree[child_tweetid]['text']
    #     cls = get_cls_from_text(tweet_text, tokeniser, model, pooling_mode, device)
    #     if torch.all(torch.eq(torch.as_tensor(cls),
    #                           torch.as_tensor(root_feat))):  # Skip if all content is the same, i.e retweet
    #         continue
    #     cls_list.append(cls)
    #     row.append(root_index)
    #     col.append(idx_counter)
    #     id2index[child_tweetid] = idx_counter
    #     idx_counter += 1
    # tweetid_check[f'{root_tweetid}'] = True
    # # Progressively construct tree
    # for tweetid in tweetids:
    #     parent_tweetid: int = tree[tweetid]['parent_tweetid']
    #     if parent_tweetid is None:  # Skip tweets without parent
    #         if tweetid != f'{root_tweetid}':
    #             no_parent_tweetids.add(tweetid)
    #         continue
    #     if tweetid != f'{root_tweetid}':  # Check that tweet ID is not root tweet ID
    #         if tweetid_check.get(f'{parent_tweetid}', False):  # Check that tweet parent is in current tree
    #             for child_tweetid in temp_graph[tweetid]:
    #                 assert type(child_tweetid) is str
    #                 try:
    #                     id2index[child_tweetid]
    #                 except:
    #                     texts.append(tree[child_tweetid]['text'])
    #                     row.append(id2index[tweetid])
    #                     col.append(idx_counter)
    #                     tweetid_check[child_tweetid] = True  # Add child tweets to current tree
    #                     id2index[child_tweetid] = idx_counter
    #                     idx_counter += 1
    #         else:
    #             missing_parent_tweetids.add(tweetid)
    #             # print(f'Node Error: {parent_tweetid} not in current tree {root_tweetid}')

    # Log for sanity checking
    if verbose:
        if len(row) != 0:
            check = False
            if max(row) < len(texts) and max(col) < len(texts):
                check = True
            print(f'Sanity check: Root ID: {root_tweetid}\tNum Tweet IDs: {len(tweetids)}\tNum Texts: {len(texts)}\t'
                  f'Max Origin Index: {max(row)}\tMax Dest Index: {max(col)}\tMax Index < Num Texts: {check}')
            print('Parents not in tree: ', missing_parent_tweetids)
            print('No parent IDs: ', no_parent_tweetids)
        else:
            print(f'Sanity check: Root ID: {root_tweetid}\tNum Tweet IDs: {len(tweetids)}\tNum Texts: {len(texts)}\t'
                  f'No Reactions')
    try:
        assert idx_counter == len(cls_list)
        assert idx_counter <= len(tweetid_list)
    except:
        pass
    processing_metadata = {'num_tweetids': len(tweetid_list),
                           'num_embeddings': len(cls_list),
                           'origin_index_max': max(row) if len(row) != 0 else None,
                           'dest_index_max': max(col) if len(col) != 0 else None,
                           'num_missing_parents': len(missing_parent_tweetids),
                           'num_no_parents': len(no_parent_tweetids),
                           'label': label}
    cls_list = np.stack(cls_list)
    user_id_list = np.stack(user_id_list)
    timestamp_list = np.stack(timestamp_list)
    return cls_list, [row, col], root_feat, root_index, label, user_id_list, timestamp_list, tweetid_list, \
           tokenised_text_list, token_ids_list, processing_metadata


def saveTree(tree, tokeniser, model, device, processing_metadata_dict, event_name, event_num):
    max_tree_size = 100
    pooling_mode = 'mean'
    data_matrix = constructDataMatrix(tree, pooling_mode, max_tree_size, tokeniser, model, device)
    if data_matrix is None:  # graph not saved
        return None
    cls, edgeindex, root_feat, root_index, label, user_ids, timestamps, tweetids, tokenised_texts, token_ids, processing_metadata = data_matrix
    root_tweetid = f'{tree["root_tweetid"]}'
    if label is None:
        print(f'{root_tweetid}: Label is None')
        return None
    try:
        processing_metadata_dict[event_name][root_tweetid] = processing_metadata
    except:
        processing_metadata_dict[event_name] = {root_tweetid: processing_metadata}
    with open(os.path.join(cwd, 'data', 'PHEME', 'raw_text', f'{root_tweetid}.json'), 'w', encoding='utf-8') as f:
        json.dump({'texts': tokenised_texts,
                   'token_ids': token_ids}, f, ensure_ascii=False, indent=4)
    edgeindex = np.array(edgeindex)
    root_index = np.array(root_index)
    label = np.array(label)
    tweetids = np.array(tweetids)
    try:
        np.savez(os.path.join(cwd, 'data', 'NewPHEMEgraph', f'{root_tweetid}.npz'),
                 cls=cls,
                 root=root_feat,
                 edgeindex=edgeindex,
                 rootindex=root_index,
                 y=label,
                 user_ids=user_ids,
                 timestamps=timestamps,
                 tweetids=tweetids,
                 event_num=event_num)
        del cls, edgeindex, root_feat, root_index, label, user_ids, timestamps, tweetids, processing_metadata, \
            data_matrix, model, tokeniser
        gc.collect()
        torch.cuda.empty_cache()
    except:
        try:
            os.makedirs(os.path.join(cwd, 'data', 'NewPHEMEgraph'))
            print(f"Created graph directory: {os.path.join(cwd, 'data', 'NewPHEMEgraph')}")
        except:
            pass
    # else:
    #     print(f'Root: {root_tweetid}\t\tTweetid: {tweetids.shape}'
    #           f'\t\tEmbeds: {x_word.shape}\t\tCLS: {cls.shape}')
    # return processing_metadata


def main():
    event_dir_path = os.path.join(cwd, 'data', 'PHEME')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased').to(device)
    tokeniser = BertTokenizer.from_pretrained('./bert-dir')
    model = BertModel.from_pretrained('./bert-dir').to(device)
    # tokeniser = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    # model = BertModel.from_pretrained('bert-base-multilingual-uncased').to(device)
    # tokeniser = None
    # model = None
    print('loading trees')
    processing_metadata_dict = {}
    for event_num, event_json in enumerate(filter(lambda x: x.find('.json') != -1, os.listdir(event_dir_path))):
        event_json_path = os.path.join(event_dir_path, event_json)
        with open(event_json_path, 'r') as event_json_file:
            event = json.load(event_json_file)
        # for root_tweetid in event.keys():
        #     tree = event[root_tweetid]
        print('loading dataset')
        event_name = event_json.split('.')[0]
        event_tweets = list(event.keys())
        # for root_tweetid in tqdm(event_tweets):
        Parallel(n_jobs=1, backend='threading')(delayed(saveTree)(
            event[root_tweetid], tokeniser, model, device,
            processing_metadata_dict, event_name, event_num) for root_tweetid in tqdm(event_tweets))
    summary = ''
    for event, event_tweet_list in processing_metadata_dict.items():
        event_num_trees = 0
        event_num_tweetids = 0
        event_num_embeddings = 0
        event_num_missing_parents = 0
        event_num_no_parents = 0
        labels = [0, 0, 0, 0]
        for _, tree_processing_metadata in event_tweet_list.items():
            event_num_trees += 1
            event_num_tweetids += tree_processing_metadata['num_tweetids']
            event_num_embeddings += tree_processing_metadata['num_embeddings']
            event_num_missing_parents += tree_processing_metadata['num_missing_parents']
            event_num_no_parents += tree_processing_metadata['num_no_parents']
            labels[tree_processing_metadata['label']] += 1
        summary += f'Event Name: {event}\n' \
                   f'Num Trees: {event_num_trees}|\tNum Tweets: {event_num_tweetids}|\t' \
                   f'Num Embeddings: {event_num_embeddings}|\n' \
                   f'Num Tweets with Parents not in Tree: {event_num_missing_parents}|\t' \
                   f'Num Tweets which are not Roots with no Parents: {event_num_no_parents}\n' \
                   f'Label Counts: {labels}\n'
    print(summary)
    return processing_metadata_dict


if __name__ == '__main__':
    main()
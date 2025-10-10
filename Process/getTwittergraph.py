# -*- coding: utf-8 -*-
import os, sys
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
import transformers
from transformers import BertTokenizer, BertModel
import json, gc
cwd=os.getcwd()

class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None

def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq=float(pair.split(':')[1])
        index=int(pair.split(':')[0])
        if index<=5000:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex

def constructMat(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            rootindex=indexC-1
            root_index=nodeC.index
            root_word=nodeC.word
    rootfeat = np.zeros([1, 5000])
    if len(root_index)>0:
        rootfeat[0, np.array(root_index)] = np.array(root_word)
    matrix=np.zeros([len(index2node),len(index2node)])
    row=[]
    col=[]
    x_word=[]
    x_index=[]
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                row.append(index_i)
                col.append(index_j)
        x_word.append(index2node[index_i+1].word)
        x_index.append(index2node[index_i+1].index)
    edgematrix=[row,col]
    return x_word, x_index, edgematrix,rootfeat,rootindex

def getfeature(x_word,x_index):
    x = np.zeros([len(x_index), 5000])
    for i in range(len(x_index)):
        if len(x_index[i])>0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x

def main(obj):
    treePath = os.path.join(cwd, 'data/' + obj + '/data.TD_RvNN.vol_5000.txt')
    print("reading twitter tree")
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]

        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
    print('tree no:', len(treeDic))

    labelPath = os.path.join(cwd, "data/" + obj + "/" + obj + "_label_All.txt")
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']

    print("loading tree label")
    event, y = [], []
    l1 = l2 = l3 = l4 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        label=label.lower()
        event.append(eid)
        if label in labelset_nonR:
            labelDic[eid]=0
            l1 += 1
        if label  in labelset_f:
            labelDic[eid]=1
            l2 += 1
        if label  in labelset_t:
            labelDic[eid]=2
            l3 += 1
        if label  in labelset_u:
            labelDic[eid]=3
            l4 += 1
    print(len(labelDic))
    print(l1, l2, l3, l4)

    def loadEid(event,id,y):
        if event is None:
            return None
        if len(event) < 2:
            return None
        if len(event)>1:
            x_word, x_index, tree, rootfeat, rootindex = constructMat(event)
            x_x = getfeature(x_word, x_index)
            rootfeat, tree, x_x, rootindex, y = np.array(rootfeat), np.array(tree), np.array(x_x), np.array(
                rootindex), np.array(y)
            # print(os.path.join(cwd, 'data', f'{obj}graph', f'{id}.npz'))
            
            try:
                np.savez(os.path.join(cwd, 'data', f'{obj}graph', f'{id}.npz'), x=x_x,root=rootfeat,edgeindex=tree,rootindex=rootindex,y=y)
            except:
                try:
                    os.makedirs(os.path.join(cwd, 'data', f'{obj}graph'))
                    print(f"Created graph directory: {os.path.join(cwd, 'data', f'{obj}graph')}")
                except:
                    pass
            return None
    print("loading dataset", )
    Parallel(n_jobs=30, backend='threading')(delayed(loadEid)(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid]) for eid in tqdm(event))
    return


def filter_threads(pooling_mode, max_tree_size, tokeniser, model, device):
    datasets = ['twitter15', 'twitter16']
    tweet_tracking_dict = {}
    event_filename = 'Twitter1516_label_All.txt'
    event_file_path = os.path.join(cwd, 'data', event_filename)
    with open(event_file_path, 'r') as f:
        lines = f.readlines()
    event_name_2_event_num = {}
    event_num_2_event_name = {}
    sid_2_event_num = {}
    event_num = 0
    for line in lines:
        _, event_name, sid = line.split('\t')[:3]
        check = event_name_2_event_num.get(event_name, None)
        if check is None:
            event_name_2_event_num[event_name] = event_num
            event_num_2_event_name[event_num] = event_name
            sid_2_event_num[sid] = event_num
            event_num += 1
        else:
            sid_2_event_num[sid] = check
    event_post_counts = {}
    label_counts = [0, 0, 0, 0]
    for dataset in datasets:
        if dataset.lower() == 'twitter15':
            # data_dir_path = os.path.join(cwd, 'data', 'Twitter15', 'split_data', 'baseline')
            data_dir_path = os.path.join(cwd, 'data', 'Twitter15', 'split_data', 'structure_v2')
        elif dataset.lower() == 'twitter16':
            # data_dir_path = os.path.join(cwd, 'data', 'Twitter15', 'split_data', 'baseline')
            data_dir_path = os.path.join(cwd, 'data', 'Twitter16', 'split_data', 'structure_v2')
        else:
            data_dir_path = ''
        for split_num in range(5):
            data_files = os.listdir(os.path.join(data_dir_path, f'split_{split_num}'))
            for filename in data_files:
                if filename.find('test') != -1 and filename.find('small') != -1:
                    continue
                with open(os.path.join(data_dir_path, f'split_{split_num}', filename), 'r') as f:
                    for line in f:
                        thread = json.loads(line)
                        root_id = thread['id_']
                        event_num = sid_2_event_num[root_id]
                        check = tweet_tracking_dict.get(root_id, None)
                        if check is None:
                            data_matrix = constructDataMatrix(thread, pooling_mode, max_tree_size,
                                                              tokeniser, model, device)
                            event_num, label, tweets, tokenised_texts, token_ids = saveTree(data_matrix, event_num)
                            try:
                                event_post_counts[event_num][label] += 1
                            except:
                                event_post_counts[event_num] = [0, 0, 0, 0]
                                event_post_counts[event_num][label] += 1
                            label_counts[label] += 1
                            tweet_tracking_dict[root_id] = 0
                            raw_text_save_dir = os.path.join(cwd, 'data', 'NewTwitter')
                            with open(os.path.join(raw_text_save_dir, f'{root_id}.json'), 'w', encoding='utf-8') as f:
                                save_obj = {'tweets': tweets,
                                            'tokenised_texts': tokenised_texts,
                                            'token_ids': token_ids}
                                json.dump(save_obj, f, ensure_ascii=False, indent=4)
                        else:
                            tweet_tracking_dict[root_id] += 1
                            print(f'{root_id} already processed, '
                                  f'duplicate in {os.path.join(data_dir_path, f"split_{split_num}", filename)}'
                                  f'duplicates: {tweet_tracking_dict[root_id]}')
    for event_num, counts in event_post_counts.items():
        print(f'event{event_num}\t{event_num_2_event_name[event_num]}\t'
              f'label counts: {counts}\ttotal posts: {sum(counts)}')
    print(f'total label counts: {label_counts}')


def saveTree(data_matrix, event_num):
    cls, edgeindex, root_index, label, root_tweetid, tweets, tokenised_texts, token_ids = data_matrix
    edgeindex = np.array(edgeindex)
    root_index = np.array(root_index)
    # label = np.array(label)
    if not os.path.exists(os.path.join(cwd, 'data', 'NewTwittergraph')):
        os.makedirs(os.path.join(cwd, 'data', 'NewTwittergraph'))
        print(f"Created graph directory: {os.path.join(cwd, 'data', 'NewTwittergraph')}")
    try:
        np.savez(os.path.join(cwd, 'data', 'NewTwittergraph', f'{root_tweetid}.npz'),
                 cls=cls,
                 edgeindex=edgeindex,
                 rootindex=root_index,
                 y=label,
                 event_num=event_num)
        with open(os.path.join(cwd, 'data', 'NewTwitter_label0.txt'), 'a') as f:
            f.write(f'{root_tweetid}\t{label}\t{event_num}\n')
        print(root_tweetid, label, event_num)
        del cls, edgeindex, root_index, data_matrix
        gc.collect()
        torch.cuda.empty_cache()
    except:
        print(f'Thread {root_tweetid}: failed to process')
    return event_num, label, tweets, tokenised_texts, token_ids


def constructDataMatrix(thread, pooling_mode, max_tree_size, tokeniser, model, device):
    # print(thread.keys())
    # tweet_ids = thread['tweet_ids']
    root_id = thread['id_']
    tweets = thread['tweets']
    # time_delay = thread['time_delay']
    label = thread['label']
    # print(len(tweet_ids), len(tweets), len(time_delay))
    structure = thread['structure']
    # 0 parent, 1 child, 2 before, 3 after, 4 self
    # print(len(tweets), len(structure), len(time_delay))

    structure = np.asarray(structure)
    if structure.shape[0] > max_tree_size:
        tweets = tweets[:max_tree_size]
        structure = structure[:max_tree_size, :max_tree_size]
    if len(tweets) < structure.shape[0]:
        structure = structure[:len(tweets), :len(tweets)]
    try:
        assert len(tweets) == structure.shape[0]
        assert len(tweets) == structure.shape[1]
    except:
        print(len(tweets), structure.shape[0], structure.shape[1])
        print(tweets)
        print(structure)
        raise Exception
    src, dst = [], []
    root_index = 0
    for i in range(structure.shape[0]):
        if i == 0:
            continue
        row = structure[i]
        parent = row.argmin()
        src.append(parent)
        dst.append(i)
    with torch.no_grad():
        cls_list = []
        tokenised_text_list = []
        token_ids_list = []
        for tweet in tweets:
            encoded_texts: transformers.BatchEncoding = tokeniser(tweet, padding='longest',
                                                                  max_length=256,
                                                                  truncation=True, return_tensors='pt')

            tokenised_text = tokeniser.tokenize(tweet)
            token_ids = encoded_texts['input_ids'][0].cpu().detach().tolist()
            if pooling_mode != 'pooler':
                embeddings = model.embeddings.word_embeddings(
                    encoded_texts['input_ids'].to(device)).cpu().detach().numpy()
                if pooling_mode == 'mean':
                    cls = embeddings.mean(-2)
                if pooling_mode == 'max':
                    cls = embeddings.max(-2)
            elif pooling_mode == 'pooler':
                cls = model(encoded_texts['input_ids'].to(device)).pooler_output.cpu().detach().numpy()
            cls_list.append(cls[0])
            tokenised_text_list.append(tokenised_text)
            token_ids_list.append(token_ids)
    return cls_list, [src, dst], root_index, label, root_id, tweets, tokenised_text_list, token_ids_list


if __name__ == '__main__':
    # obj= sys.argv[1]
    # main(obj)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokeniser = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased').to(device)
    max_tree_size = 100
    pooling_mode = 'mean'
    filter_threads(pooling_mode, max_tree_size, tokeniser, model, device)
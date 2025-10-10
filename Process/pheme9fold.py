import random
from random import shuffle
import os
import json
import copy
import numpy as np
from sklearn.model_selection import KFold

cwd = os.getcwd()


def load9foldData(obj, upsample=True):
    # labelPath = os.path.join(cwd,"data/" +obj+"/"+ obj + "_label_All.txt")
    graph_data_dir_path = os.path.join(cwd, 'data', f'{obj}graph')
    graph_data_check = {tree_id.split('.')[0]: True for tree_id in os.listdir(graph_data_dir_path)}
    data_dir_path = os.path.join(cwd, 'data', 'PHEME')
    event_jsons = sorted(list(filter(lambda x: x.find('.json') != -1, os.listdir(data_dir_path))))
    # print(event_jsons)
    # labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'nonrumor'], ['false'], ['true'], ['unverified']
    print("loading tree label")
    # NR,F,T,U = [],[],[],[]
    # l1=l2=l3=l4=0
    # labelDic = {}

    train_folds, test_folds = [], []
    for fold_num, event_json in enumerate(event_jsons):
        event_jsons_copy = copy.copy(event_jsons)
        event_jsons_copy.remove(event_json)
        train_event_ids = []
        train_event_labels = []
        train_label_split = [0, 0, 0, 0]
        train_class_ids = [[], [], [], []]
        # print(event_json, event_jsons_copy)
        for current_event in event_jsons_copy:
            event_json_path = os.path.join(data_dir_path, current_event)
            with open(event_json_path, 'r') as event:
                tweets = json.load(event)
            # print(list(filter(lambda x: graph_data_check.get(x, False), tweets.keys())))
            # make sure these graphs are actually processed, otherwise, skip
            train_event_ids += list(filter(lambda x: graph_data_check.get(x, False), tweets.keys()))
            for tweetid in tweets.keys():
                train_event_labels.append(tweets[tweetid]['label'])
                train_class_ids[tweets[tweetid]['label']].append(tweetid)
                train_label_split[tweets[tweetid]['label']] += 1
        if upsample:
            largest_class = train_label_split.index(max(train_label_split))
            for label in range(4):
                if label == largest_class:
                    continue
                shortfall = train_label_split[largest_class] - train_label_split[label]
                if shortfall > train_label_split[label]:
                    k = shortfall % train_label_split[label]
                    times = shortfall // train_label_split[label]
                    train_event_ids += train_class_ids[label] * times
                    train_event_ids += random.sample(train_class_ids[label], k)
                else:
                    train_event_ids += random.sample(train_class_ids[label], shortfall)
                train_event_labels += [label] * shortfall
        train_folds.append((train_event_ids, train_event_labels))
        event_json_path = os.path.join(data_dir_path, event_json)
        test_event_labels = []
        with open(event_json_path, 'r') as event:
            tweets = json.load(event)
            # print(list(filter(lambda x: graph_data_check.get(x, False), tweets.keys())))
            test_event_ids = list(filter(lambda x: graph_data_check.get(x, False), tweets.keys()))
            for tweetid in tweets.keys():
                test_event_labels.append(tweets[tweetid]['label'])
        test_folds.append((test_event_ids, test_event_labels))
    return list(zip(train_folds, test_folds))

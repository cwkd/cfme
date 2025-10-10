import random
from random import shuffle
import os
import numpy as np
from sklearn.model_selection import KFold
random.seed(42)

cwd = os.getcwd()


def load5foldData(obj):
    if obj == 'Twitter':
        labelPath = os.path.join(cwd, "data", "Twitter1516_label_All.txt")
        labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
        print("loading tree label")
        NR, F, T, U = [], [], [], []
        l1 = l2 = l3 = l4 = 0
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            label, eid = line.split('\t')[0], line.split('\t')[2]
            if labelDic.get(eid, None) is not None:
                # print(f'Duplicate ID: {eid}')
                continue
            labelDic[eid] = label.lower()
            if label in labelset_nonR:
                NR.append(eid)
                l1 += 1
            if labelDic[eid] in labelset_f:
                F.append(eid)
                l2 += 1
            if labelDic[eid] in labelset_t:
                T.append(eid)
                l3 += 1
            if labelDic[eid] in labelset_u:
                U.append(eid)
                l4 += 1
        print(len(labelDic))
        print(l1, l2, l3, l4)
        random.Random(0).shuffle(NR)
        random.Random(0).shuffle(F)
        random.Random(0).shuffle(T)
        random.Random(0).shuffle(U)

        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
        leng3 = int(l3 * 0.2)
        leng4 = int(l4 * 0.2)

        fold0_x_test.extend(NR[0:leng1])
        fold0_x_test.extend(F[0:leng2])
        fold0_x_test.extend(T[0:leng3])
        fold0_x_test.extend(U[0:leng4])
        fold0_x_train.extend(NR[leng1:])
        fold0_x_train.extend(F[leng2:])
        fold0_x_train.extend(T[leng3:])
        fold0_x_train.extend(U[leng4:])
        fold1_x_train.extend(NR[0:leng1])
        fold1_x_train.extend(NR[leng1 * 2:])
        fold1_x_train.extend(F[0:leng2])
        fold1_x_train.extend(F[leng2 * 2:])
        fold1_x_train.extend(T[0:leng3])
        fold1_x_train.extend(T[leng3 * 2:])
        fold1_x_train.extend(U[0:leng4])
        fold1_x_train.extend(U[leng4 * 2:])
        fold1_x_test.extend(NR[leng1:leng1*2])
        fold1_x_test.extend(F[leng2:leng2*2])
        fold1_x_test.extend(T[leng3:leng3*2])
        fold1_x_test.extend(U[leng4:leng4*2])
        fold2_x_train.extend(NR[0:leng1*2])
        fold2_x_train.extend(NR[leng1*3:])
        fold2_x_train.extend(F[0:leng2*2])
        fold2_x_train.extend(F[leng2*3:])
        fold2_x_train.extend(T[0:leng3*2])
        fold2_x_train.extend(T[leng3*3:])
        fold2_x_train.extend(U[0:leng4*2])
        fold2_x_train.extend(U[leng4*3:])
        fold2_x_test.extend(NR[leng1*2:leng1*3])
        fold2_x_test.extend(F[leng2*2:leng2*3])
        fold2_x_test.extend(T[leng3*2:leng3*3])
        fold2_x_test.extend(U[leng4*2:leng4*3])
        fold3_x_train.extend(NR[0:leng1*3])
        fold3_x_train.extend(NR[leng1*4:])
        fold3_x_train.extend(F[0:leng2*3])
        fold3_x_train.extend(F[leng2*4:])
        fold3_x_train.extend(T[0:leng3*3])
        fold3_x_train.extend(T[leng3*4:])
        fold3_x_train.extend(U[0:leng4*3])
        fold3_x_train.extend(U[leng4*4:])
        fold3_x_test.extend(NR[leng1*3:leng1*4])
        fold3_x_test.extend(F[leng2*3:leng2*4])
        fold3_x_test.extend(T[leng3*3:leng3*4])
        fold3_x_test.extend(U[leng4*3:leng4*4])
        fold4_x_train.extend(NR[0:leng1*4])
        fold4_x_train.extend(NR[leng1*5:])
        fold4_x_train.extend(F[0:leng2*4])
        fold4_x_train.extend(F[leng2*5:])
        fold4_x_train.extend(T[0:leng3*4])
        fold4_x_train.extend(T[leng3*5:])
        fold4_x_train.extend(U[0:leng4*4])
        fold4_x_train.extend(U[leng4*5:])
        fold4_x_test.extend(NR[leng1*4:leng1*5])
        fold4_x_test.extend(F[leng2*4:leng2*5])
        fold4_x_test.extend(T[leng3*4:leng3*5])
        fold4_x_test.extend(U[leng4*4:leng4*5])

    if obj == "Weibo":
        labelPath = os.path.join(cwd, "data/Weibo/weibo_id_label.txt")
        print("loading weibo label:")
        F, T = [], []
        l1 = l2 = 0
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            eid, label = line.split(' ')[0], line.split(' ')[1]
            labelDic[eid] = int(label)
            if labelDic[eid]==0:
                F.append(eid)
                l1 += 1
            if labelDic[eid]==1:
                T.append(eid)
                l2 += 1
        print(len(labelDic))
        print(l1, l2)
        random.shuffle(F)
        random.shuffle(T)

        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
        fold0_x_test.extend(F[0:leng1])
        fold0_x_test.extend(T[0:leng2])
        fold0_x_train.extend(F[leng1:])
        fold0_x_train.extend(T[leng2:])
        fold1_x_train.extend(F[0:leng1])
        fold1_x_train.extend(F[leng1 * 2:])
        fold1_x_train.extend(T[0:leng2])
        fold1_x_train.extend(T[leng2 * 2:])
        fold1_x_test.extend(F[leng1:leng1 * 2])
        fold1_x_test.extend(T[leng2:leng2 * 2])
        fold2_x_train.extend(F[0:leng1 * 2])
        fold2_x_train.extend(F[leng1 * 3:])
        fold2_x_train.extend(T[0:leng2 * 2])
        fold2_x_train.extend(T[leng2 * 3:])
        fold2_x_test.extend(F[leng1 * 2:leng1 * 3])
        fold2_x_test.extend(T[leng2 * 2:leng2 * 3])
        fold3_x_train.extend(F[0:leng1 * 3])
        fold3_x_train.extend(F[leng1 * 4:])
        fold3_x_train.extend(T[0:leng2 * 3])
        fold3_x_train.extend(T[leng2 * 4:])
        fold3_x_test.extend(F[leng1 * 3:leng1 * 4])
        fold3_x_test.extend(T[leng2 * 3:leng2 * 4])
        fold4_x_train.extend(F[0:leng1 * 4])
        fold4_x_train.extend(F[leng1 * 5:])
        fold4_x_train.extend(T[0:leng2 * 4])
        fold4_x_train.extend(T[leng2 * 5:])
        fold4_x_test.extend(F[leng1 * 4:leng1 * 5])
        fold4_x_test.extend(T[leng2 * 4:leng2 * 5])

    if obj in ['NewTwitter', 'ClsTwitter']:
        labelPath = os.path.join(cwd, "data", "NewTwitter_label.txt")
        labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
        print("loading tree label")
        NR, F, T, U = [], [], [], []
        l1 = l2 = l3 = l4 = 0
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            eid, label, event = line.split('\t')
            label = int(label)
            if labelDic.get(eid, None) is not None:
                # print(f'Duplicate ID: {eid}')
                continue
            # labelDic[eid] = label.lower()
            labelDic[eid] = label
            if label == 0:
                NR.append(eid)
                l1 += 1
            if label == 1:
                F.append(eid)
                l2 += 1
            if label == 2:
                T.append(eid)
                l3 += 1
            if label == 3:
                U.append(eid)
                l4 += 1

        print(len(labelDic))
        print(l1, l2, l3, l4)
        random.Random(0).shuffle(NR)
        random.Random(0).shuffle(F)
        random.Random(0).shuffle(T)
        random.Random(0).shuffle(U)

        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
        leng3 = int(l3 * 0.2)
        leng4 = int(l4 * 0.2)

        print('fold 0 test', len(NR[0:leng1]), len(F[0:leng2]), len(T[0:leng3]), len(U[0:leng4]))
        print('fold 0 train', l1 - len(NR[0:leng1]), l2 - len(F[0:leng2]), l3 - len(T[0:leng3]), l4 - len(U[0:leng4]))

        print('fold 1 test', len(NR[leng1:leng1*2]), len(F[leng2:leng2*2]), len(T[leng3:leng3*2]), len(U[leng4:leng4*2]))
        print('fold 1 train', l1 - len(NR[leng1:leng1*2]), l2 - len(F[leng2:leng2*2]), l3 - len(T[leng3:leng3*2]), l4 - len(U[leng4:leng4*2]))

        print('fold 2 test', len(NR[leng1*2:leng1*3]), len(F[leng2*2:leng2*3]), len(T[leng3*2:leng3*3]), len(U[leng4*2:leng4*3]))
        print('fold 2 train', l1 - len(NR[leng1*2:leng1*3]), l2 - len(F[leng2*2:leng2*3]), l3 - len(T[leng3*2:leng3*3]), l4 - len(U[leng4*2:leng4*3]))

        print('fold 3 test', len(NR[leng1*3:leng1*4]), len(F[leng2*3:leng2*4]), len(T[leng3*3:leng3*4]), len(U[leng4*3:leng4*4]))
        print('fold 3 train', l1 - len(NR[leng1*3:leng1*4]), l2 - len(F[leng2*3:leng2*4]), l3 - len(T[leng3*3:leng3*4]), l4 - len(U[leng4*3:leng4*4]))

        print('fold 4 test', len(NR[leng1*4:leng1*5]), len(F[leng2*4:leng2*5]), len(T[leng3*4:leng3*5]), len(U[leng4*4:leng4*5]))
        print('fold 4 train', l1 - len(NR[leng1*4:leng1*5]), l2 - len(F[leng2*4:leng2*5]), l3 - len(T[leng3*4:leng3*5]), l4 - len(U[leng4*4:leng4*5]))
        fold0_x_test.extend(NR[0:leng1])
        fold0_x_test.extend(F[0:leng2])
        fold0_x_test.extend(T[0:leng3])
        fold0_x_test.extend(U[0:leng4])
        fold0_x_train.extend(NR[leng1:])
        fold0_x_train.extend(F[leng2:])
        fold0_x_train.extend(T[leng3:])
        fold0_x_train.extend(U[leng4:])
        fold1_x_train.extend(NR[0:leng1])
        fold1_x_train.extend(NR[leng1 * 2:])
        fold1_x_train.extend(F[0:leng2])
        fold1_x_train.extend(F[leng2 * 2:])
        fold1_x_train.extend(T[0:leng3])
        fold1_x_train.extend(T[leng3 * 2:])
        fold1_x_train.extend(U[0:leng4])
        fold1_x_train.extend(U[leng4 * 2:])
        fold1_x_test.extend(NR[leng1:leng1*2])
        fold1_x_test.extend(F[leng2:leng2*2])
        fold1_x_test.extend(T[leng3:leng3*2])
        fold1_x_test.extend(U[leng4:leng4*2])
        fold2_x_train.extend(NR[0:leng1*2])
        fold2_x_train.extend(NR[leng1*3:])
        fold2_x_train.extend(F[0:leng2*2])
        fold2_x_train.extend(F[leng2*3:])
        fold2_x_train.extend(T[0:leng3*2])
        fold2_x_train.extend(T[leng3*3:])
        fold2_x_train.extend(U[0:leng4*2])
        fold2_x_train.extend(U[leng4*3:])
        fold2_x_test.extend(NR[leng1*2:leng1*3])
        fold2_x_test.extend(F[leng2*2:leng2*3])
        fold2_x_test.extend(T[leng3*2:leng3*3])
        fold2_x_test.extend(U[leng4*2:leng4*3])
        fold3_x_train.extend(NR[0:leng1*3])
        fold3_x_train.extend(NR[leng1*4:])
        fold3_x_train.extend(F[0:leng2*3])
        fold3_x_train.extend(F[leng2*4:])
        fold3_x_train.extend(T[0:leng3*3])
        fold3_x_train.extend(T[leng3*4:])
        fold3_x_train.extend(U[0:leng4*3])
        fold3_x_train.extend(U[leng4*4:])
        fold3_x_test.extend(NR[leng1*3:leng1*4])
        fold3_x_test.extend(F[leng2*3:leng2*4])
        fold3_x_test.extend(T[leng3*3:leng3*4])
        fold3_x_test.extend(U[leng4*3:leng4*4])
        fold4_x_train.extend(NR[0:leng1*4])
        fold4_x_train.extend(NR[leng1*5:])
        fold4_x_train.extend(F[0:leng2*4])
        fold4_x_train.extend(F[leng2*5:])
        fold4_x_train.extend(T[0:leng3*4])
        fold4_x_train.extend(T[leng3*5:])
        fold4_x_train.extend(U[0:leng4*4])
        fold4_x_train.extend(U[leng4*5:])
        fold4_x_test.extend(NR[leng1*4:leng1*5])
        fold4_x_test.extend(F[leng2*4:leng2*5])
        fold4_x_test.extend(T[leng3*4:leng3*5])
        fold4_x_test.extend(U[leng4*4:leng4*5])

    if obj in ['MaWeibo', 'NewWeibo', 'ClsWeibo']:
        labelPath = os.path.join(cwd, "data/MaWeibo/new_weibo_id_label.txt")
        # labelPath = os.path.join(cwd, "data/MaWeibo/Weibo.txt")
        print("loading weibo label:")
        F, T = [], []
        l1, l2 = 0, 0
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            eid, label = line.split(' ')[0], line.split(' ')[1]
            # eid, label = line.split('\t')[:2]
            # print(eid)
            # eid = eid[4:]
            # label = label[-1]
            labelDic[eid] = int(label)
            if labelDic[eid] == 0:
                F.append(eid)
                l1 += 1
            if labelDic[eid] == 1:
                T.append(eid)
                l2 += 1
        # print(len(labelDic))
        # print(l1, l2)
        random.shuffle(F)
        random.shuffle(T)

        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
        print('fold 0 test', len(F[0:leng1]), len(T[0:leng2]))
        print('fold 0 train', l1 - len(F[0:leng1]), l2 - len(T[0:leng2]))

        print('fold 1 test', len(F[leng1:leng1 * 2]), len(T[leng2:leng2 * 2]))
        print('fold 1 train', l1 - len(F[leng1:leng1 * 2]), l2 - len(T[leng2:leng2 * 2]))

        print('fold 2 test', len(F[leng1 * 2:leng1 * 3]), len(T[leng2 * 2:leng2 * 3]))
        print('fold 2 train', l1 - len(F[leng1 * 2:leng1 * 3]), l2 - len(T[leng2 * 2:leng2 * 3]))

        print('fold 3 test', len(F[leng1 * 3:leng1 * 4]), len(T[leng2 * 3:leng2 * 4]))
        print('fold 3 train', l1 - len(F[leng1 * 3:leng1 * 4]), l2 - len(T[leng2 * 3:leng2 * 4]))

        print('fold 4 test', len(F[leng1 * 4:leng1 * 5]), len(T[leng2 * 4:leng2 * 5]))
        print('fold 4 train', l1 - len(F[leng1 * 4:leng1 * 5]), l2 - len(T[leng2 * 4:leng2 * 5]))
        fold0_x_test.extend(F[0:leng1])
        fold0_x_test.extend(T[0:leng2])
        fold0_x_train.extend(F[leng1:])
        fold0_x_train.extend(T[leng2:])
        fold1_x_train.extend(F[0:leng1])
        fold1_x_train.extend(F[leng1 * 2:])
        fold1_x_train.extend(T[0:leng2])
        fold1_x_train.extend(T[leng2 * 2:])
        fold1_x_test.extend(F[leng1:leng1 * 2])
        fold1_x_test.extend(T[leng2:leng2 * 2])
        fold2_x_train.extend(F[0:leng1 * 2])
        fold2_x_train.extend(F[leng1 * 3:])
        fold2_x_train.extend(T[0:leng2 * 2])
        fold2_x_train.extend(T[leng2 * 3:])
        fold2_x_test.extend(F[leng1 * 2:leng1 * 3])
        fold2_x_test.extend(T[leng2 * 2:leng2 * 3])
        fold3_x_train.extend(F[0:leng1 * 3])
        fold3_x_train.extend(F[leng1 * 4:])
        fold3_x_train.extend(T[0:leng2 * 3])
        fold3_x_train.extend(T[leng2 * 4:])
        fold3_x_test.extend(F[leng1 * 3:leng1 * 4])
        fold3_x_test.extend(T[leng2 * 3:leng2 * 4])
        fold4_x_train.extend(F[0:leng1 * 4])
        fold4_x_train.extend(F[leng1 * 5:])
        fold4_x_train.extend(T[0:leng2 * 4])
        fold4_x_train.extend(T[leng2 * 5:])
        fold4_x_test.extend(F[leng1 * 4:leng1 * 5])
        fold4_x_test.extend(T[leng2 * 4:leng2 * 5])
        # print(len(fold0_x_train) + len(fold0_x_test),
        #       len(fold1_x_train) + len(fold1_x_test),
        #       len(fold2_x_train) + len(fold2_x_test),
        #       len(fold3_x_train) + len(fold3_x_test),
        #       len(fold4_x_train) + len(fold4_x_test))

    fold0_test = list(fold0_x_test)
    random.Random(0).shuffle(fold0_test)
    fold0_train = list(fold0_x_train)
    random.Random(0).shuffle(fold0_train)
    fold1_test = list(fold1_x_test)
    random.Random(0).shuffle(fold1_test)
    fold1_train = list(fold1_x_train)
    random.Random(0).shuffle(fold1_train)
    fold2_test = list(fold2_x_test)
    random.Random(0).shuffle(fold2_test)
    fold2_train = list(fold2_x_train)
    random.Random(0).shuffle(fold2_train)
    fold3_test = list(fold3_x_test)
    random.Random(0).shuffle(fold3_test)
    fold3_train = list(fold3_x_train)
    random.Random(0).shuffle(fold3_train)
    fold4_test = list(fold4_x_test)
    random.Random(0).shuffle(fold4_test)
    fold4_train = list(fold4_x_train)
    random.Random(0).shuffle(fold4_train)

    return list(fold0_test),list(fold0_train),\
           list(fold1_test),list(fold1_train),\
           list(fold2_test),list(fold2_train),\
           list(fold3_test),list(fold3_train),\
           list(fold4_test), list(fold4_train)

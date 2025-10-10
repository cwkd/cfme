import os
import random
import datetime
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader

from model.Twitter.BiGCN_Twitter import BiGCN
from model.Twitter.BiGAT_Twitter import CHGAT
from model.Twitter.EBGCN import EBGCN
from Process.process import loadBiData
from Process.pheme9fold import load9foldData
from Process.rand5fold import load5foldData

from tools.earlystopping import EarlyStopping
from tools.evaluate import *

from sklearn.utils.class_weight import compute_class_weight

# from lrp_pytorch.lrp import *
from tqdm import tqdm
import argparse

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


def get_model_copy(original_model_type, input_size, hidden_size, output_size, num_class, device, ebgcn_args):
    if original_model_type == 'BiGCN':
        base_model = BiGCN(input_size, hidden_size, output_size, num_class, device).to(device)
    elif original_model_type == 'EBGCN':
        ebgcn_args.input_features = input_size
        ebgcn_args.num_class = num_class
        base_model = EBGCN(ebgcn_args).to(device)
    elif original_model_type == 'CHGAT':
        base_model = CHGAT(input_size, hidden_size, output_size, num_class, device).to(device)
    return base_model


def train_GCN(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs,
              batchsize, datasetname, iter_num, fold, device, **kwargs):
    # version = kwargs.get('version', 2)
    log_file_path = kwargs['log_file_path']
    class_weight = kwargs.get('class_weight', None)
    split_type = kwargs.get('split_type', None)
    model_type = kwargs.get('model_type', None)
    hidden_size = kwargs.get('hidden_size', None)
    output_size = kwargs.get('output_size', None)
    ebgcn_args = kwargs.get('ebgcn_args', None)
    load_and_finetune = kwargs.get('load_and_finetune', False)
    if class_weight is not None:
        class_weight = class_weight.to(device)
    # print(datasetname, kwargs)
    if datasetname.find('PHEME') != -1 or datasetname == 'NewTwitter':
        input_size = 768
        num_class = 4
    elif datasetname.find('Weibo') != -1:
        input_size = 768
        num_class = 2
    else:
        input_size = 768
        num_class = 4
    model = get_model_copy(model_type, input_size, hidden_size, output_size, num_class, device,
                                ebgcn_args)
    if model_type in ['BiGCN', 'EBGCN']:
        BU_params = list(map(id, model.BUrumorGCN.conv1.parameters()))
        BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
        base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
        optimizer = torch.optim.Adam([
            {'params': base_params},
            {'params': model.BUrumorGCN.conv1.parameters(), 'lr': lr / 5},
            {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr / 5}
        ], lr=lr, weight_decay=weight_decay)
    elif model_type == 'CHGAT':
        optimizer = torch.optim.Adam([
            {'params': model.parameters()}
        ], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    if load_and_finetune:
        baseline_checkpoints_path = os.path.join(ROOT_DIR, 'testing', f'{datasetname}')
        filenames = os.listdir(baseline_checkpoints_path)
        for criterion in [model_type, f'i{iter_num}', f'f{fold}']:
            filenames = list(filter(lambda x: x.find(f'{criterion}'), filenames))
        checkpoint_path = os.path.join(baseline_checkpoints_path, filenames[0])
        checkpoint_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_dict['model_state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        print(f'model checkpoint loaded from {checkpoint_path}')
        print(checkpoint_dict['res'])
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    version = f'[{hidden_size},{output_size}]'
    if model_type == 'EBGCN':
        if ebgcn_args.training:
            model0 = f'{model_type}-ie'
    else:
        model0 = model_type

    if split_type is not None:
        modelname = f'{model0}-{version}-lr{lr}-wd{weight_decay}-bs{batchsize}-p{patience}'
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
            # train_loader = DataLoader(traindata_list, batch_size=16, shuffle=False, num_workers=5)  # for debugging
            test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=False, num_workers=5)
            avg_loss = []
            avg_acc = []
            batch_idx = 0
            tqdm_train_loader = tqdm(train_loader)

            for Batch_data, root_ids in tqdm_train_loader:
                try:
                    Batch_data.x = Batch_data.cls
                except:
                    pass
                Batch_data.to(device)
                if model_type == 'EBGCN':
                    train_out, TD_edge_loss, BU_edge_loss = model(Batch_data)
                    finalloss = F.nll_loss(train_out, Batch_data.y, class_weight)
                    loss = finalloss
                    if TD_edge_loss is not None:
                        loss += args.edge_loss_td * TD_edge_loss
                    if BU_edge_loss is not None:
                        loss += args.edge_loss_bu * BU_edge_loss
                else:
                    train_out = model(Batch_data)
                    finalloss = F.nll_loss(train_out, Batch_data.y, class_weight)
                    loss = finalloss
                _, pred = train_out.max(dim=-1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                correct = pred.eq(Batch_data.y).sum().item()
                train_acc = correct / len(Batch_data.y)
                avg_acc.append(train_acc)
                # print("Fold {} | Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f} | "
                #       "Train_Accuracy {:.4f}".format(fold, iter_num, epoch, batch_idx, loss.item(), train_acc))
                with open(log_file_path, 'a') as f:
                    f.write("Fold {} | Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f} | "
                      "Train_Accuracy {:.4f}\n".format(fold, iter_num, epoch, batch_idx, loss.item(), train_acc))
                batch_idx = batch_idx + 1
            train_losses.append(np.mean(avg_loss))
            train_accs.append(np.mean(avg_acc))
            train_summary = f"Iter {iter_num} | Fold {fold} | Epoch {epoch:03d} | " \
                            f"Train_Loss {train_losses[-1]:.4f} | Train_Accuracy {train_accs[-1]:.4f}\n"
            with open(log_file_path, 'a') as f:
                f.write(train_summary)
            
            temp_val_losses = []
            temp_val_accs = []
            temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
            temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
            temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
            temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            model.eval()
            tqdm_test_loader = tqdm(test_loader)
            for Batch_data, tweetid in tqdm_test_loader:
                try:
                    Batch_data.x = Batch_data.cls
                except:
                    pass
                Batch_data.to(device)
                if model_type == 'EBGCN':
                    val_out, _, _ = model(Batch_data)
                else:
                    val_out = model(Batch_data)
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
            val_summary = f"Iter {iter_num} | Fold {fold} | Epoch {epoch:03d} | Val_Loss {current_loss:.4f} | " \
                          f"Val_Accuracy {val_accs[-1]:.4f}\n"
            res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
                   'C1:{:.4f},{:.12f},{:.12f},{:.12f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                           np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
                   'C2:{:.4f},{:.12f},{:.12f},{:.12f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                           np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
                   'C3:{:.4f},{:.12f},{:.12f},{:.12f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                           np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
                   'C4:{:.4f},{:.12f},{:.12f},{:.12f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                           np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
            with open(log_file_path, 'a') as f:
                f.write(f'{val_summary}'
                        f'Results: {res}\n')
            checkpoint = {
                'model': model_type,
                'input_size': input_size,
                'hidden_size': hidden_size,
                'output_size': output_size,
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
            early_stopping(train_losses[-1], current_loss, val_accs[-1], F1, F2, F3, F4, model, modelname, datasetname,
                           checkpoint=checkpoint)
            # torch.save(checkpoint, f'savepoint_{modelname}_{datasetname}_f{fold}_i{iter_num}.pt')
            if early_stopping.early_stop:
                print("Early stopping")
                accs = early_stopping.accs
                F1 = early_stopping.F1
                F2 = early_stopping.F2
                F3 = early_stopping.F3
                F4 = early_stopping.F4
                with open(log_file_path, 'a') as f:
                    f.write(
                    "BEST LOSS: {:.4f}| BEST Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}\n"
                    .format(early_stopping.best_score, accs, F1, F2, F3, F4))
                return train_losses, val_losses, train_accs, val_accs, accs, F1, F2, F3, F4
        else:
            checkpoint = {
                'model': model_type,
                'input_size': input_size,
                'hidden_size': hidden_size,
                'output_size': output_size,
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
                f'final_{modelname}_{datasetname}_f{fold}_i{iter_num}_e{epoch:03d}_l{current_loss:.5f}.pt')
            with open(log_file_path, 'a') as f:
                f.write(
                    "Last Epoch BEST LOSS: {:.4f}| BEST Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}\n"
                  .format(early_stopping.best_score, early_stopping.accs, early_stopping.F1, early_stopping.F2,
                          early_stopping.F3, early_stopping.F4))
            torch.save(checkpoint, save_path)
    except KeyboardInterrupt:
        # Added model snapshot saving
        checkpoint = {
            'model': model_type,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
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
    return train_losses, val_losses, train_accs, val_accs, early_stopping.accs, early_stopping.F1, early_stopping.F2, \
           early_stopping.F3, early_stopping.F4


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasetname', type=str, default="Twitter", metavar='dataname',
                        help='dataset name, option: Twitter/PHEME/Weibo')
    parser.add_argument('--modelname', type=str, default="BiGCN", metavar='modeltype',
                        help='model type, option: BiGCN/EBGCN/CHGAT')
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
    parser.add_argument('--TDdroprate', type=float, default=0.2, metavar='TDdroprate',
                        help='drop rate for edges in the top-down propagation graph')
    parser.add_argument('--BUdroprate', type=float, default=0.2, metavar='BUdroprate',
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

    args = parser.parse_args()

    # some admin stuff
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.num_cuda}' if torch.cuda.is_available() else 'cpu')
    # datasetname = sys.argv[1]  # 'Twitter', 'PHEME', 'Weibo'
    datasetname = f'New{args.datasetname}'  # 'NewTwitter', 'NewPHEME', 'NewWeibo'
    # iterations = int(sys.argv[2])
    args.datasetname = datasetname
    args.input_features = 768
    args.device = device
    args.training = True

    # split_type = '9fold'  # ['5fold', '9fold']
    model = args.modelname
    treeDic = None  # Not required for PHEME

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
    edge_dropout = 0.2  # 0.2

    SAVE_DIR_PATH = os.path.join(EXPLAIN_DIR, datasetname, 'temp')
    if not os.path.exists(SAVE_DIR_PATH):
        os.makedirs(SAVE_DIR_PATH)

    # if datasetname in ['NewTwitter', 'NewWeibo', 'NewPHEME']:
    #     bert_tokeniser = BertTokenizer.from_pretrained('./bert-dir')
    #     bert_model = BertModel.from_pretrained('./bert-dir').to(device)
    #     bert_model.eval()
    # else:
    #     bert_tokeniser, bert_model = None, None

    test_accs = []
    NR_F1 = []
    FR_F1 = []
    TR_F1 = []
    UR_F1 = []

    version = f'[{hidden_size},{output_size}]'
    split_type = '5fold' if datasetname != 'NewPHEME' else '9fold'  # '5fold', '9fold'
    log_file_path = f'{datasetname}_log.txt'
    if model == 'EBGCN' and args.training:
        model0 = f'{model}-ie'
    else:
        model0 = model
    log_file_path = f'{model0}-{version}-lr{lr}-wd{weight_decay}-bs{batchsize}-p{patience}_{log_file_path}'
    summary = f'{log_file_path}\n' \
              f'{model0}:\t' \
              f'Version: {version}\t' \
              f'Dataset: {datasetname}\t' \
              f'LR: {lr}\t' \
              f'Weight Decay: {weight_decay}\t' \
              f'Batchsize: {batchsize}\t' \
              f'Patience: {patience}\t' \
              f'TDdroprate: {TDdroprate}\t' \
              f'BUdroprate: {BUdroprate}'
    start_datetime = datetime.datetime.now()
    print(start_datetime)
    with open(log_file_path, 'a') as f:
        f.write(f'{start_datetime}\n')
        f.write(f'{summary}\n')
    print(summary)
    for iter_num in range(iterations):
        torch.manual_seed(iter_num)
        np.random.seed(iter_num)
        random.seed(iter_num)
        if datasetname in ['NewTwitter', 'NewWeibo']:
            dataset_tuple = load5foldData(datasetname)
            treeDic = None  # loadTree(datasetname)
            test_accs_dict, F1_dict, F2_dict, F3_dict, F4_dict = {}, {}, {}, {}, {}
            for fold_num in range(5):
                seed = int(f'{iter_num}{fold_num}')
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                output = train_GCN(treeDic, dataset_tuple[fold_num * 2], dataset_tuple[fold_num * 2 + 1],
                                   TDdroprate, BUdroprate, lr,
                                   weight_decay, patience, n_epochs, batchsize, datasetname, iter_num,
                                   fold=fold_num, device=device, log_file_path=log_file_path, model_type=model,
                                   split_type=split_type, hidden_size=hidden_size, output_size=output_size,
                                   ebgcn_args=args)
                _, _, _, _, accs, F1, F2, F3, F4 = output
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
        elif datasetname == 'NewPHEME':
            treeDic = None
            # train_losses_dict, val_losses_dict, train_accs_dict, val_accs_dict = {}, {}, {}, {}
            test_accs_dict, F1_dict, F2_dict, F3_dict, F4_dict = {}, {}, {}, {}, {}
            for fold_num, (fold_train, fold_test) in enumerate(load9foldData(datasetname, upsample=False)):
                fold_train, fold_train_labels = fold_train
                fold_test, fold_test_labels = fold_test
                fold_train_labels = np.asarray(fold_train_labels)
                fold_test_labels = np.asarray(fold_test_labels)
                classes = np.asarray([0, 1, 2, 3])
                class_weight = compute_class_weight('balanced', classes=classes, y=fold_train_labels)
                class_weight = torch.FloatTensor(class_weight)
                seed = int(f'{iter_num}{fold_num}')
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                output = train_GCN(treeDic, fold_test, fold_train, TDdroprate, BUdroprate, lr, weight_decay,
                                   patience, n_epochs, batchsize, datasetname, iter_num, fold=fold_num,
                                   device=device, log_file_path=log_file_path, class_weight=class_weight,
                                   model_type=model, split_type=split_type, hidden_size=hidden_size,
                                   output_size=output_size, ebgcn_args=args)
                _, _, _, _, accs, F1, F2, F3, F4 = output
                F1_dict[f'F1_{fold_num}'] = F1
                F2_dict[f'F2_{fold_num}'] = F2
                F3_dict[f'F3_{fold_num}'] = F3
                F4_dict[f'F4_{fold_num}'] = F4
            test_accs.append(sum([v for k, v in test_accs_dict.items()]) / 9)
            NR_F1.append(sum([v for k, v in F1_dict.items()]) / 9)
            FR_F1.append(sum([v for k, v in F2_dict.items()]) / 9)
            TR_F1.append(sum([v for k, v in F3_dict.items()]) / 9)
            UR_F1.append(sum([v for k, v in F4_dict.items()]) / 9)
        iter_summary = f'Iter {iter_num} | Total_Test_Accuracy {test_accs[-1]:.4f} | ' \
                       f'NR F1: {NR_F1[-1]:.4f} | FR F1: {FR_F1[-1]:.4f} | TR F1: {TR_F1[-1]:.4f} | ' \
                       f'UR F1: {UR_F1[-1]:.4f}\n'
        with open(log_file_path, 'a') as f:
            f.write(f'{iter_summary}\n')
    summary = "Total_Test_Accuracy: {:.4f} | NR F1: {:.4f} | FR F1: {:.4f} | TR F1: {:.4f} | UR F1: {:.4f}".format(
        sum(test_accs) / iterations, sum(NR_F1) / iterations, sum(FR_F1) / iterations, sum(TR_F1) / iterations,
        sum(UR_F1) / iterations)
    print(summary)
    with open(log_file_path, 'a') as f:
        f.write(f'{summary}\n')
    print('End of programme')
    end_datetime = datetime.datetime.now()
    print(end_datetime)
    with open(log_file_path, 'a') as f:
        f.write(f'{end_datetime}\n')

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_train_score = None
        self.early_stop = False
        self.accs = 0
        self.F1 = 0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0
        self.val_loss_min = np.Inf
        self.checkpoint = None

    def __call__(self, train_loss, val_loss, accs, F1, F2, F3, F4, model, modelname, datasetname, checkpoint):

        train_score = train_loss
        score = val_loss
        # score = -(accs + F1 + F2 + F3 + F4) / 5

        if self.best_score is None:
            self.best_train_score = train_score
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            # self.save_checkpoint(val_loss, model, modelname, datasetname)
            self.checkpoint = checkpoint
        elif score > self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST LOSS: {:.4f}| BEST Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}"
                      .format(self.best_score, self.accs, self.F1, self.F2, self.F3, self.F4))
                self.save_checkpoint(val_loss, model, modelname, datasetname)
        else:
            self.best_train_score = train_score
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            # self.save_checkpoint(val_loss, model, modelname, datasetname)
            self.checkpoint = checkpoint
            self.counter = 0

    def save_checkpoint(self, val_loss, model, modelname, datasetname):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))
        # torch.save(model.state_dict(), modelname+datasetname+'.m')
        fold = self.checkpoint['fold']
        i = self.checkpoint['iter_num']
        epoch = self.checkpoint['epoch']
        loss = self.checkpoint['loss']
        save_path = f'best_{modelname}_{datasetname}_f{fold}_i{i}_e{epoch:03d}_l{loss:.5f}.pt'
        torch.save(self.checkpoint, save_path)
        self.val_loss_min = val_loss
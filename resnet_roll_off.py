from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, EarlyStopping
from skorch.callbacks import EpochScoring
from skorch.utils import data_from_dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import norm
from dask.distributed import Client
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import pickle
import warnings
import numpy as np
import torch
from skorch.callbacks import PrintLog
from skorch import NeuralNet
from skorch.callbacks import ProgressBar
import os
import argparse
from resnet1d import resnet18, resnet34, resnet50, resnet101

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
warnings.filterwarnings("ignore")

models = {'resnet18': resnet18, 'resnet34': resnet34,
          'resnet50': resnet50, 'resnet101': resnet101}

mods = ['roff_3', 'roff_5']
class_num = len(mods)

parser = argparse.ArgumentParser(description='ResNet Roll Off Classification')
parser.add_argument('--arch', default='resnet18', metavar='ARCH',
                    help='model architecture: resnet18, resnet34, resnet50, resnet101')

parser.add_argument('--cuda', default='0', metavar='N', type=str,
                    help='cuda id')


def import_from_mat(data, size):
    features = []
    labels = []
    for mod in mods:
        real = np.array(data[mod].real[:size])
        imag = np.array(data[mod].imag[:size])
        signal = np.concatenate([real, imag], axis=1)
        features.append(signal)
        labels.append(mods.index(mod) * np.ones([size, 1]))

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


def load_data():

    print("loading data")

    data = scipy.io.loadmat(
        "D:/qpsk_roll_3_4_5_6_snr10.dat",
    )
    features, labels = import_from_mat(data, 100000)
    features = features.astype(np.float32)
    labels = labels.astype(np.int64)
    X = features
    y = labels.reshape(-1)

    # class_num = 2
    #     # X, y = make_classification(100, 2048,
    #     #                            n_informative=5,
    #     #                            n_classes=class_num,
    #     #                            random_state=0)
    #     # X = X.astype(np.float32)
    #     # y = y.astype(np.int64)

    return X, y


class SaveBestParam(Checkpoint):

    """Save best model state"""

    def save_model(self, net):
        self.best_model_dict = copy.deepcopy(
            net.module_.state_dict()
        )
        self.best_confusion_matrix = copy.deepcopy(
            net.history[-1, "confusion_matrix"]
        )


class StopRestore(EarlyStopping):

    """Early Stop and Restore best module state"""

    def __init__(
            self,
            monitor='valid_loss',
            patience=5,
            threshold=1e-4,
            threshold_mode='rel',
            lower_is_better=True,
            sink=print,
            lr_high=10,
            lr_decay=2,
    ):
        self.monitor = monitor
        self.lower_is_better = lower_is_better
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.misses_ = 0
        self.dynamic_threshold_ = None
        self.sink = sink
        self.lr_high = lr_high
        self.lr_decay = lr_decay

    def on_epoch_end(self, net, **kwargs):
        current_score = net.history[-1, self.monitor]
        if not self._is_score_improved(current_score):
            self.misses_ += 1
        else:
            self.misses_ = 0
            self.dynamic_threshold_ = self._calc_new_threshold(current_score)

        if self.misses_ == self.lr_high:
            lr = net.get_params()['lr']
            net.set_params(lr=lr/self.lr_decay)

        if self.misses_ == self.patience:
            if net.verbose:
                self._sink("Stopping since {} has not improved in the last "
                           "{} epochs.".format(self.monitor, self.patience),
                           verbose=net.verbose)

            best_cp = net.get_params()['callbacks__best']
            net.module_.load_state_dict(best_cp.best_model_dict)
            print("Best Model State Restored")
            print("Best Confusion Matrix:\n {0}".format(
                best_cp.best_confusion_matrix))

            raise KeyboardInterrupt


class ScoreConfusionMatrix(EpochScoring):
    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        EpochScoring.on_epoch_end(self, net, dataset_train, dataset_valid)

        X_test, y_test = data_from_dataset(dataset_valid)
        y_pred = net.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sample_num = np.sum(cm, axis=1)
        sample_num = np.tile(sample_num, (sample_num.size, 1))
        cm = cm * (1/np.transpose(sample_num))
        cm = np.round(cm*100)
        cm1 = cm.astype(np.int8)
        history = net.history
        history.record("confusion_matrix", cm1)


class NeuralNetClassifierProba(NeuralNetClassifier):

    def train_step_single(self, Xi, yi, **fit_params):

        self.module_.train()
        self.optimizer_.zero_grad()
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        if math.isnan(loss):
            print("Loss is nan")
            self.get_loss(y_pred, yi, X=Xi, training=True)
        loss.backward()

        self.notify(
            'on_grad_computed',
            named_parameters=list(self.module_.named_parameters()),
            X=Xi,
            y=yi
        )

        return {
            'loss': loss,
            'y_pred': y_pred,
            }

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        if not isinstance(self.criterion_, torch.nn.NLLLoss):
            raise RuntimeError('Use NLLLoss')
        return NeuralNet.get_loss(self, y_pred, y_true, *args, **kwargs)


def train():

    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    disc = models[args.arch]()
    print(disc)

    cp = SaveBestParam(dirname='best')
    early_stop = StopRestore(patience=20, lr_decay=10, lr_high=5)
    pb = ProgressBar()
    score = ScoreConfusionMatrix(scoring="accuracy", lower_is_better=False)
    pt = PrintLog(keys_ignored="confusion_matrix")
    net = NeuralNetClassifierProba(
        disc,
        max_epochs=1000,
        lr=0.01,
        device='cuda',
        callbacks=[('best', cp),
                   ('early', early_stop)],
        iterator_train__shuffle=True,
        iterator_valid__shuffle=False
    )
    net.set_params(callbacks__valid_acc=score)
    net.set_params(callbacks__print_log=pt)

    X, y = load_data()
    net.fit(X, y)
    with open('result.pkl', 'wb') as f:
        pickle.dump(net, f)

    # param_dist = {
    #     'lr': [0.05, 0.01, 0.005],
    # }
    #
    # search = RandomizedSearchCV(net,
    #                             param_dist,
    #                             cv=StratifiedKFold(n_splits=3),
    #                             n_iter=3,
    #                             verbose=10,
    #                             scoring='accuracy')
    #
    # X, y = load_data()
    #
    # Client("127.0.0.1:8786")  # create local cluster
    #
    # with joblib.parallel_backend('dask'):
    #     search.fit(X, y)
    #
    # with open('result.pkl', 'wb') as f:
    #     pickle.dump(search, f)


if __name__ == "__main__":
    train()

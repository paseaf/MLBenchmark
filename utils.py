# Utils to support call on different ML methods and accuracy indices
from readFileExample import allModels as mlm
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
import numpy as np
import scipy.stats as ss
import pandas as pd

"""Machine Learning Methods"""
# TODO: determine parameters for all models before training
mlm_dict = {
    'lda': mlm.modelLDA,
    'knn': mlm.modelKNN,
    'randomForest': mlm.modelForest,
    'svm': mlm.modelSVM,
    'mlp': mlm.modelMLP,
    'mlpe': mlm.modelMLPE,
    'ct': mlm.modelCT,
    'b': mlm.modelBoost,
    'lr': mlm.modelLR
}


def call_train_method(train_method_name: str, train_set):
    """
    Call train_method with given string name and return the trained model.

    :param train_method_name:  The Name of training method
    :param train_set: An (X, Y) Tuple
    :return: Trained model
    """
    return mlm_dict[train_method_name](*train_set)  # train_set=(x,y)


"""Accuracy Index Methods"""


def acc(test_set_y, y_predict):
    assert (test_set_y.size == y_predict.size), "test_set_y and y_predict should have the same dimensions!"
    labels_count = test_set_y.size
    correct_count = 0
    for i in range(test_set_y.size):
        if test_set_y[i] == y_predict[i]:
            correct_count += 1
    return correct_count/labels_count


def ce(test_set_y, y_predict):
    return 1 - acc(test_set_y, y_predict)


def ber(y_true, y_pred):
    """ Returns 1 - balanced accuracy"""
    return 1 - balanced_accuracy_score(y_true, y_pred, adjusted=True)  # (y_true, y_pred)

# Cramer's V
# SOURCES:
# https://github.com/shakedzy/dython/blob/6bc2a2ba3e06faed1608c496629cc56afe5c41b3/dython/nominal.py
# https://stackoverflow.com/questions/46498455/categorical-features-correlation/46498792#46498792
# https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
def cramers_v(test_set_y, y_predict):
    confusion_matrix = pd.crosstab(test_set_y, y_predict)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    if min((kcorr-1), (rcorr-1)) == 0:
        print("denominator is 0, return 0")
        return 0
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


kappa = cohen_kappa_score   # (y_true, y_pred)

acc_idx = {
    'ACC': acc,
    'BER': ber,  # from sklearn
    'CE': ce,
    'CRAMERV': cramers_v,
    'KAPPA': kappa
}


def call_acc_idx_method(acc_idx_name, y_test, y_predict):
    assert (acc_idx_name in acc_idx), f'acc_idx_name "{acc_idx_name}" is not supported'
    return acc_idx[acc_idx_name](y_test, y_predict)

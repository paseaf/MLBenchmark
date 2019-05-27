# Utils to support call on different ML methods and accuracy indices
from readFileExample import allModels as mlm

"""Machine Learning Methods"""
# TODO: determine parameters for all models before training
mlm_dict = {
    'lda': mlm.modelLDA,
    'kknn': mlm.KNeighborsClassifier,
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


def ber(test_set_y, y_predict):
    return 1 - acc(test_set_y, y_predict)

# TODO: Define ce(test_set_y, y_predict)

# TODO: Define cramerv(test_set_y, y_predict)

# TODO: Define kappa(test_set_y, y_predict)


acc_idx = {
    'ACC': acc,
    'BER': ber,
    'CE': None,
    'CRAMERV': None,
    'KAPPA': None
}


def call_acc_idx_method(acc_idx_name, y_test, y_predict):
    assert (acc_idx_name in acc_idx), f'acc_idx_name "{acc_idx_name}" is not supported'
    return acc_idx[acc_idx_name](y_test, y_predict)

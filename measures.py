import numpy as np
from sklearn.metrics import confusion_matrix


minClass = 1  # label of minority class
majClass = 0  # label of majority class


def harmonic_mean(x, y, beta=1):
    beta *= beta
    return (beta + 1) * x * y / np.array(beta * x + y)


def get_metrics(Ytest, Ytest_pred):
    """
    Compute performance measures by comparing prediction with true labels
    :param Ytest: real label
    :param Ytest_pred:  predict label
    :return:
    """
    TN, FP, FN, TP = confusion_matrix(Ytest, Ytest_pred,
                                      labels=[majClass, minClass]).ravel()
    return TN, FP, FN, TP


def MCC(Ytest, Ytest_pred):
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    mcc = np.array([TP + FN, TP + FP, FN + TN, FP + TN]).prod()
    MCC = (TP * TN - FN * FP) / np.sqrt(mcc)
    return MCC

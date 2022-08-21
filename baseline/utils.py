import numpy as np
import torch
import dgl
import random
import seaborn
import os
from sklearn.metrics import roc_curve, roc_auc_score, \
    precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def topk_filtering(d_d: np.array, k: int):
    """Convert the Topk similarities to 1 and generate the Topk interactions.
    """
    for i in range(len(d_d)):
        sorted_idx = np.argpartition(d_d[i], -k - 1)
        d_d[i, sorted_idx[-k - 1:-1]] = 1
    return np.array(np.where(d_d == 1)).T


def remove_graph(g, test_drug_id, test_disease_id):
    etype = ('drug', 'drug-disease', 'disease')

    edges_id = g.edge_ids(torch.tensor(test_drug_id),
                          torch.tensor(test_disease_id),
                          etype=etype)
    g = dgl.remove_edges(g, edges_id, etype=etype)
    etype = ('disease', 'disease-drug', 'drug')
    edges_id = g.edge_ids(torch.tensor(test_disease_id),
                          torch.tensor(test_drug_id),
                          etype=etype)
    g = dgl.remove_edges(g, edges_id, etype=etype)
    return g


def get_metrics_auc(real_score, predict_score):
    AUC = roc_auc_score(real_score, predict_score)
    AUPR = average_precision_score(real_score, predict_score)
    return AUC, AUPR


def get_metrics(real_score, predict_score):
    """Calculate the performance metrics.
    Resource code is acquired from:
    Yu Z, Huang F, Zhao X et al.
     Predicting drug-disease associations through layer attention graph convolutional network,
     Brief Bioinform 2021;22.

    Parameters
    ----------
    real_score: true labels
    predict_score: model predictions

    Return
    ---------
    AUC, AUPR, Accuracy, F1-Score, Precision, Recall, Specificity
    """
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return auc[0, 0], aupr[0, 0], accuracy, f1_score, precision, recall, specificity


def set_seed(seed=0):
    print('seed = {}'.format(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    # dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot_result_auc(args, label, predict, auc):
    """Plot the ROC curve for predictions.
    Parameters
    ----------
    args: argumentation
    label: true labels
    predict: model predictions
    auc: calculated AUROC score
    """
    seaborn.set_style()
    fpr, tpr, threshold = roc_curve(label, predict)
    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(args.saved_path, 'result_auc.png'))
    plt.clf()


def plot_result_aupr(args, label, predict, aupr):
    """Plot the ROC curve for predictions.
    Parameters
    ----------
    args: argumentation
    label: true labels
    predict: model predictions
    aupr: calculated AUPR score
    """
    seaborn.set_style()
    precision, recall, thresholds = precision_recall_curve(label, predict)
    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 8))
    plt.plot(precision, recall, color='darkorange',
             lw=lw, label='AUPR Score (area = %0.4f)' % aupr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('RPrecision/Recall Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(args.saved_path, 'result_aupr.png'))
    plt.clf()

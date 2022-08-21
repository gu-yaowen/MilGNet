import os
import dgl
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from model_zoo import HAN
from utils import remove_graph, get_metrics_auc, \
    set_seed, topk_filtering, get_metrics, \
    plot_result_auc, plot_result_aupr


def run_han(args):
    if args.model not in os.listdir(path='../result'):
        os.mkdir('../result/{}'.format(args.model))
    os.mkdir(args.saved_path)

    if args.device != 'cpu':
        print('Training on GPU')
        args.device = torch.device('cuda:{}'.format(args.device))
    else:
        print('Training on CPU')
        args.device = torch.device('cpu')

    drug_drug = pd.read_csv('../dataset/{}/drug_drug.csv'.format(args.dataset), header=None)
    drug_drug_link = topk_filtering(drug_drug.values, 15)
    disease_disease = pd.read_csv('../dataset/{}/disease_disease.csv'.format(args.dataset), header=None)
    disease_disease_link = topk_filtering(disease_disease.values, 15)
    drug_disease = pd.read_csv('../dataset/{}/drug_disease.csv'.format(args.dataset), header=None).values
    drug_disease_link = np.array(np.where(drug_disease == 1)).T
    disease_drug_link = np.array(np.where(drug_disease.T == 1)).T
    graph_data = {('drug', 'drug-drug', 'drug'): (torch.tensor(drug_drug_link[:, 0]),
                                                  torch.tensor(drug_drug_link[:, 1])),
                  ('drug', 'drug-disease', 'disease'): (torch.tensor(drug_disease_link[:, 0]),
                                                        torch.tensor(drug_disease_link[:, 1])),
                  ('disease', 'disease-drug', 'drug'): (torch.tensor(disease_drug_link[:, 0]),
                                                        torch.tensor(disease_drug_link[:, 1])),
                  ('disease', 'disease-disease', 'disease'): (torch.tensor(disease_disease_link[:, 0]),
                                                              torch.tensor(disease_disease_link[:, 1]))}
    g = dgl.heterograph(graph_data)
    drug_feature = np.hstack((drug_drug.values, np.zeros(drug_disease.shape)))
    dis_feature = np.hstack((np.zeros(drug_disease.T.shape), disease_disease.values))
    g.nodes['drug'].data['h'] = torch.from_numpy(drug_feature).to(torch.float32)
    g.nodes['disease'].data['h'] = torch.from_numpy(dis_feature).to(torch.float32)

    set_seed(args.seed)
    data = np.array([[i, j]
                     for i in range(drug_disease.shape[0])
                     for j in range(drug_disease.shape[1])])
    data = data.astype('int64')
    data = torch.tensor(data).to(args.device)
    label = torch.tensor(drug_disease.flatten()).float().to(args.device)

    kf = StratifiedKFold(args.nfold, shuffle=True, random_state=args.seed)
    fold = 1
    pred_result = np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))

    for (train_idx, val_idx) in kf.split(data.cpu().numpy(), label.cpu().numpy()):
        print('{}-Fold Cross Validation: Fold {}'.format(args.nfold, fold))
        train_data = data[train_idx]
        train_label = label[train_idx]

        val_data = data[val_idx]
        val_label = label[val_idx]
        val_drug_id = [datapoint[0].item() for datapoint in val_data]
        val_disease_id = [datapoint[1].item() for datapoint in val_data]
        dda_idx = torch.where(val_label == 1)[0].cpu().numpy()

        val_dda_drugid = np.array(val_drug_id)[dda_idx]
        val_dda_disid = np.array(val_disease_id)[dda_idx]
        g_train = g

        g_train = remove_graph(g_train, val_dda_drugid.tolist(), val_dda_disid.tolist()).to(args.device)
        feature = {'drug': g_train.nodes['drug'].data['h'],
                   'disease': g_train.nodes['disease'].data['h']}

        model = HAN(in_feats=[feature['drug'].shape[1],
                              feature['disease'].shape[1]],
                    meta_paths=[['drug-disease', 'disease-drug'],
                                ['disease-drug', 'drug-disease'],
                                ['drug-drug', 'drug-disease', 'disease-drug'],
                                ['disease-disease', 'disease-drug', 'drug-disease']])

        model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(len(torch.where(train_label == 0)[0]) /
                                                                       len(torch.where(train_label == 1)[0])))
        print('BCE loss pos weight: {:.3f}'.format(
            len(torch.where(train_label == 0)[0]) / len(torch.where(train_label == 1)[0])))

        train_idx = torch.tensor([(i.item() - 1) * drug_disease.shape[1] + j
                                  for (i, j) in train_data]).to(args.device)
        val_idx = torch.tensor([(i.item() - 1) * drug_disease.shape[1] + j
                                for (i, j) in val_data]).to(args.device)
        for epoch in range(1, args.epoch + 1):
            model.train()

            pred = model(g_train, feature).flatten().flatten()[train_idx]
            pred_score = torch.sigmoid(pred)

            optimizer.zero_grad()
            loss = criterion(pred, train_label)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0 or epoch == args.epoch - 1:
                model.eval()
                AUC, AUPR = get_metrics_auc(train_label.detach().cpu().numpy(),
                                            pred_score.detach().cpu().numpy())

                pred = model(g_train, feature).flatten()[val_idx]
                pred_score = torch.sigmoid(pred)
                pred_result[val_drug_id, val_disease_id] = pred_score.cpu().detach().numpy()
                AUC_, AUPR_ = get_metrics_auc(val_label.detach().cpu().numpy(),
                                              pred_score.detach().cpu().numpy())

                print('Epoch {} Loss: {:.3f}; Train: AUC {:.3f}, '
                      'AUPR {:.3f}; Val: AUC {:.3f}, AUPR {:.3f}'.format(epoch, loss.item(),
                                                                         AUC, AUPR, AUC_, AUPR_))

    AUC, AUPR, Acc, F1, Pre, Rec, Spec = get_metrics(label.cpu().detach().numpy(),
                                                     pred_result.flatten())
    print('Overall: AUC {:.3f}, AUPR: {:.3f}, Accuracy: {:.3f},'
          ' F1 {:.3f}, Precision {:.3f}, Recall {:.3f}, Specificity {:.3f}'.format(
        AUC, AUPR, Acc, F1, Pre, Rec, Spec))

    with open(os.path.join(args.saved_path, 'result.txt'), 'w') as f:
        for metric, score in zip(['AUC', 'AUPR', 'Acc', 'F1', 'Pre', 'Rec', 'Spec'],
                                 [AUC, AUPR, Acc, F1, Pre, Rec, Spec]):
            f.write(metric + ':' + str(score) + '\n')
    f.close()
    pd.DataFrame(pred_result).to_csv(os.path.join(args.saved_path, 'predictions.csv'),
                                     index=False, header=False)
    plot_result_auc(args, label.cpu().detach().numpy(), pred_result.flatten(), AUC)
    plot_result_aupr(args, label.cpu().detach().numpy(), pred_result.flatten(), AUPR)

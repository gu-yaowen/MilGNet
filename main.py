import os
import torch
import pandas as pd
import numpy as np
from warnings import simplefilter
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset
from load_data import load_data, load_graph, remove_graph, \
    get_data_loaders
from model import Model
from utils import get_metrics, get_metrics_auc, set_seed, \
    plot_result_auc, plot_result_aupr, checkpoint
from args import args


def val(args, model, val_loader, val_label,
        g, feature, device):
    model.eval()
    pred_val = torch.zeros(val_label.shape).to(device)
    with torch.no_grad():
        for i, data_ in enumerate(val_loader):
            x_val, y_val = data_[0].to(device), data_[1].to(device)
            pred_, attn_ = model(g, feature, x_val)
            pred_ = pred_.squeeze(dim=1)
            score_ = torch.sigmoid(pred_)
            pred_val[args.batch_size * i: args.batch_size * i + len(y_val)] = score_.detach()
    AUC_val, AUPR_val = get_metrics_auc(val_label.cpu().detach().numpy(), pred_val.cpu().detach().numpy())
    return AUC_val, AUPR_val, pred_val


def train():
    simplefilter(action='ignore', category=UserWarning)
    print('Arguments: {}'.format(args))
    set_seed(args.seed)

    if not os.path.exists(f'result/{args.dataset}'):
        os.makedirs(f'result/{args.dataset}')
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    argsDict = args.__dict__
    with open(os.path.join(args.saved_path, 'setting.txt'), 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

    if args.device_id != 'cpu':
        print('Training on GPU')
        device = torch.device('cuda:{}'.format(args.device_id))
    else:
        print('Training on CPU')
        device = torch.device('cpu')

    g, data, label = load_data(args)
    data = torch.tensor(data).to(device)
    label = torch.tensor(label).float().to(device)

    kf = StratifiedKFold(args.nfold, shuffle=True, random_state=args.seed)
    fold = 1

    pred_result = np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))

    for (train_idx, val_idx) in kf.split(data.cpu().numpy(), label.cpu().numpy()):
        print('{}-Fold Cross Validation: Fold {}'.format(args.nfold, fold))
        train_data = data[train_idx]
        train_label = label[train_idx]
        val_data = data[val_idx]
        val_label = label[val_idx]
        val_drug_id = [datapoint[0][0].item() for datapoint in val_data]
        val_disease_id = [datapoint[0][-1].item() for datapoint in val_data]

        dda_idx = torch.where(val_label == 1)[0].cpu().numpy()
        val_dda_drugid = np.array(val_drug_id)[dda_idx]
        val_dda_disid = np.array(val_disease_id)[dda_idx]
        g_train = g
        g_train = remove_graph(g_train, val_dda_drugid.tolist(), val_dda_disid.tolist()).to(device)
        feature = {'drug': g_train.nodes['drug'].data['h'],
                   'disease': g_train.nodes['disease'].data['h']}
        train_loader = get_data_loaders(TensorDataset(train_data, train_label), args.batch_size,
                                        shuffle=True, drop=True)

        val_loader = get_data_loaders(TensorDataset(val_data, val_label), args.batch_size, shuffle=False)

        model = Model(g.etypes,
                      {'drug': feature['drug'].shape[1], 'disease': feature['disease'].shape[1]},
                      hidden_feats=args.hidden_feats,
                      num_emb_layers=args.num_layer,
                      agg_type=args.aggregate_type,
                      dropout=args.dropout,
                      bn=args.batch_norm,
                      k=args.topk)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(len(torch.where(train_label == 0)[0]) /
                                                                       len(torch.where(train_label == 1)[0])))
        print('BCE loss pos weight: {:.3f}'.format(
            len(torch.where(train_label == 0)[0]) / len(torch.where(train_label == 1)[0])))

        record_list = []
        print_list = []

        for epoch in range(1, args.epoch + 1):
            total_loss = 0
            # progress = tqdm(enumerate(train_loader), desc='Loss:', total=len(train_loader))
            # train_data, train_label = negative_sampling(ratio, train_data,
            #                                             train_label, seed=epoch)
            # train_data = torch.tensor(train_data).to(device)
            # train_label = torch.tensor(train_label).to(device)
            # train_loader = get_data_loaders(TensorDataset(train_data, train_label), args.batch_size, shuffle=True)

            pred_train, label_train = torch.zeros(train_label.shape).to(device), \
                                      torch.zeros(train_label.shape).to(device)
            for i, data_ in enumerate(train_loader):
                model.train()
                x_train, y_train = data_[0].to(device), data_[1].to(device)
                pred, attn = model(g_train, feature, x_train)
                pred = pred.squeeze(dim=1)
                score = torch.sigmoid(pred)
                optimizer.zero_grad()
                loss = criterion(pred, y_train)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() / len(train_loader)
                # progress.set_description("Loss: {:.4f}".format(total_loss / (i + 1)))
                pred_train[args.batch_size * i: args.batch_size * i + len(y_train)] = score.detach()
                label_train[args.batch_size * i: args.batch_size * i + len(y_train)] = y_train.detach()

            AUC_train, AUPR_train = get_metrics_auc(label_train.cpu().detach().numpy(),
                                                    pred_train.cpu().detach().numpy())
            # if epoch % args.print_every == 0:
            AUC_val, AUPR_val, pred_val = val(args, model, val_loader, val_label, g_train, feature, device)
            if epoch % args.print_every == 0:
                print('Epoch {} Loss: {:.5f}; Train: AUC {:.3f}, AUPR {:.3f};'
                      ' Val: AUC {:.3f}, AUPR {:.3f}'.format(epoch, total_loss, AUC_train,
                                                             AUPR_train, AUC_val, AUPR_val))
                record_list.append([total_loss, AUC_train, AUPR_train, AUC_val, AUPR_val])
            print_list.append([total_loss, AUC_train, AUPR_train])
            m = checkpoint(args, model, print_list, [total_loss, AUC_train, AUPR_train], fold)
            if m:
                best_model = m
            # print('Epoch {} Loss: {:.3f}; Train: AUC {:.3f}, AUPR {:.3f}'.format(epoch, total_loss,
            #                                                                      AUC_train, AUPR_train))

        AUC_val, AUPR_val, pred_val = val(args, best_model, val_loader, val_label, g_train, feature, device)
        pred_result[val_drug_id, val_disease_id] = pred_val.cpu().detach().numpy()
        pd.DataFrame(np.array(record_list),
                     columns=['Loss', 'AUC_train', 'AUPR_train',
                              'AUC_val', 'AUPR_val']).to_csv(os.path.join(args.saved_path,
                                                                          'training_score_{}.csv'.format(fold)),
                                                             index=False)
        fold += 1
        # break

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


if __name__ == '__main__':
    train()

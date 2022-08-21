import os
import dgl
import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def load_data(args):
    """Load dataset

        Returns
        -------
        g : dgl.graph
            Heterogeneous graph representing the drug-disease network.
        feature : dict[node_types, feature_tensors]
            Initialized node features of g.
        data : np.array
            Bags of meta-path instances.
            Given a drug d_a(id:0) and a disease d_b(id:1). The meta-path instance can be:
            [[0, 0, 1, 1],
             [0, 2, 1, 1],
             [0, 891, 32, 1]
             ...]
            In the above data, the 99999 is used as a placeholder for maintaining the dimensions.
            The 0-1 columns are the drug node ids, while the 2-3 are the disease node ids.

        label : np.array
            Labels of data.
        """
    dataset = args.dataset
    k = args.k
    drug_drug = pd.read_csv('./dataset/{}/drug_drug.csv'.format(dataset), header=None)
    drug_drug_link = topk_filtering(drug_drug.values, k)
    disease_disease = pd.read_csv('./dataset/{}/disease_disease.csv'.format(dataset), header=None)
    disease_disease_link = topk_filtering(disease_disease.values, k)
    drug_disease = pd.read_csv('./dataset/{}/drug_disease.csv'.format(dataset), header=None)
    drug_disease_link = np.array(np.where(drug_disease == 1)).T
    disease_drug_link = np.array(np.where(drug_disease.T == 1)).T
    links = {'drug-drug': drug_drug_link, 'drug-disease': drug_disease_link,
             'disease-disease': disease_disease_link}
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
    if '{}_temp_{}k'.format(args.dataset, args.k) in os.listdir():
        print('Load data and label(It takes time)...')
        data = np.load('{}_temp_{}k/data.npy'.format(args.dataset, args.k))
        label = np.load('{}_temp_{}k/label.npy'.format(args.dataset, args.k))
    else:
        os.mkdir('{}_temp_{}k'.format(args.dataset, args.k))
        data, label = [], []
        print('Generating Meta-Path Instances(It takes time)...')
        with tqdm(total=drug_disease.shape[0] * drug_disease.shape[1]) as pbar:
            pbar.set_description('Drug {} * Disease {}'.format(drug_disease.shape[0],
                                                               drug_disease.shape[1]))
            for drug_id in range(drug_disease.shape[0]):
                for disease_id in range(drug_disease.shape[1]):
                    data.append(meta_path_instance(args, drug_id, disease_id, links, k))
                    label.append(int(drug_disease.iloc[drug_id, disease_id]))
                pbar.update(drug_disease.shape[1])
        print('Preparing dataset...')
        data = np.array(data)
        label = np.array(label)
        np.save('{}_temp_{}k/data.npy'.format(args.dataset, args.k), data)
        np.save('{}_temp_{}k/label.npy'.format(args.dataset, args.k), label)
    print('Data prepared !')
    return g, data, label


def topk_filtering(d_d: np.array, k: int):
    """Convert the Topk similarities to 1 and generate the Topk interactions.
    """
    for i in range(len(d_d)):
        sorted_idx = np.argpartition(d_d[i], -k - 1)
        d_d[i, sorted_idx[-k - 1:-1]] = 1
    return np.array(np.where(d_d == 1)).T


def meta_path_instance(args, drug_id: int, disease_id: int, links: dict, k: int):
    """Generate the pseudo meta-paths as instances.
    """

    mpi = [[drug_id, drug_id, disease_id, disease_id]]
    mpi.extend([[drug_id, drug, disease_id, disease_id]
                for drug in links['drug-drug'][links['drug-drug'][:, 0] == drug_id][:, 1]])
    mpi.extend([[drug_id, drug_id, dis, disease_id]
                for dis in links['disease-disease'][links['disease-disease'][:, 0] == disease_id][:, 1]])
    mpi.extend([[drug_id, drug, dis, disease_id]
                for drug in links['drug-drug'][links['drug-drug'][:, 0] == drug_id][:, 0]
                for dis in links['disease-disease'][links['disease-disease'][:, 0] == disease_id][:, 1]])
    if len(mpi) < k * (k + 2) + 1:
        for i in range(k * (k + 2) + 1 - len(mpi)):
            random.seed(args.seed)
            mpi.append(random.choice(mpi))
    elif len(mpi) > k * (k + 2) + 1:
        mpi = mpi[:k * (k + 2) + 1]
    return mpi


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


def get_data_loaders(data, batch_size, shuffle, drop=False):
    """Build data loader for train data and test data.
    """
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop)

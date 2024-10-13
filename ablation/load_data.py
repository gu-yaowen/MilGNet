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
            Bags of meta-path instances with a form of [drug_A, drug_B, disease_A, disease_B].
            Given a drug d_a(id:0) and a disease d_b(id:1). Its meta-path instances can be:
            [[0, 0, 1, 1],
             [0, 23, 1, 1],
             [0, 0, 145, 1],
             [0, 289, 36, 1],
             ...]

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
    if '{}_temp_{}k_{}'.format(args.dataset, args.k, args.mp_instance) in os.listdir():
        print('Load data and label(It takes time)...')
        data = np.load('{}_temp_{}k_{}/data.npy'.format(
            args.dataset, args.k, args.mp_instance))
        label = np.load('{}_temp_{}k_{}/label.npy'.format(
            args.dataset, args.k, args.mp_instance))
    else:
        os.mkdir('{}_temp_{}k_{}'.format(args.dataset, args.k, args.mp_instance))
        data, label = [], []
        print('Generating Meta-Path Instances(It takes time)...')
        mp_func = {'drdrdidi': meta_path_instance, 
                   'drdi': meta_path_instance_drdi,
                   'drdrdi': meta_path_instance_drdrdi, 
                   'drdidi': meta_path_instance_drdidi,
                   'drdididi': meta_path_instance_drdididi,
                   'drdidrdi': meta_path_instance_drdidrdi,
                   'drdrdrdi': meta_path_instance_drdrdrdi,
                   'drdrdrdidi': meta_path_instance_drdrdrdidi, 
                   'drdrdididi': meta_path_instance_drdrdididi}
        with tqdm(total=drug_disease.shape[0] * drug_disease.shape[1]) as pbar:
            pbar.set_description('Drug {} * Disease {}'.format(drug_disease.shape[0],
                                                               drug_disease.shape[1]))
            for drug_id in range(drug_disease.shape[0]):
                for disease_id in range(drug_disease.shape[1]):
                    data.append(mp_func[args.mp_instance](
                        args.seed, drug_id, disease_id, links, k))
                    label.append(int(drug_disease.iloc[drug_id, disease_id]))
                pbar.update(drug_disease.shape[1])
        print('Preparing dataset...')
        data = np.array(data)
        label = np.array(label)
        np.save('{}_temp_{}k_{}/data.npy'.format(
            args.dataset, args.k, args.mp_instance), data)
        np.save('{}_temp_{}k_{}/label.npy'.format(
            args.dataset, args.k, args.mp_instance), label)
    print('Data prepared !')
    return g, data, label


def load_graph(dataset: str, k: int):
    """Construct heterogeneous drug-disease graph for given dataset.

        Parameters
        ----------
        dataset : string
            The dataset to be used, including 'B-dataset', 'C-dataset' and 'F-dataset'.
        k : int
            The topk similarities to be binaried.

        Returns
        -------
        g : dgl.graph
            Heterogeneous graph representing the drug-disease network.
    """
    drug_drug = pd.read_csv('./dataset/{}/drug_drug.csv'.format(dataset), header=None)
    drug_drug_link = topk_filtering(drug_drug.values, k)
    disease_disease = pd.read_csv('./dataset/{}/disease_disease.csv'.format(dataset), header=None)
    disease_disease_link = topk_filtering(disease_disease.values, k)
    drug_disease = pd.read_csv('./dataset/{}/drug_disease.csv'.format(dataset), header=None)
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
    return g


def topk_filtering(d_d: np.array, k: int):
    """Convert the Topk similarities to 1 and generate the Topk interactions.
    """
    for i in range(len(d_d)):
        sorted_idx = np.argpartition(d_d[i], -k - 1)
        d_d[i, sorted_idx[-k - 1:-1]] = 1
    return np.array(np.where(d_d == 1)).T


def meta_path_instance(se, drug_id: int, disease_id: int, links: dict, k: int):
    """Generate the pseudo meta-path instances.
    """
    mpi = [[drug_id, drug_id, disease_id, disease_id]]
    mpi.extend([[drug_id, drug, disease_id, disease_id]
                for drug in links['drug-drug'][links['drug-drug'][:, 0] == drug_id][:, 1]])
    mpi.extend([[drug_id, drug_id, dis, disease_id]
                for dis in links['disease-disease'][links['disease-disease'][:, 0] == disease_id][:, 1]])
    mpi.extend([[drug_id, drug, dis, disease_id]
                for drug in links['drug-drug'][links['drug-drug'][:, 0] == drug_id][:, 1]
                for dis in links['disease-disease'][links['disease-disease'][:, 0] == disease_id][:, 1]])
    if len(mpi) < k * (k + 2) + 1:
        for i in range(k * (k + 2) + 1 - len(mpi)):
            random.seed(se)
            mpi.append(random.choice(mpi))
    elif len(mpi) > k * (k + 2) + 1:
        mpi = mpi[:k * (k + 2) + 1]
    return mpi

def meta_path_instance_drdididi(se, drug_id: int, disease_id: int, links: dict, k: int):
    """Generate the pseudo meta-path instances.
    """
    mpi = [[drug_id, disease_id, disease_id, disease_id]]
    mpi.extend([[drug_id, dis2, dis1, disease_id]
                for dis1 in links['disease-disease'][links['disease-disease'][:, 0] == disease_id][:, 1]
                for dis2 in links['disease-disease'][links['disease-disease'][:, 0] == dis1][:, 1]])
    if len(mpi) < k * (k + 2) + 1:
        for i in range(k * (k + 2) + 1 - len(mpi)):
            random.seed(se)
            mpi.append(random.choice(mpi))
    elif len(mpi) > k * (k + 2) + 1:
        mpi = mpi[:k * (k + 2) + 1]
    return mpi

def meta_path_instance_drdrdrdi(se, drug_id: int, disease_id: int, links: dict, k: int):
    """Generate the pseudo meta-path instances.
    """
    mpi = [[drug_id, drug_id, drug_id, disease_id]]
    mpi.extend([[drug_id, drug1, drug2, disease_id]
                for drug1 in links['drug-drug'][links['drug-drug'][:, 0] == drug_id][:, 1]
                for drug2 in links['drug-drug'][links['drug-drug'][:, 0] == drug1][:, 1]])
    if len(mpi) < k * (k + 2) + 1:
        for i in range(k * (k + 2) + 1 - len(mpi)):
            random.seed(se)
            mpi.append(random.choice(mpi))
    elif len(mpi) > k * (k + 2) + 1:
        mpi = mpi[:k * (k + 2) + 1]
    return mpi

def meta_path_instance_drdidrdi(se, drug_id: int, disease_id: int, links: dict, k: int):
    """Generate the pseudo meta-path instances.
    """
    mpi = [[drug_id, disease_id, drug_id, disease_id]]
    mpi.extend([[drug_id, dis, drug, disease_id]
                for drug in links['drug-drug'][links['drug-drug'][:, 0] == drug_id][:, 1]
                for dis in links['disease-disease'][links['disease-disease'][:, 0] == disease_id][:, 1]])
    if len(mpi) < k * (k + 2) + 1:
        for i in range(k * (k + 2) + 1 - len(mpi)):
            random.seed(se)
            mpi.append(random.choice(mpi))
    elif len(mpi) > k * (k + 2) + 1:
        mpi = mpi[:k * (k + 2) + 1]
    return mpi

def meta_path_instance_drdi(se, drug_id: int, disease_id: int, links: dict, k: int):
    mpi = [[drug_id, disease_id]]
    return mpi

def meta_path_instance_drdrdi(se, drug_id: int, disease_id: int, links: dict, k: int):
    mpi = [[drug_id, drug_id, disease_id]]
    mpi.extend([[drug_id, drug, disease_id]
                for drug in links['drug-drug'][links['drug-drug'][:, 0] == drug_id][:, 1]])
    if len(mpi) < k * (k + 2) + 1:
        for i in range(k * (k + 2) + 1 - len(mpi)):
            random.seed(se)
            mpi.append(random.choice(mpi))
    elif len(mpi) > k * (k + 2) + 1:
        mpi = mpi[:k * (k + 2) + 1]
    return mpi

def meta_path_instance_drdidi(se, drug_id: int, disease_id: int, links: dict, k: int):
    mpi = [[drug_id, disease_id, disease_id]]
    mpi.extend([[drug_id, dis, disease_id]
                for dis in links['disease-disease'][links['disease-disease'][:, 0] == disease_id][:, 1]])
    if len(mpi) < k * (k + 2) + 1:
        for i in range(k * (k + 2) + 1 - len(mpi)):
            random.seed(se)
            mpi.append(random.choice(mpi))
    elif len(mpi) > k * (k + 2) + 1:
        mpi = mpi[:k * (k + 2) + 1]
    return mpi

def meta_path_instance_drdrdididi(se, drug_id: int, disease_id: int, links: dict, k: int):
    mpi = [[drug_id, drug_id, disease_id, disease_id, disease_id]]
    mpi.extend([[drug_id, drug, disease_id, disease_id, disease_id]
                for drug in links['drug-drug'][links['drug-drug'][:, 0] == drug_id][:, 1]])
    mpi.extend([[drug_id, drug_id, dis, disease_id, disease_id]
                for dis in links['disease-disease'][links['disease-disease'][:, 0] == disease_id][:, 1]])
    mpi.extend([[drug_id, drug_id, disease_id, dis, disease_id]
                for dis in links['disease-disease'][links['disease-disease'][:, 0] == disease_id][:, 1]])
    if len(mpi) < k * (k + 2) + 1:
        for i in range(k * (k + 2) + 1 - len(mpi)):
            random.seed(se)
            mpi.append(random.choice(mpi))
    elif len(mpi) > k * (k + 2) + 1:
        mpi = mpi[:k * (k + 2) + 1]
    return mpi

def meta_path_instance_drdrdrdidi(se, drug_id: int, disease_id: int, links: dict, k: int):
    mpi = [[drug_id, drug_id, drug_id, disease_id, disease_id]]
    mpi.extend([[drug_id, drug, drug_id, disease_id, disease_id]
                for drug in links['drug-drug'][links['drug-drug'][:, 0] == drug_id][:, 1]])
    mpi.extend([[drug_id, drug_id, drug, disease_id, disease_id]
                for drug in links['drug-drug'][links['drug-drug'][:, 0] == drug_id][:, 1]])
    mpi.extend([[drug_id, drug_id, drug_id, dis, disease_id]
                for dis in links['disease-disease'][links['disease-disease'][:, 0] == disease_id][:, 1]])
    if len(mpi) < k * (k + 2) + 1:
        for i in range(k * (k + 2) + 1 - len(mpi)):
            random.seed(se)
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

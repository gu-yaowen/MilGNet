# MilGNet
![Visits Badge](https://img.shields.io/badge/dynamic/json?label=visits&query=$.message&color=blue&url=https://hits.dwyl.com/gu-yaowen/MilGNet.json)

Codes for "Meta-Path-Based Deep Multiple Instance Learning with Heterogeneous Graph Neural Network for Drug-disease Association Prediction"

# Reference
If you make advantage of the MilGNet model or its modules proposed in our paper, please cite the following in your manuscript:

```
@inproceedings{gu2022milgnet,
title={Milgnet: a multi-instance learning-based heterogeneous graph network for drug repositioning},
author={Gu, Yaowen and Zheng, Si and Zhang, Bowen and Kang, Hongyu and Li, Jiao},
booktitle={2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
pages={430--437},
year={2022},
organization={IEEE}
}
```

# Overview
![MilGNet](https://github.com/gu-yaowen/MilGNet/blob/master/model%20structure.png)
## Environment Requirement
* torch==1.8.0
* dgl==0.5.2

## k-fold Cross Validation
    python main.py -da {DATASET} -sp {SAVED PATH}
    Main arguments:
        -da: B-dataset C-dataset F-dataset R-dataset
        -ag: Aggregation method for bag embedding [sum, mean, Linear, BiTrans]
        -nl: The number of HeteroGCN layer
        -tk: The topk similarities in heterogeneous network construction
        -k : The topk filtering in instance predictor
        -hf: The dimension of hidden feature
        -ep: The number of epoches
        -bs: Batch size
        -lr: Learning rate
        -dp: Dropout rate
    For more arguments, please see args.py
Note: please see the optimal hyperparameter settings for each benchmark dataset, and other support information in 'supplementary materials.docx'.  

## Model Intepretebility
Use the ``model_intepret.ipynb`` to easily generate topk most important **meta-path instances** for given drug-disease pair (require **pre-trained model** first).

## Baselines
[DDA-SKF](https://github.com/GCQ2119216031/DDA-SKF), [SCPMF](https://github.com/luckymengmeng/SCPMF), [NIMCGCN](https://github.com/ljatynu/NIMCGCN), [DRWBNCF](https://github.com/luckymengmeng/DRWBNCF), [REDDA](https://github.com/gu-yaowen/REDDA), [PSGCN](https://github.com/bbjy/PSGCN), [HAN](https://github.com/gu-yaowen/MilGNet/blob/master/baseline/HAN_imp.py), and [MHGNN](https://github.com/gu-yaowen/MilGNet/blob/master/baseline/MHGNN).

# Contact
We welcome you to contact us (email: gu.yaowen@imicams.ac.cn) for any questions and cooperations.

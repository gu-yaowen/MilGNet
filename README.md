# MilGNet
Codes for "MilGNet: A Multi-instance Learning-based Heterogeneous Graph Network for Drug repositioning"

# Reference
If you make advantage of the MilGNet model or its modules proposed in our paper, please cite the following in your manuscript:
TBD

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
    
## Model Intepretebility
Use the ``model_intepret.ipynb`` to easily generate topk most important **meta-path instances** for given drug-disease pair (require **pre-trained model** first).

# Contact
We welcome you to contact us (email: gu.yaowen@imicams.ac.cn) for any questions and cooperations.

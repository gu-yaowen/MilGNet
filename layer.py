import torch
import dgl.nn as dglnn
from torch import nn


class HeteroLinear(nn.Module):
    """Apply linear transformations on heterogeneous inputs.
    """

    def __init__(self, in_feats, hidden_feats, dropout=0., bn=False):
        """
        Parameters
        ----------
        in_feats : dict[key, int]
            Input feature size for heterogeneous inputs. A key can be a string or a tuple of strings.
        hidden_feats : int
            Output feature size.
        dropout : int
            The dropout rate.
        bn : bool
            Use batch normalization or not.
        """
        super(HeteroLinear, self).__init__()
        self.linears = nn.ModuleDict()
        for typ, typ_in_size in in_feats.items():
            self.linears[str(typ)] = nn.Linear(typ_in_size, hidden_feats)
            nn.init.xavier_uniform_(self.linears[str(typ)].weight)
        if bn:
            self.bn = nn.BatchNorm1d(hidden_feats)
        else:
            self.bn = False
        if dropout != 0.:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = False

    def forward(self, feat):
        out_feat = dict()
        for typ, typ_feat in feat.items():
            out_feat[typ] = self.linears[str(typ)](typ_feat)
            if self.bn:
                out_feat[typ] = self.bn(out_feat[typ])
            if self.dropout:
                out_feat[typ] = self.dropout(out_feat[typ])
        return out_feat


class Node_Embedding(nn.Module):
    """HeteroGCN block.
    """

    def __init__(self, rel_names, in_feats, hidden_feats, dropout=0., bn=False):
        """
        Parameters
        ----------
        in_feats : int
            Input feature size.
        hidden_feats : int
            Output feature size.
        dropout : int
            The dropout rate.
        bn : bool
            Use batch normalization or not.
        """
        super().__init__()
        HeteroGraphdict = {}
        for rel in rel_names:
            graphconv = dglnn.GraphConv(in_feats, hidden_feats)
            nn.init.xavier_normal_(graphconv.weight)
            HeteroGraphdict[rel] = graphconv
        self.embedding = dglnn.HeteroGraphConv(HeteroGraphdict, aggregate='sum')
        self.prelu = nn.PReLU()
        if bn:
            self.bn = nn.BatchNorm1d(hidden_feats)
        else:
            self.bn = False
        if dropout != 0.:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = False

    def forward(self, graph, inputs):
        h = self.embedding(graph, inputs)
        if self.bn:
            h = {k: self.bn(v) for k, v in h.items()}
        if self.dropout:
            h = {k: self.dropout(v) for k, v in h.items()}
        h = {k: self.prelu(v) for k, v in h.items()}
        return h


class LayerAttention(nn.Module):
    """Layer attention block.
    """

    def __init__(self, in_feats, hidden_feats=128):
        """
        Parameters
        ----------
        in_feats : int
            Input feature size.
        hidden_feats : int
            Output feature size.
        """
        super(LayerAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.Tanh(),
            nn.Linear(hidden_feats, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class MetaPathAggregator(nn.Module):
    """Aggregating the meta-path instances in each bags.
    """

    def __init__(self, in_feats, hidden_feats, agg_type='sum', dropout=0., bn=False):
        """
        Parameters
        ----------
        in_feats : int
            Input feature size.
        hidden_feats : int
            Output feature size.
        agg_type : ["sum", "mean", "Linear", "BiTrans"]
            The aggregator to be used.
        dropout : int
            The dropout rate.
        bn : bool
            Use batch normalization or not.
        """
        super(MetaPathAggregator, self).__init__()
        self.agg_type = agg_type
        if agg_type == 'sum':
            self.aggregator = torch.sum
        elif agg_type == 'mean':
            self.aggregator = torch.mean
        elif agg_type == 'Linear':
            self.aggregator = nn.Linear(in_feats * 4, hidden_feats, bias=False)
            nn.init.xavier_uniform_(self.aggregator.weight)
        elif agg_type == 'BiTrans':
            self.aggregator_drug_disease = nn.Linear(in_feats, hidden_feats, bias=False)
            self.aggregator_disease_drug = nn.Linear(in_feats, hidden_feats, bias=False)
            self.aggregator_drug = nn.Linear(in_feats, int(hidden_feats / 2), bias=False)
            self.aggregator_dis = nn.Linear(in_feats, int(hidden_feats / 2), bias=False)
            nn.init.xavier_uniform_(self.aggregator_drug_disease.weight)
            nn.init.xavier_uniform_(self.aggregator_disease_drug.weight)
            nn.init.xavier_uniform_(self.aggregator_drug.weight)
            nn.init.xavier_uniform_(self.aggregator_dis.weight)

    def forward(self, feature, mp_ins):
        mp_ins_drug = mp_ins[:, :, :2]
        mp_ins_dis = mp_ins[:, :, 2:]
        mp_ins_feat = torch.cat([feature['drug'][mp_ins_drug],
                                 feature['disease'][mp_ins_dis]], dim=2)
        if self.agg_type in ['sum', 'mean']:
            ins_emb = self.aggregator(mp_ins_feat, dim=2)
        elif self.agg_type == 'Linear':
            ins_emb = self.aggregator(mp_ins_feat.reshape(mp_ins_feat.shape[0], mp_ins_feat.shape[1],
                                                          mp_ins_feat.shape[2] * mp_ins_feat.shape[3]))
        else:
            hd_feat = mp_ins_feat.shape[3]
            mp_ins_feat = mp_ins_feat.reshape(mp_ins_feat.shape[0], mp_ins_feat.shape[1],
                                              mp_ins_feat.shape[2] * mp_ins_feat.shape[3])
            dis_feat = (((self.aggregator_drug_disease((mp_ins_feat[:, :, :hd_feat] +
                                                        mp_ins_feat[:, :, hd_feat:hd_feat * 2]) / 2)
                          + mp_ins_feat[:, :, hd_feat * 2:hd_feat * 3]) / 2)
                        + mp_ins_feat[:, :, hd_feat * 3:]) / 2
            drug_feat = (((self.aggregator_disease_drug((mp_ins_feat[:, :, hd_feat * 3:]
                                                         + mp_ins_feat[:, :, hd_feat * 2:hd_feat * 3]) / 2)
                           + mp_ins_feat[:, :, hd_feat:hd_feat * 2]) / 2)
                         + mp_ins_feat[:, :, :hd_feat]) / 2
            ins_emb = torch.cat((self.aggregator_drug(drug_feat),
                                 self.aggregator_dis(dis_feat)), dim=2)
        return ins_emb


class MILNet(nn.Module):
    """Attention based instance aggregation block for bag embedding.
    """

    def __init__(self, in_feats, hidden_feats):
        """
        Parameters
        ----------
        in_feats : int
            Input feature size.
        hidden_feats : int
            Output feature size.
        """
        super(MILNet, self).__init__()

        self.project = nn.Sequential(nn.Linear(in_feats, hidden_feats),
                                     nn.Tanh(),
                                     nn.Linear(hidden_feats, 1, bias=False))

    def forward(self, ins_emb, output_attn=True):
        attn = torch.softmax(self.project(ins_emb), dim=1)
        bag_emb = (ins_emb * attn).sum(dim=1)
        if output_attn:
            return bag_emb, attn
        else:
            return bag_emb


class InstanceNet(nn.Module):
    """Instance predictor.
    """
    def __init__(self, in_feats, k=3):
        """
        Parameters
        ----------
        in_feats : int
            Input feature size.
        k : int
            A topk filtering used in the aggregation of predictions.
        """
        super(InstanceNet, self).__init__()

        self.k = k
        self.weights = nn.Linear(int(in_feats / 2), int(in_feats / 2), bias=False)
        nn.init.xavier_uniform_(self.weights.weight)

    def forward(self, ins_emb, attn):
        drug_ins = ins_emb[:, :, :int(ins_emb.shape[-1] / 2)]
        dis_ins = ins_emb[:, :, int(ins_emb.shape[-1] / 2):]
        pred = torch.matmul(self.weights(drug_ins).reshape(drug_ins.shape[0], drug_ins.shape[1],
                                                           1, drug_ins.shape[2]),
                            dis_ins.unsqueeze(dim=-1)).squeeze(dim=3)
        attn_pred = attn * pred
        topk_out = torch.mean(attn_pred.topk(k=self.k, dim=1)[0], dim=1)
        return topk_out


class MLP(nn.Module):
    """Bag predictor.
    """
    def __init__(self, in_feats, dropout=0.):
        """
        Parameters
        ----------
        in_feats : int
            Input feature size.
        dropout : int
            The dropout rate.
        """
        super(MLP, self).__init__()
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = False
        self.linear = nn.Linear(in_feats, 1, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, bag_emb):
        if self.dropout:
            bag_emb = self.dropout(bag_emb)
        outputs = self.linear(bag_emb)
        return outputs


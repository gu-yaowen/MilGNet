import torch
import dgl.nn as dglnn
from torch import nn


class HeteroLinear(nn.Module):
    """Apply linear transformations on heterogeneous inputs.
    """

    def __init__(self, in_feats, out_feats, dropout=0., bn=False):
        """
        Parameters
        ----------
        in_feats : dict[key, int]
            Input feature size for heterogeneous inputs. A key can be a string or a tuple of strings.
        out_feats : int
            Output feature size.
        """
        super(HeteroLinear, self).__init__()
        self.linears = nn.ModuleDict()
        for typ, typ_in_size in in_feats.items():
            self.linears[str(typ)] = nn.Linear(typ_in_size, out_feats)
            nn.init.xavier_uniform_(self.linears[str(typ)].weight)
        if bn:
            self.bn = nn.BatchNorm1d(out_feats)
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
    """The base HeteroGCN layer."""

    def __init__(self, rel_names, in_feats, hidden_feats, dropout=0., bn=False):
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


class MetaPathAggregator(nn.Module):
    """Aggregating the meta-path instances in each bags.
    """

    def __init__(self, in_feats, hidden_feats, agg_type='sum', dropout=0., bn=False):
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
    """Attention based multi-instance learning layer.
    """

    def __init__(self, in_feats, hidden_feats):
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
    def __init__(self, in_feats, k):
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
    """MLP for predicting drug-disease associations.
    """

    def __init__(self, in_feats, dropout=0.):
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


class Model(nn.Module):
    def __init__(self, etypes, in_feats, hidden_feats,
                 num_emb_layers, agg_type='sum', k=0, dropout=0., bn=False):
        """
        Parameters
        ----------
        etypes : list
            e.g: ['disease-disease', 'drug-disease', 'drug-drug']
        in_feats : dict[str, int]
            Input feature size for each node type.
        hidden_feats : int
            Hidden feature size.
        num_emb_layers : int
            Number of embedding layers to be used.
        agg_type : string
            Type of meta-path aggregator to be used, including "sum", "average", "linear", and "BiTrans".
        k : int
            The topk instance predictions to be chosen.
        dropout : float
            The dropout rate to be used.
        bn : bool
            Whether to use batch normalization layer.
        """
        super(Model, self).__init__()
        self.lin_transform = HeteroLinear(in_feats, hidden_feats, dropout, bn)
        self.graph_embedding = nn.ModuleDict()
        for l in range(num_emb_layers):
            self.graph_embedding['Layer_{}'.format(l)] = Node_Embedding(etypes,
                                                                        hidden_feats,
                                                                        hidden_feats,
                                                                        dropout, bn)
        self.res_connect = nn.ModuleDict({'drug': nn.Linear(hidden_feats * (num_emb_layers + 1),
                                                            hidden_feats, bias=False),
                                          'disease': nn.Linear(hidden_feats * (num_emb_layers + 1),
                                                               hidden_feats, bias=False)})
        self.aggregator = MetaPathAggregator(hidden_feats, hidden_feats, agg_type)
        self.mil_layer = MILNet(hidden_feats, hidden_feats)
        self.bag_predict = MLP(hidden_feats, dropout=0.)
        if k > 0 and agg_type == 'BiTrans':
            self.ins_predict = InstanceNet(hidden_feats, k)
        else:
            self.ins_predict = None
        self.skip = skip
        self.mil = mil

    def forward(self, g, feature, mp_ins):
        """
        Parameters
        ----------
        g : dgl.graph
            Heterogeneous graph representing the drug-disease network.
        feature : dict[node_types, feature_tensors]
            Initialized node features of g.
        mp_ins : torch.tensor
            Bags of meta-path instances.
        """
        h_integrated = []
        h = self.lin_transform(feature)
        h_integrated.append(h)
        for emb_layer in self.graph_embedding:
            h = self.graph_embedding[emb_layer](g, h)
            h_integrated.append(h)
        h = dict(zip(['drug', 'disease'],
                 [torch.hstack([h_['drug'] for h_ in h_integrated]),
                  torch.hstack([h_['disease'] for h_ in h_integrated])]))
        h = {k: self.res_connect[k](v) for k, v in h.items()}
        ins_emb = self.aggregator(h, mp_ins)
        bag_emb, attn = self.mil_layer(ins_emb)
        pred_bag = self.bag_predict(bag_emb)
        pred_ins = self.ins_predict(ins_emb, attn)
        return (pred_ins + pred_bag) / 2, attn

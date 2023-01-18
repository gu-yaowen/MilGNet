import torch

from layer import *


class Model(nn.Module):
    def __init__(self, etypes, in_feats, hidden_feats,
                 num_emb_layers, agg_type='sum', k=0, dropout=0., bn=False,
                 skip=True, mil=True):
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
            Type of meta-path aggregator to be used, including "sum", "average", "linear", and "RotatE".
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
        self.layer_attention_drug = LayerAttention(hidden_feats,
                                                   hidden_feats)
        self.layer_attention_dis = LayerAttention(hidden_feats,
                                                  hidden_feats)
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
        h_integrated_drug, h_integrated_dis = [], []
        h = self.lin_transform(feature)
        h_integrated_drug.append(h['drug'])
        h_integrated_dis.append(h['disease'])
        for emb_layer in self.graph_embedding:
            h = self.graph_embedding[emb_layer](g, h)
            h_integrated_drug.append(h['drug'])
            h_integrated_dis.append(h['disease'])
        if self.skip:
            h = dict(zip(['drug', 'disease'],
                         [torch.stack(h_integrated_drug, dim=1),
                          torch.stack(h_integrated_dis, dim=1)]))
            h['drug'] = self.layer_attention_drug(h['drug'])
            h['disease'] = self.layer_attention_dis(h['disease'])

        ins_emb = self.aggregator(h, mp_ins)
        if self.mil:
            bag_emb, attn = self.mil_layer(ins_emb)
        else:
            # Ablation study
            bag_emb = ins_emb.sum(dim=1)
            attn = None
        pred_bag = self.bag_predict(bag_emb)
        if self.ins_predict:
            # Ablation study
            pred_ins = self.ins_predict(ins_emb, attn)
            return (pred_ins + pred_bag) / 2, attn
        return pred_bag, attn

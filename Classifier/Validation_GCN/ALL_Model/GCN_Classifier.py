
import math
import dgl
from dgl.nn.pytorch import SAGEConv
from collections import defaultdict, Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dgl.nn.pytorch import GraphNorm
    _HAS_GRAPHNORM = True
except Exception:
    _HAS_GRAPHNORM = False



class GraphGATClassifier(nn.Module):
    def __init__(self, in_feats=20, hidden=128, num_layers=2, num_classes=26,  dropout=0.2, aggregator='mean'):
        super().__init__()
        assert num_layers >= 1
        self.in_feats = in_feats
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.norm_in = nn.LayerNorm(in_feats)

        convs = []
        norms = []
        # first layer
        convs.append(SAGEConv(in_feats, hidden, aggregator_type=aggregator, feat_drop=0.0))
        norms.append(GraphNorm(hidden) if _HAS_GRAPHNORM else nn.Identity())
        # hidden layers
        for _ in range(num_layers - 1):
            convs.append(SAGEConv(hidden, hidden, aggregator_type=aggregator, feat_drop=0.0))
            norms.append(GraphNorm(hidden) if _HAS_GRAPHNORM else nn.Identity())
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

        self.cls = nn.Linear(hidden, num_classes)

    def forward(self, g: dgl.DGLGraph):
        h = g.ndata['feat']
        h = self.norm_in(h)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(g, h)
            h = F.relu(h)
            try:
                h = norm(g, h)
            except TypeError:
                h = norm(h)
            h = self.dropout(h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')  # [B, hidden]
        logits = self.cls(hg)        # [B, num_classes]
        return logits

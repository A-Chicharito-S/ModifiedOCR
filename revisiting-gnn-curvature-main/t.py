import networkx as nx
import os
import ot
import time
import torch
import pathlib
import numpy as np
import pandas as pd
import multiprocessing as mp
from torch_geometric.utils import (
    to_networkx,
    from_networkx
)
from torch_geometric.datasets import TUDataset
from GraphRicciCurvature.OllivierRicci import OllivierRicci


def _preprocess_data(data, is_undirected=False):
    # Get necessary data information
    print(data.x.shape, data.x)
    N = data.x.shape[0]
    m = data.edge_index.shape[1]

    # Compute the adjacency matrix
    if not "edge_type" in data.keys:
        edge_type = np.zeros(m, dtype=int)
    else:
        edge_type = data.edge_type

    # Convert graph to Networkx
    G = to_networkx(data)
    if is_undirected:
        G = G.to_undirected()

    return G, N, edge_type


G0 = list(TUDataset(root="data", name="PROTEINS", use_node_attr=False))[0]
G, N, edge_type = _preprocess_data(G0)
orc_OTD = OllivierRicci(G, alpha=0.5, method="OTD", verbose="INFO")
orc_OTD.compute_ricci_curvature()

edge_index, edge_weight = [[], []], []
for edge in orc_OTD.G.edges:
    start, end = edge
    weight = orc_OTD.G[start][end]['weight']
    orcurv = orc_OTD.G[start][end]['ricciCurvature']['rc_curvature']
    orcurv = (orcurv + 2) / 3  # orcurv \in [-2, 1]
    orcurv = weight * (1 / orcurv)
    # since positively curved edges enforce features to become similar (we want to "somehow" shrink the bottleneck)
    edge_index[0].append(start)
    edge_index[1].append(end)
    edge_weight.append(orcurv)

edge_index = torch.tensor(edge_index)
edge_type = torch.zeros(size=(len(G.edges),)).type(torch.LongTensor)
edge_weight = torch.tensor(edge_weight)


G0.edge_index, G0.edge_type, G0.edge_weight = edge_index, edge_type, edge_weight

from torch_geometric.nn import GCNConv
in_features, out_features = 100, 200
layer = GCNConv(in_features, out_features)
G0.x = torch.rand(42, 100)
x_new = layer(G0.x, G0.edge_index, edge_weight=G0.edge_weight)
print(x_new.shape)
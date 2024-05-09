import networkx as nx
import numpy as np
import torch
import torch.utils.data
import scipy.sparse as sp
from util import *
import util

class GraphBuild1(torch.utils.data.Dataset):
    def __init__(self, G_list, features='default', normalize=True, max_num_nodes=0):
        # Instead of preloading, just store references or paths
        self.G_list = G_list
        self.features = features
        self.normalize = normalize
        self.max_num_nodes = max_num_nodes

    def __len__(self):
        return len(self.G_list)

    def __getitem__(self, idx):
        G = self.G_list[idx]
        adj = np.array(nx.to_numpy_array(G))

        if self.normalize:
            sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
            adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

        num_nodes = G.number_of_nodes()
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        identity_matrix = sp.eye(self.max_num_nodes).toarray()  # Adjusted to always use max_num_nodes
        adj_label = adj_padded + identity_matrix

        if self.features == 'default':
            feat_dim = util.node_dict(G)[0]['feat'].shape[0]
            feats = np.zeros((self.max_num_nodes, feat_dim), dtype=float)
            for i, u in enumerate(G.nodes()):
                feats[i, :] = util.node_dict(G)[u]['feat']
        elif self.features == 'deg-num':
            degs = np.sum(adj, 1)
            if self.max_num_nodes > num_nodes:
                degs = np.expand_dims(np.pad(degs, (0, self.max_num_nodes - num_nodes), 'constant', constant_values=0), axis=1)
            else:
                degs = np.expand_dims(degs[:self.max_num_nodes], axis=1)
            feats = degs

        return {
            'adj': adj_padded,
            'feats': feats,
            'label': G.graph['label'],
            'num_nodes': num_nodes,
            'adj_label': adj_label
        }

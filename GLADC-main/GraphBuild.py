import networkx as nx
import numpy as np
import torch
import torch.utils.data
import scipy.sparse as sp
from util import *
import util

class GraphBuild(torch.utils.data.Dataset):
    def __init__(self, G_list, features='default', normalize=True, assign_feat='default', max_num_nodes=0):
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        
        self.assign_feat_all = []
        self.max_num_nodes = max_num_nodes

        if features == 'default':
            self.feat_dim = util.node_dict(G_list[0])[0]['feat'].shape[0]

        for G in G_list:
            adj = np.array(nx.to_numpy_array(G))
            # print(f"Graph {G.graph['label']} Adjacency Matrix Non-Zero Count: {np.count_nonzero(adj)}")

            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])
            if features == 'default':
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i, u in enumerate(G.nodes()):
                    f[i, :] = util.node_dict(G)[u]['feat']
                self.feature_all.append(f)
            elif features == 'deg-num':
                degs = np.sum(np.array(adj), 1)
                if self.max_num_nodes > G.number_of_nodes():
                    degs = np.expand_dims(np.pad(degs, (0, self.max_num_nodes - G.number_of_nodes()), 'constant', constant_values=0),
                                          axis=1)
                elif self.max_num_nodes < G.number_of_nodes():
                    deg_index = np.argsort(degs, axis=0)
                    deg_ind = deg_index[:G.number_of_nodes()-self.max_num_nodes]
                    degs = np.delete(degs, deg_ind, axis=0)
                    degs = np.expand_dims(degs, axis=1)
                else:
                    degs = np.expand_dims(degs, axis=1)                                        
                self.feature_all.append(degs)

            self.assign_feat_all.append(self.feature_all[-1])
            
            # Debug: Print processed information for each graph
            # print(f"Processed graph with label {G.graph['label']}: Number of nodes = {G.number_of_nodes()}, Feature dim = {self.feature_all[-1].shape[1]}")

        self.feat_dim = self.feature_all[0].shape[1]
        self.assign_feat_dim = self.assign_feat_all[0].shape[1]

    def __len__(self):
        # Debug: Print dataset size
        # print(f"Total number of graphs in dataset: {len(self.adj_all)}")
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        if self.max_num_nodes > num_nodes:
            adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
            adj_padded[:num_nodes, :num_nodes] = adj
        elif self.max_num_nodes < num_nodes:
            degs = np.sum(np.array(adj), 1)
            deg_index = np.argsort(degs, axis=0)
            deg_ind = deg_index[:num_nodes-self.max_num_nodes]
            adj_padded = np.delete(adj, deg_ind, axis=0)
            adj_padded = np.delete(adj_padded, deg_ind, axis=1)
        else:
            adj_padded = adj

        identity_matrix = sp.eye(adj_padded.shape[0]).toarray()  # Convert sparse matrix to dense array
        adj_label = adj_padded + identity_matrix

        adj = adj_label.tolist()


        # Debug: Print adjacency matrix before normalization
        # print(f"Adjacency matrix before normalization for graph index {idx}:")
        # print(np.array(adj))

        adj = normalize_adj(adj)  # Normalize adjacency matrix
        adj = adj.toarray()
        adj_label = np.array(adj_label)

        # Debug: Print details of the retrieved item
        # print(f"Retrieved graph with index {idx}: Number of nodes = {num_nodes}, Adjacency matrix shape = {adj_padded.shape}")

        return {'adj': adj_padded,
                'feats': self.feature_all[idx].copy(),
                'label': self.label_all[idx],
                'num_nodes': num_nodes,
                'adj_label': adj_label,
                'assign_feats': self.assign_feat_all[idx].copy()}

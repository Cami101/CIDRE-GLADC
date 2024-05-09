import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from packaging import version

def node_iter(G):
    if version.parse(nx.__version__) < version.parse("2.0"):
        return G.nodes_iter()
    else:
        return G.nodes()

def node_dict(G):
    if version.parse(nx.__version__) > version.parse("2.1"):
        node_dict = G.nodes
    else:
        node_dict = G.node
    return node_dict

def adj_process(adjs):
    g_num, n_num, n_num = adjs.shape
    adjs = adjs.detach()
    for i in range(g_num):
        adjs[i] += torch.eye(n_num)
        adjs[i][adjs[i]>0.] = 1.
        degree_matrix = torch.sum(adjs[i], dim=-1, keepdim=False)
        degree_matrix = torch.pow(degree_matrix,-1)
        degree_matrix[degree_matrix == float("inf")] = 0.
        degree_matrix = torch.diag(degree_matrix)
        adjs[i] = torch.mm(degree_matrix, adjs[i])
    return adjs


def NormData(adj):
    adj=adj.tolist()
    adj_norm = normalize_adj(adj )
    adj_norm = adj_norm.toarray()
    #adj = adj + sp.eye(adj.shape[0])
    #adj = adj.toarray()
    #feat = feat.toarray()
    return adj_norm



def normalize_adj(adj):

    if isinstance(adj, list):
        # print("Converting adj from list to numpy array.")
        adj = np.array(adj)
    """Symmetrically normalize adjacency matrix."""
    # print("Original adj shape:", adj.shape)  # Debugging line

    adj = sp.coo_matrix(adj)
    # print("COO adj shape:", adj.shape)  # Debugging line

    rowsum = np.array(adj.sum(1))
    # print("Rowsum shape:", rowsum.shape)  # Debugging line

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # print("d_inv_sqrt before fix:", d_inv_sqrt)  # Debugging line to check before fixing inf values

    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # print("d_inv_sqrt after fix:", d_inv_sqrt)  # Debugging line to check after fixing inf values

    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # print("d_mat_inv_sqrt shape:", d_mat_inv_sqrt.shape)  # Debugging line

    # Check shapes for the multiplication
    # print("Shape check for multiplication:")
    # print("adj shape:", adj.shape)
    # print("d_mat_inv_sqrt shape:", d_mat_inv_sqrt.shape)
    # print("d_mat_inv_sqrt transpose shape:", d_mat_inv_sqrt.transpose().shape)

    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    # print("Resulting normalized_adj shape:", normalized_adj.shape)  # Debugging line

    return normalized_adj.tocoo()


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)
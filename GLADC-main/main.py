# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.svm import OneClassSVM
import argparse
import load_data
import networkx as nx
from graph_autoencoder import *
import torch
import torch.nn as nn
import time
import graph_autoencoder
from loss import *
from util import *
from torch.autograd import Variable
from GraphBuild import GraphBuild
from numpy.random import seed
import random
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F
from sklearn.manifold import TSNE
from matplotlib import cm
from model import *
from random import shuffle
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def arg_parse():
    parser = argparse.ArgumentParser(description='G-Anomaly Arguments.')
    parser.add_argument('--datadir', dest='datadir', default ='dataset', help='Directory where benchmark is located')
    parser.add_argument('--DS', dest='DS', default ='DHFR', help='dataset name')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int, default=0, help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--num_epochs', dest='num_epochs', default=100, type=int, help='total epoch number')
    parser.add_argument('--batch-size', dest='batch_size', default=300, type=int, help='Batch size.')
    parser.add_argument('--hidden-dim', dest='hidden_dim', default=256, type=int, help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', default=128, type=int, help='Output dimension')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', default=2, type=int, help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const', const=False, default=True, help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', default=0.1, type=float, help='Dropout rate.')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--nobias', dest='bias', action='store_const', const=False, default=True, help='Whether to add bias. Default to True.')
    parser.add_argument('--feature', dest='feature', default='deg-num', help='use what node feature')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed')
    return parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def lossT(T):
    q=0
    p=0
    a=0
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            a=a + T[i][j]*T[i][j]
        a= a/T.shape[1]
        q=q+a
    p=q / T.shape[0]
    
            
    return p
        
def gen_ran_output(h0, adj, model, vice_model):
    for (adv_name,adv_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            adv_param.data = param.data + 1.0 * torch.normal(0,torch.ones_like(param.data)*param.data.std())
    x1_r,Feat_0= vice_model(h0, adj)
    return x1_r,Feat_0


def train(dataset, data_test_loader, NetG, noise_NetG, args):    
    optimizerG = torch.optim.Adam(NetG.parameters(), lr=args.lr)
    epochs=[]
    auroc_final = 0
    l_bce = nn.BCELoss()
    l_enc = l2_loss
    node_Feat=[]
    graph_Feat=[]
    max_AUC=0
    for epoch in range(args.num_epochs):
        total_time = 0
        total_lossG = 0.0
        NetG.train()
        for batch_idx, data in enumerate(dataset):           
            begin_time = time.time()
            adj = Variable(data['adj'].float(), requires_grad=False)
            h0 = Variable(data['feats'].float(), requires_grad=False)
            adj_label = Variable(data['adj_label'].float(), requires_grad=False)

            x1_r,Feat_0 = NetG.shared_encoder(h0, adj)
            x1_r_1 ,Feat_0_1= gen_ran_output(h0, adj, NetG.shared_encoder, noise_NetG)
            x_fake,s_fake,x2,Feat_1=NetG(x1_r,adj)

            
            err_g_con_s, err_g_con_x = loss_func(adj_label, s_fake, h0, x_fake)

            node_loss=torch.mean(F.mse_loss(x1_r, x2, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
            graph_loss = F.mse_loss(Feat_0, Feat_1, reduction='none').mean(dim=1).mean(dim=0)
            err_g_enc=loss_cal(Feat_0_1, Feat_0)


            lossG = err_g_con_s + err_g_con_x +node_loss+graph_loss +err_g_enc
            optimizerG.zero_grad()
            lossG.backward()
          
            optimizerG.step()
          
            total_lossG += lossG
            
            elapsed = time.time() - begin_time
            total_time += elapsed
        
        if (epoch+1)%10 == 0 and epoch > 0:
            epochs.append(epoch)
            NetG.eval()   
            loss = []
            y=[]

            false_positives = 0
            false_negatives = 0
            all_losses = []  # List to store all loss values for setting dynamic threshold

            for batch_idx, data in enumerate(data_test_loader):
               adj = Variable(data['adj'].float(), requires_grad=False)
               h0 = Variable(data['feats'].float(), requires_grad=False)

               x1_r,Feat_0 = NetG.shared_encoder(h0, adj)
         
               x_fake,s_fake,x2,Feat_1=NetG(x1_r,adj)
               
               loss_node=torch.mean(F.mse_loss(x1_r, x2, reduction='none'), dim=2).mean(dim=1).mean(dim=0)

               loss_graph = F.mse_loss(Feat_0, Feat_1, reduction='none').mean(dim=1)
            
               loss_=loss_node+loss_graph
            
               loss_ = np.array(loss_.cpu().detach())
               all_losses.extend(loss_)  # Append loss of each graph to all_losses
               
               loss.append(loss_)
               if data['label'] == 0:
                   y.append(1)
               else:
                   y.append(0) 

            threshold = np.percentile(all_losses, 95)
            print(f'Dynamic Threshold for Epoch {epoch+1}: {threshold}')

            # Now classify based on this threshold
            for loss_value, actual_label in zip(all_losses, y):
                predicted_label = 1 if loss_value > threshold else 0
                if predicted_label == 1 and actual_label == 0:
                    false_positives += 1
                elif predicted_label == 0 and actual_label == 1:
                    false_negatives += 1
            
            # Print false positives and negatives after each epoch's evaluation
            print(f'Epoch {epoch+1}: False Positives: {false_positives}, False Negatives: {false_negatives}')

            label_test = []
            for loss_ in loss:
               label_test.append(loss_)
            label_test = np.array(label_test)
            fpr_ab, tpr_ab, _ = roc_curve(y, label_test)
            test_roc_ab = auc(fpr_ab, tpr_ab)   
            print('semi-supervised abnormal detection: auroc_ab: {}'.format(test_roc_ab))
            if test_roc_ab > max_AUC:
                max_AUC=test_roc_ab
        auroc_final = max_AUC
    return auroc_final

    
if __name__ == '__main__':
    args = arg_parse()
    DS = args.DS
    setup_seed(args.seed)
    a=0
    b=0

    graphs = load_data.read_graphfile(args.datadir, args.DS, max_nodes=args.max_nodes)  
    datanum = len(graphs)
    if args.max_nodes == 0:
        max_nodes_num = max([G.number_of_nodes() for G in graphs])
    else:
        max_nodes_num = args.max_nodes
    print(datanum)
    graphs_label = [graph.graph['label'] for graph in graphs]
    for graph in graphs:
        if graph.graph['label'] == 0:
            a=a+1
        else:
            b=b+1
    print(a,b,'!!!!!!!!!!!!!!!!!!!!!!')
    
    
    kfd=StratifiedKFold(n_splits=5, random_state=args.seed, shuffle = True)
    result_auc=[]
    for k, (train_index,test_index) in enumerate(kfd.split(graphs, graphs_label)):
        
        graphs_train_ = [graphs[i] for i in train_index]
        graphs_test = [graphs[i] for i in test_index]
       
        graphs_train = []
        for graph in graphs_train_:
            if graph.graph['label'] != 0:
                graphs_train.append(graph)
        

        num_train = len(graphs_train)
        num_test = len(graphs_test)
        print(num_train, num_test)
        
        dataset_sampler_train = GraphBuild(graphs_train, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
    
        NetG= NetGe(dataset_sampler_train.feat_dim,args.hidden_dim, args.output_dim,args.dropout,args.batch_size)
        noise_NetG= Encoder(dataset_sampler_train.feat_dim,args.hidden_dim, args.output_dim,args.dropout,args.batch_size)
        
        
        data_train_loader = torch.utils.data.DataLoader(dataset_sampler_train, 
                                                    shuffle=True,
                                                    batch_size=args.batch_size)
    
        dataset_sampler_test = GraphBuild(graphs_test, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
        data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test, 
                                                        shuffle=False,
                                                        batch_size=1)
        result = train(data_train_loader, data_test_loader, NetG, noise_NetG,args)     
        result_auc.append(result)
            
    result_auc = np.array(result_auc)    
    auc_avg = np.mean(result_auc)
    auc_std = np.std(result_auc)
    print('auroc{}, average: {}, std: {}'.format(result_auc, auc_avg, auc_std))
    
    
    

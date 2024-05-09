# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.svm import OneClassSVM
from sklearn.model_selection import StratifiedKFold
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
from sklearn.utils import resample


def arg_parse():
    parser = argparse.ArgumentParser(description='G-Anomaly Arguments.')
    parser.add_argument('--datadir', dest='datadir', default ='dataset', help='Directory where benchmark is located')
    parser.add_argument('--DS', dest='DS', default ='Tox21_HSE', help='dataset name')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int, default=50, help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--clip', dest='clip', default=0.1, type=float, help='Gradient clipping.')
    parser.add_argument('--num_epochs', dest='num_epochs', default=100, type=int, help='total epoch number')
    parser.add_argument('--batch-size', dest='batch_size', default=2000, type=int, help='Batch size.')
    parser.add_argument('--hidden-dim', dest='hidden_dim', default=256, type=int, help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', default=128, type=int, help='Output dimension')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', default=2, type=int, help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const', const=False, default=True, help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', default=0.1, type=float, help='Dropout rate.')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--nobias', dest='bias', action='store_const', const=False, default=True, help='Whether to add bias. Default to True.')
    parser.add_argument('--feature', dest='feature', default='deg-num', help='use what node feature')
    parser.add_argument('--seed', dest='seed', type=int, default=2, help='seed')
    return parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
        
def gen_ran_output(h0, adj, model, vice_model):
    epsilon = 1e-8  # Small constant to prevent division by zero
    for (adv_name, adv_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            std = param.data.std().clamp(min=epsilon)  # Ensure std is always positive
            noise = torch.normal(0, torch.ones_like(param.data) * std).cuda()
            # noise = torch.normal(0, torch.ones_like(param.data) * std)
            adv_param.data = param.data + 1.0 * noise
    x1_r, Feat_0 = vice_model(h0, adj)
    return x1_r, Feat_0


def train(dataset, data_test_loader, NetG, noise_NetG, args):    
    optimizerG = torch.optim.Adam(NetG.parameters(), lr=args.lr)
    epochs=[]
    auroc_final = 0
    l_bce = nn.BCELoss()
    #l_adv= l2_loss
    l_enc = l2_loss
    node_Feat=[]
    graph_Feat=[]
    false_positives = []  # Initialize a list to store false positives
    epoch_aucs = []  # List to store AUC for each epoch
    anom_count = []


    max_AUC=0
    for epoch in range(args.num_epochs):
        total_time = 0
        total_lossG = 0.0
        NetG.train()
        false_positives = []
        false_negatives = []
        all_optimal_tpr = []
        for batch_idx, data in enumerate(dataset):           
            begin_time = time.time()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            # adj = Variable(data['adj'].float(), requires_grad=False)


            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            # h0 = Variable(data['feats'].float(), requires_grad=False)



            adj_label = Variable(data['adj_label'].float(), requires_grad=False).cuda()
            # adj_label = Variable(data['adj_label'].float(), requires_grad=False)


            x1_r,Feat_0 = NetG.shared_encoder(h0, adj)
            x1_r_1 ,Feat_0_1= gen_ran_output(h0, adj, NetG.shared_encoder, noise_NetG)

            x_fake,s_fake,x2,Feat_1=NetG(x1_r,adj)
            
            err_g_con_s, err_g_con_x = loss_func(adj_label, s_fake, h0, x_fake)

            node_loss=torch.mean(F.mse_loss(x1_r, x2, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
            graph_loss = F.mse_loss(Feat_0, Feat_1, reduction='none').mean(dim=1).mean(dim=0)
            #err_g_enc=l_enc(Feat_0, Feat_1)
            err_g_enc=loss_cal(Feat_0_1, Feat_0)


            lossG = err_g_con_s + err_g_con_x + graph_loss +node_loss+err_g_enc
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
            all_fpr = []
            all_tpr = []
            num_anomalous = 0

            all_predictions = []
            for batch_idx, data in enumerate(data_test_loader):
               adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            #    adj = Variable(data['adj'].float(), requires_grad=False)
               h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            #    h0 = Variable(data['feats'].float(), requires_grad=False)


               label = data['label']
               
               # Forward pass to get outputs
               x1_r,Feat_0 = NetG.shared_encoder(h0, adj)
               x_fake,s_fake,x2,Feat_1=NetG(x1_r,adj)

               loss_node=torch.mean(F.mse_loss(x1_r, x2, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
               loss_graph = F.mse_loss(Feat_0, Feat_1, reduction='none').mean(dim=1)
               loss_=loss_node+loss_graph
               loss_ = np.array(loss_.cpu().detach())
               loss.append(loss_)

               if data['label'] == 0:
                   y.append(1)
               else:
                   y.append(0)
                
            label_test = []
            for loss_ in loss:
               label_test.append(loss_)
            label_test = np.array(label_test)

            fpr_ab, tpr_ab, thresholds = roc_curve(y, label_test)

            all_fpr.append(fpr_ab)
            all_tpr.append(tpr_ab)
            test_roc_ab = auc(fpr_ab, tpr_ab)
            epoch_aucs.append(test_roc_ab)  # Store the AUC of this epoch
            num_anomalous = sum(np.array(label_test) > optimal_threshold)
            anom_count.append(num_anomalous)

            print(f'Epoch {epoch+1}: AUROC = {test_roc_ab:.4f}')

            if test_roc_ab > auroc_final:
                auroc_final = test_roc_ab
            max_AUC= auroc_final 
        #if epoch == (args.num_epochs-1):
            #auroc_final =  test_roc_ab
            print(max_AUC)

    return epoch_aucs, all_fpr, all_tpr, anom_count

if __name__ == '__main__':
    args = arg_parse()
    setup_seed(args.seed)

    # Load the entire dataset
    all_graphs = load_data.read_graphfile(args.datadir, args.DS, max_nodes=args.max_nodes)

    print("All graphs loaded")

    # Initialize an empty list to store labels
    labels = []

    # Extract labels from all_graphs
    for G in all_graphs:
        labels.append(G.graph['label'])

    # Convert labels to numpy array
    labels = np.array(labels)

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    all_fold_aucs = []  # Store AUCs from all folds
    total_anom_count = 0  # Initialize total anomalous count


    # Loop over folds
    for fold, (train_index, test_index) in enumerate(skf.split(all_graphs, labels)):
        # Split dataset into train and test for this fold
        train_graphs = [all_graphs[i] for i in train_index]
        test_graphs = [all_graphs[i] for i in test_index]

        # Remove abnormal graphs from training set
        train_graphs = [G for G in train_graphs if G.graph['label'] == 0]


        # Prepare the dataset samplers for training and testing
        dataset_sampler_train = GraphBuild(train_graphs, features=args.feature, normalize=False, max_num_nodes=args.max_nodes)
        data_train_loader = torch.utils.data.DataLoader(dataset_sampler_train, shuffle=True, batch_size=args.batch_size)

        dataset_sampler_test = GraphBuild(test_graphs, features=args.feature, normalize=False, max_num_nodes=args.max_nodes)
        data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test, shuffle=False, batch_size=1)

        # Initialize models
        NetG = NetGe1(dataset_sampler_train.feat_dim, args.hidden_dim, args.output_dim, args.num_gc_layers, num_layers=3, bn=args.bn, args=args).cuda()
        # NetG = NetGe1(dataset_sampler_train.feat_dim, args.hidden_dim, args.output_dim, args.num_gc_layers, num_layers=3, bn=args.bn, args=args)
        noise_NetG = Encoder1(dataset_sampler_train.feat_dim, args.hidden_dim, args.output_dim, args.num_gc_layers,num_layers=3, bn=args.bn, args=args).cuda()
        # noise_NetG = Encoder1(dataset_sampler_train.feat_dim, args.hidden_dim, args.output_dim, args.num_gc_layers,num_layers=3, bn=args.bn, args=args)

        # Train the model
        result_auc, all_fpr, all_tpr, anom_count = train(data_train_loader, data_test_loader, NetG, noise_NetG, args)
        all_fold_aucs.append(result_auc)
        total_anom_count += sum(anom_count)  # Add up all counts from current fold

        plt.figure(figsize=(10, 8))  # Adjust the size as needed
        plt.plot(all_fpr[-1], all_tpr[-1], color='darkorange',
                lw=2, label='ROC curve (area = %0.2f)' % result_auc[-1])
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Specify the filename and directory to save the figure
        plt.savefig('ROC_Curve.png')
        plt.close()  # Close the figure to free up memory



    
    result_auc = np.array(result_auc)    
    auc_avg = np.mean(result_auc)
    auc_std = np.std(result_auc)
    #Calculate the average number of anomalous graphs detected across all folds
    average_anom_count = total_anom_count / len(all_fold_aucs)
    print(f'Average number of anomalous graphs detected across all folds: {average_anom_count}')

    print('auroc{}, average: {}, std: {}'.format(result_auc, auc_avg, auc_std))


    for i, auc_scores in enumerate(all_fold_aucs):
        plt.plot(auc_scores, label=f'Fold {i+1}')
    plt.xlabel('Epoch (x10)')
    plt.ylabel('AUC')
    plt.title('AUC per Epoch Across Folds')
    plt.legend()
    plt.savefig(f'auc_fold_{i+1}.png')  # Save each figure to a file
    plt.close()  # Close the figure to free up memory

    
    
    

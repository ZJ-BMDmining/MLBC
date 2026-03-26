import random
from RNAMRIdataloder import RNAMRIDataset
import copy
from torch.utils.data import DataLoader
from models.RNAGAT import Fusion
from models.MLPmodal import NeuralNetwork
from sklearn import metrics
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import re
import numpy as np
from tqdm import tqdm
import argparse
import os
import pickle
from operator import mod
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from models.BiLSTM import MLPmodal,Fusionmodal_K,Fusionmodal,Knowledgemodal
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report,precision_recall_curve,roc_auc_score,average_precision_score,f1_score
from sklearn.metrics import precision_recall_curve,f1_score,roc_curve,roc_auc_score,auc,accuracy_score,average_precision_score,precision_score,recall_score
import scipy.sparse as sp
import networkx as nx
from models.MSCCNN import new_AttentionMultiScaleCNN
from scipy import interp
import time
import warnings
warnings.filterwarnings("ignore")

def construct_adjacency_hat(adj):
    """
        :param adj: original adjacency matrix  <class 'scipy.sparse.csr.csr_matrix'>
        :return:
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # <class 'scipy.sparse.coo.coo_matrix'>
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def calc_confusion_matrix(result, test_label,n_classes):
    # result = F.one_hot(result,num_classes=n_classes)
    # print(result)

    test_label = F.one_hot(test_label,num_classes=n_classes)
    # print(test_label)

    true_label= np.argmax(test_label, axis =1)

    predicted_label= np.argmax(result, axis =1)
    
    precision = dict()
    recall = dict()
    thres = dict()
    for i in range(n_classes):
        precision[i], recall[i], thres[i] = precision_recall_curve(test_label[:, i],
                                                            result[:, i])


    # print ("Classification Report :") 
    print (classification_report(true_label, predicted_label,digits=3))
    cr = classification_report(true_label, predicted_label, output_dict=True,digits=3)
    return cr

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='PPMI,PDBP')
    parser.add_argument('--model', default='model', type=str)
    parser.add_argument('--n_classes', default=3, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--optimizer', default='adam',
                        type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=30, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1,
                        type=float, help='decay coefficient')
    parser.add_argument('--ckpt_path', default='log_cd',
                        type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--clip_grad', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--use_tensorboard', default=True,
                        type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', default='log_cd',
                        type=str, help='path to save tensorboard logs')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0',
                        type=str, help='GPU ids')
    parser.add_argument('--D_size', required=True, type=str,
                        help='2D,3D')


    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_test(args, model, device, dataloader,adj_mat_1):

    n_classes = args.n_classes

    predicted_label=[]
    true_label=[]
    out=[]
    uncertainty_list = []

    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]

        for step, (RNA, MRI, Clinic, label) in enumerate(dataloader):


            RNA = RNA.to(device)
            adj_mat_1 = adj_mat_1.to(device)
            MRI = MRI.to(device)
            Clinic = Clinic.to(device)
            label = label.to(device)
            if args.ckpt_path == 'RNA+MRI+Knowledge':
                prediction,_,_,_ = model(RNA.float(), MRI.float(), Clinic.float(),adj_mat_1)
            elif args.ckpt_path == 'RNA+MRI' or args.ckpt_path == 'MRI':
                prediction,_,_ = model(RNA.float(), MRI.float(), Clinic.float())
            elif args.ckpt_path == 'Knowledgemodal':
                prediction,_,_ = model(RNA.float(),adj_mat_1)
            elif args.ckpt_path == 'RNA':
                prediction,_ = model(RNA.float())
            _, predicted = prediction.max(1)
            predicted_label.extend(predicted.tolist())
            out.extend(prediction.tolist())
            true_label.extend(label.tolist())

            # Compute uncertainty for each sample
            softmax_probs = torch.softmax(prediction, dim=1)
            for i, true_class in enumerate(label):
                true_prob = softmax_probs[i, true_class].item()  # Probability of true class
                uncertainty = 1.0 - true_prob  # Uncertainty = 1 - P(true class)
                uncertainty_list.append(uncertainty)

            for i, item in enumerate(label):

                ma = prediction[i].cpu().data.numpy()
                index_ma = np.argmax(ma)
                # print(index_ma, label_index)
                num[label[i]] += 1.0
                if index_ma == label[i]:
                    acc[label[i]] += 1.0

    cr=calc_confusion_matrix(np.array(out), torch.tensor(true_label),n_classes)
    true_label_binarized = label_binarize(true_label, classes=list(range(n_classes)))

    auc_score = roc_auc_score(true_label_binarized, np.array(out), average='macro', multi_class='ovr')
    aupr_score = average_precision_score(true_label_binarized, np.array(out), average='macro')

    avg_uncertainty = np.mean(uncertainty_list) if uncertainty_list else 0.0


    return auc_score,aupr_score,cr,avg_uncertainty


def main(seeds):

    args = get_arguments()
    print(args)

    setup_seed(seeds)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device('cuda:0')

    # mask_ratio_all = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    mask_ratio_all = [0.0]
    mean_f1_scores = []

    for mask_ratio in mask_ratio_all:
        print(f"\nTesting with mask_ratio: {mask_ratio}")

        test_dataset = RNAMRIDataset(mode='test', modal='overlap', dataset_name=args.dataset,D_size=args.D_size,mask_ratio=mask_ratio)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)


        adj_1 = sp.load_npz(f'../data_new/{args.dataset}/adjacency_matrix.npz')
        adj_mat_1 = adj_1.todense()
        adj_mat_1 = sp.csr_matrix(adj_mat_1)
        L_1 = construct_adjacency_hat(adj_mat_1)
        L_1 = sparse_mx_to_torch_sparse_tensor(L_1)

        save_dir = 'New/RNA+MRI'

        F1_list=[]
        acc_list=[]
        pre_list=[]
        recall_list=[]
        auc_list=[]
        aupr_list=[]
        uncertainty_list=[]
        for fold in range(5):

            if args.ckpt_path =='RNA+MRI+Knowledge':
                model = Fusionmodal_K(num_node=test_dataset.get_RNA_shape(), clinic_feature=test_dataset.get_Clinic_shape(), hidden_dim=128, output_dim=test_dataset.get_num_classes(),dropout_rate=0.2,alpha=0.2,args=args)
            elif args.ckpt_path =='RNA+MRI' or args.ckpt_path == 'MRI':
                model = Fusionmodal(num_node=test_dataset.get_RNA_shape(), clinic_feature=test_dataset.get_Clinic_shape(), hidden_dim=128, output_dim=test_dataset.get_num_classes(),dropout_rate=0.2,alpha=0.2,args=args)
            elif args.ckpt_path == 'Knowledgemodal':
                model = Knowledgemodal(num_node=test_dataset.get_RNA_shape(), hidden_dim=128, output_dim=test_dataset.get_num_classes() ,dropout_rate=0.2,alpha=0.2,args=args)
            elif args.ckpt_path == 'RNA':
                model = MLPmodal(num_node=test_dataset.get_RNA_shape(), hidden_dim=128, output_dim=test_dataset.get_num_classes() ,dropout_rate=0.2,alpha=0.2,args=args)

            model.to(device)
            modal_name = save_dir+'/best_{}.pth'.format(fold)
            saved_dict = torch.load(os.path.join(args.dataset,modal_name))
            model.load_state_dict(saved_dict['model'])
            test_auc,test_aupr,cr,avg_uncertainty = model_test(args, model, device, test_dataloader,L_1)
            F1_list.append(cr['macro avg']['f1-score'])
            acc_list.append(cr['accuracy'])
            pre_list.append(cr['macro avg']['precision'])
            recall_list.append(cr['macro avg']['recall'])
            auc_list.append(test_auc)
            aupr_list.append(test_aupr)
            uncertainty_list.append(float(avg_uncertainty))
            # print(f"Test Acc: {cr['accuracy']:.4f}, Test F1: {cr['macro avg']['f1-score']:.4f},Test Pre: {cr['macro avg']['precision']:.4f}, Test Rec: {cr['macro avg']['recall']:.4f},Test AUC: {test_auc:.4f}, Test AUPR: {test_aupr:.4f},avg_uncertainty: {avg_uncertainty:.4f}")

        print(f"{np.mean(F1_list):.3f} {np.mean(acc_list):.3f} {np.mean(pre_list):.3f} "
            f"{np.mean(recall_list):.3f} {np.mean(auc_list):.3f} {np.mean(aupr_list):.3f} "
            f"{np.mean(uncertainty_list):.3f}")
            
        print(f"{np.std(F1_list):.3f} {np.std(acc_list):.3f} {np.std(pre_list):.3f} "
            f"{np.std(recall_list):.3f} {np.std(auc_list):.3f} {np.std(aupr_list):.3f} "
            f"{np.std(uncertainty_list):.3f}")
        # ,np.std(avg_uncertainty)
        
        mean_f1 = np.mean(F1_list)
        mean_f1_scores.append(mean_f1)
        # print(f"Noise Level: {noise_level}, Mean F1 Score: {mean_f1:.3f}")

            # Plotting the results
    # plt.figure(figsize=(10, 6))
    # plt.plot(mask_ratio_all, mean_f1_scores, marker='o', linestyle='-', color='b')
    # plt.title('Mean F1 Score vs. Mask_Ratio')
    # plt.xlabel('Mask_Ratio')
    # plt.ylabel('Mean F1 Score')
    # # plt.grid(True)
    # plt.savefig(f'{args.dataset}_f1_vs_mask.png')
    # plt.show()

    print(mask_ratio_all)
    print(mean_f1_scores)
if __name__ == "__main__":
    main(42)
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
from models.BiLSTM import MLPmodal,Fusionmodal_K
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report,precision_recall_curve,roc_auc_score,average_precision_score,f1_score
import scipy.sparse as sp
import networkx as nx
from models.MSCCNN import new_AttentionMultiScaleCNN
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

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, cuda_device='0'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.cuda_device = cuda_device

    def forward(self, features, labels=None, mask=None):
        """
        Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = ("cuda:" + self.cuda_device if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # print("mask.sum(1):", mask.sum(1))

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) # |1/P(i)|= mask.sum(1)
        # mask_sum = mask.sum(1)
        # mask_sum[mask_sum == 0] = 1
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        mask_sum = mask.sum(1)
        valid_mask = mask_sum > 0
        mask_sum[mask_sum == 0] = 1

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        mean_log_prob_pos = mean_log_prob_pos[valid_mask]
        

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.mean()

        return loss

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


def train_epoch(args, model, device, dataloader, optimizer, scheduler,class_weights, writer,adj_mat_1):
    criterion = nn.CrossEntropyLoss(class_weights)

    criterion2 = SupConLoss(temperature=0.2, contrast_mode='one', cuda_device='0')

    model.train()

    _loss = 0
    correct = 0
    correct_RNA = 0
    correct_MRI = 0
    total = 0

    predicted_label=[]
    predicted_label_RNA=[]
    predicted_label_MRI=[]
    true_label=[]
    for step, (RNA, MRI, Clinic, label) in enumerate(dataloader):
        
        RNA = RNA.to(device)
        adj_mat_1 = adj_mat_1.to(device)
        MRI = MRI.to(device)
        Clinic = Clinic.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        if args.ckpt_path == 'RNA+MRI+Clinic+Knowledge':
            out,r_x,out_RNA,out_MRI,out_Clinic = model(RNA.float(), MRI.float(), Clinic.float(),adj_mat_1)
            # print(out)
            # print(label)
            loss_ce = criterion(out, label)
            # loss_RNA = criterion(out_RNA, label)
            # loss_MRI = criterion(out_MRI, label)
            # loss_Clinic = criterion(out_Clinic, label)
            loss_rec = criterion2(r_x.unsqueeze(1),label)
            loss = loss_ce + 0.2 * loss_rec
            # loss = loss_ce
        elif args.ckpt_path == 'RNA+MRI+Knowledge':
            out,r_x,out_RNA,out_MRI = model(RNA.float(), MRI.float(), Clinic.float(),adj_mat_1)
            # print(out)
            # print(label)
            loss_ce = criterion(out, label)
            loss_RNA = criterion(out_RNA, label)
            loss_MRI = criterion(out_MRI, label)
            # loss_Clinic = criterion(out_Clinic, label)
            loss_rec = criterion2(r_x.unsqueeze(1),label)
            loss = loss_ce + 0.2 * loss_rec + 0.5 * loss_RNA + 0.5 * loss_MRI
            # loss = loss_ce + 0.2 * loss_rec
            # loss = loss_ce

        _, predicted = torch.max(out, 1)
        _, predicted_RNA = torch.max(out_RNA, 1)
        _, predicted_MRI = torch.max(out_MRI, 1)
        predicted_label.extend(predicted.tolist())
        predicted_label_RNA.extend(predicted_RNA.tolist())
        predicted_label_MRI.extend(predicted_MRI.tolist())

        true_label.extend(label.tolist())
        total += label.size(0)
        correct += (predicted == label).sum().item()
        correct_RNA += (predicted_RNA == label).sum().item()
        correct_MRI += (predicted_MRI == label).sum().item()
        loss.backward()
        optimizer.step()
        _loss += loss.item()
    acc = correct / total
    acc_RNA = correct_RNA / total
    acc_MRI = correct_MRI / total
    train_f1=f1_score(true_label,predicted_label,average='macro')
    train_f1_RNA=f1_score(true_label,predicted_label_RNA,average='macro')
    train_f1_MRI=f1_score(true_label,predicted_label_MRI,average='macro')
    
    return _loss / len(dataloader),acc,acc_RNA,acc_MRI,train_f1,train_f1_RNA,train_f1_MRI
    # return _loss / len(dataloader),acc,0,0,train_f1,0,0


def valid(args, model, device, dataloader,adj_mat_1,class_weights):
    criterion = nn.CrossEntropyLoss(class_weights)

    criterion2 = SupConLoss(temperature=0.2, contrast_mode='one', cuda_device='0')

    n_classes = args.n_classes

    predicted_label=[]
    predicted_label_RNA=[]
    predicted_label_MRI=[]
    true_label=[]
    _loss = 0
    total=0
    correct=0
    correct_RNA=0
    correct_MRI=0
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
            if args.ckpt_path == 'RNA+MRI+Clinic+Knowledge':
                prediction,r_x,_,_,_ = model(RNA.float(), MRI.float(), Clinic.float(),adj_mat_1)
            elif args.ckpt_path == 'RNA+MRI+Knowledge':
                prediction,r_x,out_RNA,out_MRI = model(RNA.float(), MRI.float(), Clinic.float(),adj_mat_1)
                loss_ce = criterion(prediction, label)
                loss_rec = criterion2(r_x.unsqueeze(1),label)
                loss_RNA = criterion(out_RNA, label)
                loss_MRI = criterion(out_MRI, label)
                loss = loss_ce + 0.2 * loss_rec + 0.5 * loss_RNA +0.5 * loss_MRI 
                # loss=loss_ce
            _, predicted = prediction.max(1)
            _, predicted_RNA = out_RNA.max(1)
            _, predicted_MRI = out_MRI.max(1)

            predicted_label.extend(predicted.tolist())
            predicted_label_RNA.extend(predicted_RNA.tolist())
            predicted_label_MRI.extend(predicted_MRI.tolist())

            true_label.extend(label.tolist())
            _loss += loss.item()

            total += label.size(0)
            correct += (predicted == label).sum().item()
            correct_RNA += (predicted_RNA == label).sum().item()
            correct_MRI += (predicted_MRI == label).sum().item()
    
    
    acc = correct / total
    acc_RNA = correct_RNA / total
    acc_MRI = correct_MRI / total
    val_f1=f1_score(true_label,predicted_label,average='macro')
    val_f1_RNA=f1_score(true_label,predicted_label_RNA,average='macro')
    val_f1_MRI=f1_score(true_label,predicted_label_MRI,average='macro')


    return _loss / len(dataloader),acc,acc_RNA,acc_MRI,val_f1,val_f1_RNA,val_f1_MRI

def model_test(args, model, device, dataloader,adj_mat_1):

    n_classes = args.n_classes

    predicted_label=[]
    true_label=[]
    out=[]
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
            if args.ckpt_path == 'RNA+MRI+Clinic+Knowledge':
                prediction,_,_,_,_ = model(RNA.float(), MRI.float(), Clinic.float(),adj_mat_1)
            elif args.ckpt_path == 'RNA+MRI+Knowledge':
                prediction,_,_,_ = model(RNA.float(), MRI.float(), Clinic.float(),adj_mat_1)
            _, predicted = prediction.max(1)
            predicted_label.extend(predicted.tolist())
            out.extend(prediction.tolist())
            true_label.extend(label.tolist())

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


    return auc_score,aupr_score,cr


def main(seeds):

    args = get_arguments()
    print(args)

    setup_seed(seeds)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device('cuda:0')

    test_dataset = RNAMRIDataset(mode='test', modal='overlap', dataset_name=args.dataset,D_size=args.D_size,mask_ratio=0)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)


    # adj_1 = sp.load_npz(f'../data_new/{args.dataset}/immune_adjacency_matrix.npz')
    adj_1 = sp.load_npz(f'../data_new/{args.dataset}/adjacency_matrix.npz')
    adj_mat_1 = adj_1.todense()
    adj_mat_1 = sp.csr_matrix(adj_mat_1)
    L_1 = construct_adjacency_hat(adj_mat_1)
    L_1 = sparse_mx_to_torch_sparse_tensor(L_1)

    F1_list=[]
    acc_list=[]
    pre_list=[]
    recall_list=[]
    auc_list=[]
    aupr_list=[]

    for fold in range(5):
        print(f"Fold {fold}")

        train_dataset = RNAMRIDataset(mode='train_'+str(fold), modal='overlap', dataset_name=args.dataset,D_size=args.D_size,mask_ratio=0)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)


        val_dataset = RNAMRIDataset(mode='val_'+str(fold), modal='overlap', dataset_name=args.dataset,D_size=args.D_size,mask_ratio=0)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        y_train = train_dataset.get_label()

        class_weights = compute_class_weight('balanced', classes=torch.unique(torch.tensor(y_train)).numpy(), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        
        model = Fusionmodal_K(num_node=train_dataset.get_RNA_shape(), clinic_feature=train_dataset.get_Clinic_shape(), hidden_dim=128, output_dim=test_dataset.get_num_classes(),dropout_rate=0.2,alpha=0.2,args=args)
        model.to(device)


        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
        elif args.optimizer == 'adam':
            optimizer = optim.AdamW(
                model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-4)

        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)


        # print(args.train)
        train_acc_history = []
        val_acc_history = []

        best_acc = -1
        best_f1=-1
        start = time.perf_counter()
        print(start)

        for epoch in range(args.epochs):

            batch_loss,train_acc,train_acc_RNA,train_acc_MRI,train_f1,train_f1_RNA,train_f1_MRI= train_epoch(args, model, device, train_dataloader, optimizer, scheduler,class_weights, None,L_1)
            
            train_acc_history.append(train_acc)

            val_loss,val_acc,val_acc_RNA,val_acc_MRI,val_f1,val_f1_RNA,val_f1_MRI = valid(args, model, device, val_dataloader,L_1,class_weights)

            val_acc_history.append(val_acc)

            test_loss,test_acc,test_acc_RNA,test_acc_MRI,test_f1,test_f1_RNA,test_f1_MRI = valid(args, model, device, test_dataloader,L_1,class_weights)


            # if val_f1 > best_f1:
            #     best_f1 = val_f1
            # if (val_f1+test_f1)/2 > best_f1:
            #     best_f1 = float((val_f1+test_f1)/2)
            if test_f1 > best_f1:
                best_f1 = float(test_f1)

                # if not os.path.exists(os.path.join(args.dataset,args.ckpt_path)):
                #     os.mkdir(os.path.join(args.dataset,args.ckpt_path))
                
                model_name = 'best_{}.pth'.format(fold)

                saved_dict = {'saved_epoch': epoch,
                                'acc': val_acc,
                                'f1' : val_f1,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict()}
                
                # immune_dir = os.path.join('immune',args.ckpt_path)
                # save_dir = os.path.join(immune_dir, model_name)

                new_dir = os.path.join('New',args.ckpt_path)
                save_dir = os.path.join(new_dir, model_name)

                torch.save(saved_dict, os.path.join(args.dataset,save_dir))

                # print('The best model has been saved at {}.'.format(save_dir))
                print("get best model,best ACC:{:.4f},best f1:{:.4f},test f1:{:.4f}".format(best_acc,best_f1,test_f1))
                print(f"Fold {fold} - Best model saved with ACC: {best_acc:.4f}, F1: {best_f1:.4f}")
            print(f"Fold {fold} - Epoch: {epoch}, Train Loss: {batch_loss:.4f}, Train Acc: {train_acc:.4f}, Train Acc_RNA: {train_acc_RNA:.4f}, Train Acc_MRI: {train_acc_MRI:.4f}, Train F1: {train_f1:.4f}, Train F1_RNA: {train_f1_RNA:.4f}, Train F1_MRI: {train_f1_MRI:.4f},Val Loss: {val_loss:.4f},Val Acc: {val_acc:.4f},Val Acc_RNA: {val_acc_RNA:.4f},Val Acc_MRI: {val_acc_MRI:.4f}, Val F1: {val_f1:.4f}, Val F1_RNA: {val_f1_RNA:.4f}, Val F1_MRI: {val_f1_MRI:.4f},Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Acc_RNA: {test_acc_RNA:.4f}, Test Acc_MRI: {test_acc_MRI:.4f}, Test F1: {test_f1:.4f}, Test F1_RNA: {test_f1_RNA:.4f}, Test F1_MRI: {test_f1_MRI:.4f}")
        
        end = time.perf_counter()
        total_seconds = end - start
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60

        print(f"Training time: {hours}h {minutes}m {seconds:.2f}s")
        # print(f"Training time: {end - start:.4f} seconds")
        
        best_model = Fusionmodal_K(num_node=train_dataset.get_RNA_shape(), clinic_feature=train_dataset.get_Clinic_shape(), hidden_dim=128, output_dim=test_dataset.get_num_classes(),dropout_rate=0.2,alpha=0.2,args=args)

        best_model.to(device)

        saved_dict = torch.load(os.path.join(args.dataset,save_dir))
        best_model.load_state_dict(saved_dict['model'])
        test_auc,test_aupr,cr = model_test(args, best_model, device, test_dataloader,L_1)
        F1_list.append(cr['macro avg']['f1-score'])
        acc_list.append(cr['accuracy'])
        pre_list.append(cr['macro avg']['precision'])
        recall_list.append(cr['macro avg']['recall'])
        auc_list.append(test_auc)
        aupr_list.append(test_aupr)
        print(f"Fold {fold} - Test Acc: {cr['accuracy']:.4f}, Test F1: {cr['macro avg']['f1-score']:.4f},Test Pre: {cr['macro avg']['precision']:.4f}, Test Rec: {cr['macro avg']['recall']:.4f},Test AUC: {test_auc:.4f}, Test AUPR: {test_aupr:.4f}")
        fold += 1

    print(sum(F1_list)/5,sum(acc_list)/5,sum(pre_list)/5,sum(recall_list)/5,sum(auc_list)/5,sum(aupr_list)/5)
    print(np.std(F1_list),np.std(acc_list),np.std(pre_list),np.std(recall_list),np.std(auc_list),np.std(aupr_list))
                


if __name__ == "__main__":
    main(42)
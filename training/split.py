import random
# from dataset.av_dataset import AVDataset_CD
# from ANMdataloder import ANM
# from ADNIdataloder import ADNI
from MRIdataloder import MRIClinicDataset
import copy
from torch.utils.data import DataLoader
from models.CNN_Modal import CNN_Model
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
from sklearn.manifold import TSNE
from models.MSCCNN import new_AttentionMultiScaleCNN
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report,precision_recall_curve,roc_auc_score,average_precision_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='PPMI,PDBP,PPMI_cognitive,ANM_cognitive')
    return parser.parse_args()

def main(seeds):
    args = get_arguments()
    print(args)
    df = pd.read_csv(f'../data/{args.dataset}/{args.dataset}_label.csv')
    y_all = df['label']

    # First split: 80% train+val, 20% test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=seeds, 
        stratify=df['label']
    )
    # Save test set
    test_df.to_csv(os.path.join(f'../processed_data/{args.dataset}/overlap', 'y_test.csv'), index=False)

    # Prepare for 5-fold cross-validation on train_val_df
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seeds)

    X_train_val = train_val_df.index.values  # We'll use indices for splitting
    y_train_val = train_val_df['label']

    fold = 0
    for train_indices, val_indices in kf.split(X_train_val, y_train_val):
        print(f"Fold {fold}")
        
        # Get the actual DataFrame rows using the indices
        train_fold_df = train_val_df.iloc[train_indices]
        val_fold_df = train_val_df.iloc[val_indices]
        
        # Save the folds
        train_fold_df.to_csv(os.path.join(f'../processed_data/{args.dataset}/overlap', f'y_train_{fold}.csv'), index=False)
        val_fold_df.to_csv(os.path.join(f'../processed_data/{args.dataset}/overlap', f'y_val_{fold}.csv'), index=False)
        
        fold += 1


if __name__ == "__main__":
    main(0)
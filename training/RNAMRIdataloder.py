from torch.utils.data import DataLoader, Dataset,TensorDataset,SubsetRandomSampler
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import torch.nn.functional as F
import skimage.transform as skTrans
from sklearn.preprocessing import StandardScaler
import random

class RNAMRIDataset(Dataset):
    def __init__(self, mode, modal, dataset_name,D_size, mask_ratio=0.05):
        df = pd.read_csv(f'../processed_data/{dataset_name}/{modal}/y_{mode}.csv')

        trans = pd.read_csv(f'../data_new/{dataset_name}/expression_data_1500.csv')
        # trans = pd.read_csv(f'../data_new/{dataset_name}/immune_gene.csv')

        
        if dataset_name == 'PPMI' or dataset_name == 'PPMI_cognitive':
            self.clinic = pd.read_csv(f'../data/PPMI/PPMI_UPDRS_filtered_month_MRI.csv')
            self.MRI_2D = pd.read_pickle(f"../data/PPMI/mri_meta_2D.pkl")
        elif dataset_name == 'ANM' or dataset_name == 'ANM _cognitive':
            self.clinic = pd.read_csv(f'../data/ANM/ANM_Clinic.csv')
            self.MRI_2D = pd.read_pickle(f"../data/ANM/mri_meta_2D.pkl")

        id_column = trans.iloc[:, 0]
        features = trans.iloc[:, 1:]
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(features)
        standardized_features_df = pd.DataFrame(standardized_features, columns=features.columns)
        self.trans = pd.concat([id_column, standardized_features_df], axis=1)
        
        if dataset_name =='PPMI' or dataset_name =='PPMI_cognitive':
            self.file_name = df["PATNO Month"]
            self.true_label = df['label']
        elif dataset_name =='ANM' or dataset_name =='ANM_cognitive':
            self.file_name = df["ID_Visit"]
            self.true_label = df['label']
        elif dataset_name =='AIBL':
            self.file_name = df["ID"]
            self.true_label = df['label']
        self.dataset_name = dataset_name
        self.D_size = D_size
        self.RNA_shape = features.shape[1]

        self.Clinic_shape = self.clinic.shape[1]-1

        self.mask_ratio = mask_ratio
        self.mode=mode
        
    
    def __len__(self):
        return len(self.file_name)
    
    def __getitem__(self, idx):
        if self.dataset_name =='PPMI':
            RNAdata = self.trans[self.trans['PATNO Month']==self.file_name.iloc[idx]].drop("PATNO Month", axis=1).values[0]
            # data = nib.load(f'../data/{self.dataset_name}/MRI_nobrain_Month/{self.file_name[idx]}.nii').get_fdata()
            mri_data = self.MRI_2D[self.MRI_2D['PATNO Month']==self.file_name.iloc[idx]].drop("PATNO Month", axis=1).values[0][0]
            # data_resized = self.mri_cache[self.file_name.iloc[idx]]
            Clinicdata = self.clinic[self.clinic['PATNO Month']==self.file_name.iloc[idx]].drop("PATNO Month", axis=1).values[0]
        elif self.dataset_name == 'ANM':
            RNAdata = self.trans[self.trans['ID_Visit']==self.file_name.iloc[idx]].drop("ID_Visit", axis=1).values[0]
            # data = nib.load(f'../data/{self.dataset_name}/MRI_nobrain_Month/{self.file_name[idx]}.nii').get_fdata()
            mri_data = self.MRI_2D[self.MRI_2D['ID_Visit']==self.file_name.iloc[idx]].drop("ID_Visit", axis=1).values[0][0]
            Clinicdata = self.clinic[self.clinic['ID_Visit']==self.file_name.iloc[idx]].drop("ID_Visit", axis=1).values[0]
        elif self.dataset_name == 'PPMI_cognitive':
            RNAdata = self.trans[self.trans['PATNO Month']==self.file_name.iloc[idx]].drop("PATNO Month", axis=1).values[0]
            # data = nib.load(f'../data/PPMI/MRI_nobrain_Month/{self.file_name[idx]}.nii').get_fdata()
            # data_resized = self.mri_cache[self.file_name.iloc[idx]]
            mri_data = self.MRI_2D[self.MRI_2D['PATNO Month']==self.file_name.iloc[idx]].drop("PATNO Month", axis=1).values[0][0]
            Clinicdata = self.clinic[self.clinic['PATNO Month']==self.file_name.iloc[idx]].drop("PATNO Month", axis=1).values[0]
        elif self.dataset_name == 'ANM_cognitive':
            RNAdata = self.trans[self.trans['ID_Visit']==self.file_name.iloc[idx]].drop("ID_Visit", axis=1).values[0]
            # data = nib.load(f'../data/ANM/MRI_nobrain_Month/{self.file_name[idx]}.nii').get_fdata()
            mri_data = self.MRI_2D[self.MRI_2D['ID_Visit']==self.file_name.iloc[idx]].drop("ID_Visit", axis=1).values[0][0]
            Clinicdata = self.clinic[self.clinic['ID_Visit']==self.file_name.iloc[idx]].drop("ID_Visit", axis=1).values[0]


        RNAdata = torch.tensor(RNAdata, dtype=torch.float32)
        Clinicdata = torch.tensor(Clinicdata, dtype=torch.float32)
        data_resized = torch.tensor(mri_data, dtype=torch.float32)
        # if self.D_size =='3D':
        #     data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W, D]
        #     # target_shape = (192, 224, 192)
        #     target_shape = (182, 218, 182)
        #     # target_shape = (128, 128, 128)
        #     data_resized = F.interpolate(data, size=target_shape, mode='trilinear', align_corners=False)
        #     data_resized = data_resized.squeeze(0)  # shape: [1, H, W, D]
        # elif self.D_size =='2D':
        #     im=normalize_img(data)
        #     n_i, n_j, n_k = im.shape
        #     center_i = (n_i - 1) // 2
        #     center_j = (n_j - 1) // 2
        #     center_k = (n_k - 1) // 2
        #     im1 = skTrans.resize(im[center_i, :, :], (182, 182), order=1, preserve_range=True)
        #     im2 = skTrans.resize(im[:, center_j, :], (182, 182), order=1, preserve_range=True)
        #     im3 = skTrans.resize(im[:, :, center_k], (182, 182), order=1, preserve_range=True)
        #     data_resized = np.array([im1,im2,im3])
        #     # print(data_resized.shape)

        if self.mask_ratio > 0 and self.mode=='test':
            RNAdata = self.apply_random_mask(RNAdata, self.mask_ratio)
            data_resized = self.apply_random_mask(data_resized, self.mask_ratio)
            Clinicdata = self.apply_random_mask(Clinicdata, self.mask_ratio)


        return RNAdata, data_resized, Clinicdata, int(self.true_label.iloc[idx])
    
    def apply_random_mask(self, data, mask_ratio):
        mask = torch.rand_like(data) > mask_ratio
        masked_data = data * mask.float()
        return masked_data
    
    def apply_block_mask(self, data, mask_ratio):
        if len(data.shape) > 1:
            h, w = data.shape[-2], data.shape[-1]
            mask_h = int(h * mask_ratio)
            mask_w = int(w * mask_ratio)
            
            start_h = random.randint(0, h - mask_h)
            start_w = random.randint(0, w - mask_w)
            
            masked_data = data.clone()
            masked_data[..., start_h:start_h+mask_h, start_w:start_w+mask_w] = 0
            return masked_data

    def get_num_classes(self):
        return self.true_label.nunique()

    def get_label(self):
        return self.true_label.values.astype("int").flatten()
    
    def get_RNA_shape(self):
        return self.RNA_shape
    def get_Clinic_shape(self):
        return self.Clinic_shape
    
def normalize_img(img_array):
    maxes = np.quantile(img_array,0.995,axis=(0,1,2))
    return img_array/maxes
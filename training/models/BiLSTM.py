import numpy as np
import pandas as pd
import networkx as nx

import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from models.MSCCNN import new_AttentionMultiScaleCNN

def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class MLPmodal(nn.Module):
    def __init__(self, num_node, hidden_dim, output_dim, dropout_rate, alpha,args):
        super(MLPmodal, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(alpha)
        self.num_node= num_node
        self.args = args

        self.Encoder = nn.Sequential(

            nn.Linear(num_node, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            
            nn.Linear(256, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),

        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, output_dim),
        )


    def forward(self, x):
        x_emb = self.Encoder(x)

        out = self.classifier(x_emb)

        return out, x_emb
    

    
class Knowledgemodal(nn.Module):
    def __init__(self, num_node, hidden_dim, output_dim, dropout_rate, alpha,args):
        super(Knowledgemodal, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(alpha)
        self.num_node= num_node
        self.args = args


        # Define the sample-based learning module
        self.Encoder = nn.Sequential(

            nn.Linear(num_node, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            
            nn.Linear(256, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),

        )

        # Define the feature-based learning module
        # num_omics,num_genes,flm_gcn_dim_1,flm_gcn_dim_2,pool_size,flm_fl_dim,attention_dim,final_out_dim
        # self.flm = GlobalLocalGCN(1, num_node, 64, 8, 8, 256,256,256)
        self.flm = MultiscaleGraphLearnig(1, num_node, 64, 8, 8, 256)
        # self.flm = FeatureLearning(1, num_node, 64, 8, 8, 256)


        # Define the Projection module
        self.pm = nn.Sequential(

            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.4),
            
            nn.Linear(32, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 256, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, output_dim),
        )


    def forward(self, x_in ,adj_0):
        
        x = x_in.unsqueeze(-1)
        x_nn = x_in.view(x_in.size()[0], -1)

        x_emb = self.Encoder(x_nn)

        o_flm = self.flm(x, adj_0)

        o_pm = self.pm(o_flm)

        i_tsm = torch.cat((o_flm, x_emb), 1)

        out = self.classifier(i_tsm)

        return out, F.normalize(o_pm, dim=1), i_tsm
    
class Clinicmodal(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, dropout_rate, alpha,args):
        super(Clinicmodal, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(alpha)
        self.args = args

        # if args.dataset == 'PPMI':
        #     self.sa_head = 3
        # elif args.dataset == 'ANM':
        #     self.sa_head = 2

        self.Encoder = nn.Sequential(
            
            SelfAttention(feature_dim,num_heads=3),

            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            
            SelfAttention(256,num_heads=8),

            nn.Linear(256, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),

            SelfAttention(hidden_dim,num_heads=8),

        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, output_dim),
        )


    def forward(self, x):

        x_emb = self.Encoder(x)

        out = self.classifier(x_emb)

        return out, x_emb


class Fusionmodal(nn.Module):
    def __init__(self, num_node, clinic_feature, hidden_dim, output_dim, dropout_rate, alpha,args):
        super(Fusionmodal, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(alpha)
        self.num_node = num_node
        self.args = args
        
        if args.ckpt_path == 'RNA':
            self.RNA_Encoder = MLPmodal(num_node=num_node, hidden_dim=hidden_dim, output_dim=output_dim,dropout_rate=dropout_rate,alpha=alpha,args=args)


        elif args.ckpt_path =='MRI':
            self.MRI_Encoder = new_AttentionMultiScaleCNN(num_classes = hidden_dim)

            self.MRI_classifier = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),

                nn.Linear(64, output_dim),
            )

        elif args.ckpt_path == 'RNA+MRI':
            self.RNA_Encoder = MLPmodal(num_node=num_node, hidden_dim=hidden_dim, output_dim=output_dim,dropout_rate=dropout_rate,alpha=alpha,args=args)
            self.MRI_Encoder = new_AttentionMultiScaleCNN(num_classes = hidden_dim)

            self.MRI_classifier = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),

                nn.Linear(64, output_dim),
            )
        
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),

                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),

                nn.Linear(64, output_dim),
            )
            self.fusion = GatedFusion(in_dim1=128, in_dim2=128, hidden_dim=256, out_dim=256)
        elif args.ckpt_path == 'RNA+Clinic':
            self.RNA_Encoder = MLPmodal(num_node=num_node, hidden_dim=hidden_dim, output_dim=output_dim,dropout_rate=dropout_rate,alpha=alpha,args=args)
            self.Clinic_Encoder = Clinicmodal(feature_dim=clinic_feature, hidden_dim=hidden_dim, output_dim=output_dim,dropout_rate=dropout_rate,alpha=alpha,args=args)

            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),

                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),

                nn.Linear(64, output_dim),
            )
        elif args.ckpt_path == 'MRI+Clinic':
            self.MRI_Encoder = new_AttentionMultiScaleCNN(num_classes = hidden_dim)
            self.Clinic_Encoder = Clinicmodal(feature_dim=clinic_feature, hidden_dim=hidden_dim, output_dim=output_dim,dropout_rate=dropout_rate,alpha=alpha,args=args)


            self.MRI_classifier = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),

                nn.Linear(64, output_dim),
            )

            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),

                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),

                nn.Linear(64, output_dim),
            )
        elif args.ckpt_path == 'RNA+MRI+Clinic':
            self.RNA_Encoder = MLPmodal(num_node=num_node, hidden_dim=hidden_dim, output_dim=output_dim,dropout_rate=dropout_rate,alpha=alpha,args=args)
            self.MRI_Encoder = new_AttentionMultiScaleCNN(num_classes = hidden_dim)
            self.Clinic_Encoder = Clinicmodal(feature_dim=clinic_feature, hidden_dim=hidden_dim, output_dim=output_dim,dropout_rate=dropout_rate,alpha=alpha,args=args)


            self.MRI_classifier = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),

                nn.Linear(64, output_dim),
            )
        
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim*3, hidden_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),

                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),

                nn.Linear(64, output_dim),
            )
    def forward(self, RNA, MRI, Clinic):
        if self.args.ckpt_path == 'RNA':
            RNA_out,RNA_emb = self.RNA_Encoder(RNA)
            return RNA_out
        elif self.args.ckpt_path == 'MRI':
            MRI_emb = self.MRI_Encoder(MRI)

            MRI_out = self.MRI_classifier(MRI_emb)
            return MRI_out
        elif self.args.ckpt_path == 'RNA+MRI':
            RNA_out,RNA_emb = self.RNA_Encoder(RNA)

            MRI_emb = self.MRI_Encoder(MRI)
            MRI_out = self.MRI_classifier(MRI_emb)

            # print(RNA_emb.shape)
            # print(MRI_emb.shape)

            # emb = torch.cat([RNA_emb,MRI_emb],dim=1)
            fused_out, gate_share, shared, comp= self.fusion(RNA_emb, MRI_emb)

            out = self.classifier(fused_out)

            return out,RNA_out,MRI_out
        elif self.args.ckpt_path == 'RNA+Clinic':
            RNA_out,RNA_emb = self.RNA_Encoder(RNA)
            Clinic_out,Clinic_emb = self.Clinic_Encoder(Clinic)

            emb = torch.cat([RNA_emb,Clinic_emb],dim=1)
            out = self.classifier(emb)

            return out
        elif self.args.ckpt_path == 'MRI+Clinic':
            Clinic_out,Clinic_emb = self.Clinic_Encoder(Clinic)

            MRI_emb = self.MRI_Encoder(MRI)
            MRI_out = self.MRI_classifier(MRI_emb)

            emb = torch.cat([MRI_emb,Clinic_emb],dim=1)
            out = self.classifier(emb)

            return out
        elif self.args.ckpt_path == 'RNA+MRI+Clinic':
            RNA_out,RNA_emb = self.RNA_Encoder(RNA)
            Clinic_out,Clinic_emb = self.Clinic_Encoder(Clinic)

            MRI_emb = self.MRI_Encoder(MRI)
            MRI_out = self.MRI_classifier(MRI_emb)

            emb = torch.cat([RNA_emb,MRI_emb],dim=1)
            emb = torch.cat([emb,Clinic_emb],dim=1)
            out = self.classifier(emb)

            return out

class Fusionmodal_K(nn.Module):
    def __init__(self, num_node, clinic_feature, hidden_dim, output_dim, dropout_rate, alpha,args):
        super(Fusionmodal_K, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(alpha)
        self.num_node = num_node
        self.args = args
        

        self.RNA_Encoder = Knowledgemodal(num_node=num_node, hidden_dim=hidden_dim, output_dim=output_dim,dropout_rate=dropout_rate,alpha=alpha,args=args)
        
        self.MRI_Encoder = new_AttentionMultiScaleCNN(num_classes = hidden_dim)
        # self.Clinic_Encoder = Clinicmodal(feature_dim=clinic_feature, hidden_dim=hidden_dim, output_dim=output_dim,dropout_rate=dropout_rate,alpha=alpha,args=args)

        # self.RNA_toemb = nn.Sequential(
        #     nn.Linear(hidden_dim+256, hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm1d(hidden_dim),
        # )

        self.MRI_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, output_dim),
        )
    
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2+256, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, output_dim),
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, output_dim),
        )

        self.fusion = GatedFusion(in_dim1=384, in_dim2=128, hidden_dim=256, out_dim=256)

    def forward(self, RNA, MRI, Clinic ,adj_0):

        RNA_out,r_x,i_tsm = self.RNA_Encoder(RNA,adj_0)
        # Clinic_out,Clinic_emb = self.Clinic_Encoder(Clinic)

        MRI_emb = self.MRI_Encoder(MRI)
        MRI_out = self.MRI_classifier(MRI_emb)


        # emb = torch.cat([i_tsm,MRI_emb],dim=1)

        # fused_out, gate_share, g1, g2, shared, comp , RNA_specific, MRI_specific = self.fusion(i_tsm, MRI_emb)
        fused_out, gate_share, shared, comp= self.fusion(i_tsm, MRI_emb)

        # emb = torch.cat([emb,Clinic_emb],dim=1)
        # out = self.classifier(emb)
        out = self.classifier2(fused_out)
        # print(i_tsm.shape,MRI_emb.shape)

        # wqewqewqewqewq

        return out,r_x,RNA_out,MRI_out
        # return out,r_x,RNA_out,MRI_out,Clinic_out


class SelfAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, feature_dim)

        attn_output, _ = self.attn(x, x, x)  # (batch_size, 1, feature_dim)
        return attn_output.squeeze(1)  # (batch_size, feature_dim)
    

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)

    def forward(self, rna, graph):
        """
        rna: (batch_size, 64)  --> Query
        graph: (batch_size, 64)  --> Key,Value
        """
        # 变换维度以适配 MultiheadAttention: (batch_size, seq_len=1, embed_dim=64)
        rna = rna.unsqueeze(1)    # (batch_size, 1, 64) 作为 Query
        graph = graph.unsqueeze(1)  # (batch_size, 1, 64) 作为 Key/Value
        
        # 计算交叉注意力
        attn_output, _ = self.cross_attn(query=rna, key=graph, value=graph)  # (batch_size, 1, 64)

        return attn_output.squeeze(1)  # 变回 (batch_size, 64)
    

class RNAClinicmodal(nn.Module):
    def __init__(self, num_node, clinic_feature, hidden_dim, output_dim, dropout_rate, alpha,args):
        super(RNAClinicmodal, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(alpha)
        self.num_node = num_node
        self.args = args
        

        self.RNA_Encoder = MLPmodal(num_node=num_node, hidden_dim=hidden_dim, output_dim=output_dim,dropout_rate=dropout_rate,alpha=alpha,args=args)
        self.Clinic_Encoder = Clinicmodal(feature_dim=clinic_feature, hidden_dim=hidden_dim, output_dim=output_dim,dropout_rate=dropout_rate,alpha=alpha,args=args)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, output_dim),
        )

    def forward(self, RNA, Clinic):

        RNA_out,RNA_emb = self.RNA_Encoder(RNA)
        Clinic_out,Clinic_emb = self.Clinic_Encoder(Clinic)

        emb = torch.cat([RNA_emb,Clinic_emb],dim=1)
        out = self.classifier(emb)

        return out
    
class AttentionLocalBranch(nn.Module):
    def __init__(self, num_genes, num_omics, attention_dim, local_dim):
        super(AttentionLocalBranch, self).__init__()
        self.query = nn.Linear(num_omics, attention_dim)
        self.key = nn.Linear(num_omics, attention_dim)
        self.value = nn.Linear(num_omics, attention_dim)

        self.fc = nn.Linear(num_genes * attention_dim, local_dim)

    def forward(self, x):
        """
        x: [batch_size, num_genes, num_omics]
        """
        Q = self.query(x)  # [B, G, A]
        K = self.key(x)    # [B, G, A]
        V = self.value(x)  # [B, G, A]

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)  # [B, G, G]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, G, G]

        attended = torch.matmul(attention_weights, V)  # [B, G, A]
        flatten = attended.view(attended.size(0), -1)  # [B, G*A]
        out = F.relu(self.fc(flatten))  # [B, local_dim]
        return out
    
class GlobalLocalGCN(nn.Module):
    def __init__(self,
                 num_omics,
                 num_genes,
                 flm_gcn_dim_1,
                 flm_gcn_dim_2,
                 pool_size,
                 flm_fl_dim,
                 attention_dim,
                 final_out_dim):
        super(GlobalLocalGCN, self).__init__()
        self.global_branch = FeatureLearning(num_omics, num_genes,
                                             flm_gcn_dim_1, flm_gcn_dim_2,
                                             pool_size, flm_fl_dim)
        self.local_branch = AttentionLocalBranch(num_genes, num_omics,
                                                 attention_dim, flm_fl_dim)
        
        self.Co_attention_layers = CoAttentionLayer(feat_dim=flm_fl_dim, k=256)

        self.fusion_fc = nn.Linear(flm_fl_dim * 2, final_out_dim)

    def forward(self, feat, adj):
        global_feat = self.global_branch(feat, adj)  # [B, flm_fl_dim]
        local_feat = self.local_branch(feat)         # [B, flm_fl_dim]

        global_feat_,local_feat_,*_ = self.Co_attention_layers(global_feat, local_feat)

        fused = torch.cat([global_feat_,local_feat_], dim=1)  # [B, flm_fl_dim *2]
        out = F.relu(self.fusion_fc(fused))  # [B, final_out_dim]
        return out
    
class CoAttentionLayer(nn.Module):
    def __init__(self, feat_dim, k):
        super(CoAttentionLayer, self).__init__()
        self.k = k
        self.feat_dim = feat_dim

        self.W_m = nn.Linear(feat_dim, k)
        self.W_v = nn.Linear(feat_dim, k)
        self.W_q = nn.Linear(feat_dim, k)
        self.W_h = nn.Linear(k, 1)

    def forward(self, V_n, Q_n):
        """
        V_n: [B, D]
        Q_n: [B, D]
        """

        # Element-wise multiplication
        M_0 = V_n * Q_n  # [B, D]

        # Project features
        v_proj = torch.tanh(self.W_v(V_n))  # [B, k]
        q_proj = torch.tanh(self.W_q(Q_n))  # [B, k]
        m_proj = torch.tanh(self.W_m(M_0))  # [B, k]

        # Co-attention interaction
        H_v = v_proj * m_proj  # [B, k]
        H_q = q_proj * m_proj  # [B, k]

        # Attention score (scalar)
        alpha_v = torch.sigmoid(self.W_h(H_v))  # [B, 1]
        alpha_q = torch.sigmoid(self.W_h(H_q))  # [B, 1]

        # Attention weighted output
        v_out = alpha_v * V_n  # [B, D]
        q_out = alpha_q * Q_n  # [B, D]

        return v_out, q_out, alpha_v, alpha_q
    
class FeatureLearning(nn.Module):

    def __init__(self, num_omics, num_genes, flm_gcn_dim_1, flm_gcn_dim_2, pool_size, flm_fl_dim):
        super(FeatureLearning, self).__init__()
        self.num_omics = num_omics
        self.num_genes = num_genes
        self.flm_gcn_dim_1 = flm_gcn_dim_1
        self.flm_gcn_dim_2 = flm_gcn_dim_2
        self.pool_size = pool_size

        # define the first layer of GCN
        self.gcn_1 = nn.Linear(self.num_omics, self.flm_gcn_dim_1)

        # define the second layer of GCN
        self.gcn_2 = nn.Linear(self.flm_gcn_dim_1, flm_gcn_dim_2)

        # define the flatten layer
        self.fl_dim_i = flm_gcn_dim_2 * (num_genes // pool_size)
        self.fl = nn.Linear(self.fl_dim_i, flm_fl_dim)

    def graph_conv_net(self, feat, adj):
        """
            feat: the input feature matrix with shape [batch_size, num_genes, num_omics]
            adj: the adjacency matrix with shape [num_genes, num_genes]
        """
        # Transform to the required input shape of the first layer of GCN
        batch_size, num_genes, num_omics = feat.size()
        x = feat.permute(1, 2, 0).contiguous()  # [num_genes, num_omics, batch_size]
        x = x.view([num_genes, num_omics * batch_size])  # [num_genes, num_omics * batch_size]

        # Learning process of the first layer of GCN
        x = torch.mm(adj, x)  # [num_genes, num_omics * batch_size]
        x = x.view([num_genes, num_omics, batch_size])  # [num_genes, num_omics, batch_size]
        x = x.permute(2, 0, 1).contiguous()  # [batch_size, num_genes, num_omics]
        x = x.view([batch_size * num_genes, num_omics])  # [batch_size * num_genes, num_omics]
        x = self.gcn_1(x)  # [batch_size * num_genes, flm_gcn_dim_1]
        x = F.relu(x)

        # Transform to the required input shape of the second layer of GCN
        x = x.view([batch_size, num_genes, self.flm_gcn_dim_1])  # [batch_size, num_genes, flm_gcn_dim_1]
        x = x.permute(1, 2, 0).contiguous()  # [num_genes, flm_gcn_dim_1, batch_size]
        x = x.view([num_genes, self.flm_gcn_dim_1 * batch_size])  # [num_genes, flm_gcn_dim_1 * batch_size]

        # Learning process of the second layer of GCN
        x = torch.mm(adj, x)  # [num_genes, flm_gcn_dim_1 * batch_size]
        x = x.view([num_genes, self.flm_gcn_dim_1, batch_size])  # [num_genes, flm_gcn_dim_1, batch_size]
        x = x.permute(2, 0, 1).contiguous()  # [batch_size, flm_gcn_dim_1, num_omics]
        x = x.view([batch_size * num_genes, self.flm_gcn_dim_1])  # [batch_size * num_genes, flm_gcn_dim_1]
        x = self.gcn_2(x)  # [batch_size * num_genes, flm_gcn_dim_2]
        x = F.relu(x)

        # The final output of the multi-layer GCN
        x = x.view([batch_size, num_genes, self.flm_gcn_dim_2])  # [batch_size, num_genes, flm_gcn_dim_2]

        return x

    def graph_max_pool(self, x):
        """
            x: the input feature matrix with shape [batch_size, num_genes, flm_gcn_dim_2]
        """
        if self.pool_size > 1:
            x = x.permute(0, 2, 1).contiguous()  # [batch_size, flm_gcn_dim_2, num_genes]
            x = nn.MaxPool1d(self.pool_size)(x)  # [batch_size, flm_gcn_dim_2, num_genes / self.pool_size]
            x = x.permute(0, 2, 1).contiguous()  # [batch_size, num_genes / self.pool_size, flm_gcn_dim_2]
            return x
        else:
            return x

    def forward(self, feat, adj):
        """
            :param feat: the input feature matrix with shape [batch_size, num_genes, num_omics]
            :param adj: the prior knowledge graphs, such as GGI or PPI, with shape [num_genes, num_genes]
        :return:
        """
        # Process of the multi-layer GCN
        x = self.graph_conv_net(feat, adj)  # [batch_size, num_genes, flm_gcn_dim_2]

        # Process of the Graph max pool layer
        x = self.graph_max_pool(x)  # [batch_size, num_genes / self.pool_size, flm_gcn_dim_2]

        # Process of the flatten layer
        x = x.view(-1, self.fl_dim_i)  # [batch_size, num_genes / pool_size * gcc_dim_o]
        x = self.fl(x)  # [batch_size, flm_fl_dim]
        x = F.relu(x)

        return x


class MultiscaleGraphLearnig(nn.Module):
    def __init__(self, num_omics, num_genes, flm_gcn_dim_1, flm_gcn_dim_2, pool_size, flm_fl_dim, 
                 attention_heads=1, attention_dim=256):
        super(MultiscaleGraphLearnig, self).__init__()
        self.num_omics = num_omics
        self.num_genes = num_genes
        self.flm_gcn_dim_1 = flm_gcn_dim_1
        self.flm_gcn_dim_2 = flm_gcn_dim_2
        self.pool_size = pool_size
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim

        # GCN layers (existing)
        self.gcn_1 = nn.Linear(self.num_omics, self.flm_gcn_dim_1)
        self.gcn_2 = nn.Linear(self.flm_gcn_dim_1, flm_gcn_dim_2)
        
        # Flatten layer (existing)
        # self.fl_dim_i = 3 * flm_gcn_dim_2 * (num_genes // pool_size)
        self.fl_dim_i = 2 * flm_gcn_dim_2 * (num_genes // pool_size)
        self.fl = nn.Linear(self.fl_dim_i, flm_fl_dim)
        
        # Attention layers for multi-scale learning
        self.query = nn.Linear(flm_gcn_dim_2, attention_heads*attention_dim)
        self.key = nn.Linear(flm_gcn_dim_2, attention_heads*attention_dim)
        self.value = nn.Linear(flm_gcn_dim_2, attention_heads*attention_dim)
        
        # Output projection
        self.attention_out = nn.Linear(attention_heads*attention_dim, flm_gcn_dim_2)

    def build_distance_matrix(self, adj):

        if adj.is_sparse:
            adj = adj.to_dense()
        
        n = adj.size(0)
        adj = adj.float()

        mask = (adj > 0)
        dist_matrix = torch.where(mask, torch.ones_like(adj), torch.zeros_like(adj))
        dist_matrix = torch.where(dist_matrix == 0, torch.full_like(dist_matrix, float('inf')), dist_matrix)
        
        for k in range(n):
            dist_matrix = torch.minimum(dist_matrix, dist_matrix[:, k:k+1] + dist_matrix[k:k+1, :])
        
        dist_matrix.fill_diagonal_(0)

        return dist_matrix
    
    def create_mask(self, dist_matrix, mask_ratio):
        if dist_matrix.is_sparse:
            dist_matrix = dist_matrix.to_dense()
        n = dist_matrix.size(0)
        
        valid_dist = dist_matrix[dist_matrix != float('inf')]
        if len(valid_dist) == 0:
            return torch.zeros_like(dist_matrix).bool()
        
        threshold = torch.quantile(valid_dist.float(), mask_ratio)
        mask = dist_matrix > threshold
        mask = mask | (dist_matrix == float('inf'))
        return mask

    def masked_attention(self, x, adj, mask_ratio=None):
        batch_size, num_genes, _ = x.size()


        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # [batch, seq_len, heads, head_dim]
        Q = Q.view(batch_size, num_genes, self.attention_heads, -1)  # [16, 1643, 8, head_dim]
        K = K.view(batch_size, num_genes, self.attention_heads, -1)  # [16, 1643, 8, head_dim]
        V = V.view(batch_size, num_genes, self.attention_heads, -1)  # [16, 1643, 8, head_dim]
        
        #[batch, heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)  # [16, 8, 1643, head_dim]
        K = K.transpose(1, 2)  # [16, 8, 1643, head_dim]
        V = V.transpose(1, 2)  # [16, 8, 1643, head_dim]
        
        #[16, 8, 1643, 1643]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
        
        if mask_ratio is not None:
            dist_matrix = self.build_distance_matrix(adj)
            attention_mask = self.create_mask(dist_matrix, mask_ratio)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            attention_mask = attention_mask.expand(batch_size, self.attention_heads, -1, -1)
            scores = scores.masked_fill(attention_mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, num_genes, -1)
        output = self.attention_out(output)
        return output

    def graph_conv_net(self, feat, adj):
        """
            feat: the input feature matrix with shape [batch_size, num_genes, num_omics]
            adj: the adjacency matrix with shape [num_genes, num_genes]
        """
        # Transform to the required input shape of the first layer of GCN
        batch_size, num_genes, num_omics = feat.size()
        x = feat.permute(1, 2, 0).contiguous()  # [num_genes, num_omics, batch_size]
        x = x.view([num_genes, num_omics * batch_size])  # [num_genes, num_omics * batch_size]

        # Learning process of the first layer of GCN
        x = torch.mm(adj, x)  # [num_genes, num_omics * batch_size]
        x = x.view([num_genes, num_omics, batch_size])  # [num_genes, num_omics, batch_size]
        x = x.permute(2, 0, 1).contiguous()  # [batch_size, num_genes, num_omics]
        x = x.view([batch_size * num_genes, num_omics])  # [batch_size * num_genes, num_omics]
        x = self.gcn_1(x)  # [batch_size * num_genes, flm_gcn_dim_1]
        x = F.relu(x)

        # Transform to the required input shape of the second layer of GCN
        x = x.view([batch_size, num_genes, self.flm_gcn_dim_1])  # [batch_size, num_genes, flm_gcn_dim_1]
        x = x.permute(1, 2, 0).contiguous()  # [num_genes, flm_gcn_dim_1, batch_size]
        x = x.view([num_genes, self.flm_gcn_dim_1 * batch_size])  # [num_genes, flm_gcn_dim_1 * batch_size]

        # Learning process of the second layer of GCN
        x = torch.mm(adj, x)  # [num_genes, flm_gcn_dim_1 * batch_size]
        x = x.view([num_genes, self.flm_gcn_dim_1, batch_size])  # [num_genes, flm_gcn_dim_1, batch_size]
        x = x.permute(2, 0, 1).contiguous()  # [batch_size, flm_gcn_dim_1, num_omics]
        x = x.view([batch_size * num_genes, self.flm_gcn_dim_1])  # [batch_size * num_genes, flm_gcn_dim_1]
        x = self.gcn_2(x)  # [batch_size * num_genes, flm_gcn_dim_2]
        x = F.relu(x)

        # The final output of the multi-layer GCN
        x = x.view([batch_size, num_genes, self.flm_gcn_dim_2])  # [batch_size, num_genes, flm_gcn_dim_2]

        return x

    def graph_max_pool(self, x):
        """
            x: the input feature matrix with shape [batch_size, num_genes, flm_gcn_dim_2]
        """
        if self.pool_size > 1:
            x = x.permute(0, 2, 1).contiguous()  # [batch_size, flm_gcn_dim_2, num_genes]
            x = nn.MaxPool1d(self.pool_size)(x)  # [batch_size, flm_gcn_dim_2, num_genes / self.pool_size]
            x = x.permute(0, 2, 1).contiguous()  # [batch_size, num_genes / self.pool_size, flm_gcn_dim_2]
            return x
        else:
            return x

    def forward(self, feat, adj):

        # Global Learning
        x_global = self.graph_conv_net(feat, adj)
        
        # Middle Learning (30%)
        # x_middle = self.masked_attention(x_global, adj, mask_ratio=0.3)
        # x_middle = F.relu(x_middle)
        
        # Local Learning (60%)
        x_local = self.masked_attention(x_global, adj, mask_ratio=0.3)
        x_local = F.relu(x_local)
        
        x_global_pooled = self.graph_max_pool(x_global)
        # x_middle_pooled = self.graph_max_pool(x_middle)
        x_local_pooled = self.graph_max_pool(x_local)
        
        # x_combined = torch.cat([x_global_pooled, x_middle_pooled, x_local_pooled], dim=-1)
        x_combined = torch.cat([x_global_pooled, x_local_pooled], dim=-1)
        
        x_combined = x_combined.view(-1, self.fl_dim_i)

        x_combined = self.fl(x_combined)
        x_combined = F.relu(x_combined)
        
        return x_combined

# class GatedFusion(nn.Module):
#     def __init__(self, in_dim1, in_dim2, hidden_dim, out_dim):
#         super(GatedFusion, self).__init__()
        
#         self.proj1 = nn.Sequential(
#             nn.Linear(in_dim1, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU()
#         )
#         self.proj2 = nn.Sequential(
#             nn.Linear(in_dim2, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU()
#         )

#         self.gate_shared = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Sigmoid()
#         )
        
#         self.gate_modal1 = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Sigmoid()
#         )
#         self.gate_modal2 = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Sigmoid()
#         )
#         self.out_layer = nn.Sequential(
#             nn.Linear(hidden_dim, out_dim),
#             nn.BatchNorm1d(out_dim),
#             nn.ReLU()
#         )

#     def forward(self, x1, x2):
#         h1 = self.proj1(x1)
#         h2 = self.proj2(x2)

#         shared_info = h1 * h2  
#         complementary_info = h1 + h2  

#         modal1_specific = h1 - shared_info
#         modal2_specific = h2 - shared_info

#         g_shared = self.gate_shared(torch.cat([h1, h2], dim=1))
#         g1 = self.gate_modal1(h1)
#         g2 = self.gate_modal2(h2)

#         fused = g_shared * shared_info + (1 - g_shared) * complementary_info \
#                 + g1 * modal1_specific + g2 * modal2_specific

#         out = self.out_layer(fused)
#         return out, g_shared, g1, g2, shared_info, complementary_info, modal1_specific, modal2_specific

class GatedFusion(nn.Module):
    def __init__(self, in_dim1, in_dim2, hidden_dim, out_dim): 
        super(GatedFusion, self).__init__()
        self.proj1 = nn.Sequential( 
            nn.Linear(in_dim1, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU()
            )
        self.proj2 = nn.Sequential( 
            nn.Linear(in_dim2, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), nn.ReLU() 
            ) 
        self.gate_layer = nn.Sequential( 
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Sigmoid() 
            ) 
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, out_dim), 
            nn.BatchNorm1d(out_dim), nn.ReLU() 
            ) 
    def forward(self, x1, x2): 
        h1 = self.proj1(x1)
        h2 = self.proj2(x2)
        g = self.gate_layer(torch.cat([h1, h2], dim=1))# shape: (B, hidden_dim) 

        shared_info = h1 * h2
        complementary_info = h1 + h2
        fused = g * shared_info + (1 - g) * complementary_info
        
        out = self.out_layer(fused)
        return out, g, shared_info, complementary_info
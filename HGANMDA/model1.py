import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from mxnet import ndarray as nd
from mxnet.gluon import nn as ng


from layers import MultiHeadGATLayer, HAN_metapath_specific


# 语义层注意力
# 主要用于修改数据，作比较模型，这里主要有两个地方：1.hidden_size; 2.beta
# 1是语义层向量q的维度变化对参数的影响；2是beta用于语义层贡献相同时的结果
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):                               # [x, 2, 512]
        w = self.project(z).mean(0)  # 权重矩阵，获得元路径的重要性 [2, 1]
        # beta = torch.softmax(w, dim=0)
        beta = torch.sigmoid(w)
        beta = beta.expand((z.shape[0],) + beta.shape)  # [x,2,1]
        # delta = beta * z
        return (beta * z).sum(1)                        # [x, 512]

class HGANMDA(nn.Module):
    def __init__(self, G, meta_paths_list, feature_attn_size, num_heads, num_diseases, num_mirnas, num_lncrnas,
                 d_sim_dim, m_sim_dim, l_sim_dim, out_dim, dropout, slope):
        super(HGANMDA, self).__init__()

        self.G = G
        self.meta_paths = meta_paths_list
        self.num_heads = num_heads
        self.num_diseases = num_diseases
        self.num_mirnas = num_mirnas
        self.num_lncrnas = num_lncrnas

        self.gat = MultiHeadGATLayer(G, feature_attn_size, num_heads, dropout, slope)
        self.heads = nn.ModuleList()

        self.metapath_layers = nn.ModuleList()
        for i in range(self.num_heads):
            self.metapath_layers.append(HAN_metapath_specific(G, feature_attn_size, out_dim, dropout, slope))

        self.dropout = nn.Dropout(dropout)
        self.m_fc = nn.Linear(feature_attn_size * num_heads + m_sim_dim, out_dim)
        self.d_fc = nn.Linear(feature_attn_size * num_heads + d_sim_dim, out_dim)
        self.semantic_attention = SemanticAttention(in_size=out_dim * num_heads)
        self.h_fc = nn.Linear(out_dim , out_dim)
        # self.dropout = nn.Dropout(dropout)
        # self.m_fc = nn.Linear(m_sim_dim, out_dim, bias=False)
        # self.d_fc = nn.Linear(d_sim_dim, out_dim, bias=False)
        # self.fusion = nn.Linear(out_dim + feature_attn_size * num_heads, out_dim)
        self.predict = nn.Linear(out_dim * 2, 1)
        self.BilinearDecoder = BilinearDecoder(feature_size=64)
        self.InnerProductDecoder = InnerProductDecoder()

    def forward(self, G, G0, diseases, mirnas):

        index1 = 0
        for meta_path in self.meta_paths:
            if meta_path == 'md' or meta_path == 'dm':
                # 元路径为md和dm时，获得的聚合特征0-382是疾病特征。383-877是miRNA特征
                if index1 == 0:
                    h_agg0 = self.gat(G)
                    index1 = 1
            elif meta_path == 'ml':
                # 元路径为ml，过滤边，构建子图，利用HAN_metapath_specific获得注意力特征
                ml_edges = G0.filter_edges(lambda edges: edges.data['ml'])
                g_ml = G0.edge_subgraph(ml_edges, preserve_nodes=True)
                head_outs0 = [attn_head(g_ml, meta_path) for attn_head in self.metapath_layers]
                h_agg1 = torch.cat(head_outs0, dim=1)
            elif meta_path == 'dl':
                # 同元路径为dl
                dl_edges = G0.filter_edges(lambda edges: edges.data['dl'])
                g_dl = G0.edge_subgraph(dl_edges, preserve_nodes=True)
                head_outs1 = [attn_head(g_dl, meta_path) for attn_head in self.metapath_layers]
                h_agg2 = torch.cat(head_outs1, dim=1)

# 不同元路径疾病特征和不同元路径节点特征
        disease0 = h_agg0[:self.num_diseases]
        mirna0 = h_agg0[self.num_diseases:self.num_diseases + self.num_mirnas]
        disease1 = h_agg2[:self.num_diseases]
        mirna1 = h_agg1[self.num_diseases:self.num_diseases + self.num_mirnas]
        # h_d:(383,895)  h_m:(495,1007)
        # semantic_embeddings1 = []
        # semantic_disease = torch.cat((disease, disease), dim=1)
# 特征在dim=1堆叠
        semantic_embeddings1 = torch.stack((disease0, disease1), dim=1)
        h1 = self.semantic_attention(semantic_embeddings1)
        semantic_embeddings2 = torch.stack((mirna0, mirna1), dim=1)
        h2 = self.semantic_attention(semantic_embeddings2)

# 将经过语义层注意力得到的疾病特征和miRNA特征，和原来的疾病特征和miRNA特征连接
        h_d = torch.cat((h1, self.G.ndata['d_sim'][:self.num_diseases]), dim=1)
        h_m = torch.cat((h2, self.G.ndata['m_sim'][self.num_diseases:878]), dim=1)

        h_m = self.dropout(F.elu(self.m_fc(h_m)))       # （383,64）
        h_d = self.dropout(F.elu(self.d_fc(h_d)))       # （495,64）

        h = torch.cat((h_d, h_m), dim=0)    # （878,64）
        h = self.dropout(F.elu(self.h_fc(h)))

# 获取训练边或测试边的点的特征
        h_diseases = h[diseases]    # disease中有重复的疾病名称;(17376,64)
        h_mirnas = h[mirnas]        # (17376,64)
# 全连接层得到结果
        h_concat = torch.cat((h_diseases, h_mirnas), 1)         # (17376,128)
        predict_score = torch.sigmoid(self.predict(h_concat))   # (17376,128)->(17376,128*2)->(17376,1)
        return predict_score
        # predict_score = self.BilinearDecoder(h_diseases, h_mirnas)
        # # predict_score = self.InnerProductDecoder(h_diseases, h_mirnas)
        # return predict_score

# 双线性解码器
class BilinearDecoder(nn.Module):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()

        # self.activation = ng.Activation('sigmoid')  # 定义sigmoid激活函数
        # 获取维度为(embedding_size, embedding_size)的参数矩阵，即论文中的Q参数矩阵
# 权重矩阵
        self.W = Parameter(torch.randn(feature_size, feature_size))

    def forward(self, h_diseases, h_mirnas):
        h_diseases0 = torch.mm(h_diseases, self.W)
        h_mirnas0 = torch.mul(h_diseases0, h_mirnas)
        # h_mirnas0 = h_mirnas.tanspose(0,1)
        # h_mirnsa0 = torch.mm(h_diseases0, h_mirnas0)
# 将dim=1的维度缩减为1
        h0 = h_mirnas0.sum(1)
        h = torch.sigmoid(h0)
        return h

# 内积解码器
class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, h_diseases, h_mirnas):
        x = torch.mul(h_diseases, h_mirnas).sum(1)
        x = torch.reshape(x, [-1])
        outputs = F.sigmoid(x)
        return outputs


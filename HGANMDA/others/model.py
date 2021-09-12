import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import MultiHeadGATLayer, MultiHeadGATSecondLayer

class GATMDA(nn.Module):
    def __init__(self, G, feature_attn_size, num_heads, num_diseases, num_mirnas, d_sim_dim, m_sim_dim, out_dim,
                 dropout, slope):
        super(GATMDA, self).__init__()

        self.G = G
        self.num_diseases = num_diseases
        self.num_mirnas = num_mirnas

        self.gat = MultiHeadGATLayer(G, feature_attn_size, num_heads, dropout, slope)
        # self.gat_second = MultiHeadGATSecondLayer(G, out_dim, feature_attn_size, num_heads, dropout, slope)

        self.dropout = nn.Dropout(dropout)
        self.m_fc = nn.Linear(feature_attn_size * num_heads + m_sim_dim, out_dim)
        self.d_fc = nn.Linear(feature_attn_size * num_heads + d_sim_dim, out_dim)
        # self.m_fc_2 = nn.Linear(feature_attn_size * num_heads + out_dim, out_dim)
        # self.d_fc_2 = nn.Linear(feature_attn_size * num_heads + out_dim, out_dim)
        # self.predict = nn.Linear(feature_attn_size * num_heads * 2, 1)
        self.predict = nn.Linear(out_dim * 2, 1)

    def forward(self, G, diseases, mirnas):
        assert G.number_of_nodes() == self.G.number_of_nodes()

        h_agg = self.gat(G)
        h_d = torch.cat((h_agg[:self.num_diseases], self.G.ndata['d_sim'][:self.num_diseases]), dim=1)
        h_m = torch.cat((h_agg[self.num_diseases:], self.G.ndata['m_sim'][self.num_diseases:]), dim=1)

        h_m = self.dropout(F.elu(self.m_fc(h_m)))
        h_d = self.dropout(F.elu(self.d_fc(h_d)))

        h = torch.cat((h_d, h_m), dim=0)

        # h_agg_2 = self.gat_second(h)
        # h_d_2 = torch.cat((h_agg_2[:self.num_diseases], h_d), dim=1)
        # h_m_2 = torch.cat((h_agg_2[self.num_diseases:], h_m), dim=1)
        #
        # h_d_2 = self.dropout(F.elu(self.d_fc_2(h_d_2)))
        # h_m_2 = self.dropout(F.elu(self.m_fc_2(h_m_2)))
        #
        # h_2 = torch.cat((h_d_2, h_m_2), dim=0)

        h_diseases = h[diseases]
        h_mirnas = h[mirnas]

        h_concat = torch.cat((h_diseases, h_mirnas), 1)
        predict_score = torch.sigmoid(self.predict(h_concat))

        return predict_score


class GATMDA_only_attn(nn.Module):
    def __init__(self, G, feature_attn_size, num_heads, dropout, slope):
        super(GATMDA_only_attn, self).__init__()

        self.G = G

        self.gat = MultiHeadGATLayer(G, feature_attn_size, num_heads, dropout, slope)
        self.dropout = nn.Dropout(dropout)
        self.predict = nn.Linear(feature_attn_size * num_heads * 2, 1)

    def forward(self, G, diseases, mirnas):
        assert G.number_of_nodes() == self.G.number_of_nodes()

        h_agg = self.gat(G)
        h_concat = torch.cat((h_agg[diseases], h_agg[mirnas]), 1)
        predict_score = torch.sigmoid(self.predict(h_concat))

        return predict_score
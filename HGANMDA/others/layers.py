import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, G, feature_attn_size, dropout, slope):
        super(GATLayer, self).__init__()

        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)

        self.G = G
        self.slope = slope

        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], feature_attn_size, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], feature_attn_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attn_fc = nn.Linear(feature_attn_size * 2, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.m_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a, negative_slope=self.slope)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)

        return {'h': F.elu(h)}

    def forward(self, G):
        assert G.number_of_nodes() == self.G.number_of_nodes()

        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)

        self.G.apply_edges(self.edge_attention)
        self.G.update_all(self.message_func, self.reduce_func)

        return self.G.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, G, feature_attn_size, num_heads, dropout, slope, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()

        self.G = G
        self.dropout = dropout
        self.slope = slope
        self.merge = merge

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(G, feature_attn_size, dropout, slope))

    def forward(self, G):
        head_outs = [attn_head(G) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs), dim=0)


class GATSecondLayer(nn.Module):
    def __init__(self, G, in_dim, out_dim, dropout, slope):
        super(GATSecondLayer, self).__init__()

        self.G = G
        self.slope = slope

        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(out_dim * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a, negative_slope=self.slope)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': F.elu(h)}

    def forward(self, h):
        self.G.ndata['z'] = self.fc(h)
        self.G.apply_edges(self.edge_attention)
        self.G.update_all(self.message_func, self.reduce_func)
        return self.G.ndata.pop('h')


class MultiHeadGATSecondLayer(nn.Module):
    def __init__(self, G, in_dim, out_dim, num_heads, dropout, slope, merge='cat'):
        super(MultiHeadGATSecondLayer, self).__init__()

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATSecondLayer(G, in_dim, out_dim, dropout, slope))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]

        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))
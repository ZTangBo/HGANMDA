import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import metrics
import torch
import torch.nn as nn
import dgl

# 数据读取
def load_data(directory, random_seed):
    D_SSM1 = np.loadtxt(directory + '/D_SSM1.txt')
    D_SSM2 = np.loadtxt(directory + '/D_SSM2.txt')
    D_GSM = np.loadtxt(directory + '/D_GSM.txt')
    M_FSM = np.loadtxt(directory + '/M_FSM.txt')
    M_GSM = np.loadtxt(directory + '/M_GSM.txt')
    IL = np.loadtxt(directory + '/lncRNA-fuction.txt')
    all_associations = pd.read_csv(directory + '/all_mirna_disease_pairs.csv', names=['miRNA', 'disease', 'label'])
    ml_associations = pd.read_csv(directory + '/miRNA-lncRNA.csv', names=['miRNA', 'lncRNA', 'label'])
    ld_associations = pd.read_csv(directory + '/disease-lncRNA.csv', names=['lncRNA', 'disease', 'label'])
    D_SSM = (D_SSM1 + D_SSM2) / 2

    ID = D_SSM
    IM = M_FSM
    for i in range(D_SSM.shape[0]):
        for j in range(D_SSM.shape[1]):
            if ID[i][j] == 0:
                ID[i][j] = D_GSM[i][j]

    for i in range(M_FSM.shape[0]):
        for j in range(M_FSM.shape[1]):
            if IM[i][j] == 0:
                IM[i][j] = M_GSM[i][j]
# 筛选miRNA-disease正样本和与正样本数相同的负样本
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

# 筛选miRNA-lncRNA和disease-lncRNA已知关联
    ml_associations1 = ml_associations.loc[ml_associations['label'] == 1]
    ld_associations1 = ld_associations.loc[ld_associations['label'] == 1]
    sample_df = known_associations.append(random_negative)

# 指针重置
    sample_df.reset_index(drop=True, inplace=True)
    ml_associations1.reset_index(drop=True, inplace=True)
    ld_associations1.reset_index(drop=True, inplace=True)
    samples = sample_df.values      # 获得重新编号的新样本
    ml_associations = ml_associations1.values
    ld_associations = ld_associations1.values

    return ID, IM, IL, samples, ml_associations, ld_associations      # 未知关联数量较多，选择和已知关联数目相同的未知关联组成样本

# miRNA-disease异质图和miRNA-disease-lncRNA异质图的构建
def build_graph(directory, random_seed):
    ID, IM, IL, samples, ml_associations, ld_associations = load_data(directory, random_seed)
    # miRNA-disease二元异质图
    g = dgl.DGLGraph()
    g.add_nodes(ID.shape[0] + IM.shape[0])
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
    node_type[: ID.shape[0]] = 1
    g.ndata['type'] = node_type
# 0-382设为疾病节点，并传入特征
    d_sim = torch.zeros(g.number_of_nodes(), ID.shape[1])
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))
    g.ndata['d_sim'] = d_sim
# 383-877设为miRNA节点，并传入特征
    m_sim = torch.zeros(g.number_of_nodes(), IM.shape[1])
    m_sim[ID.shape[0]: ID.shape[0]+IM.shape[0], :] = torch.from_numpy(IM.astype('float32'))
    g.ndata['m_sim'] = m_sim
# 让指针从0开始，原本节点标签从1开始
    disease_ids = list(range(1, ID.shape[0]+1))
    mirna_ids = list(range(1, IM.shape[0]+1))

    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}
    mirna_ids_invmap = {id_: i for i, id_ in enumerate(mirna_ids)}

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]
    sample_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]

    g.add_edges(sample_disease_vertices, sample_mirna_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    g.add_edges(sample_mirna_vertices, sample_disease_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    g.readonly()

    # miRNA-disease-lncRNA三元异质图
    g0 = dgl.DGLGraph()
    g0.add_nodes(ID.shape[0] + IM.shape[0] + IL.shape[0])
    node_type = torch.zeros(g0.number_of_nodes(), dtype=torch.int64) # 返回一个878全为0的tensor 878+467=1345
    node_type[: ID.shape[0]] = 1            # disease383标记为1，miRNA标记为0，lncRNA标记为2
    node_type[ID.shape[0] + IM.shape[0]:] = 2
    g0.ndata['type'] = node_type             # 将图中疾病的节点记为1

    d_sim = torch.zeros(g0.number_of_nodes(), ID.shape[1])       # （1345，383）
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))
    g0.ndata['d_sim'] = d_sim

    m_sim = torch.zeros(g0.number_of_nodes(), IM.shape[1])       # （1345,495）
    m_sim[ID.shape[0]: ID.shape[0]+IM.shape[0], :] = torch.from_numpy(IM.astype('float32'))
    g0.ndata['m_sim'] = m_sim        # 每一行表示一个miRNA的特征

    l_sim = torch.zeros(g0.number_of_nodes(), IL.shape[1])       # （1345,64）
    l_sim[ID.shape[0]+IM.shape[0]: ID.shape[0]+IM.shape[0]+IL.shape[0], :] = torch.from_numpy(IL.astype('float32'))
    g0.ndata['l_sim'] = l_sim
    # lncRNA标签
    lncrna_ids = list(range(1, IL.shape[0]+1))
    lncrna_ids_invmap = {id_: i for i, id_ in enumerate(lncrna_ids)}

    ml_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in ml_associations[:, 0]]
    ml_lncrna_vertices = [lncrna_ids_invmap[id_] + ID.shape[0] + IM.shape[0] for id_ in ml_associations[:, 1]]
    ld_lncrna_vertices = [disease_ids_invmap[id_] for id_ in ld_associations[:, 0]]
    ld_disease_vertices = [lncrna_ids_invmap[id_] + ID.shape[0] + IM.shape[0] for id_ in ld_associations[:, 1]]

    g0.add_edges(sample_disease_vertices, sample_mirna_vertices,         # 0-10859
                data={'dm': torch.from_numpy(samples[:, 2].astype('float32'))})
    g0.add_edges(sample_mirna_vertices, sample_disease_vertices,         # 10860-21719
                data={'md': torch.from_numpy(samples[:, 2].astype('float32'))})
    g0.add_edges(ml_mirna_vertices, ml_lncrna_vertices,                  # 21720-22396
                data={'ml': torch.from_numpy(ml_associations[:, 2].astype('float32'))})
    g0.add_edges(ml_lncrna_vertices, ml_mirna_vertices,                  # 22397-23073
                data={'lm': torch.from_numpy(ml_associations[:, 2].astype('float32'))})
    g0.add_edges(ld_lncrna_vertices, ld_disease_vertices,                # 23074-23092
                data={'ld': torch.from_numpy(ld_associations[:, 2].astype('float32'))})
    g0.add_edges(ld_disease_vertices, ld_lncrna_vertices,                # 23093-23111
                data={'dl': torch.from_numpy(ld_associations[:, 2].astype('float32'))})
    g0.readonly()

    return g, g0, sample_disease_vertices, sample_mirna_vertices, ID, IM, IL, samples, ml_associations, ld_associations


def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Mean AUC: %.4f $\pm$ %.4f' % (mean_auc, auc_std))

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)

    # std_tpr = np.std(tpr, axis=0)
    # tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='LightSkyBlue', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.savefig(directory+'/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.close()


def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AP: %.4f' % (i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='BlueViolet', alpha=0.9,
             label='Mean AP: %.4f $\pm$ %.4f' % (mean_prc, prc_std))  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc='lower left')
    plt.savefig(directory + '/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.close()
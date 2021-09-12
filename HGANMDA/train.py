import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn import metrics

from utils import load_data, build_graph, weight_reset
from model1 import HGANMDA


def Train(directory, epochs, attn_size, attn_heads, out_dim, dropout, slope, lr, wd, random_seed, cuda, model_type):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        if not cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed(random_seed)

    # if cuda:
    #     context = torch.device('cuda:0')
    # else:
    #     context = torch.device('cpu')
    context = torch.device('cpu')
    # g为miRNA和disease图，g0为miRNA、disease和lncRNA图
    g, g0, disease_vertices, mirna_vertices, ID, IM, IL, samples, ml_associations, ld_associations = build_graph(directory, random_seed)
    samples_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])

    print('## vertices:', g.number_of_nodes())
    print('## edges:', g.number_of_edges())
    print('## disease nodes:', torch.sum(g.ndata['type'] == 1).numpy())
    print('## mirna nodes: ', torch.sum(g.ndata['type'] == 0).numpy())
    print('## lncrna nodes: ', torch.sum(g0.ndata['type'] == 2).numpy())
    g.to(context)

    auc_result = []
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []
    prc_result = []

    fprs = []
    tprs = []
    precisions = []
    recalls = []

# 设置五折交叉验证
    i = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    for train_idx, test_idx in kf.split(samples[:, 2]):
        i += 1
        print('Training for Fold', i)
# 将训练集的指针标记为1，其余为0
        samples_df['train'] = 0
        samples_df['train'].iloc[train_idx] = 1

        train_tensor = torch.from_numpy(samples_df['train'].values.astype('int64'))

        edge_data = {'train': train_tensor}
# 对两个异质图的训练边进行标记
        g.edges[disease_vertices, mirna_vertices].data.update(edge_data)
        g.edges[mirna_vertices, disease_vertices].data.update(edge_data)

        g0.edges[disease_vertices, mirna_vertices].data.update(edge_data)
        g0.edges[mirna_vertices, disease_vertices].data.update(edge_data)
        # 由于图的边的添加训练的原因，g0和g不包含lncRNA节点的边的标记相同，然后分别根据训练边构建子图
        train_eid = g.filter_edges(lambda edges: edges.data['train'])
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)
        g_train0 = g0.edge_subgraph(train_eid, preserve_nodes=True)
        # g_train.copy_from_parent() dgl0.4之后的版本没有这个语句

        # 训练标签和测试标签用于最后对比
        label_train = g_train.edata['label'].unsqueeze(1)
        src_train, dst_train = g_train.all_edges()

        test_eid = g.filter_edges(lambda edges: edges.data['train'] == 0)
        src_test, dst_test = g.find_edges(test_eid)
        label_test = g.edges[test_eid].data['label'].unsqueeze(1)


        print('## Training edges:', len(train_eid))
        print('## Testing edges:', len(test_eid))

        if model_type == 'HGANMDA':
            model = HGANMDA(G=g_train0,
                           meta_paths_list=['md', 'dm', 'ml', 'dl'],
                           feature_attn_size=attn_size,
                           num_heads=attn_heads,
                           num_diseases=ID.shape[0],
                           num_mirnas=IM.shape[0],
                           num_lncrnas=IL.shape[0],
                           d_sim_dim=ID.shape[1],
                           m_sim_dim=IM.shape[1],
                           l_sim_dim=IL.shape[1],
                           out_dim=out_dim,
                           dropout=dropout,
                           slope=slope,
                           )


        model.apply(weight_reset)
        model.to(context)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        loss = nn.BCELoss()

        for epoch in range(epochs):
            start = time.time()

            model.train()
            with torch.autograd.set_detect_anomaly(True):
                score_train = model(g_train0, g0, src_train, dst_train)
                loss_train = loss(score_train, label_train)

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                score_val = model(g, g0, src_test, dst_test)
                loss_val = loss(score_val, label_test)

            score_train_cpu = np.squeeze(score_train.cpu().detach().numpy())
            score_val_cpu = np.squeeze(score_val.cpu().detach().numpy())
            label_train_cpu = np.squeeze(label_train.cpu().detach().numpy())
            label_val_cpu = np.squeeze(label_test.cpu().detach().numpy())

            train_auc = metrics.roc_auc_score(label_train_cpu, score_train_cpu)
            val_auc = metrics.roc_auc_score(label_val_cpu, score_val_cpu)

            pred_val = [0 if j < 0.5 else 1 for j in score_val_cpu]
            acc_val = metrics.accuracy_score(label_val_cpu, pred_val)
            pre_val = metrics.precision_score(label_val_cpu, pred_val)
            recall_val = metrics.recall_score(label_val_cpu, pred_val)
            f1_val = metrics.f1_score(label_val_cpu, pred_val)

            end = time.time()

            if (epoch + 1) % 10 == 0:
                print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(),
                      'Val Loss: %.4f' % loss_val.cpu().detach().numpy(),
                      'Acc: %.4f' % acc_val, 'Pre: %.4f' % pre_val, 'Recall: %.4f' % recall_val, 'F1: %.4f' % f1_val,
                      'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc, 'Time: %.2f' % (end - start))

        model.eval()
        with torch.no_grad():
            score_test = model(g, g0, src_test, dst_test)

        score_test_cpu = np.squeeze(score_test.cpu().detach().numpy())
        label_test_cpu = np.squeeze(label_test.cpu().detach().numpy())

        fpr, tpr, thresholds = metrics.roc_curve(label_test_cpu, score_test_cpu)
        precision, recall, _ = metrics.precision_recall_curve(label_test_cpu, score_test_cpu)
        test_auc = metrics.auc(fpr, tpr)
        test_prc = metrics.auc(recall, precision)

        pred_test = [0 if j < 0.5 else 1 for j in score_test_cpu]
        acc_test = metrics.accuracy_score(label_test_cpu, pred_test)
        pre_test = metrics.precision_score(label_test_cpu, pred_test)
        recall_test = metrics.recall_score(label_test_cpu, pred_test)
        f1_test = metrics.f1_score(label_test_cpu, pred_test)

        print('Fold: ', i, 'Test acc: %.4f' % acc_test, 'Test Pre: %.4f' % pre_test,
              'Test Recall: %.4f' % recall_test, 'Test F1: %.4f' % f1_test, 'Test PRC: %.4f' % test_prc,
              'Test AUC: %.4f' % test_auc)

        auc_result.append(test_auc)
        acc_result.append(acc_test)
        pre_result.append(pre_test)
        recall_result.append(recall_test)
        f1_result.append(f1_test)
        prc_result.append(test_prc)

        fprs.append(fpr)
        tprs.append(tpr)
        precisions.append(precision)
        recalls.append(recall)

    print('## Training Finished !')
    print('-----------------------------------------------------------------------------------------------')
    print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
          'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))

    return fprs, tprs, auc_result, precisions, recalls, prc_result
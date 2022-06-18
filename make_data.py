import torch
from torch_geometric.data import Data
import pickle
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

path_sbm_train = "data/SBM/SBM_CLUSTER_train.pkl"
path_sbm_val = "data/SBM/SBM_CLUSTER_val.pkl"
path_sbm_test = "data/SBM/SBM_CLUSTER_test.pkl"
with open(path_sbm_train, 'rb') as f:
    sbm_train = pickle.load(f)
with open(path_sbm_val, 'rb') as f:
    sbm_val = pickle.load(f)
with open(path_sbm_test, 'rb') as f:
    sbm_test = pickle.load(f)

K = 1000
NBTRAIN = len(sbm_train)
NBVAL = len(sbm_val)
NBTEST = len(sbm_test)

for k in range(K):
    print(k)
    idx_base = 0
    edge_index = torch.LongTensor([[], []])
    x = torch.tensor([])
    y = torch.LongTensor([])
    for idx_graph in range(k*int(NBTRAIN/K), (k+1)*int(NBTRAIN/K)):
        sub_graph = sbm_train[idx_graph]
        edge_index = torch.cat(
            (edge_index, torch.nonzero(sub_graph.W).T+idx_base), 1)
        x = torch.cat((x, torch.unsqueeze(sub_graph.node_feat, 1)), 0)
        y = torch.cat((y, sub_graph.node_label), 0)
        idx_base += len(sub_graph.node_feat)
    # print(x.size(), edge_index, idx_base)
    idx_train = idx_base
    for idx_graph in range(k*int(NBVAL/K), (k+1)*int(NBVAL/K)):
        sub_graph = sbm_val[idx_graph]
        edge_index = torch.cat(
            (edge_index, torch.nonzero(sub_graph.W).T+idx_base), 1)
        x = torch.cat((x, torch.unsqueeze(sub_graph.node_feat, 1)), 0)
        y = torch.cat((y, sub_graph.node_label), 0)
        idx_base += len(sub_graph.node_feat)
    # print(x.size(), edge_index, idx_base)
    idx_val = idx_base
    for idx_graph in range(k*int(NBTEST/K), (k+1)*int(NBTEST/K)):
        sub_graph = sbm_test[idx_graph]
        edge_index = torch.cat(
            (edge_index, torch.nonzero(sub_graph.W).T+idx_base), 1)
        x = torch.cat((x, torch.unsqueeze(sub_graph.node_feat, 1)), 0)
        y = torch.cat((y, sub_graph.node_label), 0)
        idx_base += len(sub_graph.node_feat)
    # print(x.size(), edge_index, idx_base)
    idx_test = idx_base

    x = torch.zeros(x.shape[0], 7).scatter(
        1, torch.tensor(x, dtype=torch.int64), 1)[:, 1:]
    train_mask = torch.BoolTensor([0]*idx_base)
    train_mask[0:idx_train] = True
    val_mask = torch.BoolTensor([0]*idx_base)
    val_mask[idx_train:idx_val] = True
    test_mask = torch.BoolTensor([0]*idx_base)
    test_mask[idx_val:idx_test] = True
    labels = y[train_mask]
    weight = (labels.size(0) - torch.bincount(labels)).float() / labels.size(0)
    with open("data/SBM/{}_uncopynode.pkl".format(k), "wb") as f:
        pickle.dump(Data(edge_index=edge_index, x= x, y = y, train_mask = train_mask, val_mask = val_mask, test_mask = test_mask, weight=weight), f)
    # break

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
import utils
from torch import optim
from torch.autograd import Variable
import pickle
import numpy as np

import warnings
warnings.filterwarnings("ignore")


HIDDENSIZE = 32
INPUT_SIZE = HIDDENSIZE
LAYERS = 1
LR_GRAPHNAS = 0.00035
EPOCHS = 200


class Identity(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, x1):
        return x


class GMMConv(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """

    def __init__(self, insize, outsize):
        super(GMMConv, self).__init__()
        self.gmmconv = gnn.GMMConv(insize, outsize, 2, 4)
        return

    def forward(self, x, edge_index):
        edge_attr = utils.cal_edge_attr_for_gmmconv(edge_index)
        x = x + F.relu(self.gmmconv(x, edge_index, edge_attr))
        return x


class GINConv(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """

    def __init__(self, nfeat, nclass):
        super(GINConv, self).__init__()
        self.ginconv = gnn.GINConv(
            nn.Sequential(nn.Linear(nfeat, nfeat),
                          nn.BatchNorm1d(nfeat),
                          nn.ReLU(),
                          nn.Linear(nfeat, nclass)),
            train_eps=True)
        return

    def forward(self, x, edge_index):
        x = x + F.relu(self.ginconv(x, edge_index))
        return x


class Gat(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Gat, self).__init__()
        self.gat1 = gnn.GATConv(nfeat, nfeat)
        # self.gat2 = gnn.GATConv(nfeat, nclass)
        return

    def forward(self, x, edge_index):
        x = x + F.elu(self.gat1(x, edge_index))
        # x = x + F.elu(self.gat2(x, edge_index))
        return x


class Sage(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Sage, self).__init__()
        self.sage1 = gnn.SAGEConv(nfeat, nfeat)
        # self.sage2 = gnn.SAGEConv(nfeat, nclass)
        # self.sage3 = gnn.SAGEConv(nfeat, nclass)
        # self.sage4 = gnn.SAGEConv(nfeat, nclass)
        return

    def forward(self, x, edge_index):
        x = x + F.relu(self.sage1(x, edge_index))
        # x = x + F.relu(self.sage2(x, edge_index))
        # x = x + F.relu(self.sage3(x, edge_index))
        # x = x + F.relu(self.sage4(x, edge_index))
        return x


class Gcn(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Gcn, self).__init__()
        self.gcn1 = gnn.GCNConv(nfeat, nfeat)
        # self.gcn2 = gnn.GCNConv(nfeat, nclass)
        return

    def forward(self, x, edge_index):
        x = x + F.relu(self.gcn1(x, edge_index))
        # x = x + F.relu(self.gcn2(x, edge_index))
        return x


class Sgc(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Sgc, self).__init__()
        self.sgc1 = gnn.SGConv(nfeat, nclass, 1, False)
        # self.sgc2 = gnn.SGConv(nfeat, nclass, 1, False)
        # self.sgc3 = gnn.SGConv(nfeat, nclass, 1, False)
        # self.sgc4 = gnn.SGConv(nfeat, nclass, 1, False)
        return

    def forward(self, x, edge_index):
        x = x + F.relu(self.sgc1(x, edge_index))
        # x = x + F.relu(self.sgc2(x, edge_index))
        # x = x + F.relu(self.sgc3(x, edge_index))
        # x = x + F.relu(self.sgc4(x, edge_index))
        return x


class Appnp(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Appnp, self).__init__()
        self.appnp1 = gnn.APPNP(K=1, alpha=0.1)
        # self.appnp2 = gnn.APPNP(K=1, alpha=0.1)
        return

    def forward(self, x, edge_index):
        x = x + F.relu(self.appnp1(x, edge_index))
        # x = x + F.relu(self.appnp2(x, edge_index))
        return x


class Agnn(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Agnn, self).__init__()
        self.agnn1 = gnn.AGNNConv()
        # self.agnn2 = gnn.AGNNConv()
        return

    def forward(self, x, edge_index):
        x = x + F.relu(self.agnn1(x, edge_index))
        # x = x + F.relu(self.agnn2(x, edge_index))
        return x


class Arma(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Arma, self).__init__()
        self.arma1 = gnn.ARMAConv(nfeat, nfeat, num_stacks=2)
        # self.arma2 = gnn.ARMAConv(nfeat, nclass, num_stacks=2)
        return

    def forward(self, x, edge_index):
        x = x + F.relu(self.arma1(x, edge_index))
        # x = x + F.relu(self.arma2(x, edge_index))
        return x


class FeaStConv(nn.Module):
    def __init__(self, nfeat, nclass):
        super(FeaStConv, self).__init__()
        self.feastconv1 = gnn.FeaStConv(nfeat, nfeat, 2)
        # self.feastconv2 = gnn.FeaStConv(nfeat, nclass, 2)
        return

    def forward(self, x, edge_index):
        x = x + F.relu(self.feastconv1(x, edge_index))
        # x = x + F.relu(self.feastconv2(x, edge_index))
        return x


class GENConv(nn.Module):
    def __init__(self, nfeat, nclass):
        super(GENConv, self).__init__()
        self.genconv1 = gnn.GENConv(nfeat, nfeat)
        # self.genconv2 = gnn.GENConv(nfeat, nclass)
        return

    def forward(self, x, edge_index):
        x = x + F.relu(self.genconv1(x, edge_index))
        # x = x + F.relu(self.genconv2(x, edge_index))
        return x


class GatedGraphConv(nn.Module):
    def __init__(self, nfeat, nclass):
        super(GatedGraphConv, self).__init__()
        self.gatedgraph1 = gnn.GatedGraphConv(nfeat, 1)
        # self.gatedgraph2 = gnn.GatedGraphConv(nfeat, 1)
        return

    def forward(self, x, edge_index):
        x = x + F.relu(self.gatedgraph1(x, edge_index))
        # x = x + F.relu(self.gatedgraph2(x, edge_index))
        return x


class Structure():
    def __init__(self):
        return

    def set_y(self, len=64):
        self.y_1 = 'Gat(self.len, self.len)'  # 6:755
        self.len = len
        self.y_2_local_len = self.len
        self.y_2 = 'GINConv(self.len, self.len)'  # 729
        self.y_3 = 'Sage(self.len, self.len)'  # 714
        self.y_4 = 'Gcn(self.len, self.len)'  # 683
        self.y_5 = 'Sgc(self.len, self.len)'  # 77,759
        self.y_6 = 'Appnp(self.len, self.len)'  # 781
        self.y_7 = 'Agnn(self.len, self.len)'  # 734,686
        self.y_8 = 'Arma(self.len, self.len)'  # 612,556
        self.y_9 = 'FeaStConv(self.len, self.len)'  # 2:728,693
        self.y_10 = 'GENConv(self.len, self.len)'  # 722,67
        self.y_11 = 'GMMConv(self.len, self.len)'  # 59,676
        self.y_12 = 'GatedGraphConv(self.len,self.len)'
        return


class SonNet(nn.Module):
    def __init__(self, nfeat, nclass):
        with open('tmp.pkl', 'rb') as f:
            self.supermask = pickle.load(f)
        super(SonNet, self).__init__()
        self.hdim = 64
        self.x_blink = nn.Identity()
        self.z_blink = nn.Identity()

        self.x = nn.Linear(nfeat, self.hdim)
        self.x_1 = F.sigmoid
        self.x_2 = F.tanh
        self.x_3 = F.relu
        self.x_4 = F.softmax
        self.x_5 = nn.Identity()

        Structure.set_y(self)

        for j in range(1, 6 + 1):
            for i in range(0, j):
                for t in range(1, 12 + 1):
                    layer = 'self.y{}{}_{}'.format(i, j, t)
                    exec("{}=eval(self.y_{})".format(layer, t))

        self.z = nn.Linear(self.hdim, nclass)
        self.z_1 = F.sigmoid
        self.z_2 = F.tanh
        self.z_3 = F.relu
        self.z_4 = F.softmax
        self.z_5 = nn.Identity()
        self.first = True
        return

    def forward(self, x, edge_index):
        supermask = self.supermask
        local_mask = torch.ones(7)
        for i in range(1, 7):
            local_mask[int((supermask[i]-1)/12)] = 0
            if supermask[i] == 0:
                local_mask[i] = 0
        if self.first:
            print(local_mask)
            self.first = False
        x = self.x_blink(x) if supermask[0] == 0 else eval(
            'self.x_{}'.format(supermask[0]))(self.x(x))

        if supermask[1] != 0:
            l1_input = x
            l1_output = eval('self.y01_{}'.format(
                supermask[1]))(l1_input, edge_index)
        else:
            l1_output = x

        if supermask[2] != 0:
            l2_input = x if (
                1 <= supermask[2] and supermask[2] <= 12) else l1_output
            l2_output = eval('self.y{}2_{}'.format(
                int((supermask[2] - 1)/12), (supermask[2] - 1) % 12 + 1))(l2_input, edge_index)
        else:
            l2_output = x

        if supermask[3] != 0:
            l3_input = x if (1 <= supermask[3] and supermask[3] <= 12) else (
                l1_output if (13 <= supermask[3] and supermask[3] <= 24) else l2_output)
            l3_output = eval('self.y{}3_{}'.format(
                int((supermask[3] - 1)/12), (supermask[3] - 1) % 12 + 1))(l3_input, edge_index)
        else:
            l3_output = x

        if supermask[4] != 0:
            l4_input = x if (supermask[4] in [1, 12]) else (l1_output if (supermask[4] in [
                12+1, 12*2]) else (l2_output if (supermask[4] in [24+1, 12*3]) else l3_output))
            l4_output = eval('self.y{}4_{}'.format(
                int((supermask[4] - 1)/12), (supermask[4] - 1) % 12 + 1))(l4_input, edge_index)
        else:
            l4_output = x

        if supermask[5] != 0:
            l5_input = x if (supermask[5] in [1, 12]) else (l1_output if (supermask[5] in [12+1, 12*2]) else (
                l2_output if (supermask[5] in [24+1, 12*3]) else (l3_output if (supermask[5] in [36+1, 12*4]) else l4_output)))
            l5_output = eval('self.y{}5_{}'.format(
                int((supermask[5] - 1)/12), (supermask[5] - 1) % 12 + 1))(l5_input, edge_index)
        else:
            l5_output = x

        if supermask[6] != 0:
            l6_input = x if (supermask[6] in [1, 12]) else (l1_output if (supermask[6] in [12+1, 12*2]) else (l2_output if (supermask[6] in [
                24+1, 12*3]) else (l3_output if (supermask[6] in [36+1, 12*4]) else (l4_output if (supermask[6] in [48+1, 12*5]) else l5_output))))
            l6_output = eval('self.y{}6_{}'.format(
                int((supermask[6] - 1)/12), (supermask[6] - 1) % 12 + 1))(l6_input, edge_index)
        else:
            l6_output = x
        x = (l1_output*local_mask[1]+l2_output*local_mask[2]+l3_output *
             local_mask[3]+l4_output*local_mask[4]+l5_output*local_mask[5]+l6_output*local_mask[6])/local_mask.sum()

        return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(self.z(x))


class DynamicSonNet(nn.Module, Structure):
    def __init__(self, nfeat, nclass):
        # self.supermask = [5, 10, 6, 1, 2, 1, 2, 5]  # graphnas
        # self.supermask = [2, 2, 1, 16, 5, 7, 42, 1] # random
        # self.supermask = [3, 1, 1, 10, 26, 10, 10, 4] # darts
        # self.supermask = [4, 1, 13, 25, 10, 26, 14, 4] # fednas
        # self.supermask = [5, 10, 16, 0, 29, 0, 0, 5]  # agcns
        self.supermask = [5, 5, 5+12, 5+12*2, 5, 5+12*4, 5+12*5, 5]  # try
        super(DynamicSonNet, self).__init__()
        self.hdim = 64
        self.x_blink = nn.Identity()
        self.z_blink = nn.Identity()

        self.x = nn.Linear(nfeat, self.hdim)
        self.x_1 = F.sigmoid
        self.x_2 = F.tanh
        self.x_3 = F.relu
        self.x_4 = F.softmax
        self.x_5 = nn.Identity()

        Structure.set_y(self)

        for j in range(1, 6 + 1):
            for i in range(0, j):
                for t in range(1, 12 + 1):
                    layer = 'self.y{}{}_{}'.format(i, j, t)
                    exec("{}=eval(self.y_{})".format(layer, t))

        self.z = nn.Linear(self.hdim, nclass)
        self.z_1 = F.sigmoid
        self.z_2 = F.tanh
        self.z_3 = F.relu
        self.z_4 = F.softmax
        self.z_5 = nn.Identity()
        self.first = True
        return

    def forward(self, x, edge_index):
        supermask = self.supermask
        local_mask = torch.ones(7)
        for i in range(1, 7):
            local_mask[int((supermask[i]-1)/12)] = 0
            if supermask[i] == 0:
                local_mask[i] = 0
        if self.first:
            print(local_mask)
            self.first = False
        x = self.x_blink(x) if supermask[0] == 0 else eval(
            'self.x_{}'.format(supermask[0]))(self.x(x))

        if supermask[1] != 0:
            l1_input = x
            l1_output = eval('self.y01_{}'.format(
                supermask[1]))(l1_input, edge_index)
        else:
            l1_output = x

        if supermask[2] != 0:
            l2_input = x if (
                1 <= supermask[2] and supermask[2] <= 12) else l1_output
            l2_output = eval('self.y{}2_{}'.format(
                int((supermask[2] - 1)/12), (supermask[2] - 1) % 12 + 1))(l2_input, edge_index)
        else:
            l2_output = x

        if supermask[3] != 0:
            l3_input = x if (1 <= supermask[3] and supermask[3] <= 12) else (
                l1_output if (13 <= supermask[3] and supermask[3] <= 24) else l2_output)
            l3_output = eval('self.y{}3_{}'.format(
                int((supermask[3] - 1)/12), (supermask[3] - 1) % 12 + 1))(l3_input, edge_index)
        else:
            l3_output = x

        if supermask[4] != 0:
            l4_input = x if (supermask[4] in [1, 12]) else (l1_output if (supermask[4] in [
                12+1, 12*2]) else (l2_output if (supermask[4] in [24+1, 12*3]) else l3_output))
            l4_output = eval('self.y{}4_{}'.format(
                int((supermask[4] - 1)/12), (supermask[4] - 1) % 12 + 1))(l4_input, edge_index)
        else:
            l4_output = x

        if supermask[5] != 0:
            l5_input = x if (supermask[5] in [1, 12]) else (l1_output if (supermask[5] in [12+1, 12*2]) else (
                l2_output if (supermask[5] in [24+1, 12*3]) else (l3_output if (supermask[5] in [36+1, 12*4]) else l4_output)))
            l5_output = eval('self.y{}5_{}'.format(
                int((supermask[5] - 1)/12), (supermask[5] - 1) % 12 + 1))(l5_input, edge_index)
        else:
            l5_output = x

        if supermask[6] != 0:
            l6_input = x if (supermask[6] in [1, 12]) else (l1_output if (supermask[6] in [12+1, 12*2]) else (l2_output if (supermask[6] in [
                24+1, 12*3]) else (l3_output if (supermask[6] in [36+1, 12*4]) else (l4_output if (supermask[6] in [48+1, 12*5]) else l5_output))))
            l6_output = eval('self.y{}6_{}'.format(
                int((supermask[6] - 1)/12), (supermask[6] - 1) % 12 + 1))(l6_input, edge_index)
        else:
            l6_output = x
        x = (l1_output*local_mask[1]+l2_output*local_mask[2]+l3_output *
             local_mask[3]+l4_output*local_mask[4]+l5_output*local_mask[5]+l6_output*local_mask[6])/local_mask.sum()

        return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(self.z(x))


class SuperNet(nn.Module, Structure):
    def __init__(self, nfeat, nclass):
        super(SuperNet, self).__init__()
        self.hdim = 64
        self.x_blink = nn.Identity()
        self.z_blink = nn.Identity()

        self.x = nn.Linear(nfeat, self.hdim)
        self.x_1 = F.sigmoid
        self.x_2 = F.tanh
        self.x_3 = F.relu
        self.x_4 = F.softmax
        self.x_5 = nn.Identity()

        Structure.set_y(self)

        for j in range(1, 6 + 1):
            for i in range(0, j):
                for t in range(1, 12 + 1):
                    layer = 'self.y{}{}_{}'.format(i, j, t)
                    exec("{}=eval(self.y_{})".format(layer, t))

        self.z = nn.Linear(self.hdim, nclass)
        self.z_1 = F.sigmoid
        self.z_2 = F.tanh
        self.z_3 = F.relu
        self.z_4 = F.softmax
        self.z_5 = nn.Identity()
        return

    def forward(self, x, edge_index, supermask):
        local_mask = torch.ones(7)
        for i in range(1, 7):
            local_mask[int((supermask[i]-1)/12)] = 0
            if supermask[i] == 0:
                local_mask[i] = 0
        x = self.x_blink(x) if supermask[0] == 0 else eval(
            'self.x_{}'.format(supermask[0]))(self.x(x))

        if supermask[1] != 0:
            l1_input = x
            l1_output = eval('self.y01_{}'.format(
                supermask[1]))(l1_input, edge_index)
        else:
            l1_output = x

        if supermask[2] != 0:
            l2_input = x if (
                1 <= supermask[2] and supermask[2] <= 12) else l1_output
            l2_output = eval('self.y{}2_{}'.format(
                int((supermask[2] - 1)/12), (supermask[2] - 1) % 12 + 1))(l2_input, edge_index)
        else:
            l2_output = x

        if supermask[3] != 0:
            l3_input = x if (1 <= supermask[3] and supermask[3] <= 12) else (
                l1_output if (13 <= supermask[3] and supermask[3] <= 24) else l2_output)
            l3_output = eval('self.y{}3_{}'.format(
                int((supermask[3] - 1)/12), (supermask[3] - 1) % 12 + 1))(l3_input, edge_index)
        else:
            l3_output = x

        if supermask[4] != 0:
            l4_input = x if (supermask[4] in [1, 12]) else (l1_output if (supermask[4] in [
                12+1, 12*2]) else (l2_output if (supermask[4] in [24+1, 12*3]) else l3_output))
            l4_output = eval('self.y{}4_{}'.format(
                int((supermask[4] - 1)/12), (supermask[4] - 1) % 12 + 1))(l4_input, edge_index)
        else:
            l4_output = x

        if supermask[5] != 0:
            l5_input = x if (supermask[5] in [1, 12]) else (l1_output if (supermask[5] in [12+1, 12*2]) else (
                l2_output if (supermask[5] in [24+1, 12*3]) else (l3_output if (supermask[5] in [36+1, 12*4]) else l4_output)))
            l5_output = eval('self.y{}5_{}'.format(
                int((supermask[5] - 1)/12), (supermask[5] - 1) % 12 + 1))(l5_input, edge_index)
        else:
            l5_output = x

        if supermask[6] != 0:
            l6_input = x if (supermask[6] in [1, 12]) else (l1_output if (supermask[6] in [12+1, 12*2]) else (l2_output if (supermask[6] in [
                24+1, 12*3]) else (l3_output if (supermask[6] in [36+1, 12*4]) else (l4_output if (supermask[6] in [48+1, 12*5]) else l5_output))))
            l6_output = eval('self.y{}6_{}'.format(
                int((supermask[6] - 1)/12), (supermask[6] - 1) % 12 + 1))(l6_input, edge_index)
        else:
            l6_output = x
        x = (l1_output*local_mask[1]+l2_output*local_mask[2]+l3_output *
             local_mask[3]+l4_output*local_mask[4]+l5_output*local_mask[5]+l6_output*local_mask[6])/local_mask.sum()

        return self.z_blink(x) if supermask[7] == 0 else eval('self.z_{}'.format(supermask[7]))(self.z(x))


class Darts(nn.Module, Structure):
    def __init__(self, nfeat, nclass):
        super(Darts, self).__init__()
        self.len = 64
        self.hdim = self.len
        self.x_blink = nn.Identity()
        self.z_blink = nn.Identity()

        self.x = nn.Linear(nfeat, self.hdim)
        self.x_1 = F.sigmoid
        self.x_2 = F.tanh
        self.x_3 = F.relu
        self.x_4 = F.softmax
        self.x_5 = nn.Identity()

        Structure.set_y(self, self.len)

        for j in range(1, 6 + 1):
            for i in range(0, j):
                for t in range(1, 12 + 1):
                    layer = 'self.y{}{}_{}'.format(i, j, t)
                    exec("{}=eval(self.y_{})".format(layer, t))

        self.z = nn.Linear(self.hdim, nclass)
        self.z_1 = F.sigmoid
        self.z_2 = F.tanh
        self.z_3 = F.relu
        self.z_4 = F.softmax
        self.z_5 = nn.Identity()

        self.tau = 10
        self.register_parameter('alpha', nn.Parameter(
            Variable(torch.ones(5)/2, requires_grad=True)))
        self.register_parameter('gamma', nn.Parameter(
            Variable(torch.ones(5)/2, requires_grad=True)))
        for j in range(1, 7):
            self.register_parameter('beta{}'.format(j), nn.Parameter(
                eval('Variable(torch.ones({})/2,requires_grad=True)'.format(j*12+1))))
        return

    def forward(self, x, edge_index):
        device = self.alpha.device
        tmp_lis = []
        tmp_alpha = F.softmax(self.alpha)
        for i in range(5):
            tmp_lis.append(
                eval('tmp_alpha[{}] * self.x_{}(self.x(x))'.format(i, i+1)))
        x = torch.sum(torch.stack(tmp_lis, axis=0), dim=0)
        y0 = x
        for j in range(1, 6 + 1):
            tmp_lis = []
            exec('tmp_beta{} = torch.rand(j*12+1)'.format(j))
            for i in range(j):
                exec('tmp_beta{}[{}:{}] = F.softmax(self.beta{}[{}:{}])'.format(
                    j, i*12+1, i*12+13, j, i*12+1, i*12+13))
            for i in range(1, j*12+1):
                tmp_lis.append(eval('self.y{}{}_{}(y{},edge_index)*tmp_beta{}[i]'.format(
                    int((i-1)/12), j, (i-1) % 12+1, int((i-1)/12), j)))
            exec(
                'y{} = torch.sum(torch.stack(tmp_lis,axis=0),dim=0) + 0*self.beta{}[0]**2'.format(j, j))
        x = eval('y1') + eval('y2') + eval('y3') + \
            eval('y4') + eval('y5') + eval('y6')
        tmp_lis = []
        tmp_gamma = F.softmax(self.gamma)
        for i in range(5):
            tmp_lis.append(
                eval('tmp_gamma[{}] * self.z_{}(self.z(x))'.format(i, i + 1)))
        x = torch.sum(torch.stack(tmp_lis, axis=0), dim=0)
        return x

    def generate_supermask(self):
        device = self.alpha.device
        supermask = [0, 0, 0, 0, 0, 0, 0, 0]
        max_alpha = torch.Tensor([-1000000]).to(device)
        arg_max_alpha = None
        for idx in range(5):
            if eval('torch.abs(self.alpha[{}])'.format(idx)) > max_alpha:
                max_alpha = eval('torch.abs(self.alpha[{}])'.format(idx))
                arg_max_alpha = idx
        supermask[0] = arg_max_alpha + 1

        max_alpha = torch.Tensor([-1000000]).to(device)
        arg_max_alpha = None
        for idx in range(5):
            if eval('torch.abs(self.gamma[{}])'.format(idx)) > max_alpha:
                max_alpha = eval('torch.abs(self.gamma[{}])'.format(idx))
                arg_max_alpha = idx
        supermask[7] = arg_max_alpha + 1

        for j in range(1, 6 + 1):
            max_alpha = torch.Tensor([-1000000]).to(device)
            arg_max_alpha = None
            for i in range(j*12+1):
                if eval('torch.abs(self.beta{}[{}])'.format(j, i)) > max_alpha:
                    max_alpha = eval('torch.abs(self.beta{}[{}])'.format(j, i))
                    arg_max_alpha = i
            supermask[j] = arg_max_alpha
        return supermask

    def get_parameters(self):
        for name, param in self.named_parameters():
            if name.__contains__("alpha") or name.__contains__("beta") or name.__contains__("gamma"):
                continue
            yield param

    def get_arc_params(self):
        for name, param in self.named_parameters():
            if name.__contains__("alpha") or name.__contains__("beta") or name.__contains__("gamma"):
                yield param


class FedNas(nn.Module, Structure):
    def __init__(self, nfeat, nclass):
        super(FedNas, self).__init__()
        self.len = 64
        self.hdim = self.len
        self.x_blink = nn.Identity()
        self.z_blink = nn.Identity()

        self.x = nn.Linear(nfeat, self.hdim)
        self.x_1 = F.sigmoid
        self.x_2 = F.tanh
        self.x_3 = F.relu
        self.x_4 = F.softmax
        self.x_5 = nn.Identity()

        Structure.set_y(self, self.len)

        for j in range(1, 6 + 1):
            for i in range(0, j):
                for t in range(1, 12 + 1):
                    layer = 'self.y{}{}_{}'.format(i, j, t)
                    exec("{}=eval(self.y_{})".format(layer, t))

        self.z = nn.Linear(self.hdim, nclass)
        self.z_1 = F.sigmoid
        self.z_2 = F.tanh
        self.z_3 = F.relu
        self.z_4 = F.softmax
        self.z_5 = nn.Identity()

        self.tau = 10
        self.register_parameter('alpha', nn.Parameter(
            Variable(torch.ones(5)/2, requires_grad=True)))
        self.register_parameter('gamma', nn.Parameter(
            Variable(torch.ones(5)/2, requires_grad=True)))
        for j in range(1, 7):
            self.register_parameter('beta{}'.format(j), nn.Parameter(
                eval('Variable(torch.ones({})/2,requires_grad=True)'.format(j*12+1))))
        return

    def forward(self, x, edge_index):
        device = self.alpha.device
        tmp_lis = []
        tmp_alpha = F.softmax(self.alpha)
        for i in range(5):
            tmp_lis.append(
                eval('tmp_alpha[{}] * self.x_{}(self.x(x))'.format(i, i+1)))
        x = torch.sum(torch.stack(tmp_lis, axis=0), dim=0)
        y0 = x
        for j in range(1, 6 + 1):
            tmp_lis = []
            exec('tmp_beta{} = torch.rand(j*12+1)'.format(j))
            for i in range(j):
                exec('tmp_beta{}[{}:{}] = F.softmax(self.beta{}[{}:{}])'.format(
                    j, i*12+1, i*12+13, j, i*12+1, i*12+13))
            for i in range(1, j*12+1):
                tmp_lis.append(eval('self.y{}{}_{}(y{},edge_index)*tmp_beta{}[i]'.format(
                    int((i-1)/12), j, (i-1) % 12+1, int((i-1)/12), j)))
            exec(
                'y{} = torch.sum(torch.stack(tmp_lis,axis=0),dim=0) + 0*self.beta{}[0]**2'.format(j, j))
        x = eval('y1') + eval('y2') + eval('y3') + \
            eval('y4') + eval('y5') + eval('y6')
        tmp_lis = []
        tmp_gamma = F.softmax(self.gamma)
        for i in range(5):
            tmp_lis.append(
                eval('tmp_gamma[{}] * self.z_{}(self.z(x))'.format(i, i + 1)))
        x = torch.sum(torch.stack(tmp_lis, axis=0), dim=0)
        return x

    def generate_supermask(self):
        device = self.alpha.device
        supermask = [0, 0, 0, 0, 0, 0, 0, 0]
        max_alpha = torch.Tensor([-1000000]).to(device)
        arg_max_alpha = None
        for idx in range(5):
            if eval('torch.abs(self.alpha[{}])'.format(idx)) > max_alpha:
                max_alpha = eval('torch.abs(self.alpha[{}])'.format(idx))
                arg_max_alpha = idx
        supermask[0] = arg_max_alpha + 1

        max_alpha = torch.Tensor([-1000000]).to(device)
        arg_max_alpha = None
        for idx in range(5):
            if eval('torch.abs(self.gamma[{}])'.format(idx)) > max_alpha:
                max_alpha = eval('torch.abs(self.gamma[{}])'.format(idx))
                arg_max_alpha = idx
        supermask[7] = arg_max_alpha + 1

        for j in range(1, 6 + 1):
            max_alpha = torch.Tensor([-1000000]).to(device)
            arg_max_alpha = None
            for i in range(j*12+1):
                if eval('torch.abs(self.beta{}[{}])'.format(j, i)) > max_alpha:
                    max_alpha = eval('torch.abs(self.beta{}[{}])'.format(j, i))
                    arg_max_alpha = i
            supermask[j] = arg_max_alpha
        return supermask

    def get_parameters(self):
        for name, param in self.named_parameters():
            if name.__contains__("alpha") or name.__contains__("beta") or name.__contains__("gamma"):
                continue
            yield param

    def get_arc_params(self):
        for name, param in self.named_parameters():
            if name.__contains__("alpha") or name.__contains__("beta") or name.__contains__("gamma"):
                yield param


class GraphNas(nn.Module):
    def __init__(self, input_size):
        super(GraphNas, self).__init__()
        self.input_size = input_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=HIDDENSIZE,
            num_layers=LAYERS,
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDENSIZE,
        )

        self.out1 = nn.Linear(HIDDENSIZE, 5)
        self.out2 = nn.Linear(HIDDENSIZE, 12)
        self.out8 = nn.Linear(HIDDENSIZE, 5)
        for i in range(3, 7 + 1):
            exec("self.out{} = nn.Linear(HIDDENSIZE, 13)".format(i))

        self.b = 0
        self.beta = 0.9
        return

    def generate_code(self):
        h_state = [torch.FloatTensor(np.zeros((1, 1, self.input_size))),
                   torch.FloatTensor(np.zeros((1, 1, self.input_size)))]
        if torch.cuda.is_available():
            h_state[0] = h_state[0].cuda()
            h_state[1] = h_state[1].cuda()
        x = h_state[0]
        res = []
        for i in range(1, 8 + 1):
            r_out, h_state = self.lstm(x, h_state)
            res.append(
                F.softmax(eval("self.out{}".format(i))(r_out[0]), dim=1))
            x = h_state[0]
        return res

    def parse_code(self, dummy_code):
        supermask = []
        idx = 1
        for code in dummy_code:
            tmp = code.multinomial(num_samples=1).cpu().numpy()
            tmp = tmp[0][0]
            if idx == 1 or idx == 8:
                supermask.append(tmp + 1)
            else:
                supermask.append(tmp)
            idx += 1
        return supermask

    def get_loss(self, dummy_code, supermask, R):
        losses = []
        for i in range(len(dummy_code)):
            index = supermask[i]
            if i in [0, 7]:
                index -= 1
            losses.append(torch.log(dummy_code[i][0][index]) * (R - self.b))
        self.b = self.beta * self.b + (1 - self.beta) * R
        return torch.stack(losses).mean()


if __name__ == "__main__":
    supermask = [5, 10, 16, 9, 29, 0, 0, 5]
    local_mask = torch.ones(7)
    for i in range(1, 7):
        local_mask[int((supermask[i]-1)/12)] = 0
        if supermask[i] == 0:
            local_mask[i] = 0
    print(local_mask)
    pass

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import torch

# class Gat(nn.Module):
#     def __init__(self,nfeat, nclass):
#         super(Gat,self).__init__()
#         self.gat1 = gnn.GATConv(nfeat,8,8,True)
#         self.gat2 = gnn.GATConv(64,nclass)
#         return

#     def forward(self, x,edge_index):
#         x = self.gat1(x,edge_index)
#         x = F.elu(x)
#         x = self.gat2(x,edge_index)
#         return x

# class Sage(nn.Module):
#     def __init__(self,nfeat, nclass):
#         super(Sage, self).__init__()
#         self.sage1 = gnn.SAGEConv(nfeat,64)
#         self.sage2 = gnn.SAGEConv(64,nclass)
#         return

#     def forward(self, x,edge_index):
#         x = self.sage1(x,edge_index)
#         x = F.sigmoid(x)
#         x = self.sage2(x,edge_index)
#         return x

# class Gcn(nn.Module):
#     def __init__(self,nfeat, nclass):
#         super(Gcn,self).__init__()
#         self.gcn1 = gnn.GCNConv(nfeat,64)
#         self.gcn2 = gnn.GCNConv(64,nclass)
#         self.gcn = gnn.GCNConv(nfeat,nclass)
#         return

#     def forward(self, x,edge_index):
#         x = self.gcn1(x,edge_index)
#         x = F.relu(x)
#         x = self.gcn2(x,edge_index)
#         x = F.softmax(x)
#         return x

# class Sgc(nn.Module):
#     def __init__(self,nfeat, nclass):
#         super(Sgc,self).__init__()
#         self.sgc = gnn.SGConv(nfeat,nclass,2,True)
#         return

#     def forward(self, x,edge_index):
#         x = self.sgc(x,edge_index)
#         return x

# class Appnp(nn.Module):
#     def __init__(self,nfeat, nclass):
#         super(Appnp,self).__init__()
#         self.linear = nn.Linear(nfeat,64)
#         self.linear2 = nn.Linear(64,nclass)
#         self.appnp = gnn.APPNP(K=10, alpha=0.1)
#         return

#     def forward(self, x,edge_index):
#         x = F.relu(self.linear(x))
#         x = self.linear2(x)
#         x = self.appnp(x,edge_index)
#         return F.log_softmax(x, dim=-1)

# class Agnn(nn.Module):
#     def __init__(self,nfeat, nclass):
#         super(Agnn,self).__init__()
#         self.linear = nn.Linear(nfeat,16)
#         self.agnn1 = gnn.AGNNConv()
#         self.agnn2 = gnn.AGNNConv()
#         self.agnn3 = gnn.AGNNConv()
#         self.agnn4 = gnn.AGNNConv()
#         self.linear2 = nn.Linear(16,nclass)
#         return

#     def forward(self, x,edge_index):
#         x = F.relu(self.linear(x))
#         x = self.agnn1(x,edge_index)
#         x = self.agnn2(x,edge_index)
#         x = self.agnn3(x,edge_index)
#         # x = self.agnn4(x,edge_index)
#         x = F.softmax(self.linear2(x))
#         return x

# class Arma(nn.Module):
#     def __init__(self,nfeat, nclass):
#         super(Arma,self).__init__()
#         self.arma1 = gnn.ARMAConv(nfeat, 16,num_stacks =2)
#         self.arma2 = gnn.ARMAConv(16, nclass,num_stacks =2)
#         return

#     def forward(self, x,edge_index):
#         x = self.arma1(x,edge_index)
#         x = self.arma2(x,edge_index)
#         return x

# class Gated(nn.Module):
#     def __init__(self,nfeat, nclass):
#         super(Gated,self).__init__()
#         self.linear1 = nn.Linear(nfeat,64)
#         self.gated = gnn.GatedGraphConv(64,3)
#         self.linear = nn.Linear(64,nclass)
#         return

#     def forward(self, x,edge_index):
#         x = F.relu(self.linear1(x))
#         x = self.gated(x,edge_index)
#         x = self.linear(x)
#         return F.softmax(x)

# reinformenced model implementation
torch.cuda.manual_seed(0)


class Gcn(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Gcn, self).__init__()
        hdim = 64
        self.lnin = nn.Linear(nfeat, hdim)
        self.convs = nn.ModuleList([gnn.GCNConv(hdim, hdim) for _ in range(4)])
        # self.lnout1 = nn.Linear(hdim, int(hdim/2), bias=True)
        # self.lnout2 = nn.Linear(int(hdim/2), int(hdim/4), bias=True)
        # self.lnout3 = nn.Linear(int(hdim/4), nclass, bias=True)
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        for conv in self.convs:
            x = x+F.relu(conv(x, edge_index))
        # x = F.relu(self.lnout1(x))
        # x = F.relu(self.lnout2(x))
        # x = self.lnout3(x)
        x = self.lnout(x)
        return x


class Sage(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Sage, self).__init__()
        hdim = 64
        self.lnin = nn.Linear(nfeat, hdim)
        self.convs = nn.ModuleList(
            [gnn.SAGEConv(hdim, hdim) for _ in range(4)])
        # self.lnout1 = nn.Linear(hdim, int(hdim/2), bias=True)
        # self.lnout2 = nn.Linear(int(hdim/2), int(hdim/4), bias=True)
        # self.lnout3 = nn.Linear(int(hdim/4), nclass, bias=True)
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        for conv in self.convs:
            h_in = x
            x = F.relu(conv(x, edge_index))
            x = h_in + x
        # x = F.relu(self.lnout1(x))
        # x = F.relu(self.lnout2(x))
        # x = self.lnout3(x)
        x = self.lnout(x)
        return x


class Gat(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Gat, self).__init__()
        hdim = 64
        self.lnin = nn.Linear(nfeat, hdim)
        self.convs = nn.ModuleList(
            [gnn.GATConv(hdim, hdim) for _ in range(4)])
        # self.lnout1 = nn.Linear(hdim, int(hdim/2), bias=True)
        # self.lnout2 = nn.Linear(int(hdim/2), int(hdim/4), bias=True)
        # self.lnout3 = nn.Linear(int(hdim/4), nclass, bias=True)
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        for conv in self.convs:
            h_in = x
            x = F.elu(conv(x, edge_index))
            x = h_in + x
        # x = F.relu(self.lnout1(x))
        # x = F.relu(self.lnout2(x))
        # x = self.lnout3(x)
        x = self.lnout(x)
        return x


class Sgc(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Sgc, self).__init__()
        hdim = 64
        self.lnin = nn.Linear(nfeat, hdim)
        self.convs = nn.ModuleList(
            [gnn.SGConv(hdim, hdim, 1) for _ in range(4)])
        # self.lnout1 = nn.Linear(hdim, int(hdim/2), bias=True)
        # self.lnout2 = nn.Linear(int(hdim/2), int(hdim/4), bias=True)
        # self.lnout3 = nn.Linear(int(hdim/4), nclass, bias=True)
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        for conv in self.convs:
            h_in = x
            x = F.relu(conv(x, edge_index))
            x = h_in + x
        # x = F.relu(self.lnout1(x))
        # x = F.relu(self.lnout2(x))
        # x = self.lnout3(x)
        x = self.lnout(x)
        return x


class Appnp(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Appnp, self).__init__()
        hdim = 64
        self.lnin = nn.Linear(nfeat, hdim)
        self.convs = nn.ModuleList(
            [gnn.APPNP(K=1, alpha=0.1) for _ in range(4)])
        self.appnp = gnn.APPNP(K=4, alpha=0.1)
        # self.lnout1 = nn.Linear(hdim, int(hdim/2), bias=True)
        # self.lnout2 = nn.Linear(int(hdim/2), int(hdim/4), bias=True)
        # self.lnout3 = nn.Linear(int(hdim/4), nclass, bias=True)
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        # for conv in self.convs:
        #     h_in = x
        #     x = F.relu(conv(x, edge_index))
        #     x = h_in + x
        x = F.relu(self.appnp(x, edge_index))
        # x = F.relu(self.lnout1(x))
        # x = F.relu(self.lnout2(x))
        # x = self.lnout3(x)
        x = self.lnout(x)
        return x


class Agnn(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Agnn, self).__init__()
        hdim = 64
        self.lnin = nn.Linear(nfeat, hdim)
        self.convs = nn.ModuleList(
            [gnn.AGNNConv() for _ in range(4)])
        # self.lnout1 = nn.Linear(hdim, int(hdim/2), bias=True)
        # self.lnout2 = nn.Linear(int(hdim/2), int(hdim/4), bias=True)
        # self.lnout3 = nn.Linear(int(hdim/4), nclass, bias=True)
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        for conv in self.convs:
            h_in = x
            x = F.relu(conv(x, edge_index))
            x = h_in + x
        # x = F.relu(self.lnout1(x))
        # x = F.relu(self.lnout2(x))
        # x = self.lnout3(x)
        x = self.lnout(x)
        return x


class Arma(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Arma, self).__init__()
        hdim = 64
        self.lnin = nn.Linear(nfeat, hdim)
        self.convs = nn.ModuleList(
            [gnn.ARMAConv(hdim, hdim, 2) for _ in range(4)])
        # self.lnout1 = nn.Linear(hdim, int(hdim/2), bias=True)
        # self.lnout2 = nn.Linear(int(hdim/2), int(hdim/4), bias=True)
        # self.lnout3 = nn.Linear(int(hdim/4), nclass, bias=True)
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        for conv in self.convs:
            h_in = x
            x = F.relu(conv(x, edge_index))
            x = h_in + x
        # x = F.relu(self.lnout1(x))
        # x = F.relu(self.lnout2(x))
        # x = self.lnout3(x)
        x = self.lnout(x)
        return x


class Gated(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Gated, self).__init__()
        hdim = 64
        self.lnin = nn.Linear(nfeat, hdim)
        self.convs = nn.ModuleList(
            [gnn.GatedGraphConv(hdim, 1) for _ in range(4)])
        # self.lnout1 = nn.Linear(hdim, int(hdim/2), bias=True)
        # self.lnout2 = nn.Linear(int(hdim/2), int(hdim/4), bias=True)
        # self.lnout3 = nn.Linear(int(hdim/4), nclass, bias=True)
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        for conv in self.convs:
            h_in = x
            x = F.relu(conv(x, edge_index))
            x = h_in + x
        # x = F.relu(self.lnout1(x))
        # x = F.relu(self.lnout2(x))
        # x = self.lnout3(x)
        x = self.lnout(x)
        return x


if __name__ == "__main__":
    # model = GraphNas(32)
    # if torch.cuda.is_available():
    #     model.cuda()
    # dummy_code = model.generate_code()
    # supermask = model.parse_code(dummy_code)
    # print(dummy_code)
    # print(supermask)
    model = Gat(6, 6)
    # for name, param in model.named_parameters():
    #     print(name, param)
    print(model.convs[0].lin_src._parameters)
    pass

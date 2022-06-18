import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import torch

torch.cuda.manual_seed(0)


class Gcn(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Gcn, self).__init__()
        hdim = 64
        self.lnin = nn.Linear(nfeat, hdim)
        self.convs = nn.ModuleList([gnn.GCNConv(hdim, hdim) for _ in range(4)])
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        for conv in self.convs:
            x = x+F.relu(conv(x, edge_index))
        x = self.lnout(x)
        return x


class Sage(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Sage, self).__init__()
        hdim = 64
        self.lnin = nn.Linear(nfeat, hdim)
        self.convs = nn.ModuleList(
            [gnn.SAGEConv(hdim, hdim) for _ in range(4)])
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        for conv in self.convs:
            h_in = x
            x = F.relu(conv(x, edge_index))
            x = h_in + x
        x = self.lnout(x)
        return x


class Gat(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Gat, self).__init__()
        hdim = 64
        self.lnin = nn.Linear(nfeat, hdim)
        self.convs = nn.ModuleList(
            [gnn.GATConv(hdim, hdim) for _ in range(4)])
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        for conv in self.convs:
            h_in = x
            x = F.elu(conv(x, edge_index))
            x = h_in + x
        x = self.lnout(x)
        return x


class Sgc(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Sgc, self).__init__()
        hdim = 64
        self.lnin = nn.Linear(nfeat, hdim)
        self.convs = nn.ModuleList(
            [gnn.SGConv(hdim, hdim, 1) for _ in range(4)])
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        for conv in self.convs:
            h_in = x
            x = F.relu(conv(x, edge_index))
            x = h_in + x
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
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        x = F.relu(self.appnp(x, edge_index))
        x = self.lnout(x)
        return x


class Agnn(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Agnn, self).__init__()
        hdim = 64
        self.lnin = nn.Linear(nfeat, hdim)
        self.convs = nn.ModuleList(
            [gnn.AGNNConv() for _ in range(4)])
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        for conv in self.convs:
            h_in = x
            x = F.relu(conv(x, edge_index))
            x = h_in + x
        x = self.lnout(x)
        return x


class Arma(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Arma, self).__init__()
        hdim = 64
        self.lnin = nn.Linear(nfeat, hdim)
        self.convs = nn.ModuleList(
            [gnn.ARMAConv(hdim, hdim, 2) for _ in range(4)])
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        for conv in self.convs:
            h_in = x
            x = F.relu(conv(x, edge_index))
            x = h_in + x
        x = self.lnout(x)
        return x


class Gated(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Gated, self).__init__()
        hdim = 64
        self.lnin = nn.Linear(nfeat, hdim)
        self.convs = nn.ModuleList(
            [gnn.GatedGraphConv(hdim, 1) for _ in range(4)])
        self.lnout = nn.Linear(hdim, nclass)
        return

    def forward(self, x, edge_index):
        x = self.lnin(x)
        for conv in self.convs:
            h_in = x
            x = F.relu(conv(x, edge_index))
            x = h_in + x
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

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

class Gat(nn.Module):
    def __init__(self,nfeat, nclass):
        super(Gat,self).__init__()
        self.gat1 = gnn.GATConv(nfeat,8,8,True)
        self.gat2 = gnn.GATConv(64,nclass)
        return

    def forward(self, x,edge_index):
        x = self.gat1(x,edge_index)
        x = F.elu(x)
        x = self.gat2(x,edge_index)
        return x

class Sage(nn.Module):
    def __init__(self,nfeat, nclass):
        super(Sage, self).__init__()
        self.sage1 = gnn.SAGEConv(nfeat,64)
        self.sage2 = gnn.SAGEConv(64,nclass)
        return

    def forward(self, x,edge_index):
        x = self.sage1(x,edge_index)
        x = F.sigmoid(x)
        x = self.sage2(x,edge_index)
        return x

class Gcn(nn.Module):
    def __init__(self,nfeat, nclass):
        super(Gcn,self).__init__()
        self.gcn1 = gnn.GCNConv(nfeat,64)
        self.gcn2 = gnn.GCNConv(64,nclass)
        self.gcn = gnn.GCNConv(nfeat,nclass)
        return

    def forward(self, x,edge_index):
        x = self.gcn1(x,edge_index)
        x = F.relu(x)
        x = self.gcn2(x,edge_index)
        x = F.softmax(x)
        return x

class Sgc(nn.Module):
    def __init__(self,nfeat, nclass):
        super(Sgc,self).__init__()
        self.sgc = gnn.SGConv(nfeat,nclass,2,True)
        return

    def forward(self, x,edge_index):
        x = self.sgc(x,edge_index)
        return x

class Appnp(nn.Module):
    def __init__(self,nfeat, nclass):
        super(Appnp,self).__init__()
        self.linear = nn.Linear(nfeat,64)
        self.linear2 = nn.Linear(64,nclass)
        self.appnp = gnn.APPNP(K=10, alpha=0.1)
        return

    def forward(self, x,edge_index):
        x = F.relu(self.linear(x))
        x = self.linear2(x)
        x = self.appnp(x,edge_index)
        return F.log_softmax(x, dim=-1)

class Agnn(nn.Module):
    def __init__(self,nfeat, nclass):
        super(Agnn,self).__init__()
        self.linear = nn.Linear(nfeat,16)
        self.agnn1 = gnn.AGNNConv()
        self.agnn2 = gnn.AGNNConv()
        self.agnn3 = gnn.AGNNConv()
        self.agnn4 = gnn.AGNNConv()
        self.linear2 = nn.Linear(16,nclass)
        return

    def forward(self, x,edge_index):
        x = F.relu(self.linear(x))
        x = self.agnn1(x,edge_index)
        x = self.agnn2(x,edge_index)
        x = self.agnn3(x,edge_index)
        # x = self.agnn4(x,edge_index)
        x = F.softmax(self.linear2(x))
        return x

class Arma(nn.Module):
    def __init__(self,nfeat, nclass):
        super(Arma,self).__init__()
        self.arma1 = gnn.ARMAConv(nfeat, 16,num_stacks =2)
        self.arma2 = gnn.ARMAConv(16, nclass,num_stacks =2)
        return

    def forward(self, x,edge_index):
        x = self.arma1(x,edge_index)
        x = self.arma2(x,edge_index)
        return x

class Gated(nn.Module):
    def __init__(self,nfeat, nclass):
        super(Gated,self).__init__()
        self.linear1 = nn.Linear(nfeat,64)
        self.gated = gnn.GatedGraphConv(64,3)
        self.linear = nn.Linear(64,nclass)
        return

    def forward(self, x,edge_index):
        x = F.relu(self.linear1(x))
        x = self.gated(x,edge_index)
        x = self.linear(x)
        return F.softmax(x)

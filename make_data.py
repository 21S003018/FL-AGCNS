import pickle
from random import randint
from torch_geometric.datasets import Coauthor, CoraFull, Planetoid
from utils import PartitionTool
import torch
CORAFULL = "corafull"
PHYSICS = "Physics"


class Maker():
    def __init__(self) -> None:
        self.partitioner = PartitionTool()
        pass

    def get_data(self, name):
        if name == CORAFULL:
            root = "data/corafull/"
            dataset = CoraFull(root=root)
        elif name == PHYSICS:
            root = "data/"
            dataset = Coauthor(root=root, name="Physics")
        elif name == "cora":
            root = "data/"
            dataset = Planetoid(root="data/", name="cora")
        elif name == "citeseer":
            root = "data/"
            dataset = Planetoid(root="data/", name="citeseer")
        elif name == "pubmed":
            root = "data/"
            dataset = Planetoid(root="data/", name="pubmed")
        return dataset.data

    def get_mask(self, data):
        visit = torch.ones((len(data.y))) * 20
        train_mask = torch.zeros(len(data.y)).bool()
        for index in range(len(data.y)):
            label = data.y[index]
            if visit[label] > 0:
                train_mask[index] = True
                visit[label] -= 1
        val_mask = torch.zeros(len(data.y)).bool()
        num = 0
        while True:
            index = randint(0, len(data.y)-1)
            if train_mask[index] == False and val_mask[index] == False:
                val_mask[index] = True
                num += 1
            if num == 500:
                break
        test_mask = torch.zeros(len(data.y)).bool()
        num = 0
        while True:
            index = randint(0, len(data.y)-1)
            if train_mask[index] == False and val_mask[index] == False and test_mask[index] == False:
                test_mask[index] = True
                num += 1
            if num == 1000:
                break
        return train_mask, val_mask, test_mask

    def partition(self, name, k=3, copy_node=False):
        data = self.get_data(name)
        train_mask, val_mask, test_mask = self.get_mask(data)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        print(train_mask.sum(), val_mask.sum(), test_mask.sum())
        datas = self.partitioner.partition_subgraph(data, k, copy_node)
        for i in range(k):
            path = "data/{}/{}_{}copynode.pkl".format(
                name, i, "" if copy_node else "un")
            with open(path, "wb") as f:
                pickle.dump(datas[i], f)
        for data in datas:
            print(data, data.train_mask.sum(),
                  data.val_mask.sum(), data.test_mask.sum())
        return


if __name__ == '__main__':
    maker = Maker()
    maker.partition('cora', 3, True)
    # maker.partition('citeseer', 3, True)
    # maker.partition('pubmed', 12, True)
    # maker.partition(CORAFULL, 16, True)
    # maker.partition(PHYSICS, 18, True)
    pass

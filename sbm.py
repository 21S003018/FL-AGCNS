from models import SonNet, DynamicSonNet
import pickle
import torch
import time
from torch import optim
import torch.nn.functional as F
import numpy as np


def num_params(model):
    params = list(model.named_parameters())
    k = 0
    for _, i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    return k


def accuracy(output, labels):
    '''
    get accuracy
    :param output:
    :param labels:
    :return:
    '''
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum().item()
    return correct / len(labels)


# load data
datas = []
for i in range(1000):
    path = f'data/SBM/{i}_uncopynode.pkl'
    with open(path, 'rb') as f:
        datas.append(pickle.load(f).to('cuda:0'))


EPOCHS = 50


def train_single_epoch(model):
    model.train()
    optimizer = optim.Adam(
        model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    st_time = time.time()
    best_val_accu = 0
    for epoch in range(EPOCHS):
        losses = 0
        train_accu = 0
        val_accu = 0
        test_accu = 0
        for iter, data in enumerate(datas):
            preds = model(data.x, data.edge_index)
            labels = data.y[data.train_mask]
            loss = F.cross_entropy(
                preds[data.train_mask], data.y[data.train_mask], weight=data.weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            train_accu += accuracy(preds[data.train_mask],
                                   data.y[data.train_mask])
            val_accu += accuracy(preds[data.val_mask],
                                 data.y[data.val_mask])
            test_accu += accuracy(preds[data.test_mask],
                                  data.y[data.test_mask])
        if val_accu > best_val_accu:
            best_val_accu = val_accu
            torch.save(model.state_dict(), 'sbm_model.pth')
            print('\tIter{}, loss:{},train_acc:{},val_acc:{},test_acc:{},use_time:{}'.format(
                epoch+1, losses/(iter+1), train_accu/(iter+1), val_accu/(iter+1), test_accu/(iter+1), round(time.time()-st_time)))
        scheduler.step(losses/(iter+1))

    test_accu = 0
    model.load_state_dict(torch.load('sbm_model.pth'))
    for iter, data in enumerate(datas):
        preds = model(data.x, data.edge_index)
        test_accu += accuracy(preds[data.test_mask],
                              data.y[data.test_mask])*len(data.y[data.test_mask])/116820
    return test_accu


def train(supermask):
    with open('tmp.pkl', 'wb') as f:
        pickle.dump(supermask, f)
    accus = []
    for i in range(10):
        print(f"{i}-th random evaluation")
        model = SonNet(6, 6).to('cuda:0')
        accus.append(train_single_epoch(model))
    accus = np.array(accus)
    print(f"{supermask}, test_accu_mean:{accus.mean()},test_accu_sd:{accus.std()}, num_params:{num_params(model)}")


# code
# supermask = [5, 2, 1+12*1, 4+12*2, 5+12*3, (5+12*4)*0, (5+12*5)*0, 5]  # random sure
# supermask = [3, 4, 5+12*1, 9+12*2, 2, 12+12*4, 0, 2] # agcns old
# supermask = [5, 7, 4+12*1, 8+12*2, 2+12*3, 10+12*1, 3+12*2, 5] # new agcns 1
# supermask = [5, 7, 4+12*1, 8+12*2, 2+12*3, 6+12*1, 2+12*2, 5]  # new agcns 2
# supermask = [5, 10, 6+12*1, 1+12*2, 2, 0, 0, 5]  # graphnas sure
# supermask = [3, 1, 1+12*1, 10+12*2, 2+12*2, 0, 0, 4]  # darts sure
# supermask = [4, 1, 1+12*1, 1+12*2, 10, 2, 0, 4]  # fednas sure
# supermask = [5, 8, 3+12*1, 1+12*2, 10+12*1, 0, 0, 4] # rl sure

supermasks = [[5, 2, 1+12*1, 4+12*2, 5+12*3, (5+12*4)*0, (5+12*5)*0, 5],
              [3, 4, 5+12*1, 9+12*2, 2, 12+12*4, 0, 2],
              [5, 7, 4+12*1, 8+12*2, 2+12*3, 10+12*1, 3+12*2, 5],
              [5, 7, 4+12*1, 8+12*2, 2+12*3, 6+12*1, 2+12*2, 5],
              [5, 10, 6+12*1, 1+12*2, 2, 0, 0, 5],
              [3, 1, 1+12*1, 10+12*2, 2+12*2, 0, 0, 4],
              [4, 1, 1+12*1, 1+12*2, 10, 2, 0, 4],
              [5, 8, 3+12*1, 1+12*2, 10+12*1, 0, 0, 4]]


def main():
    for supermask in supermasks:
        print('evaluate supermask: {}'.format(supermask))
        train(supermask)


if __name__ == '__main__':
    main()

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HIDDENSIZE = 32
INPUT_SIZE = HIDDENSIZE
LAYERS = 1
LR = 0.02


class GraphNas(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(GraphNas, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDENSIZE,
            num_layers=LAYERS,
            # batch_first=True
        )
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDENSIZE,
            num_layers=LAYERS,
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
        h_state = torch.FloatTensor(np.zeros((1, 1, INPUT_SIZE))).cpu()
        if torch.cuda.is_available():
            h_state = h_state.cuda()
        else:
            h_state = h_state.cpu()
        x = h_state
        res = []
        for i in range(1, 8 + 1):
            r_out, h_state = self.lstm(x, h_state)
            res.append(
                F.softmax(eval("self.out{}".format(i))(r_out[0]), dim=1))
            x = h_state
        return res

    def get_loss(self, dummy_code, supermask, R):
        code = dummy_code[0]
        # loss = torch.log(torch.max(code)) * (R - self.b)
        # loss.backward()
        # loss_sum += loss.item()
        losses = []
        for i in range(len(dummy_code)):
            index = supermask[i]
            if i not in [0, 7]:
                index -= 1
            losses.append(torch.log(code[index]) * (R - self.b))
        # idx = 1
        # for code in dummy_code:
        #     exec(
        #         "loss{} = loss{} + (torch.log(torch.max(code))*({} - self.b))".format(idx, idx - 1, R))
        #     idx += 1
        self.b = self.beta * self.b + (1 - self.beta) * R
        return torch.stack(losses).mean()

    def parse_code(self, dummy_code):
        # supermask = []
        # idx = 1
        # for code in dummy_code:
        #     if idx == 1 or idx == 8:
        #         supermask.append(np.argmax(code) + 1)
        #     else:
        #         supermask.append(np.argmax(code))
        #     idx += 1
        supermask = dummy_code.multinomial(
            num_samples=1).reshape(len(dummy_code))
        supermask[1:-1] += 1
        return supermask


def get_reward(supermask):
    return 0.5


def train(EPOCHS=100):
    supermasks = []
    model = GraphNas(INPUT_SIZE)
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()

    for i in range(EPOCHS):
        dummy_code = model.generate_code()
        supermask = model.parse_code(dummy_code)
        supermasks.append(supermask)
        R = get_reward(supermask)
        loss = model.get_loss(dummy_code, supermask, R)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return supermasks


if __name__ == '__main__':
    print(torch.__version__)
    pass

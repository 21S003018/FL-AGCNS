from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.data import ClusterData, Data, ClusterLoader
from numpy.random import randint, random
import pickle
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OBJ_END = 'eofeofeof'
IP_PORT_BASE = 9000
ADDR = "addr"

# ip oprt


def reset_ipport():
    for label in [''] + list(range(100)):
        path = '{}/ip_oprt{}.pkl'.format(ADDR, label)
        with open(path, 'wb') as f:
            pickle.dump(('localhost', IP_PORT_BASE + 100), f)
    return


def read_ipport():
    for label in [''] + list(range(100)):
        path = '{}/ip_oprt{}.pkl'.format(ADDR, label)
        with open(path, 'rb') as f:
            print(pickle.load(f))
    return


def get_ip_port(id=''):
    path = '{}/ip_oprt{}.pkl'.format(ADDR, id)
    with open(path, 'rb') as f:
        ans = pickle.load(f)
    if ans[1] >= IP_PORT_BASE + 1000:
        new_ip_port = (ans[0], IP_PORT_BASE)
    else:
        new_ip_port = (ans[0], ans[1] + 100)
    with open(path, 'wb') as f:
        pickle.dump(new_ip_port, f)
    return ans


# net connect
def socket_recv_with_response(socket):
    '''
    recv message with specified socket and response okk
    :param socket:
    :return: decoded data
    '''
    data = b''
    while True:
        packet = socket.recv(1024*4)
        data += packet
        if data[-9:].__contains__(OBJ_END.encode()):
            data = data[:-9]
            break
    socket.send('okk'.encode())
    if data.__sizeof__() >= 10*1024*1024:
        # logger.info('recv big obj:{:.2f}M'.format(
        #     data.__sizeof__()/(1024*1024)))
        print('recv big obj:{:.2f}M'.format(data.__sizeof__()/(1024*1024)))
    return pickle.loads(data)


def socket_send_with_waiting_response(socket, mess):
    '''
    send out message and waiting for okk as response
    :param socket:
    :param mess:
    :return:
    '''
    mess = pickle.dumps(mess)
    socket.send(mess+OBJ_END.encode())
    socket.recv(1024)
    return


def socket_recv(socket):
    '''
    only recv message
    not send okk
    :param socket:
    :return:
    '''
    data = b''
    while True:
        packet = socket.recv(1024*4)
        # if packet.__contains__(OBJ_END.encode()):
        #     data += packet.replace(OBJ_END.encode(), b'')
        #     break
        data += packet
        if data[-9:].__contains__(OBJ_END.encode()):
            data = data[:-9]
            break
    if data.__sizeof__() >= 10*1024*1024:
        # logger.info('recv big obj:{:.2f}M'.format(
        #     data.__sizeof__()/(1024*1024)))
        print('recv big obj:{:.2f}M'.format(data.__sizeof__()/(1024*1024)))
    return pickle.loads(data)


def socket_send(socket, mess):
    '''
    only send message
    not waiting for response
    :param socket:
    :param mess:
    :return:
    '''
    mess = pickle.dumps(mess)
    socket.send(mess+OBJ_END.encode())
    return


class Contacter():
    '''
    use this class as a interface to process data recv and send
    '''

    def __init__(self):
        self.socket = None
        return

    def recv_with_res(self):
        data = socket_recv_with_response(self.socket)
        return data

    def recv(self):
        data = socket_recv(self.socket)
        return data

    def send_with_waiting_res(self, mess):
        socket_send_with_waiting_response(self.socket, mess)
        return

    def send(self, mess):
        socket_send(self.socket, mess)
        return


# serealize data
def setalize(supermasks):
    '''
    remove the same supermask, ank keep only one copy for each supermask
    :param supermasks:
    :return:
    '''
    sm_set = set()
    for tmp in supermasks:
        sm_set.add(str(tmp))
    setlized_sms = []
    for tmp in sm_set:
        setlized_sms.append(eval(tmp))
    return setlized_sms


def serialize_model(model):
    '''
    encode all the parameters in a nn model
    :param model: nn.Module
    :return: dic
    '''
    param_dict = {}
    for name, param in model.named_parameters():
        param_dict[name] = param.to('cpu')
    import utils
    return param_dict


# cal edge_feat
def cal_edge_attr_for_gmmconv(edge_index):
    '''
    contruct edge_attr on specified dataset for gmmconv
    :param edge_index:
    :return:
    '''
    embedding_matrix = torch.bincount(edge_index[0]).unsqueeze(1)
    edge_attr = F.embedding(
        edge_index, embedding_matrix).pow(-0.5).T.squeeze(0)
    return edge_attr.to(edge_index.device)


# evo related
MASKRANGE = [[1, 5], [1, 12], [13, 24], [
    25, 36], [1, 48], [1, 60], [1, 72], [1, 5]]


def rand_element(idx):
    zero_p = [0, 0, 0, 0, 0.2, 0.2, 0.2, 0]
    rand_seed = zero_p[idx]
    if random() < rand_seed:
        return 0
    return randint(MASKRANGE[idx][0], MASKRANGE[idx][1] + 1)


def random_supermask():
    '''
    generate random supermask
    :return:
    '''
    mask = []
    idx = 0
    for r in MASKRANGE:
        mask.append(rand_element(idx))
        idx += 1
    return mask


def cross_over(original, target):
    son = []
    for i in range(len(MASKRANGE)):
        if np.random.random() > 0.5:
            son.append(target[i])
        else:
            son.append(original[i])
    return son


def mutate(original):
    c = 0.125
    son = []
    for i, range in enumerate(MASKRANGE):
        if np.random.random() < c:
            son.append(rand_element(i))
        else:
            son.append(original[i])
    return son


# metrics
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


def num_correct(output, labels):
    '''
    get the number of correct nodes that has been predicted correctly
    :param output:
    :param labels:
    :return:
    '''
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum().item()
    return int(correct)


def num_params(model):
    params = list(model.named_parameters())
    k = 0
    for _, i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    return k


if __name__ == "__main__":
    p = 0.2
    print(p**3, 3*p**2*(1-p), 3*p*(1-p)**2, (1-p)**3)
    pass

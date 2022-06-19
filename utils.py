import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import ClusterData, Data
from numpy.random import randint, random
import pickle
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OBJ_END = 'eofeofeof'
IP_PORT_BASE = 8000
ADDR = "addr"


# ip port
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


# net contact
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
        data += packet
        if data[-9:].__contains__(OBJ_END.encode()):
            data = data[:-9]
            break
    if data.__sizeof__() >= 10*1024*1024:
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
    # print(param_dict)
    return param_dict


# partition datasets
class PartitionTool():
    def __init__(self):
        return

    def partition_subgraph(self, data, k, copy_node=False):
        '''
        partition a data into k parts
        :param data: torch_geometric.data.Data
        :param k:
        :param copy_node:
        :return:
        '''
        cluster = ClusterData(data, k)
        print(len(cluster.perm))
        # cluster.perm = torch.Tensor(np.random.shuffle(np.array(cluster.perm)))
        # return
        # cluster.perm =
        for i in range(k):
            exec('self.idx_{} = cluster.perm[cluster.partptr[{}]:cluster.partptr[{}]]'.format(
                i, i, i + 1))
        print('cluster over!')
        # expand
        if copy_node == True:
            def expand(edge_index, idx):
                new_edge_index_u = []
                new_edge_index_v = []
                extra_node = set()
                for tmp in idx:
                    for v in v2e[int(tmp)]:
                        extra_node.add(v)
                        extra_node.add(int(tmp))
                    new_edge_index_u += [int(tmp)
                                         for i in range(len(v2e[int(tmp)]))]
                    new_edge_index_v += v2e[int(tmp)]
                # print(len(id_set))
                print('over for loop')
                return torch.LongTensor(list(extra_node)), torch.LongTensor([new_edge_index_u, new_edge_index_v])
        else:
            def expand(edge_index, idx):
                new_edge_index_u = []
                new_edge_index_v = []
                id_set = set()
                for tmp in idx:
                    id_set.add(int(tmp))
                for tmp in idx:
                    for v in v2e[int(tmp)]:
                        if id_set.__contains__(v):
                            new_edge_index_u.append(int(tmp))
                            new_edge_index_v.append(v)
                return torch.LongTensor(list(idx)), torch.LongTensor([new_edge_index_u, new_edge_index_v])

        edge_index = np.array(data.edge_index)
        v2e = {}
        # for i in range(len(edge_index[0])):
        #     u, v = edge_index[0][i], edge_index[1][i]
        #     v2e[u] = []
        #     v2e[v] = []
        for u in range(len(data.x)):
            v2e[u] = []
        for i in range(len(edge_index[0])):
            u, v = edge_index[0][i], edge_index[1][i]
            v2e[u].append(v)

        for i in range(k):
            print(i)
            tmp_idx_set = set()
            for tmp in eval('self.idx_{}'.format(i)):
                tmp_idx_set.add(int(tmp))
            exec('self.idx_{}, self.edge_index_{} = expand(edge_index, self.idx_{})'.format(
                i, i, i))
            new_idx_set = set()
            for tmp in eval('self.idx_{}'.format(i)):
                new_idx_set.add(int(tmp))
            exec('self.extra_idx_{} = list(new_idx_set.difference(tmp_idx_set))'.format(i))
        print('subgraph expands over')

        def hashing(idx, edge_index):
            idx = np.array(idx)
            edge_index = np.array(edge_index)
            idx = np.sort(idx)
            idx_map = np.zeros(int(np.max(idx)) + 1)
            for i in range(len(idx)):
                idx_map[int(idx[i])] = i
            # print(idx_map)
            for i in range(len(edge_index[0])):
                edge_index[0][i], edge_index[1][i] = idx_map[int(
                    edge_index[0][i])], idx_map[int(edge_index[1][i])]
            return torch.LongTensor(edge_index)

        def hash_extra_idx(idx, extra_idx):
            idx = np.array(idx)
            idx = np.sort(idx)
            idx_map = np.zeros(int(np.max(idx)) + 1)
            for i in range(len(idx)):
                idx_map[int(idx[i])] = i
            for i in range(len(extra_idx)):
                extra_idx[i] = idx_map[extra_idx[i]]
            return torch.LongTensor(extra_idx)

        for i in range(k):
            print(i)
            exec(
                'self.edge_index_{} = hashing(self.idx_{},self.edge_index_{})'.format(i, i, i))
            exec('self.extra_idx_{} = hash_extra_idx(self.idx_{},self.extra_idx_{})'.format(
                i, i, i))
        print('hash over')
        x = data.x
        y = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        datas = []
        for i in range(k):
            print(i)
            exec('idx_{}_bool_tensor = torch.zeros(len(x)).bool()'.format(i))
            exec(
                'for idx in self.idx_{}:idx_{}_bool_tensor[idx] = True'.format(i, i))
            exec('x_{} = x[idx_{}_bool_tensor]'.format(i, i))
            exec('y_{} = y[idx_{}_bool_tensor]'.format(i, i))
            exec('train_mask_{} = train_mask[idx_{}_bool_tensor]'.format(i, i))
            exec('val_mask_{} = val_mask[idx_{}_bool_tensor]'.format(i, i))
            exec('test_mask_{} = test_mask[idx_{}_bool_tensor]'.format(i, i))

            for tmp in eval('self.extra_idx_{}'.format(i)):
                exec('train_mask_{}[tmp] = False'.format(i))
                exec('val_mask_{}[tmp] = False'.format(i))
                exec('test_mask_{}[tmp] = False'.format(i))

            exec('data_{} = Data(x=x_{},y=y_{},edge_index=self.edge_index_{},train_mask=train_mask_{},val_mask=val_mask_{},test_mask=test_mask_{})'.format(
                i, i, i, i, i, i, i))
            datas.append(eval('data_{}'.format(i)))
        print('sub graph pack over')
        return datas


# cal edge_feat
def cal_edge_attr_for_gmmconv(edge_index):
    '''
    contruct edge_attr on specified dataset for gmmconv
    :param edge_index:
    :return:
    '''
    embedding_matrix = torch.bincount(
        edge_index[0], minlength=edge_index.max()+1).unsqueeze(1)+1
    edge_attr = F.embedding(
        edge_index, embedding_matrix).pow(-0.5).T.squeeze(0)
    return edge_attr.to(edge_index.device)


# evo related
MASKRANGE = [[1, 5], [1, 12], [13, 24], [
    25, 36], [1, 48], [1, 60], [1, 72], [1, 5]]


def rand_element(idx):
    zero_p = [0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0]
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
    # partitioner = PartitionTool()
    # partitioner.partition_subgraph()
    for i in range(18):
        with open('data/Physics/{}_{}copynode.pkl'.format(i, ''), 'rb') as f:
            data = pickle.load(f)
        print(data.edge_index.max(), data, data.train_mask.sum(),
              data.val_mask.sum(), data.test_mask.sum())
    # print(read_ipport())
    # x = torch.rand(5, 1)
    # y = torch.ones(5, 1)
    # print(x, x.where(x > 0.5, y))
    pass

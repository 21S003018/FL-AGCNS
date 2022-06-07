import socket
import random
from models import *
from external_models import *
from utils import Contacter
import utils
import time
import pickle
from numpy.random import randint
import numpy as np
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# static configuration
IP_PORT = ('localhost', 5100)
# BASE = 9
# gpu = [9, 10, 11]
BASE = 0
gpu = [1, 2, 3, 4, 5, 6, 7, 8]
# CLIENTBASE = 1
# gpu = [BASE, CLIENTBASE, CLIENTBASE]
COPY_NODE = False
# hyper-parameter
LR = 0.02
EVO_EPOCH = 250
SAMPLE_EPOCH = 5
NUM_POP = 60
SAMPLE_SIZE = 20
# mode
DEBUG = False


class Controller():
    def __init__(self, num_client):
        '''
        static configuration for num of clients, socket, device
        :param num_client:
        '''
        self.num_client = num_client
        # self.device = torch.device('cpu')
        self.device = torch.device('cuda:{}'.format(
            BASE) if torch.cuda.is_available() else 'cpu')
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(utils.get_ip_port())
        self.socket.listen(3)
        self.best_supermask = None
        self.best_accu = 0
        return

    def configure(self, model_name, dataset, nfeat, nclass):
        '''
        dynamic configuration for model,dataset,optimizer
        accept the connect request from client
        :param model_name:
        :param dataset:
        :param nfeat:
        :param nclass:
        :return:
        '''
        self.model = eval(model_name)(nfeat, nclass).to(self.device)
        self.nfeat = nfeat
        self.nclass = nclass
        self.model_name = model_name
        self.dataset = dataset
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=LR, weight_decay=5e-6)
        # log
        print('controller configures over, waiting for connect request from clients')
        self.clients = []
        for idx in range(self.num_client):
            client, addr = self.socket.accept()
            self.clients.append(client)
        return

    def work(self):
        '''
        begin train the model with the following command sequence:
        dataset
        model
        loop:train controlled by son class
        ...
        :return:
        '''
        # configure
        self.broadcast_with_waiting_res('dataset')
        self.broadcast_with_waiting_res(self.dataset)
        print('controller broadcasts datasetname:{} over'.format(self.dataset))
        # time.sleep(1)
        self.broadcast_with_waiting_res('model')
        self.broadcast_with_waiting_res(self.model_name)
        # self.broadcast_with_waiting_res(utils.serialize_model(self.model));print('controller broadcasts model over')
        self.broadcast(utils.serialize_model(self.model))
        self.blink_aggregate()
        print('controller broadcasts model over')
        # train
        # ...
        return

    def broadcast(self, message):
        '''
        broadcast the message to clients
        without waiting for the response
        :param message:
        :return:
        '''
        for socket in self.clients:
            utils.socket_send(socket, message)
        return

    def broadcast_with_waiting_res(self, message):
        '''
        send out the message and waiting for okk as response
        :param message:
        :return:
        '''
        for socket in self.clients:
            utils.socket_send_with_waiting_response(socket, message)
        return

    def aggregate(self):
        '''
        collect message from all the clients
        without response after recv message
        :return:
        '''
        res = []
        for socket in self.clients:
            res.append(utils.socket_recv(socket))
        return res

    def blink_aggregate(self):
        '''
        use as blocking the controller
        and fitting for the interaction protocol
        :return:
        '''
        for socket in self.clients:
            socket.recv(1024)
        return

    def aggregate_grad(self, grad_dicts):
        '''
        aggregate the gradient
        :param grad_dicts:
        :return:
        '''
        avg_grad_dict = {}
        for name in grad_dicts[0].keys():
            # print(name,grad_dicts[0][name],grad_dicts[1][name],grad_dicts[2][name])
            if not grad_dicts[0][name] == None:
                for i in range(self.num_client):
                    if avg_grad_dict.__contains__(name):
                        avg_grad_dict[name] += grad_dicts[i][name].to(
                            self.device)
                    else:
                        avg_grad_dict[name] = grad_dicts[i][name].to(
                            self.device)
                # avg_grad_dict[name] = grad_dicts[0][name].to(self.device)\
                #     + grad_dicts[1][name].to(self.device)\
                #     + grad_dicts[2][name].to(self.device)
                # + grad_dicts[3][name].to(self.device)
                # + grad_dicts[4][name].to(self.device)\
                # + grad_dicts[5][name].to(self.device)\
                # + grad_dicts[6][name].to(self.device)\
                # + grad_dicts[7][name].to(self.device)
            else:
                avg_grad_dict[name] = None
        self.update_grad(avg_grad_dict)
        return avg_grad_dict

    def update_grad(self, grad_dict):
        '''
        update the model with grad_dict
        :param grad_dict:
        :return:
        '''
        for name, grad in grad_dict.items():
            if not name.__contains__('.') and (name in ['alpha', 'gamma'] or name.__contains__('beta')):
                exec('self.model.{}.grad = grad'.format(name))
                continue

            name += 'ending'
            slots = name.split('.')
            layer = name.replace('.{}'.format(slots[len(slots)-1]), '')
            if slots[len(slots)-2].isdigit():
                layer = layer.replace('.{}'.format(
                    slots[len(slots)-2]), '[{}]'.format(slots[len(slots)-2]))
            label = slots[len(slots)-1].replace('ending', '')
            exec("self.model.{}._parameters['{}'].grad = grad".format(
                layer, label))
        self.optimizer.step()
        return

    def end_analyse(self):
        '''
        analyse the structure of the model
        output its parameters
        :return:
        '''
        idx = 0
        for name, param in self.model.named_parameters():
            print(name, param)
            idx += 1
            if idx > 1:
                break
        return

    def close(self):
        self.socket.close()
        return


class Client(Contacter):
    def __init__(self, id):
        '''
        static configuration for device, socket
        :param id:
        '''
        super(Client, self).__init__()
        self.id = id
        # self.device = torch.device('cpu')
        self.device = torch.device('cuda:{}'.format(
            gpu[id]) if torch.cuda.is_available() else 'cpu')
        self.ip_port = utils.get_ip_port(self.id)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return

    def configure(self, model, dataset, copy_node=COPY_NODE):
        '''
        dynamic configuration for model,optimizer,rate,dataset
        :param model: nn.Module
        :param dataset:
        :param copy_node:
        :return:
        '''
        self.model = model
        self.optimizer = optim.Adam(
            model.parameters(), lr=LR, weight_decay=5e-6)
        self.loss = None
        path = ''
        if dataset.lower() in ['cora', 'citeseer', 'pubmed', 'corafull', 'physics']:
            path = 'data/{}/{}_{}copynode.pkl'.format(
                dataset, self.id, ''if copy_node else'un')
            with open(path, 'rb') as f:
                self.data = pickle.load(f).to(self.device)
            print(self.data.x.device, self.data.y.device, self.data.edge_index.device,
                  self.data.train_mask.device, self.data.val_mask.device, self.data.test_mask.device)
        elif dataset.lower() == 'reddit':
            path = 'data/{}/subsubg{}_{}copynode.pkl'.format(
                dataset, self.id, ''if copy_node else'un')
            with open(path, 'rb') as f:
                self.data = pickle.load(f)
                for idx in range(len(self.data)):
                    self.data[idx].to(self.device)
        self.cal_rate(copy_node)
        return

    def work(self):
        '''
        connect to controller
        loop: process command
        :return:
        '''
        self.socket.connect(self.ip_port)
        command = None
        while self.is_continue(command):
            # print('client{} get {}'.format(self.id,command))
            # ;print('client{} recv "{}",processing...'.format(self.id,command))
            command = self.recv_with_res()
            # ;print('client{} processes "{}" over'.format(self.id,command))
            self.process(command)
        self.socket.close()
        print('client{} closes'.format(self.id))
        # self.end_analyse()
        return

    def is_continue(self, command):
        '''
        'ending' denotes the end of connection
        :param command:
        :return:
        '''
        if command == None:
            return True
        if command.lower() == 'ending':
            return False
        return True

    def process(self, command):
        if command == 'dataset':
            self.process_dataset()
        elif command == 'val':
            self.process_val()
        elif command == 'loss':
            self.process_loss()
        return

    def process_dataset(self):
        '''
        configure for a dataset with its: nfeat,nclass,nnode,ntrain_node,nval_node,ntest_node
        :return:
        '''
        self.dataset_name = self.recv_with_res()
        if self.dataset_name.lower() == 'cora':
            self.nfeat, self.nclass = 1433, 7
            self.nnode = 2708
            self.num_train_node, self.num_val_node, self.num_test_node = 140, 500, 1000
        elif self.dataset_name.lower() == 'citeseer':
            self.nfeat, self.nclass = 3703, 6
            self.nnode = 3327
            self.num_train_node, self.num_val_node, self.num_test_node = 120, 500, 1000
        elif self.dataset_name.lower() == 'pubmed':
            self.nfeat, self.nclass = 500, 3
            self.nnode = 19717
            self.num_train_node, self.num_val_node, self.num_test_node = 60, 500, 1000
        elif self.dataset_name.lower() == 'corafull':
            self.nfeat, self.nclass = 8710, 70
            self.nnode = 19793
            self.num_train_node, self.num_val_node, self.num_test_node = 1395, 500, 1000
        elif self.dataset_name.lower() == 'physics':
            self.nfeat, self.nclass = 8415, 5
            self.nnode = 34493
            self.num_train_node, self.num_val_node, self.num_test_node = 100, 500, 1000

        elif self.dataset_name.lower() == 'reddit':
            self.nfeat, self.nclass = 602, 41
            self.nnode = 232965
            self.num_train_node, self.num_val_node, self.num_test_node = 153431, 23831, 55703
        return

    def process_model(self):
        '''
        configure model{name,parameter},
        :return:
        '''
        print('process model')
        model_name = self.recv_with_res()
        model = eval(model_name)(self.nfeat, self.nclass)
        param_dict = self.recv_with_res()
        # chuange the parameters of model
        for name, param in param_dict.items():
            if not name.__contains__('.') and (name in ['alpha', 'gamma'] or name.__contains__('beta')):
                exec('model.{} = param'.format(name))
                continue
            name += 'ending'
            slots = name.split('.')  # ;print('slots',slots)
            # ;print('layer',layer)
            layer = name.replace('.{}'.format(slots[len(slots)-1]), '')
            if slots[len(slots)-2].isdigit():
                layer = layer.replace('.{}'.format(
                    slots[len(slots)-2]), '[{}]'.format(slots[len(slots)-2]))
            label = slots[len(slots)-1].replace('ending', '')
            exec("model.{}._parameters['{}'] = param".format(layer, label))
        model = model.to(self.device)
        self.configure(model, self.dataset_name)
        return

    def cal_rate(self, copy_node):
        '''
        calculate the proportion of the {train,val,test} set and the complete dataset's
        :param copy_node:
        :return:
        '''
        if self.dataset_name.lower() in ['cora', 'citeseer', 'pubmed', 'corafull', 'physics']:
            # path = 'data/reddit/subg_{}copynode.pkl'.format('' if copy_node else 'un')
            # with open(path, 'rb') as f:
            #     data = pickle.load(f)
            self.train_rate = float(
                torch.sum(self.data.train_mask))/self.num_train_node
            self.val_rate = float(
                torch.sum(self.data.val_mask))/self.num_val_node
            self.test_rate = float(
                torch.sum(self.data.test_mask))/self.num_test_node
        elif self.dataset_name.lower() == 'reddit':
            path = 'data/reddit/subg_{}copynode.pkl'.format(
                ''if copy_node else'un')
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.train_rate = float(
                torch.sum(data[self.id].train_mask)) / self.num_train_node
            self.val_rate = float(
                torch.sum(data[self.id].val_mask)) / self.num_val_node
            self.test_rate = float(
                torch.sum(data[self.id].test_mask)) / self.num_test_node
        # print(self.train_rate,self.val_rate,self.test_rate)
        # print(torch.sum(self.data.test_mask))
        return

    def get_grad_dict(self):
        '''as the name of function'''
        grad = {}
        for name, params in self.model.named_parameters():
            if not name.__contains__('.') and (name in ['alpha', 'gamma'] or name.__contains__('beta')):
                grad[name] = eval('self.model.{}.grad'.format(name))
                continue

            name += 'ending'
            slots = name.split('.')
            layer = name.replace('.{}'.format(slots[len(slots)-1]), '')
            name = name.replace('ending', '')
            if slots[len(slots)-2].isdigit():
                layer = layer.replace('.{}'.format(
                    slots[len(slots)-2]), '[{}]'.format(slots[len(slots)-2]))
            label = slots[len(slots)-1].replace('ending', '')
            grad[name] = eval(
                "self.model.{}._parameters['{}'].grad".format(layer, label))
            if not grad[name] == None:
                grad[name] *= self.train_rate
        return grad

    def update_grad(self, grad_dict):
        for name, grad in grad_dict.items():
            if not grad == None:
                grad = grad.to(self.device)

            if not name.__contains__('.') and (name in ['alpha', 'gamma'] or name.__contains__('beta')):
                exec('self.model.{}.grad = grad'.format(name))
                continue

            name += 'ending'
            slots = name.split('.')
            layer = name.replace('.{}'.format(slots[len(slots)-1]), '')
            if slots[len(slots)-2].isdigit():
                layer = layer.replace('.{}'.format(
                    slots[len(slots)-2]), '[{}]'.format(slots[len(slots)-2]))
            label = slots[len(slots)-1].replace('ending', '')
            exec("self.model.{}._parameters['{}'].grad = grad".format(
                layer, label))
        self.optimizer.step()
        return

    def process_val(self):
        return

    def process_loss(self):
        '''
        block process and send loss
        :return:
        '''
        self.recv()
        self.send(self.loss * self.train_rate)
        return

    def end_analyse(self):
        print(self.id)
        idx = 0
        for name, param in self.model.named_parameters():
            # print(name, param)
            idx += 1
            if idx > 1:
                break
        return


class ControllerSuperNet(Controller):
    def __init__(self, num_client):
        super(ControllerSuperNet, self).__init__(num_client)
        return

    def work(self, evo_epochs=EVO_EPOCH, sample_epochs=SAMPLE_EPOCH, num_pop=NUM_POP, sample_size=SAMPLE_SIZE):
        '''
        command sequence:
        supermasks
        loop:
            loop:train
            population
            loss
            val
        :param evo_epochs:
        :param sample_epochs:
        :param num_pop:
        :param sample_size:
        :return:
        '''
        Controller.work(self)
        supermasks = [utils.random_supermask() for i in range(num_pop)]
        self.supermasks = supermasks
        self.broadcast_with_waiting_res('supermasks')
        self.broadcast_with_waiting_res(self.supermasks)
        for epoch in range(evo_epochs):
            st_time = time.time()
            for sample_epoch in range(sample_epochs):
                self.broadcast_with_waiting_res('train')
                sample_supermasks = random.sample(self.supermasks, sample_size)
                self.broadcast(sample_supermasks)
                grad_dicts = self.aggregate()
                # self.broadcast_with_waiting_res(self.aggregate_grad(grad_dicts))
                self.broadcast(self.aggregate_grad(grad_dicts))
                self.blink_aggregate()
                # logger.info('sample epoch~{}'.format(sample_epoch))
                print('sample epoch~{}'.format(sample_epoch))
            self.broadcast_with_waiting_res('population')
            self.broadcast('get')  # ;print('client processing pop')
            populations = self.aggregate()
            print('controller evoing')
            self.evo()
            # print('controller evo over')
            self.update_pop2(0.5*pow(0.99, epoch), populations)
            # loss
            # print("brocasting loss")
            self.broadcast_with_waiting_res('loss')
            # print("brocast loss over")
            self.broadcast('get')
            # print("broadcase get")
            loss = 0
            losses = self.aggregate()
            # print("aggregate loss over")
            for idx in range(self.num_client):
                loss += losses[idx]
            print('train evo-epoch~{},loss={},use time:{}, current best supermask:{} with accu:{}\n'.format(
                epoch, loss, time.time() - st_time, self.best_supermask, self.best_accu))

            if DEBUG:
                for supermask in self.supermasks:
                    self.broadcast_with_waiting_res('val')
                    self.broadcast(supermask)
                    accu = 0
                    accus = self.aggregate()
                    for idx in range(self.num_client):
                        accu += float(accus[idx])
                    print('use sonnet {} with accu {} on val dataset'.format(
                        supermask, accu))
        self.get_best_supermask()
        return

    def aggregate_accu(self, supermask):
        self.broadcast_with_waiting_res('val')
        self.broadcast(supermask)
        accu = 0
        accus = self.aggregate()
        for idx in range(self.num_client):
            accu += float(accus[idx])
        return accu

    def evo(self):
        supermasks = self.supermasks
        num_pop = len(supermasks)
        t = 0.5
        volumn = int(num_pop * t)  # volumn
        reservoir = []
        while len(reservoir) < volumn:
            j = 0
            while j < volumn and j < num_pop:
                reservoir.append(supermasks[j])
                j += 1
            while j < num_pop:
                temp = randint(0, j)
                if temp < volumn:
                    reservoir[temp] = supermasks[j]
                j += 1

        new_supermasks = []
        l = 0.5
        for original in reservoir:
            if np.random.random() < l:
                son = utils.cross_over(original, reservoir[randint(0, volumn)])
                new_supermasks.append(son)
            else:
                # mutation
                son = utils.mutate(original)
                new_supermasks.append(son)

        result_supermasks = supermasks + new_supermasks
        result_supermasks = utils.setalize(result_supermasks)
        performance = []
        for supermask in result_supermasks:
            accu = self.aggregate_accu(supermask)
            if accu > self.best_accu:
                self.best_accu, self.best_supermask = accu, supermask
            performance.append(accu)
            print("supermask:{},accu:{}".format(supermask, accu), end=";")
        print()
        result = sorted(range(len(performance)), key=lambda k: performance[k])
        result.reverse()
        result = result[: num_pop]
        self.supermasks = [result_supermasks[idx] for idx in result]
        # print(performance)
        # print(result)
        # if DEBUG:
        #     # print(self.supermasks)
        #     print([performance[idx] for idx in result])
        # self.send(self.supermasks[0:int(self.train_rate * len(self.supermasks)) + 1])
        return

    def update_pop2(self, d, populations):
        # get the supermask from the controller
        num_reserved = int((1-d) * len(self.supermasks)) + 1
        # candidates = utils.setalize(self.supermasks)
        # performance = []
        # for supermask in candidates:
        #     performance.append(self.aggregate_accu(supermask))
        # result = sorted(range(len(performance)), key=lambda k: performance[k])
        # result.reverse()
        # result = result[: num_reserved]
        # candidates_from_controller = [candidates[idx] for idx in result]
        candidates_from_controller = self.supermasks[:num_reserved]

        # get the supermask from the clients
        # pop1 = populations[0]
        # pop2 = populations[1]
        # pop3 = populations[2]
        candidates = []
        num = 0
        idx = 0
        upper = len(self.supermasks)
        while num < upper:
            for i in range(self.num_client):
                if idx < len(populations[i]):
                    candidates.append(populations[i][idx])
                    num += 1
            idx += 1
        candidate = candidates_from_controller + utils.setalize(candidates)
        self.supermasks = candidate[0:len(self.supermasks)]
        return

    def update_pop(self, populations):
        pop1 = populations[0]
        pop2 = populations[1]
        pop3 = populations[2]
        candidate = []
        num = 0
        idx = 0
        upper = len(self.supermasks)
        while num < upper:
            if idx < len(pop1):
                candidate.append(pop1[idx])
                num += 1
            if idx < len(pop2):
                candidate.append(pop2[idx])
                num += 1
            if idx < len(pop3):
                candidate.append(pop3[idx])
                num += 1
            idx += 1
        self.supermasks = candidate[0:len(self.supermasks)]
        return

    def get_best_supermask(self):
        best_supermask = None
        best_val_accu = -1
        for supermask in self.supermasks:
            self.broadcast_with_waiting_res('val')
            self.broadcast(supermask)
            accus = self.aggregate()
            accu = 0
            for idx in range(self.num_client):
                accu += float(accus[idx])
            if accu > best_val_accu:
                best_val_accu = accu
                best_supermask = supermask
        self.output = best_supermask
        return
    # def output_best_population(self):
    #     self.output = self.supermasks
    #     return


class ClientSuperNet(Client):
    def __init__(self, id):
        Client.__init__(self, id)
        return

    def process(self, command):
        Client.process(self, command)
        if command == 'model':
            self.process_model()
        if command == 'supermasks':
            self.process_supermasks()
        if command == 'train':
            self.process_train()
        elif command == 'population':
            self.process_population()
        return

    def process_supermasks(self):
        self.supermasks = self.recv_with_res()
        return

    def process_train(self):
        self.model.train()
        supermasks = self.recv()  # ;print('client~{} get supermasks'.format(self.id))
        sm_idx = 0  # ;st_time=time.time()
        for supermask in supermasks:
            y_predict = self.model(
                self.data.x, self.data.edge_index, supermask)
            if sm_idx == 0:
                exec('loss_{}=F.cross_entropy(y_predict[self.data.train_mask],self.data.y[self.data.train_mask])'.format(
                    sm_idx))
            else:
                exec(
                    'loss_{}=loss_{} + F.cross_entropy(y_predict[self.data.train_mask],self.data.y[self.data.train_mask])'.format(sm_idx, sm_idx - 1))
            sm_idx += 1
        # loss = eval('loss_{}'.format(sm_idx - 1)) / len(supermasks)
        # ;print('factor{}'.format(factor),time.time()-st_time)
        factor = len(supermasks)
        loss = eval('loss_{}'.format(sm_idx - 1)) / \
            factor  # ;st_time=time.time()
        self.optimizer.zero_grad()
        loss.backward()  # ;print(time.time()-st_time)
        grad = self.get_grad_dict()
        self.send(grad)  # ;print('client~{} send out grad'.format(self.id))
        aggr_grad = self.recv_with_res()  # ;print('client~{} recv grad'.format(self.id))
        self.update_grad(aggr_grad)
        # self.end_analyse()
        self.loss = loss.item()
        # print('client~{} has loss:{}'.format(self.id,loss.item()))
        return

    def process_population(self):
        def val(supermask):
            self.model.eval()
            return utils.accuracy(self.model(self.data.x, self.data.edge_index, supermask)[self.data.val_mask], self.data.y[self.data.val_mask])
        self.recv()
        supermasks = self.supermasks
        num_pop = len(supermasks)
        t = 0.5
        volumn = int(num_pop * t)  # volumn
        reservoir = []
        while len(reservoir) < volumn:
            j = 0
            while j < volumn and j < num_pop:
                reservoir.append(supermasks[j])
                j += 1
            while j < num_pop:
                temp = randint(0, j)
                if temp < volumn:
                    reservoir[temp] = supermasks[j]
                j += 1
        # print('pop 1')
        new_supermasks = []
        l = 0.5
        for original in reservoir:
            if np.random.random() < l:
                son = utils.cross_over(original, reservoir[randint(0, volumn)])
                new_supermasks.append(son)
            else:
                # mutation
                son = utils.mutate(original)
                new_supermasks.append(son)
        # print('pop 2')
        result_supermasks = supermasks + new_supermasks
        result_supermasks = utils.setalize(result_supermasks)
        performance = []
        for supermask in result_supermasks:
            performance.append(val(supermask))
        # print('pop 3')
        result = sorted(range(len(performance)), key=lambda k: performance[k])
        result.reverse()
        result = result[: num_pop]
        self.supermasks = [result_supermasks[idx] for idx in result]
        # print(performance)
        # print(result)
        # if DEBUG:
        #     # print(self.supermasks)
        #     print([performance[idx] for idx in result])
        self.send(self.supermasks[0:int(
            self.train_rate * len(self.supermasks)) + 1])
        return

    def process_val(self):
        supermask = self.recv()
        self.model.eval()
        accu = utils.accuracy(self.model(self.data.x, self.data.edge_index, supermask)[self.data.val_mask],
                              self.data.y[self.data.val_mask])
        # print(accu,self.val_rate)
        self.send(accu * self.val_rate)
        return


class ControllerCommonNet(Controller):
    def __init__(self, num_client):
        super(ControllerCommonNet, self).__init__(num_client)
        return

    def work(self, epochs=200):
        Controller.work(self)
        optval = -1
        for epoch in range(epochs):
            self.broadcast_with_waiting_res('train')
            # self.broadcast(pickle.dumps('train'))#;print('controller broadcasts command:"{}" over'.format('train'))
            self.broadcast('get')
            grad_dicts = self.aggregate()  # ;print('controller aggregates grads over')
            # self.broadcast_with_waiting_res(self.aggregate_grad(grad_dicts))#;print('controller broadcasts grads over')
            self.broadcast(self.aggregate_grad(grad_dicts))
            self.blink_aggregate()  # ;print('controller broadcasts grads over')

            # save model
            self.broadcast_with_waiting_res('val')
            self.broadcast('get')
            accu = 0
            accus = self.aggregate()
            for idx in range(self.num_client):
                accu += float(accus[idx])
            if accu > optval:
                optval = accu
                torch.save(self.model.state_dict(), 'model.pth')
            print(epoch)

            # val
            if DEBUG:
                if epoch % 20 == 0:
                    self.broadcast_with_waiting_res('val')
                    self.broadcast('get')
                    accu = 0
                    accus = self.aggregate()
                    for idx in range(self.num_client):
                        accu += accus[idx]
                    self.broadcast_with_waiting_res('loss')
                    self.broadcast('get')
                    loss = 0
                    losses = self.aggregate()
                    for idx in range(self.num_client):
                        loss += losses[idx]
                    print('train epoch~{} with accu {} on val dataset,loss={}'.format(
                        epoch, accu, loss))

        self.model.load_state_dict(torch.load('model.pth'))
        # get metrics
        res = {}
        st_time = time.time()
        self.broadcast_with_waiting_res('model')
        self.broadcast_with_waiting_res(self.model_name)
        self.broadcast(utils.serialize_model(self.model))
        self.blink_aggregate()
        self.broadcast_with_waiting_res('test')
        self.broadcast('get')
        accu = 0
        accus = self.aggregate()
        ed_time = time.time()

        for idx in range(self.num_client):
            accu += float(accus[idx])

        params = list(self.model.named_parameters())
        k = 0
        for name, i in params:
            l = 1
            for j in i.size():
                l *= j
            k = k + l
        num_params = k

        res['accu'] = accu
        res['time'] = ed_time - st_time
        res['num_params'] = num_params
        return res


class ClientCommonNet(Client):
    def __init__(self, id):
        Client.__init__(self, id)
        return

    def process(self, command):
        answer = Client.process(self, command)
        if command == 'model':
            self.process_model()
        elif command == 'train':
            self.process_train()
        elif command == 'test':
            self.process_test()
        return answer

    def process_train(self):
        self.recv()
        self.model.train()  # ;print('client{} change state to train'.format(self.id))
        # ;print('client{} get output of model'.format(self.id))
        y_predict = self.model(self.data.x, self.data.edge_index)
        # ;print('client{} get loss'.format(self.id))
        loss = F.cross_entropy(
            y_predict[self.data.train_mask], self.data.y[self.data.train_mask])
        self.optimizer.zero_grad()  # ;print('client{} zero grad'.format(self.id))
        loss.backward()  # ;print('client{} backward'.format(self.id))
        grad = self.get_grad_dict()  # ;print('client{} get grad'.format(self.id))
        self.send(grad)  # ;print('client{} upload grad'.format(self.id))
        # ;print('client{} get aggr_grad'.format(self.id))
        aggr_grad = self.recv_with_res()
        # ;print('client{} update model'.format(self.id))
        self.update_grad(aggr_grad)
        self.loss = loss.item()
        # print('client~{} has loss:{}'.format(self.id,loss.item()))
        return

    def process_val(self):
        self.recv()
        self.model.eval()
        accu = utils.accuracy(self.model(self.data.x, self.data.edge_index)[
                              self.data.val_mask], self.data.y[self.data.val_mask])
        print(accu, self.val_rate)
        self.send(accu * self.val_rate)
        return

    def process_test(self):
        self.recv()
        self.model.eval()
        accu = utils.accuracy(self.model(self.data.x, self.data.edge_index)[
                              self.data.test_mask], self.data.y[self.data.test_mask])
        # print(utils.num_correct(self.model(self.data.x, self.data.edge_index)[self.data.test_mask],self.data.y[self.data.test_mask]))
        self.send(accu * self.test_rate)
        return


class ControllerDarts(Controller):
    def __init__(self, num_client):
        super(ControllerDarts, self).__init__(num_client)
        return

    def work(self, evo_epochs=125, sample_epochs=SAMPLE_EPOCH, num_pop=NUM_POP, sample_size=SAMPLE_SIZE):
        '''
        command sequence:
        supermasks
        loop:
            loop:train
            population
            loss
            val
        :param evo_epochs:
        :param sample_epochs:
        :param num_pop:
        :param sample_size:
        :return:
        '''
        Controller.work(self)
        for epoch in range(evo_epochs):
            st_time = time.time()
            # for sample_epoch in range(sample_epochs):
            self.broadcast_with_waiting_res('train')
            self.broadcast('get')
            grad_dicts = self.aggregate()
            self.broadcast(self.aggregate_grad(grad_dicts))
            self.blink_aggregate()
            # print('sample epoch~{}'.format(sample_epoch))
            # print('controller evoing')
            # loss
            self.broadcast_with_waiting_res('loss')
            self.broadcast('get')
            loss = 0
            losses = self.aggregate()
            for idx in range(self.num_client):
                loss += losses[idx]
            print('train -epoch~{},loss={},use time:{}'.format(epoch,
                  loss, time.time() - st_time))
            if DEBUG:
                self.broadcast_with_waiting_res('val')
                self.broadcast('get')
                accu = 0
                accus = self.aggregate()
                for idx in range(self.num_client):
                    accu += float(accus[idx])
                print('accu {} on val dataset'.format(accu))
        return

    def aggregate_accu(self, supermask):
        self.broadcast_with_waiting_res('val')
        self.broadcast(supermask)
        accu = 0
        accus = self.aggregate()
        for idx in range(self.num_client):
            accu += float(accus[idx])
        return accu


class ClientDarts(Client):
    def __init__(self, id):
        Client.__init__(self, id)
        return

    def process(self, command):
        Client.process(self, command)
        if command == 'model':
            self.process_model()
        if command == 'train':
            self.process_train()
        return

    def configure(self, model, dataset, copy_node=COPY_NODE):
        self.model = model
        self.optimizer = optim.Adam(
            model.get_parameters(), lr=LR, weight_decay=5e-6)
        self.a_optimizer = optim.Adam(
            model.get_arc_params(), lr=LR, weight_decay=5e-6)
        self.loss = None
        path = ''
        if dataset.lower() in ['cora', 'citeseer', 'pubmed', 'corafull', 'physics']:
            path = 'data/{}/{}_{}copynode.pkl'.format(
                dataset, self.id, ''if copy_node else'un')
            with open(path, 'rb') as f:
                self.data = pickle.load(f).to(self.device)
            print(self.data.x.device, self.data.y.device, self.data.edge_index.device,
                  self.data.train_mask.device, self.data.val_mask.device, self.data.test_mask.device)
        elif dataset.lower() == 'reddit':
            path = 'data/{}/subsubg{}_{}copynode.pkl'.format(
                dataset, self.id, ''if copy_node else'un')
            with open(path, 'rb') as f:
                self.data = pickle.load(f)
                for idx in range(len(self.data)):
                    self.data[idx].to(self.device)
        self.cal_rate(copy_node)
        return

    def process_train(self):
        self.model.train()
        self.recv()  # ;print('client~{} get supermasks'.format(self.id))

        y_predict = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(
            y_predict[self.data.train_mask], self.data.y[self.data.train_mask])
        self.optimizer.zero_grad()
        loss.backward()  # ;print(time.time()-st_time)

        y_predict = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(
            y_predict[self.data.train_mask], self.data.y[self.data.train_mask])
        self.a_optimizer.zero_grad()
        loss.backward()  # ;print(time.time()-st_time)

        grad = self.get_grad_dict()
        self.send(grad)  # ;print('client~{} send out grad'.format(self.id))
        aggr_grad = self.recv_with_res()  # ;print('client~{} recv grad'.format(self.id))
        self.update_grad(aggr_grad)
        # self.end_analyse()
        self.loss = loss.item()
        # print('client~{} has loss:{}'.format(self.id,loss.item()))
        return

    def process_val(self):
        self.recv()
        self.model.eval()
        accu = utils.accuracy(self.model(self.data.x, self.data.edge_index)[self.data.val_mask],
                              self.data.y[self.data.val_mask])
        self.send(accu * self.val_rate)
        return


class ControllerFedNas(Controller):
    def __init__(self, num_client):
        super(ControllerFedNas, self).__init__(num_client)
        return

    def work(self, evo_epochs=250, sample_epochs=SAMPLE_EPOCH, num_pop=NUM_POP, sample_size=SAMPLE_SIZE):
        '''
        command sequence:
        supermasks
        loop:
            loop:train
            population
            loss
            val
        :param evo_epochs:
        :param sample_epochs:
        :param num_pop:
        :param sample_size:
        :return:
        '''
        Controller.work(self)
        for epoch in range(evo_epochs):
            st_time = time.time()
            # for sample_epoch in range(sample_epochs):
            self.broadcast_with_waiting_res('train')
            self.broadcast('get')
            grad_dicts = self.aggregate()
            self.broadcast(self.aggregate_grad(grad_dicts))
            self.blink_aggregate()
            # print('sample epoch~{}'.format(sample_epoch))
            # print('controller evoing')
            # loss
            self.broadcast_with_waiting_res('loss')
            self.broadcast('get')
            loss = 0
            losses = self.aggregate()
            for idx in range(self.num_client):
                loss += losses[idx]
            print('train -epoch~{},loss={},use time:{}'.format(epoch,
                  loss, time.time() - st_time))
            if DEBUG:
                self.broadcast_with_waiting_res('val')
                self.broadcast('get')
                accu = 0
                accus = self.aggregate()
                for idx in range(self.num_client):
                    accu += float(accus[idx])
                print('accu {} on val dataset'.format(accu))

        return

    def aggregate_accu(self, supermask):
        self.broadcast_with_waiting_res('val')
        self.broadcast(supermask)
        accu = 0
        accus = self.aggregate()
        for idx in range(self.num_client):
            accu += float(accus[idx])
        return accu


class ClientFedNas(Client):
    def __init__(self, id):
        Client.__init__(self, id)
        self.lamda = 0.1
        return

    def process(self, command):
        Client.process(self, command)
        if command == 'model':
            self.process_model()
        if command == 'train':
            self.process_train()
        return

    def configure(self, model, dataset, copy_node=COPY_NODE):
        self.model = model
        self.optimizer = optim.Adam(
            model.parameters(), lr=LR, weight_decay=5e-6)
        self.a_optimizer = optim.Adam(
            model.get_arc_params(), lr=LR, weight_decay=5e-6)
        self.loss = None
        path = ''
        if dataset.lower() in ['cora', 'citeseer', 'pubmed', 'corafull', 'physics']:
            path = 'data/{}/{}_{}copynode.pkl'.format(
                dataset, self.id, ''if copy_node else'un')
            with open(path, 'rb') as f:
                self.data = pickle.load(f).to(self.device)
            print(self.data.x.device, self.data.y.device, self.data.edge_index.device,
                  self.data.train_mask.device, self.data.val_mask.device, self.data.test_mask.device)
        elif dataset.lower() == 'reddit':
            path = 'data/{}/subsubg{}_{}copynode.pkl'.format(
                dataset, self.id, ''if copy_node else'un')
            with open(path, 'rb') as f:
                self.data = pickle.load(f)
                for idx in range(len(self.data)):
                    self.data[idx].to(self.device)
        self.cal_rate(copy_node)
        return

    def process_train(self):
        self.model.train()
        self.recv()  # ;print('client~{} get supermasks'.format(self.id))

        # calculate the loss of training data
        y_predict = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(
            y_predict[self.data.train_mask], self.data.y[self.data.train_mask])
        self.optimizer.zero_grad()
        loss.backward()  # ;print(time.time()-st_time)

        y_predict = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(
            y_predict[self.data.train_mask], self.data.y[self.data.train_mask])
        self.a_optimizer.zero_grad()
        loss.backward()  # ;print(time.time()-st_time)

        grads = self.get_grad_dict()

        # calculate the loss of the eval data
        y_predict = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(
            y_predict[self.data.val_mask], self.data.y[self.data.val_mask])
        self.optimizer.zero_grad()
        loss.backward()  # ;print(time.time()-st_time)

        y_predict = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(
            y_predict[self.data.val_mask], self.data.y[self.data.val_mask])
        self.a_optimizer.zero_grad()
        loss.backward()  # ;print(time.time()-st_time)

        grad_eval = self.get_grad_dict()

        # merge the grad of training data and val data
        for name, grad in grads.items():
            if grads[name] == None:
                continue
            else:
                grads[name] += self.lamda * grad_eval[name]

        self.send(grads)  # ;print('client~{} send out grad'.format(self.id))
        aggr_grad = self.recv_with_res()  # ;print('client~{} recv grad'.format(self.id))
        self.update_grad(aggr_grad)
        # self.end_analyse()
        self.loss = loss.item()
        # print('client~{} has loss:{}'.format(self.id,loss.item()))
        return

    def process_val(self):
        self.recv()
        self.model.eval()
        accu = utils.accuracy(self.model(self.data.x, self.data.edge_index)[self.data.val_mask],
                              self.data.y[self.data.val_mask])
        self.send(accu * self.val_rate)
        return

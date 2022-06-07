import random
import multiprocessing as mp
import time
import pickle
import os
from federal import ControllerCommonNet, ClientCommonNet

dataset_name = 'Physics'


def server_run(supermask, epoch, dataset):
    nfeat, nclass = 0, 0
    if dataset == 'cora':
        nfeat, nclass = 1433, 7
    elif dataset == 'citeseer':
        nfeat, nclass = 3703, 6
    elif dataset == 'pubmed':
        nfeat, nclass = 500, 3
    elif dataset == 'corafull':
        nfeat, nclass = 8710, 70
    elif dataset == 'Physics':
        nfeat, nclass = 8415, 5
    client = 3
    if dataset == "Physics":
        client = 8
    with open('tmp.pkl', 'wb') as f:
        pickle.dump(supermask, f)
    controller = ControllerCommonNet(client)
    controller.configure('SonNet', dataset, nfeat, nclass)
    res = controller.work(epochs=epoch)
    print('Test on sonnet of {}, get the result as\n{}'.format(dataset, res))
    controller.broadcast_with_waiting_res('ending')
    controller.close()
    path = "rl_"+dataset+"_accu.pkl"
    with open(path, 'wb') as f:
        pickle.dump(res["accu"], f)
    return


def client_run(dataset):
    clients = []
    client = 3
    if dataset == "Physics":
        client = 8
    for j in range(client):
        clients.append(ClientCommonNet(j))
    processes = []
    for client in clients:
        process = mp.Process(target=client.work)
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    return


def getAcc(supermask, epoch=5):

    server = mp.Process(target=server_run, args=[
                        supermask, epoch, dataset_name])
    server.start()
    time.sleep(10)
    client = mp.Process(target=client_run, args=[dataset_name])
    client.start()
    server.join()

    # read accuracy
    path = "rl_"+dataset_name+"_accu.pkl"
    with open(path, 'rb') as f:
        acc = pickle.load(f)
    return acc


class q_learning:

    def __init__(self, shape):
        """
        initialize a q-learning model
        """

        self.table = {}
        actionNum = 1
        for node in shape:
            layer = node[0]
            if layer == 1:
                # first layer
                for action in range(actionNum):
                    self.table[(layer-1, action)] = [0]*node[1]
            elif layer == len(shape):
                for action in range(1, actionNum+1):
                    self.table[(layer-1, action)] = [0]*node[1]
                self.table[(layer-1, 0)] = [0]
            else:
                for action in range(1, actionNum+1):
                    self.table[(layer-1, action)] = [0]*(node[1]+1)
                self.table[(layer-1, 0)] = [0]
            actionNum = node[1]

        self.n = len(shape)
        # epsilon greedy
        self.epsilon = 0.5
        # alpha q table
        self.alpha = 0.5

    def get_best_action(self, state):
        """
        get best action 
        """
        actions = self.table[state]
        index = actions.index(max(actions))
        return index

    def get_action(self, state, epsilon=0.5):
        """
        use epsilon greedy algorithm to choose an action from state's actions
        """
        actions = self.table[state]
        if random.random() > epsilon:
            # then choose action with max q value
            index = actions.index(max(actions))
            return index
        # randomly choose an action
        index = random.randint(0, len(actions)-1)
        print("actions:{}".format(actions))
        print("randomint:{}".format(random.randint(0, len(actions)-1)))
        print("randomint:{}".format(random.randint(0, len(actions)-1)))
        print("randomint:{}".format(random.randint(0, len(actions)-1)))
        return index

    def update(self, reward, state, action, next_state, states=None):
        """
        use reward update q[state][action]
        we do reward shape with states when arrive at last layer
        states not contain state, otherState contains [nodeindex, action]
        """
        # q = q + alpha[reward + max(q') -q]
        # reward shape
        if state[0] == self.n-1:
            meanReward = reward / (len(states)+1)
            for otherState in states:
                self.table[otherState[:2]][otherState[-1]] = self.table[otherState[:2]][otherState[-1]] + self.alpha*(
                    meanReward+max(self.table[(otherState[0]+1, otherState[2])])-self.table[otherState[:2]][otherState[-1]])
            self.table[state][action] = self.table[state][action] + \
                self.alpha*(meanReward-self.table[state][action])
        else:
            self.table[state][action] = self.table[state][action] + self.alpha * \
                (reward+max(self.table[next_state])-self.table[state][action])

    def output(self):
        """
        output the best result
        """


class env:
    def __init__(self, shape):
        """
        docstring
        """
        self.state = (0, 0)
        self.reward = 0
        self.done = False
        self.shape = shape
        # trace is states contain state like (nodeid, las_action, action)
        self.trace = []
        self.isShape = False
        self.trainNum = 0

    def reset(self):
        """
        reset the environment
        """
        self.state = (0, 0)
        self.done = False
        self.reward = 0
        self.trace = []
        self.isShape = False
        return self.state

    def mov(self, action):
        """
        interact with environment
        return reward,next state
        """
        # record trace
        self.trace.append((self.state[0], self.state[1], action))
        # update state
        self.state = (self.state[0]+1, action)

        if self.state[0] == len(self.shape):
            # reach final state
            trace = [item[2] for item in self.trace]
            trace[0] += 1
            trace[-1] += 1
            self.trace = self.trace[:-1]
            self.trainNum += 1
            print("training trace: {},trainNum: {}".format(trace, self.trainNum))
            self.reward = getAcc(trace)
            self.done = True
            self.isShape = True
        else:
            self.reward = 0


def createShape(raw):
    """
    produce shape automatically
    raw: [1,2,3] choices list
    """
    shape = []
    number = 1
    for i in raw:
        item = [number, i]
        number = number*i
        shape.append(item)
    return shape


def readResult():
    with open("./result", "rb") as result:
        actions = pickle.load(result)
        acc = pickle.load(result)
        time = pickle.load(result)
        print(acc)
        print(actions)
        print(time)
        return (actions, acc, time)


def writeResult(actions, acc, time):
    with open("./result", 'wb') as result:
        pickle.dump(actions, result)
        pickle.dump(acc, result)
        pickle.dump(time, result)


def save_model(env, model, num):
    if os.path.exists('rl_model.pkl'):
        with open('rl_model.pkl', 'rb') as f:
            map = pickle.load(f)
    else:
        map = dict()

    with open('rl_model.pkl', 'wb') as f:
        map[num] = (env, model)
        pickle.dump(map, f)


def read_model():
    with open('rl_model.pkl', 'rb') as f:
        map = pickle.load(f)
    return map


if __name__ == '__main__':
    startTime = time.time()
    mp.set_start_method('spawn')
    random.seed(719)

    shape = [[1, 5], [2, 12], [3, 24], [4, 36],
             [5, 48], [6, 60], [7, 72], [8, 5]]
    model = q_learning(shape)
    envi = env(shape)
    episodes = 1000

    for i in range(episodes):
        print("RL: episodes = {}".format(i))
        state = envi.reset()
        epsilon = epsilon = 0.9/(episodes ** 2) * (episodes ** 2 - i**2) + 0.1
        while True:
            action = model.get_action(state, epsilon=epsilon)
            envi.mov(action)
            if envi.isShape:
                model.update(envi.reward, state, action,
                             envi.state, envi.trace)
            else:
                model.update(envi.reward, state, action, envi.state)
            if envi.done == True:
                break
            state = envi.state

        if i % 100 == 0:
            save_model(envi, model, i)
            print("{} step timeLength".format(i), time.time()-startTime)
        if time.time()-startTime - 10*(i+1) > 6*60*60:
            print("end episodes", i)
            break

    envi.reset()
    state = (0, 0)
    actions = []
    for i in range(len(shape)):
        action = model.get_best_action(state)
        actions.append(action)
        state = (state[0]+1, action)
    actions[0] = actions[0]+1
    actions[-1] = actions[-1]+1
    endTime = time.time()
    TimeLength = endTime-startTime
    acc = getAcc(actions, 50)
    print(acc)
    print(actions)
    print(TimeLength)
    writeResult(actions=actions, acc=acc, time=TimeLength)
